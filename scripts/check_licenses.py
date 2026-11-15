"""Fail on a copyleft runtime dependency that nobody reviewed (#225).

Walks the installed dependency closure of this project (``[project].dependencies``
plus the extras and packages named in ``scripts/license_allowlist.toml``), reads
each distribution's declared license, and fails when:

- a package declares a copyleft license and is not in the allowlist;
- an allowlisted package declares a copyleft family the allowlist does not name
  (its license changed);
- a package the closure needs is not installed (the check would be blind);
- the table in ``THIRD_PARTY_NOTICES.md`` differs from the allowlist.

``--write`` rewrites that table from the allowlist. Run it in an environment
that has the project and its extras installed::

    python scripts/check_licenses.py
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import re
import sys
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "scripts" / "license_allowlist.toml"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
PYPROJECT = ROOT / "pyproject.toml"
TABLE_START = "<!-- license-table:start (scripts/check_licenses.py --write) -->"
TABLE_END = "<!-- license-table:end -->"
_TABLE_HEADER = "| Package | License | Why it is installed |\n|---|---|---|\n"

# Family -> how it is written in license metadata. Order matters only for
# readability; each family is matched on its own.
_FAMILIES: dict[str, re.Pattern[str]] = {
    "AGPL": re.compile(r"AGPL|Affero", re.IGNORECASE),
    "LGPL": re.compile(r"LGPL|Lesser General Public", re.IGNORECASE),
    "GPL": re.compile(
        r"(?<![AL])GPL|(?<!Lesser )(?<!Affero )General Public License",
        re.IGNORECASE,
    ),
    "MPL": re.compile(r"\bMPL\b|MPL-|Mozilla Public", re.IGNORECASE),
    "EPL": re.compile(r"\bEPL\b|EPL-|Eclipse Public", re.IGNORECASE),
    "EUPL": re.compile(r"EUPL|European Union Public", re.IGNORECASE),
    "CDDL": re.compile(r"CDDL|Common Development and Distribution", re.IGNORECASE),
    "OSL": re.compile(r"\bOSL-|Open Software License", re.IGNORECASE),
    "CeCILL": re.compile(r"CeCILL", re.IGNORECASE),
    "OFL": re.compile(r"\bOFL\b|OFL-|Open Font License", re.IGNORECASE),
}
# A License field longer than this is the license *text* (it can quote other
# licenses it is compatible with); only the expression and classifiers count.
_SHORT_LICENSE = 100


@dataclass(frozen=True)
class Dist:
    """What the check needs from one installed distribution."""

    name: str
    families: frozenset[str]
    requires: tuple[str, ...]


Lookup = Callable[[str], Dist | None]


def copyleft_families(
    expression: str | None, license_field: str | None, classifiers: Iterable[str]
) -> frozenset[str]:
    """Copyleft families a distribution declares in its metadata."""
    texts = [c for c in classifiers if c.startswith("License ::")]
    if expression:
        texts.append(expression)
    elif license_field and "\n" not in license_field.strip():
        if len(license_field.strip()) <= _SHORT_LICENSE:
            texts.append(license_field)
    return frozenset(
        family
        for family, pattern in _FAMILIES.items()
        if any(pattern.search(text) for text in texts)
    )


def installed(name: str) -> Dist | None:
    """Read one distribution from this environment, or None if absent."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    meta = dist.metadata
    return Dist(
        name=canonicalize_name(meta["Name"]),
        families=copyleft_families(
            meta.get("License-Expression"),
            meta.get("License"),
            meta.get_all("Classifier") or [],
        ),
        requires=tuple(dist.requires or ()),
    )


def _wanted(requirement: Requirement, extras: Iterable[str]) -> bool:
    if requirement.marker is None:
        return True
    return any(requirement.marker.evaluate({"extra": e}) for e in extras)


def walk_closure(
    roots: Iterable[str], lookup: Lookup
) -> tuple[dict[str, Dist], list[str]]:
    """Every distribution the roots need here, and the ones that are missing."""
    found: dict[str, Dist] = {}
    missing: list[str] = []
    stack = [Requirement(r) for r in roots]
    while stack:
        requirement = stack.pop()
        name = canonicalize_name(requirement.name)
        if name in found or name in missing:
            continue
        dist = lookup(name)
        if dist is None:
            missing.append(name)
            continue
        found[name] = dist
        extras = requirement.extras or {""}
        for raw in dist.requires:
            child = Requirement(raw)
            if _wanted(child, extras):
                stack.append(child)
    return found, sorted(missing)


def roots_from(pyproject: dict[str, Any], closure: dict[str, Any]) -> list[str]:
    """``[project].dependencies``, the chosen extras and the extra packages."""
    project = pyproject["project"]
    roots = list(project["dependencies"])
    for extra in closure.get("extras", []):
        roots.extend(project["optional-dependencies"][extra])
    roots.extend(closure.get("extra_packages", []))
    return roots


def check_closure(found: dict[str, Dist], allowed: dict[str, Any]) -> list[str]:
    """Problems with the copyleft packages of the closure."""
    problems: list[str] = []
    for name, dist in sorted(found.items()):
        if not dist.families:
            continue
        entry = allowed.get(name)
        if entry is None:
            families = ", ".join(sorted(dist.families))
            problems.append(f"{name}: copyleft ({families}) and not in the allowlist")
            continue
        unexpected = dist.families - set(entry["families"])
        if unexpected:
            problems.append(
                f"{name}: now declares {', '.join(sorted(unexpected))}, "
                f"allowlisted for {', '.join(entry['families'])}"
            )
    return problems


def render_table(allowed: dict[str, Any]) -> str:
    """The notices table, one row per allowlisted package, in file order."""
    rows = "".join(
        f"| `{name}` | {entry['license']} | {entry['why']} |\n"
        for name, entry in allowed.items()
    )
    return _TABLE_HEADER + rows


def _split_notices(text: str) -> tuple[str, str, str]:
    start = text.index(TABLE_START) + len(TABLE_START) + 1
    end = text.index(TABLE_END)
    return text[:start], text[start:end], text[end:]


def check_notices(text: str, allowed: dict[str, Any]) -> list[str]:
    """Empty when the notices table is exactly what the allowlist renders."""
    if TABLE_START not in text or TABLE_END not in text:
        return [f"{NOTICES.name}: table markers not found"]
    _, table, _ = _split_notices(text)
    if table != render_table(allowed):
        return [f"{NOTICES.name}: table differs from the allowlist (run --write)"]
    return []


def write_notices(text: str, allowed: dict[str, Any]) -> str:
    """The notices with their table rendered from the allowlist."""
    head, _, tail = _split_notices(text)
    return head + render_table(allowed) + tail


def run(
    pyproject: dict[str, Any],
    allowlist: dict[str, Any],
    notices: str,
    lookup: Lookup | None = None,
) -> list[str]:
    """Every problem found; empty means the gate passes."""
    lookup = lookup or installed
    allowed: dict[str, Any] = {
        canonicalize_name(k): v for k, v in allowlist["packages"].items()
    }
    found, missing = walk_closure(
        roots_from(pyproject, allowlist.get("closure", {})), lookup
    )
    problems = [f"{name}: required but not installed" for name in missing]
    problems += check_closure(found, allowed)
    problems += check_notices(notices, allowlist["packages"])
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copyleft license gate (#225).")
    parser.add_argument(
        "--write", action="store_true", help="rewrite the notices table"
    )
    arguments = parser.parse_args(argv)
    allowlist = tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))
    notices = NOTICES.read_text(encoding="utf-8")
    if arguments.write:
        updated = write_notices(notices, allowlist["packages"])
        NOTICES.write_bytes(updated.encode("utf-8"))
        notices = updated
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    problems = run(pyproject, allowlist, notices)
    for problem in problems:
        print(f"license gate: {problem}", file=sys.stderr)
    if not problems:
        print(f"license gate: ok ({len(allowlist['packages'])} allowlisted)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
