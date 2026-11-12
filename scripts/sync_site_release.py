"""Show where the published site, INSTALL and CITATION.cff lag behind a release.

After a release the site, ``docs/INSTALL.md`` and the screenshots can stay on the
old version for days (0.21.0 shipped while the site still said 0.20.2). This
script reads the version the world sees (PyPI, the latest GitHub Release, the
Zenodo concept record), counts the version literals in the places that repeat
it, and prints the diff a maintainer would apply. It never writes a file and
never calls the site's repository: ``--dry-run`` is the only mode, kept as a
flag so the command in ``docs/dev/RELEASING.md`` says what it does.

``--warn`` is for CI: when the site lacks the version it prints a GitHub
``::warning::`` annotation and still exits 0, so the release job stays green.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

PYPI_URL = "https://pypi.org/pypi/visionforge-studio/json"
RELEASE_URL = "https://api.github.com/repos/marcus-vreis/VisionForge/releases/latest"
ZENODO_URL = "https://zenodo.org/api/records/23425765/versions/latest"
SITE_URL = "https://marcus-vreis.github.io/visionforge-site/"
TIMEOUT_SECONDS = 20
# Releases are 0.x.y until 1.0; the pattern skips Python, CUDA and Node versions (3.13.0,
# 12.8.0) that also sit in INSTALL.md. Widen it when the package reaches 1.0.
VERSION_PATTERN = re.compile(r"(?<![\d.])0\.\d+\.\d+(?![\d.])")
NOISE = frozenset({"0.0.0"})

Fetch = Callable[[str], str]


def fetch_text(url: str) -> str:
    """GET ``url`` and return the body; the one place that touches the network."""
    request = urllib.request.Request(  # noqa: S310 - fixed https URLs above
        url, headers={"User-Agent": "visionforge-release-check"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        body: bytes = response.read()
    return body.decode("utf-8", errors="replace")


def _json_field(fetch: Fetch, url: str, *path: str) -> str | None:
    """Return a string at ``path`` of the JSON body, or None if unreachable or absent."""
    try:
        node: object = json.loads(fetch(url))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, str) else None


def published_facts(fetch: Fetch) -> dict[str, str | None]:
    """What PyPI, GitHub and Zenodo report for the latest release (None if unknown)."""
    tag = _json_field(fetch, RELEASE_URL, "tag_name")
    return {
        "pypi": _json_field(fetch, PYPI_URL, "info", "version"),
        "github release": tag.removeprefix("v") if tag else None,
        "zenodo doi": _json_field(fetch, ZENODO_URL, "doi"),
    }


def version_counts(text: str) -> dict[str, int]:
    """How many times each release-like version literal appears in ``text``."""
    counts: dict[str, int] = {}
    for found in VERSION_PATTERN.findall(text):
        if found not in NOISE:
            counts[found] = counts.get(found, 0) + 1
    return counts


def stale_lines(name: str, text: str, target: str) -> list[str]:
    """Diff lines for one document: each other version that should become ``target``."""
    return [
        f"  - {name}: {old} -> {target} ({times}x)"
        for old, times in sorted(version_counts(text).items())
        if old != target
    ]


def local_documents(root: Path) -> dict[str, str]:
    """The checkout files that repeat the version, by name (missing ones are skipped)."""
    names = ("CITATION.cff", "docs/INSTALL.md")
    return {
        n: (root / n).read_text(encoding="utf-8") for n in names if (root / n).is_file()
    }


class Report(NamedTuple):
    """The printed lines plus what the exit status and the warning depend on."""

    lines: list[str]
    target: str | None
    site_has_target: bool
    has_diff: bool


def build_report(fetch: Fetch, root: Path, target: str | None) -> Report:
    """Compare the published versions with the checkout and the site home page."""
    facts = published_facts(fetch)
    resolved = target or facts["pypi"]
    lines = ["Published:"] + [f"  {k}: {v or 'unavailable'}" for k, v in facts.items()]
    if resolved is None:
        lines.append("No target version: pass --version or check PyPI.")
        return Report(lines, None, False, True)
    lines.append(f"Target version: {resolved}")
    documents = local_documents(root)
    diff: list[str] = []
    try:
        site = fetch(SITE_URL)
    except (urllib.error.URLError, OSError):
        site = None
        diff.append(f"  ! site {SITE_URL}: unreachable")
    site_has_target = site is not None and resolved in version_counts(site)
    if site is not None:
        documents["site (home)"] = site
        if not version_counts(site):
            diff.append("  ! site (home): shows no version at all")
    for name, text in documents.items():
        diff.extend(stale_lines(name, text, resolved))
    lines.append(
        "Proposed changes (nothing was written):" if diff else "Nothing to change."
    )
    return Report(lines + diff, resolved, site_has_target, bool(diff))


def main(argv: list[str] | None = None, fetch: Fetch = fetch_text) -> int:
    """Print the report; with ``--warn`` turn a missing site version into a warning."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="print only (the only mode)"
    )
    parser.add_argument("--version", help="target version (default: latest on PyPI)")
    parser.add_argument(
        "--warn", action="store_true", help="emit a ::warning:: and exit 0"
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="checkout to read"
    )
    args = parser.parse_args(argv)
    report = build_report(fetch, args.root, args.version)
    print("\n".join(report.lines))
    if args.warn:
        if not report.site_has_target:
            shown = report.target or "the latest release"
            print(
                f"::warning::The site {SITE_URL} does not show version {shown} yet; "
                "follow the post-release checklist in docs/dev/RELEASING.md."
            )
        return 0
    return 1 if report.has_diff or not report.site_has_target else 0


if __name__ == "__main__":
    sys.exit(main())
