"""The module map in ARCHITECTURE.md must match the modules in the package."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "visionforge"
ARCHITECTURE = ROOT / "docs" / "dev" / "ARCHITECTURE.md"
PYPROJECT = ROOT / "pyproject.toml"

# Modules written as `name.py` in the doc that are not part of the package.
PHRASE = "extras that install something"
NOT_IN_PACKAGE = {
    "my_task",  # example researcher task in user_tasks/
}


def _package_modules() -> set[str]:
    """Basenames of every tracked or on-disk module of the package."""
    modules = {
        path.stem
        for path in PACKAGE.rglob("*.py")
        if "static" not in path.relative_to(PACKAGE).parts
    }
    modules.discard("__init__")
    return modules


def _doc_text() -> str:
    return ARCHITECTURE.read_text(encoding="utf-8")


def _mentioned(module: str, text: str) -> bool:
    return (
        re.search(rf"(?<![\w]){re.escape(module)}(?:\.py)?(?![\w])", text) is not None
    )


def test_every_module_is_in_the_map() -> None:
    text = _doc_text()
    missing = sorted(m for m in _package_modules() if not _mentioned(m, text))
    assert not missing, (
        f"{len(missing)} modules of src/visionforge are missing from "
        f"docs/dev/ARCHITECTURE.md: {missing}"
    )


def test_every_listed_module_exists() -> None:
    listed = set(re.findall(r"`(?:[\w/{},]+/)?([a-z_][a-z0-9_]*)\.py`", _doc_text()))
    unknown = sorted(listed - _package_modules() - NOT_IN_PACKAGE)
    assert not unknown, f"ARCHITECTURE.md lists modules that do not exist: {unknown}"


def test_extras_policy_matches_pyproject() -> None:
    """The doc names the extras that install something and the empty ones."""
    text = _doc_text()
    in_extras = False
    non_empty: set[str] = set()
    empty: set[str] = set()
    for line in PYPROJECT.read_text(encoding="utf-8").splitlines():
        if line.startswith("[project.optional-dependencies]"):
            in_extras = True
            continue
        if in_extras and line.startswith("["):
            break
        match = re.match(r"^([A-Za-z0-9_-]+)\s*=\s*(\[.*)$", line)
        if in_extras and match:
            name, value = match.groups()
            (empty if value.strip() == "[]" else non_empty).add(name)
    assert empty, "no empty extras found in pyproject.toml"
    policy = text.split("## Dependency policy", 1)[1].split("\n## ", 1)[0]
    before, found, after = policy.partition(PHRASE)
    assert found, f"the Dependency policy lacks {PHRASE!r}"
    for name in sorted(empty):
        assert f"`{name}`" in before, f"empty extra {name!r} is not named as empty"
    for name in sorted(non_empty):
        assert f"`{name}`" in after, (
            f"extra {name!r} is not named as installing something"
        )
