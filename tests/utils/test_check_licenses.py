"""The copyleft license gate (#225), on synthetic metadata and the real files."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from scripts import check_licenses as gate
from scripts.check_licenses import Dist

_AGPL = (
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)"
)
_LGPL = "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)"
_GPL = "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)"
_MIT = "License :: OSI Approved :: MIT License"


def _dist(name: str, families: set[str] | None = None, *requires: str) -> Dist:
    return Dist(name=name, families=frozenset(families or ()), requires=requires)


def _lookup(*dists: Dist) -> gate.Lookup:
    table = {d.name: d for d in dists}
    return table.get


_PYPROJECT: dict[str, Any] = {
    "project": {
        "dependencies": ["app-core>=1"],
        "optional-dependencies": {"detection": ["yolo-lib"]},
    }
}


def _allowlist(**packages: dict[str, Any]) -> dict[str, Any]:
    return {"closure": {"extras": ["detection"]}, "packages": packages}


def _notices(allowlist: dict[str, Any]) -> str:
    table = gate.render_table(allowlist["packages"])
    return f"# Notices\n\n{gate.TABLE_START}\n{table}{gate.TABLE_END}\n\nFonts.\n"


_YOLO_ENTRY = {"families": ["AGPL"], "license": "AGPL-3.0", "why": "detection"}


@pytest.mark.parametrize(
    "expression, license_field, classifiers, families",
    [
        (None, None, [_AGPL], {"AGPL"}),
        (None, "BSD-3-Clause", [_LGPL], {"LGPL"}),
        (None, "Artistic License", [_GPL], {"GPL"}),
        ("MPL-2.0 AND MIT", None, [], {"MPL"}),
        (None, "MPL-2.0", [], {"MPL"}),
        ("MIT", "GPL-3.0", [_MIT], set()),  # the expression wins over License
        (None, "Permission is granted...\nunlike the GPL, this...", [], set()),
        (None, "x" * 101 + " GPL", [], set()),
        (None, None, [_MIT], set()),
    ],
)
def test_families_come_from_expression_short_license_and_classifiers(
    expression: str | None,
    license_field: str | None,
    classifiers: list[str],
    families: set[str],
) -> None:
    assert gate.copyleft_families(expression, license_field, classifiers) == families


def test_closure_follows_requested_extras_and_reports_missing() -> None:
    found, missing = gate.walk_closure(
        ["app-core[fast]"],
        _lookup(
            _dist(
                "app-core",
                None,
                "speedup; extra == 'fast'",
                "docs-tool; extra == 'docs'",
                "absent-lib",
            ),
            _dist("speedup"),
            _dist("docs-tool", {"GPL"}),
        ),
    )
    assert sorted(found) == ["app-core", "speedup"]
    assert missing == ["absent-lib"]


def test_copyleft_outside_the_allowlist_fails() -> None:
    allowlist = _allowlist()
    lookup = _lookup(_dist("app-core"), _dist("yolo-lib", {"AGPL"}))
    problems = gate.run(_PYPROJECT, allowlist, _notices(allowlist), lookup)
    assert problems == ["yolo-lib: copyleft (AGPL) and not in the allowlist"]


def test_copyleft_inside_the_allowlist_passes() -> None:
    allowlist = _allowlist(**{"yolo-lib": _YOLO_ENTRY})
    lookup = _lookup(_dist("app-core"), _dist("yolo-lib", {"AGPL"}))
    assert gate.run(_PYPROJECT, allowlist, _notices(allowlist), lookup) == []


def test_a_license_that_gains_a_family_fails() -> None:
    allowlist = _allowlist(**{"yolo-lib": _YOLO_ENTRY})
    lookup = _lookup(_dist("app-core"), _dist("yolo-lib", {"AGPL", "GPL"}))
    problems = gate.run(_PYPROJECT, allowlist, _notices(allowlist), lookup)
    assert problems == ["yolo-lib: now declares GPL, allowlisted for AGPL"]


def test_a_missing_dependency_fails_instead_of_passing_blind() -> None:
    allowlist = _allowlist()
    problems = gate.run(_PYPROJECT, allowlist, _notices(allowlist), _lookup())
    assert problems == [
        "app-core: required but not installed",
        "yolo-lib: required but not installed",
    ]


def test_notices_table_must_match_the_allowlist() -> None:
    allowlist = _allowlist(**{"yolo-lib": _YOLO_ENTRY})
    stale = _notices(_allowlist())
    assert gate.check_notices(stale, allowlist["packages"]) == [
        "THIRD_PARTY_NOTICES.md: table differs from the allowlist (run --write)"
    ]
    assert gate.check_notices("# no table\n", allowlist["packages"]) == [
        "THIRD_PARTY_NOTICES.md: table markers not found"
    ]
    rewritten = gate.write_notices(stale, allowlist["packages"])
    assert rewritten == _notices(allowlist)


def test_repository_notices_match_the_allowlist() -> None:
    """The committed THIRD_PARTY_NOTICES.md is what the allowlist renders."""
    allowlist = tomllib.loads(gate.ALLOWLIST.read_text(encoding="utf-8"))
    notices = gate.NOTICES.read_text(encoding="utf-8")
    assert gate.check_notices(notices, allowlist["packages"]) == []
    assert gate.write_notices(notices, allowlist["packages"]) == notices


def test_allowlist_closure_names_real_extras() -> None:
    allowlist = tomllib.loads(gate.ALLOWLIST.read_text(encoding="utf-8"))
    pyproject = tomllib.loads(gate.PYPROJECT.read_text(encoding="utf-8"))
    roots = gate.roots_from(pyproject, allowlist["closure"])
    assert "ultralytics>=8.3" in roots
    assert {"torch", "torchvision"} <= set(roots)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The gate's three files in a scratch folder, with a synthetic environment."""
    allowlist = tmp_path / "license_allowlist.toml"
    allowlist.write_text(
        '[closure]\nextras = ["detection"]\n\n[packages.yolo-lib]\n'
        'families = ["AGPL"]\nlicense = "AGPL-3.0"\nwhy = "detection"\n',
        encoding="utf-8",
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\ndependencies = ["app-core"]\n\n'
        '[project.optional-dependencies]\ndetection = ["yolo-lib"]\n',
        encoding="utf-8",
    )
    notices = tmp_path / "THIRD_PARTY_NOTICES.md"
    notices.write_text(_notices(_allowlist()), encoding="utf-8")
    monkeypatch.setattr(gate, "ALLOWLIST", allowlist)
    monkeypatch.setattr(gate, "PYPROJECT", pyproject)
    monkeypatch.setattr(gate, "NOTICES", notices)
    return tmp_path


def test_main_exits_1_on_problems_and_0_after_write(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        gate, "installed", _lookup(_dist("app-core"), _dist("yolo-lib", {"AGPL"}))
    )
    assert gate.main([]) == 1
    assert "table differs" in capsys.readouterr().err
    assert gate.main(["--write"]) == 0
    assert "`yolo-lib` | AGPL-3.0" in (repo / "THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )


def test_installed_reads_this_environment() -> None:
    dist = gate.installed("pytest")
    assert dist is not None
    assert dist.name == "pytest"
    assert dist.families == frozenset()
    assert gate.installed("no-such-distribution-here") is None
