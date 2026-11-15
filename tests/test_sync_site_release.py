"""scripts/sync_site_release.py prints the version diff and never writes or dispatches."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest
import yaml

from scripts import sync_site_release as sync

ROOT = Path(__file__).resolve().parents[1]
ENCODING = "utf-8"
DOCS_DIR = "docs"
ROOT_FLAG = "--root"
WARN_FLAG = "--warn"
VERSION_FLAG = "--version"
OLD = "0.20.2"
NEW = "0.21.0"
ZENODO_DOI = "10.5281/zenodo.23926437"


def _fetcher(site_version: str | None, *, site_up: bool = True, pypi_up: bool = True):
    """A fake HTTP layer: PyPI/GitHub/Zenodo say NEW, the site says ``site_version``."""
    bodies = {
        sync.PYPI_URL: json.dumps({"info": {"version": NEW}}),
        sync.RELEASE_URL: json.dumps({"tag_name": f"v{NEW}"}),
        sync.ZENODO_URL: json.dumps({"doi": ZENODO_DOI}),
        sync.SITE_URL: f"<p>VisionForge {site_version}, python 3.13.2, cuda 12.8.0</p>",
    }
    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        if (url == sync.SITE_URL and not site_up) or (
            url == sync.PYPI_URL and not pypi_up
        ):
            raise urllib.error.URLError("down")
        return bodies[url]

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def _checkout(tmp_path: Path, version: str = NEW) -> Path:
    (tmp_path / DOCS_DIR).mkdir()
    (tmp_path / "CITATION.cff").write_text(f"version: {version}\n", encoding=ENCODING)
    (tmp_path / DOCS_DIR / "INSTALL.md").write_text(
        f"pip install visionforge-studio=={OLD}\nPython 3.13.2\n", encoding=ENCODING
    )
    return tmp_path


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p): p.read_text(encoding=ENCODING)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_dry_run_prints_the_diff_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _checkout(tmp_path)
    before = _snapshot(root)
    fetch = _fetcher(OLD)
    code = sync.main(["--dry-run", ROOT_FLAG, str(root)], fetch=fetch)
    out = capsys.readouterr().out
    assert code == 1
    assert f"pypi: {NEW}" in out
    assert f"github release: {NEW}" in out
    assert f"zenodo doi: {ZENODO_DOI}" in out
    assert f"site (home): {OLD} -> {NEW} (1x)" in out
    assert f"docs/INSTALL.md: {OLD} -> {NEW} (1x)" in out
    assert "3.13.2" not in out
    assert "12.8.0" not in out
    assert "nothing was written" in out
    assert _snapshot(root) == before
    assert set(fetch.calls) <= {
        sync.PYPI_URL,
        sync.RELEASE_URL,
        sync.ZENODO_URL,
        sync.SITE_URL,
    }


def test_site_on_the_new_version_exits_zero(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    assert (
        sync.main([ROOT_FLAG, str(root)], fetch=_fetcher(NEW)) == 1
    )  # INSTALL still old
    (root / DOCS_DIR / "INSTALL.md").write_text(f"=={NEW}\n", encoding=ENCODING)
    assert sync.main([ROOT_FLAG, str(root)], fetch=_fetcher(NEW)) == 0


def test_warn_mode_emits_a_warning_and_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = sync.main(
        [WARN_FLAG, VERSION_FLAG, NEW, ROOT_FLAG, str(_checkout(tmp_path))],
        fetch=_fetcher(OLD),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"::warning::The site {sync.SITE_URL} does not show version {NEW}" in out


def test_warn_mode_is_quiet_when_the_site_has_the_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sync.main(
        [WARN_FLAG, VERSION_FLAG, NEW, ROOT_FLAG, str(_checkout(tmp_path))],
        fetch=_fetcher(NEW),
    )
    assert "::warning::" not in capsys.readouterr().out


def test_unreachable_site_is_reported_and_warned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = sync.main(
        [WARN_FLAG, VERSION_FLAG, NEW, ROOT_FLAG, str(_checkout(tmp_path))],
        fetch=_fetcher(NEW, site_up=False),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "unreachable" in out
    assert "::warning::" in out


def test_without_pypi_and_without_version_there_is_no_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = sync.main(
        [ROOT_FLAG, str(_checkout(tmp_path))], fetch=_fetcher(NEW, pypi_up=False)
    )
    assert code == 1
    assert "pypi: unavailable" in capsys.readouterr().out


def test_malformed_json_counts_as_unavailable() -> None:
    assert sync._json_field(lambda _url: "not json", sync.PYPI_URL, "info") is None
    assert (
        sync._json_field(lambda _url: '{"info": 3}', sync.PYPI_URL, "info", "version")
        is None
    )


def test_version_counts_ignore_python_and_cuda_versions() -> None:
    text = "0.21.0 0.21.0 0.20.2 3.13.2 12.8.0 10.0.26300 0.0.0"
    assert sync.version_counts(text) == {"0.21.0": 2, "0.20.2": 1}


def test_cd_workflow_has_a_non_blocking_site_check() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "cd.yml").read_text(encoding=ENCODING)
    )
    job = workflow["jobs"]["site-version-check"]
    commands = " ".join(step.get("run", "") for step in job["steps"])
    assert "scripts/sync_site_release.py" in commands
    assert WARN_FLAG in commands
    assert VERSION_FLAG in commands
    assert job["needs"] == "github-release"
    assert "repository_dispatch" not in (
        ROOT / ".github" / "workflows" / "cd.yml"
    ).read_text(encoding=ENCODING)


def test_releasing_lists_each_post_release_item_with_a_command() -> None:
    text = (ROOT / DOCS_DIR / "dev" / "RELEASING.md").read_text(encoding=ENCODING)
    section = text.split("## After publishing", 1)[1].split("\n## ", 1)[0]
    rows = {
        line.split("|")[1].strip(): line
        for line in section.splitlines()
        if line.startswith("| ") and not line.startswith("| Item")
    }
    expected = {"PyPI", "GitHub Release", "Zenodo", "CITATION.cff", "INSTALL"}
    assert expected | {"Screenshots", "Site", "Milestone"} == set(rows)
    for item, line in rows.items():
        assert line.count("`") >= 2, f"{item} has no command"
    assert "sync_site_release.py --dry-run" in rows["Site"]
