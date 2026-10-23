"""Sonar metadata follows the analyzed checkout without importing the package."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.sonar_project_version import project_version

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("version", ["0.17.0", "0.18.0rc1", "1!2.0.dev3+cpu.1"])
def test_version_follows_pyproject_changes(tmp_path: Path, version: str) -> None:
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    assert project_version(metadata) == version


@pytest.mark.parametrize(
    "version", [None, 17, "", " 0.17.0", "0.17.0 extra", "0.17.0\nother=value"]
)
def test_invalid_version_cannot_inject_scanner_or_workflow_output(
    tmp_path: Path, version: object
) -> None:
    metadata = tmp_path / "pyproject.toml"
    value = "" if version is None else f"version = {json.dumps(version)}\n"
    metadata.write_text(f"[project]\n{value}", encoding="utf-8")
    with pytest.raises(ValueError, match="project.version"):
        project_version(metadata)


def test_cli_reads_current_checkout_without_installed_package(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "9.8.7"\n', encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/sonar_project_version.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "version=9.8.7\n"
    assert result.stderr == ""


def test_missing_metadata_fails_instead_of_using_a_fallback(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        project_version(tmp_path / "pyproject.toml")


def test_scan_receives_generated_version_and_existing_coverage() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["sonarqube"]["steps"]
    metadata = next(step for step in steps if step.get("id") == "sonar-version")
    assert (
        metadata["run"] == 'python scripts/sonar_project_version.py >> "$GITHUB_OUTPUT"'
    )
    scan = next(step for step in steps if step.get("name") == "SonarCloud scan")
    assert steps.index(metadata) < steps.index(scan)
    assert (
        "-Dsonar.projectVersion=${{ steps.sonar-version.outputs.version }}"
        in scan["with"]["args"]
    )
    assert "-Dsonar.python.coverage.reportPaths=coverage.xml" in scan["with"]["args"]
    properties = (ROOT / "sonar-project.properties").read_text()
    assert not any(
        line.startswith("sonar.projectVersion=") for line in properties.splitlines()
    )
