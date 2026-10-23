"""Emit the analyzed checkout's package version for the Sonar workflow."""

import re
import tomllib
from pathlib import Path


def project_version(pyproject: Path) -> str:
    """Read a static version safe for a scanner argument and GitHub step output."""
    with pyproject.open("rb") as source:
        version = tomllib.load(source).get("project", {}).get("version")
    if not isinstance(version, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version
    ):
        raise ValueError("project.version must be a nonempty, single-token string")
    return version


def main() -> None:
    """Write version metadata from the workflow's current checkout."""
    print(f"version={project_version(Path('pyproject.toml'))}")


if __name__ == "__main__":
    main()
