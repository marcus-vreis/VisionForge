"""The GUI sweeps orphaned filtered-dataset copies when it starts (ADR-084, ADR-130)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionforge.core import materialized_dataset
from visionforge.gui.server import app


def test_startup_sweeps_default_and_profile_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs/profiles/b/models/_filtered").mkdir(parents=True)
    (tmp_path / "outputs/profiles/a/models/_filtered").mkdir(parents=True)
    swept: list[Path] = []
    monkeypatch.setattr(materialized_dataset, "sweep_orphans", swept.append)

    with TestClient(app):
        assert swept == [
            Path("outputs/models/_filtered"),
            Path("outputs/profiles/a/models/_filtered"),
            Path("outputs/profiles/b/models/_filtered"),
        ]
