"""Offering "continue" has to mean it: the answer comes from what is on disk.

ADR-092 made the resume file's presence the answer to "can this be resumed", so
these check the derivation rather than a stored flag — including the two cases
where the file is the wrong thing to look at: Ultralytics keeps its own state
(ADR-093), and a sweep's parent directory holds no training at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.resume import ResumeState, save_resume_state
from visionforge.gui.api.routes import _resume_status


def _run(
    tmp_path: Path,
    config: dict[str, Any],
    *,
    total_epochs: int = 1,
    resume_file: bool = False,
    last_pt: bool = False,
) -> tuple[Path, dict[str, Any]]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume_file:
        # A real payload: an unreadable file is a separate case below.
        save_resume_state(
            run_dir,
            ResumeState(
                epoch=total_epochs,
                model={},
                optimizer={},
                scheduler=None,
                scaler=None,
                best_metric=0.1,
                best_epoch=1,
                patience_counter=0,
                history=[],
            ),
        )
    if last_pt:
        (run_dir / "weights").mkdir(exist_ok=True)
        (run_dir / "weights" / "last.pt").write_bytes(b"x")
    data = {
        "experiment": "e",
        "config": config,
        "metrics": {"total_epochs": total_epochs},
        "history": [],
        "artifacts": {},
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir, data


def _classification(epochs: int = 5, block: str = "classification") -> dict[str, Any]:
    return {
        "task": "multiclass",
        "block": block,
        "training": {"epochs": epochs},
        "model": {"name": "resnet18"},
    }


class TestDerivedFromDisk:
    def test_a_stopped_run_with_state_can_be_continued(self, tmp_path: Path) -> None:
        run_dir, data = _run(tmp_path, _classification(), resume_file=True)

        assert _resume_status(run_dir, data) == (True, 5)

    def test_without_the_file_there_is_nothing_to_continue(
        self, tmp_path: Path
    ) -> None:
        run_dir, data = _run(tmp_path, _classification())

        assert _resume_status(run_dir, data) == (False, 5)

    def test_an_unreadable_file_is_not_offered(self, tmp_path: Path) -> None:
        """`load_resume_state` returns None for it, and so must this."""
        run_dir, data = _run(tmp_path, _classification(), resume_file=True)
        (run_dir / "resume.pt").write_bytes(b"not a checkpoint")

        assert _resume_status(run_dir, data)[0] is False

    @pytest.mark.parametrize("block", ["grid_search", "random_search", "cross_val"])
    def test_a_sweeps_parent_directory_is_never_offered(
        self, tmp_path: Path, block: str
    ) -> None:
        """Its trainings live in sub-directories; continuing here continues nothing."""
        run_dir, data = _run(tmp_path, _classification(block=block), resume_file=True)

        assert _resume_status(run_dir, data)[0] is False

    def test_a_config_without_epochs_is_not_offered(self, tmp_path: Path) -> None:
        run_dir, data = _run(tmp_path, {"task": "multiclass", "training": {}})

        assert _resume_status(run_dir, data) == (False, None)


class TestUltralyticsKeepsItsOwnState:
    """A YOLO run writes no resume.pt, so judging it by that would say "no"."""

    @staticmethod
    def _detection(backend: str = "ultralytics", epochs: int = 10) -> dict[str, Any]:
        return {
            "task": "detection",
            "training": {"epochs": epochs},
            "model": {"name": "yolo11n", "backend": backend},
        }

    def test_last_pt_and_an_unfinished_history_can_be_continued(
        self, tmp_path: Path
    ) -> None:
        run_dir, data = _run(tmp_path, self._detection(), total_epochs=4, last_pt=True)

        assert _resume_status(run_dir, data) == (True, 10)

    def test_a_finished_run_is_not_offered(self, tmp_path: Path) -> None:
        run_dir, data = _run(tmp_path, self._detection(), total_epochs=10, last_pt=True)

        assert _resume_status(run_dir, data)[0] is False

    def test_without_last_pt_there_is_nothing_to_continue(self, tmp_path: Path) -> None:
        run_dir, data = _run(tmp_path, self._detection(), total_epochs=4)

        assert _resume_status(run_dir, data)[0] is False

    def test_the_torchvision_backend_is_judged_by_the_resume_file(
        self, tmp_path: Path
    ) -> None:
        """It runs our loop, so `last.pt` says nothing about it."""
        run_dir, data = _run(
            tmp_path,
            self._detection(backend="torchvision"),
            total_epochs=4,
            last_pt=True,
        )

        assert _resume_status(run_dir, data)[0] is False


class TestResumeEndpoint:
    @staticmethod
    def _client_and_routes() -> tuple[Any, Any]:
        from fastapi.testclient import TestClient

        from visionforge.gui.api import routes as routes_mod
        from visionforge.gui.server import app

        return TestClient(app), routes_mod

    def test_an_unknown_run_is_a_404(self) -> None:
        client, _ = self._client_and_routes()

        assert client.post("/api/runs/never-existed/resume").status_code == 404

    def test_a_run_with_nothing_left_is_a_409(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, routes_mod = self._client_and_routes()
        run_dir, _ = _run(tmp_path / "models" / "e", _classification())
        monkeypatch.setattr(routes_mod, "_MODELS_DIR", tmp_path / "models")

        resp = client.post(f"/api/runs/{run_dir.name}/resume")

        assert resp.status_code == 409
        assert "continue" in resp.json()["detail"]
