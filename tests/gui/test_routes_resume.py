"""Offering "continue" has to mean it: the answer comes from what is on disk.

ADR-092 made the resume file's presence the answer to "can this be resumed", so
these check the derivation rather than a stored flag — including the two cases
where the file is the wrong thing to look at: Ultralytics keeps its own state
(ADR-093), and a sweep's parent directory holds no training at all.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.resume import ResumeState, save_resume_state
from visionforge.gui.api.routes import _resume_status
from visionforge.utils.config import ExperimentConfig


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


def _classification_full(base_dir: Path, epochs: int = 5) -> dict[str, Any]:
    """A config ExperimentConfig validates, for tests that reach the executor.

    ``base_dir`` must exist: the config validator checks it.
    """
    return {
        "name": "e",
        "task": "multiclass",
        "block": "classification",
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "data": {"base_dir": str(base_dir)},
        "training": {"epochs": epochs, "learning_rate": 1e-3},
        "device": {"kind": "cpu"},
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

    def test_a_stopped_run_is_submitted_in_its_own_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.gui.api.schemas import RunResponse

        client, routes_mod = self._client_and_routes()
        stored = _classification_full(tmp_path)
        run_dir, _ = _run(tmp_path / "models" / "e", stored, resume_file=True)
        monkeypatch.setattr(routes_mod, "_MODELS_DIR", tmp_path / "models")
        submitted: dict[str, Any] = {}

        def fake_submit(job_id, label, task, strategy, start, **_kwargs):  # type: ignore[no-untyped-def]
            submitted.update(
                job_id=job_id, label=label, task=task, strategy=strategy, start=start
            )
            return RunResponse(run_id=job_id)

        monkeypatch.setattr(routes_mod, "_submit_job", fake_submit)
        seen: dict[str, Any] = {}

        async def fake_execute(config, job_id, *, resume_dir):  # type: ignore[no-untyped-def]
            seen.update(config=config, job_id=job_id, resume_dir=resume_dir)

        monkeypatch.setattr(routes_mod, "_execute_experiment", fake_execute)

        resp = client.post(f"/api/runs/{run_dir.name}/resume")

        assert resp.status_code == 200
        assert submitted["task"] == "classification"
        assert "retomando" in submitted["label"]
        assert submitted["strategy"] == "simple"
        asyncio.run(submitted["start"]())
        assert seen["resume_dir"] == run_dir
        assert seen["job_id"] == submitted["job_id"] == resp.json()["run_id"]
        assert seen["config"] == ExperimentConfig.model_validate(stored)

    def test_a_stored_config_that_no_longer_validates_is_a_400(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, routes_mod = self._client_and_routes()
        broken = _classification_full(tmp_path)
        broken["training"]["learning_rate"] = -1.0
        run_dir, _ = _run(tmp_path / "models" / "e", broken, resume_file=True)
        monkeypatch.setattr(routes_mod, "_MODELS_DIR", tmp_path / "models")

        resp = client.post(f"/api/runs/{run_dir.name}/resume")

        assert resp.status_code == 400
        assert "no longer valid" in resp.json()["detail"]

    def test_an_unreadable_run_json_is_a_500(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, routes_mod = self._client_and_routes()
        run_dir = tmp_path / "models" / "e" / "20260923_000000_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(routes_mod, "_MODELS_DIR", tmp_path / "models")

        resp = client.post(f"/api/runs/{run_dir.name}/resume")

        assert resp.status_code == 500
        assert "run.json" in resp.json()["detail"]
