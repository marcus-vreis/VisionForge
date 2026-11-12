"""POST /api/runs/{run_id}/mlflow/retry (ADR-170), with the in-memory MLflow fake."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import visionforge.gui.api.routes as routes_mod
from visionforge.gui.server import app

from ..fake_mlflow import FakeMlflowClient

_RUN_ID = "20261108_120000_000000"


@pytest.fixture
def outputs(tmp_path: Path) -> Iterator[Path]:
    """A throwaway outputs/ with profiles/ and models/, wired into the routes."""
    profiles = tmp_path / "profiles"
    models = tmp_path / "models"
    profiles.mkdir()
    models.mkdir()
    with (
        patch.object(routes_mod, "_PROFILES_DIR", profiles),
        patch.object(routes_mod, "_MODELS_DIR", models),
    ):
        yield tmp_path


def _enable_mlflow(outputs: Path, enabled: bool = True) -> None:
    meta = {"name": "Ana", "settings": {"mlflow": {"enabled": enabled}}}
    (outputs / "profiles" / "default.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _run(outputs: Path, *, task: str = "binary", status: str = "FAILED") -> Path:
    run_dir = outputs / "models" / "exp" / _RUN_ID
    run_dir.mkdir(parents=True)
    record: dict[str, Any] = {
        "id": f"exp_{_RUN_ID}",
        "status": "completed",
        "config": {
            "name": "exp",
            "task": task,
            "training": {"epochs": 1},
        },
        "history": [{"epoch": 1, "train_loss": 0.4}],
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "tracking.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "error": "OSError",
                "run_id": None,
                "identity": "x",
                "status": status,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _retry() -> Any:
    return TestClient(app).post(f"/api/runs/{_RUN_ID}/mlflow/retry")


def test_unknown_run_is_404(outputs: Path) -> None:
    _enable_mlflow(outputs)
    assert _retry().status_code == 404


def test_profile_without_mlflow_is_409(outputs: Path) -> None:
    _enable_mlflow(outputs, enabled=False)
    _run(outputs)
    assert _retry().status_code == 409


def test_run_still_running_is_409(outputs: Path) -> None:
    _enable_mlflow(outputs)
    _run(outputs, status="RUNNING")
    assert _retry().status_code == 409


def test_run_without_tracking_record_is_409(outputs: Path) -> None:
    _enable_mlflow(outputs)
    (_run(outputs) / "tracking.json").unlink()
    assert _retry().status_code == 409


def test_unreadable_record_is_500(outputs: Path) -> None:
    _enable_mlflow(outputs)
    (_run(outputs) / "run.json").write_text("{not json", encoding="utf-8")
    assert _retry().status_code == 500


def test_record_without_config_is_409(outputs: Path) -> None:
    _enable_mlflow(outputs)
    (_run(outputs) / "run.json").write_text('{"history": []}', encoding="utf-8")
    assert _retry().status_code == 409


def _identity(run_dir: Path) -> str:
    # The tracker ties a sidecar to its profile and run folder (ADR-170).
    return hashlib.sha256(f"default\0{run_dir.resolve()}".encode()).hexdigest()


def test_retry_exports_a_failed_run(
    outputs: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    _enable_mlflow(outputs)
    run_dir = _run(outputs)
    sidecar = json.loads((run_dir / "tracking.json").read_text())
    sidecar["identity"] = _identity(run_dir)
    (run_dir / "tracking.json").write_text(json.dumps(sidecar), encoding="utf-8")

    response = _retry()

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "finished"
    assert body["status"] == "FAILED"
    (run,) = fake_mlflow.runs.values()
    assert body["run_id"] == run.run_id
    assert run.status == "FAILED"
    assert run.artifacts == ["run.json"]
    assert [(m.step, m.value) for m in run.metrics["loss/train"]] == [(1, 0.4)]


@pytest.mark.parametrize("task", ["detection", "regression", "segmentation"])
def test_other_tasks_retry_too(
    outputs: Path, task: str, fake_mlflow: type[FakeMlflowClient]
) -> None:
    # #286: every task exports, so every task can be retried (ADR-197).
    _enable_mlflow(outputs)
    run_dir = _run(outputs, task=task)
    sidecar = json.loads((run_dir / "tracking.json").read_text())
    sidecar["identity"] = _identity(run_dir)
    (run_dir / "tracking.json").write_text(json.dumps(sidecar), encoding="utf-8")

    response = _retry()

    assert response.status_code == 200
    (run,) = fake_mlflow.runs.values()
    assert run.params["task"] == json.dumps(task)
    assert run.artifacts == ["run.json"]


def test_record_of_another_run_is_not_adopted(
    outputs: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    _enable_mlflow(outputs)
    _run(outputs)  # its sidecar identity "x" belongs to no local run

    response = _retry()

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "failed"
    assert body["error"] == "ValueError"
    assert fake_mlflow.runs == {}
