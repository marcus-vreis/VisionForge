from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import Field
from torch import nn

from visionforge.tasks import (
    BaseTaskConfig,
    TaskSpec,
    clear_task_registry,
    register_task,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_task_registry()
    yield
    clear_task_registry()


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


class ToyConfig(BaseTaskConfig):
    scale: float = Field(default=1.0, gt=0)


def _register_toy() -> None:
    @register_task(
        key="toyapi",
        label="Toy API",
        accent="#2dd4bf",
        description="toy",
        metrics={"mae": "lower"},
        primary_metric="mae",
    )
    class ToyApiTask(TaskSpec):
        Config = ToyConfig

        def build_model(self, cfg: Any) -> nn.Module:
            return nn.Linear(3, 1)

        def build_loaders(self, cfg: Any):
            batches = [(torch.randn(4, 3), torch.randn(4, 1)) for _ in range(2)]
            return batches, batches, None

        def compute_loss(self, model: nn.Module, batch: Any, cfg: Any):
            inputs, targets = batch
            return nn.functional.mse_loss(model(inputs), targets)

        def compute_metrics(self, model: nn.Module, loader: Any, cfg: Any):
            errs = [(model(x) - y).abs().mean().item() for x, y in loader]
            return {"mae": sum(errs) / len(errs)}


def _payload(tmp_path: Path) -> dict:
    return {
        "name": "toy_api_run",
        "data": {"base_dir": str(tmp_path)},
        "training": {"epochs": 1, "batch_size": 4},
        "output": {"models_dir": str(tmp_path / "models")},
        "device": {"kind": "cpu"},
    }


class TestTaskList:
    def test_lists_builtins_plus_registered_customs(
        self, app_and_routes: tuple
    ) -> None:
        _register_toy()
        app, _ = app_and_routes
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        tasks = {t["key"]: t for t in resp.json()["tasks"]}
        for builtin in (
            "classification",
            "detection",
            "regression",
            "segmentation",
            "anomaly",
        ):
            assert builtin in tasks
            assert tasks[builtin]["custom"] is False
        toy = tasks["toyapi"]
        assert toy["custom"] is True
        assert toy["accent"] == "#2dd4bf"
        assert toy["primary_metric"] == "mae"


class TestCustomSchema:
    def test_schema_exposes_task_config(self, app_and_routes: tuple) -> None:
        _register_toy()
        app, _ = app_and_routes
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/custom/toyapi/schema")
        assert resp.status_code == 200
        props = resp.json()["properties"]
        assert {"name", "data", "training", "scale"} <= set(props)

    def test_unknown_key_404(self, app_and_routes: tuple) -> None:
        app, _ = app_and_routes
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/custom/nope/schema")
        assert resp.status_code == 404


class TestCustomRun:
    def test_run_trains_and_reports(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _register_toy()
        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        # Context manager = one persistent event loop across requests. Without
        # it TestClient spins a loop per request and the genuinely-async
        # background training is destroyed when the POST's loop closes.
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post("/api/custom/toyapi/run", json=_payload(tmp_path))
            assert resp.status_code == 200, resp.text
            run_id = resp.json()["run_id"]

            status = {"status": "running"}
            for _ in range(600):
                status = client.get("/api/experiment/status").json()
                if status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert status["status"] == "completed", status

            report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
            assert report["task"] == "custom:toyapi"
            assert "mae" in report["metrics"]
            assert report["total_epochs"] == 1
            run_dir = Path(report["run_dir"])
            assert (run_dir / "run.json").is_file()
        routes_mod._current_run = None

    def test_conflict_when_already_running(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _register_toy()
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/custom/toyapi/run", json=_payload(tmp_path))
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None

    def test_invalid_config_422(self, app_and_routes: tuple, tmp_path: Path) -> None:
        _register_toy()
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        cfg = _payload(tmp_path)
        cfg["scale"] = -1  # violates gt=0 on the task's own Config
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/custom/toyapi/run", json=cfg)
        assert resp.status_code == 422

    def test_unknown_key_404(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/custom/ghost/run", json=_payload(tmp_path))
        assert resp.status_code == 404
