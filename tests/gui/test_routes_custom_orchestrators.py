"""Routes for sweep/replicates over custom tasks (ADR-058, brick 4).

Orchestration itself is covered by tests/tasks/test_runner.py — here the
orchestrators are mocked and only the route semantics are asserted
(dispatch, default metric from the task's declaration, 404/409/422).
"""

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

from .conftest import occupy_queue, release_queue


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
        key="toyorch",
        label="Toy Orchestrated",
        accent="#2dd4bf",
        metrics={"mae": "lower"},
        primary_metric="mae",
    )
    class ToyOrchTask(TaskSpec):
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
        "name": "toy_orch_run",
        "data": {"base_dir": str(tmp_path)},
        "training": {"epochs": 1, "batch_size": 4},
        "output": {"models_dir": str(tmp_path / "models")},
        "device": {"kind": "cpu"},
    }


class TestRoutesExist:
    def test_custom_sweep_and_replicates_routes_registered(
        self, app_and_routes: tuple
    ) -> None:
        # Probe the APIRouter directly: depending on the Starlette version,
        # app.routes may hide included-router children entirely.
        _, routes_mod = app_and_routes
        paths = {route.path for route in routes_mod.router.routes}
        assert "/api/custom/{key}/sweep" in paths
        assert "/api/custom/{key}/replicates" in paths
        # deliberately absent — model.name is not guaranteed by BaseTaskConfig
        assert "/api/custom/{key}/compare" not in paths


class TestCustomReplicates:
    def test_dispatch_uses_declared_primary_metric(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.replicates import ReplicateTrial

        _register_toy()
        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}

        def fake_replicates(runner, base, seeds, metric, progress_callback=None):  # type: ignore[no-untyped-def]
            captured["seeds"] = seeds
            captured["metric"] = metric
            captured["runner"] = type(runner).__name__
            return [
                ReplicateTrial(seeds[0], "success", {"mae": 0.4}, 0.1),
                ReplicateTrial(seeds[1], "success", {"mae": 0.6}, 0.1),
            ]

        orig = routes_mod.run_replicates
        routes_mod.run_replicates = fake_replicates
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/custom/toyorch/replicates",
                json={"config": _payload(tmp_path), "seeds": [7, 8]},
            )
            assert resp.status_code == 200, resp.text
            run_id = resp.json()["run_id"]

            status = {"status": "running"}
            for _ in range(50):
                status = client.get("/api/experiment/status").json()
                if status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert status["status"] == "completed", status

            assert captured["seeds"] == [7, 8]
            assert captured["metric"] == "mae"  # from @register_task, not the request
            assert captured["runner"] == "CustomTaskRunner"

            report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
            assert report["metric"] == "mae"
            assert report["total_replicates"] == 2
        finally:
            routes_mod.run_replicates = orig
            routes_mod._current_run = None

    def test_unknown_key_404(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/custom/ghost/replicates",
            json={"config": _payload(tmp_path), "seeds": [1, 2]},
        )
        assert resp.status_code == 404

    def test_queues_behind_a_running_job(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _register_toy()
        app, routes_mod = app_and_routes
        occupy_queue(routes_mod)
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/custom/toyorch/replicates",
                json={"config": _payload(tmp_path)},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "queued"
        finally:
            release_queue(routes_mod)

    def test_invalid_config_422(self, app_and_routes: tuple, tmp_path: Path) -> None:
        _register_toy()
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        cfg = _payload(tmp_path)
        cfg["scale"] = -1  # violates gt=0 on the task's own Config
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/custom/toyorch/replicates",
            json={"config": cfg, "seeds": [1, 2]},
        )
        assert resp.status_code == 422


class TestCustomSweep:
    def test_grid_sweep_over_the_tasks_own_field(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.sweep import SweepTrial

        _register_toy()
        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}

        def fake_sweep(
            runner, base, space, *, mode, metric, n_trials, seed, progress_callback=None
        ):  # type: ignore[no-untyped-def]
            captured["space"] = space
            captured["metric"] = metric
            captured["runner"] = type(runner).__name__
            return [
                SweepTrial(0, {"scale": 1.0}, "success", {"mae": 0.4}),
                SweepTrial(1, {"scale": 2.0}, "success", {"mae": 0.8}),
            ]

        orig = routes_mod.run_sweep
        routes_mod.run_sweep = fake_sweep
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/custom/toyorch/sweep",
                json={
                    "config": _payload(tmp_path),
                    "mode": "grid",
                    # `scale` only exists on this task's Config — the sweep
                    # space is validated against it, not a built-in config.
                    "search_space": {"scale": [1.0, 2.0]},
                },
            )
            assert resp.status_code == 200, resp.text
            run_id = resp.json()["run_id"]

            status = {"status": "running"}
            for _ in range(50):
                status = client.get("/api/experiment/status").json()
                if status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert status["status"] == "completed", status

            assert captured["space"] == {"scale": [1.0, 2.0]}
            assert captured["metric"] == "mae"
            assert captured["runner"] == "CustomTaskRunner"

            report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
            assert report["mode"] == "grid"
            assert report["total_trials"] == 2
        finally:
            routes_mod.run_sweep = orig
            routes_mod._current_run = None

    def test_sweep_rejects_path_unknown_to_the_task(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _register_toy()
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/custom/toyorch/sweep",
            json={
                "config": _payload(tmp_path),
                "mode": "grid",
                "search_space": {"model.name": ["a", "b"]},
            },
        )
        assert resp.status_code == 422

    def test_unknown_key_404(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/custom/ghost/sweep",
            json={"config": _payload(tmp_path), "search_space": {"scale": [1.0]}},
        )
        assert resp.status_code == 404
