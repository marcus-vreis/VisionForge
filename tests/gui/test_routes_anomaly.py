from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from .conftest import occupy_queue, release_queue


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


def _payload(tmp_path: Path) -> dict:
    base = tmp_path / "ds"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "name": "anom_run",
        "model": {"name": "autoencoder", "latent_dim": 16},
        "data": {"base_dir": str(base)},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }


class TestAnomalySchema:
    def test_schema_exposes_anomaly_config(self, app_and_routes: tuple) -> None:
        app, _ = app_and_routes
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/anomaly/schema")
        assert resp.status_code == 200
        schema = resp.json()
        assert "properties" in schema
        assert {"model", "data", "training"} <= set(schema["properties"])


class TestAnomalyRun:
    def test_run_dispatches_anomaly_block_with_callback(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.blocks.anomaly import AnomalyBlock

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}
        orig_setup = AnomalyBlock.setup
        orig_run = AnomalyBlock.run
        orig_report = AnomalyBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["class"] = type(self).__name__
            captured["model"] = config.model.name  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            captured["has_callback"] = self._progress_callback is not None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {"train": {"best_auroc": 0.9, "run_dir": None}}

        AnomalyBlock.setup = fake_setup  # type: ignore[method-assign]
        AnomalyBlock.run = fake_run  # type: ignore[method-assign]
        AnomalyBlock.report = fake_report  # type: ignore[method-assign]

        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/anomaly/run", json=_payload(tmp_path))
            assert resp.status_code == 200, resp.text

            for _ in range(50):
                if "class" in captured:
                    break
                time.sleep(0.05)
            assert captured.get("class") == "AnomalyBlock"
            assert captured.get("model") == "autoencoder"
            assert captured.get("has_callback") is True
        finally:
            AnomalyBlock.setup = orig_setup  # type: ignore[method-assign]
            AnomalyBlock.run = orig_run  # type: ignore[method-assign]
            AnomalyBlock.report = orig_report  # type: ignore[method-assign]
            routes_mod._current_run = None

    def test_queues_behind_a_running_job(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        occupy_queue(routes_mod)
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/anomaly/run", json=_payload(tmp_path))
            assert resp.status_code == 200
            assert resp.json()["status"] == "queued"
        finally:
            release_queue(routes_mod)

    def test_run_rejects_invalid_config(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        payload = _payload(tmp_path)
        # coreset_ratio out of (0, 1] -> 422
        payload["model"] = {"name": "patchcore", "coreset_ratio": 2.0}
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/anomaly/run", json=payload)
        assert resp.status_code == 422


class TestAnomalyCompareSweep:
    def test_compare_dispatches_and_reports(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.comparison import ComparisonTrial

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        def fake_compare(runner, config_dict, model_names, metric):  # type: ignore[no-untyped-def]
            return [
                ComparisonTrial("patchcore", "success", {"auroc": 0.95}),
                ComparisonTrial("autoencoder", "success", {"auroc": 0.88}),
            ]

        orig = routes_mod.run_model_comparison
        routes_mod.run_model_comparison = fake_compare
        try:
            client = TestClient(app, raise_server_exceptions=True)
            payload = {
                "config": _payload(tmp_path),
                "model_names": ["autoencoder", "patchcore"],
            }
            resp = client.post("/api/anomaly/compare", json=payload)
            assert resp.status_code == 200, resp.text
            run_id = resp.json()["run_id"]

            status = {"status": "running"}
            for _ in range(50):
                status = client.get("/api/experiment/status").json()
                if status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert status["status"] == "completed", status

            report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
            assert report["metric"] == "auroc"  # defaulted from primary_metric
            assert report["top_3"][0]["model_arch"] == "patchcore"
        finally:
            routes_mod.run_model_comparison = orig
            routes_mod._current_run = None

    def test_sweep_dispatches_and_reports(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.sweep import SweepTrial

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        def fake_sweep(
            runner, base, space, *, mode, metric, n_trials, seed, progress_callback=None
        ):  # type: ignore[no-untyped-def]
            return [SweepTrial(0, {"model.latent_dim": 32}, "success", {"auroc": 0.9})]

        orig = routes_mod.run_sweep
        routes_mod.run_sweep = fake_sweep
        try:
            client = TestClient(app, raise_server_exceptions=True)
            payload = {
                "config": _payload(tmp_path),
                "mode": "grid",
                "search_space": {"model.latent_dim": [16, 32]},
            }
            resp = client.post("/api/anomaly/sweep", json=payload)
            assert resp.status_code == 200, resp.text
            run_id = resp.json()["run_id"]

            status = {"status": "running"}
            for _ in range(50):
                status = client.get("/api/experiment/status").json()
                if status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert status["status"] == "completed", status

            report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
            assert report["mode"] == "grid"
            assert report["metric"] == "auroc"
        finally:
            routes_mod.run_sweep = orig
            routes_mod._current_run = None
