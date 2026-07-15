from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


def _payload(tmp_path: Path) -> dict:
    base = tmp_path / "ds"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "name": "rep_run",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(base), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }


class TestReplicatesEndpoints:
    def test_all_five_tasks_expose_a_replicates_route(
        self, app_and_routes: tuple
    ) -> None:
        # Probe the APIRouter directly: depending on the Starlette version,
        # app.routes may hide included-router children entirely.
        _, routes_mod = app_and_routes
        paths = {route.path for route in routes_mod.router.routes}
        for task in (
            "classification",
            "regression",
            "segmentation",
            "detection",
            "anomaly",
        ):
            assert f"/api/{task}/replicates" in paths


class TestRegressionReplicates:
    def test_replicates_dispatch_and_aggregate_report(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.replicates import ReplicateTrial

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}

        def fake_replicates(runner, base, seeds, metric, progress_callback=None):  # type: ignore[no-untyped-def]
            captured["seeds"] = seeds
            captured["metric"] = metric
            return [
                ReplicateTrial(seeds[0], "success", {"r2": 0.80}, 0.1),
                ReplicateTrial(seeds[1], "success", {"r2": 0.90}, 0.1),
                ReplicateTrial(seeds[2], "success", {"r2": 1.00}, 0.1),
            ]

        orig = routes_mod.run_replicates
        routes_mod.run_replicates = fake_replicates
        try:
            client = TestClient(app, raise_server_exceptions=True)
            payload = {
                "config": _payload(tmp_path),
                "seeds": [7, 8, 9],
                "metric": "r2",
            }
            resp = client.post("/api/regression/replicates", json=payload)
            assert resp.status_code == 200, resp.text
            run_id = resp.json()["run_id"]

            status = {"status": "running"}
            for _ in range(50):
                status = client.get("/api/experiment/status").json()
                if status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert status["status"] == "completed", status

            assert captured["seeds"] == [7, 8, 9]
            assert captured["metric"] == "r2"

            report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
            assert report["metric"] == "r2"
            assert report["seeds"] == [7, 8, 9]
            assert report["total_replicates"] == 3
            assert report["successful_replicates"] == 3
            headline = report["headline"]
            assert abs(headline["mean"] - 0.9) < 1e-9
            assert headline["n"] == 3
            assert headline["ci95_low"] < 0.9 < headline["ci95_high"]
            # trials stay in seed order — replicates are not a ranking
            assert [t["seed"] for t in report["trials"]] == [7, 8, 9]
            # the aggregate is persisted to outputs/reports for later reference
            assert (Path(report["report_dir"]) / "replicates_summary.json").exists()
            assert (Path(report["report_dir"]) / "replicates_ranking.csv").exists()
        finally:
            routes_mod.run_replicates = orig
            routes_mod._current_run = None

    def test_seeds_derived_from_config_seed_when_omitted(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.replicates import ReplicateTrial

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}

        def fake_replicates(runner, base, seeds, metric, progress_callback=None):  # type: ignore[no-untyped-def]
            captured["seeds"] = seeds
            return [ReplicateTrial(s, "success", {"r2": 0.9}, 0.1) for s in seeds]

        orig = routes_mod.run_replicates
        routes_mod.run_replicates = fake_replicates
        try:
            client = TestClient(app, raise_server_exceptions=True)
            payload = {"config": _payload(tmp_path), "n_replicates": 3}
            resp = client.post("/api/regression/replicates", json=payload)
            assert resp.status_code == 200, resp.text

            for _ in range(50):
                if "seeds" in captured:
                    break
                time.sleep(0.05)
            # RegressionConfig defaults training.seed to 42 → 42, 43, 44
            assert captured["seeds"] == [42, 43, 44]
        finally:
            routes_mod.run_replicates = orig
            routes_mod._current_run = None

    def test_duplicate_seeds_rejected(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/replicates",
            json={"config": _payload(tmp_path), "seeds": [1, 1, 2]},
        )
        assert resp.status_code == 422

    def test_single_seed_rejected_by_schema(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/replicates",
            json={"config": _payload(tmp_path), "seeds": [1]},
        )
        assert resp.status_code == 422

    def test_conflict_when_already_running(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/replicates",
                json={"config": _payload(tmp_path)},
            )
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None

    def test_rejects_invalid_config(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        cfg = _payload(tmp_path)
        # num_targets=2 with a single target column → invalid RegressionConfig
        cfg["model"] = {"name": "resnet18", "num_targets": 2, "pretrained": False}
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/replicates",
            json={"config": cfg, "seeds": [1, 2]},
        )
        assert resp.status_code == 422
