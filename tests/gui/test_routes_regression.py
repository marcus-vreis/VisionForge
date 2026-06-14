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
        "name": "reg_run",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(base), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }


class TestRegressionSchema:
    def test_schema_exposes_regression_config(self, app_and_routes: tuple) -> None:
        app, _ = app_and_routes
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/regression/schema")
        assert resp.status_code == 200
        schema = resp.json()
        assert "properties" in schema
        assert {"model", "data", "training"} <= set(schema["properties"])


class TestRegressionRun:
    def test_run_dispatches_regression_block_with_callback(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.blocks.regression import RegressionBlock

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}
        orig_setup = RegressionBlock.setup
        orig_run = RegressionBlock.run
        orig_report = RegressionBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["class"] = type(self).__name__
            captured["model"] = config.model.name  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            captured["has_callback"] = self._progress_callback is not None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {"train": {"best_val_loss": 0.1, "run_dir": None}}

        RegressionBlock.setup = fake_setup  # type: ignore[method-assign]
        RegressionBlock.run = fake_run  # type: ignore[method-assign]
        RegressionBlock.report = fake_report  # type: ignore[method-assign]

        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/regression/run", json=_payload(tmp_path))
            assert resp.status_code == 200, resp.text

            for _ in range(50):
                if "class" in captured:
                    break
                time.sleep(0.05)
            assert captured.get("class") == "RegressionBlock"
            assert captured.get("model") == "resnet18"
            assert captured.get("has_callback") is True
        finally:
            RegressionBlock.setup = orig_setup  # type: ignore[method-assign]
            RegressionBlock.run = orig_run  # type: ignore[method-assign]
            RegressionBlock.report = orig_report  # type: ignore[method-assign]
            routes_mod._current_run = None

    def test_run_conflict_when_already_running(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/regression/run", json=_payload(tmp_path))
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None

    def test_run_rejects_invalid_config(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        payload = _payload(tmp_path)
        # num_targets (2) must equal len(target_columns) (1) -> 422
        payload["model"] = {"name": "resnet18", "num_targets": 2, "pretrained": False}
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/regression/run", json=payload)
        assert resp.status_code == 422


class TestRegressionCompare:
    def test_compare_dispatches_and_reports(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.comparison import ComparisonTrial

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        def fake_compare(runner, config_dict, model_names, metric):  # type: ignore[no-untyped-def]
            return [
                ComparisonTrial(
                    "resnet50", "success", {"r2": 0.9}, training_time_s=2.0
                ),
                ComparisonTrial(
                    "resnet18", "success", {"r2": 0.7}, training_time_s=1.0
                ),
            ]

        orig = routes_mod.run_model_comparison
        routes_mod.run_model_comparison = fake_compare
        try:
            client = TestClient(app, raise_server_exceptions=True)
            payload = {
                "config": _payload(tmp_path),
                "model_names": ["resnet18", "resnet50"],
                "metric": "r2",
            }
            resp = client.post("/api/regression/compare", json=payload)
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
            assert report["metric"] == "r2"
            assert report["total_ran"] == 2
            assert report["failed_count"] == 0
            assert report["top_3"][0]["model_arch"] == "resnet50"
            # the ranking is persisted to outputs/reports for later reference
            assert (Path(report["report_dir"]) / "comparison_summary.json").exists()
            assert (Path(report["report_dir"]) / "comparison_ranking.csv").exists()
        finally:
            routes_mod.run_model_comparison = orig
            routes_mod._current_run = None

    def test_compare_conflict_when_already_running(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/compare",
                json={"config": _payload(tmp_path), "model_names": ["a", "b"]},
            )
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None

    def test_compare_rejects_invalid_config(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        cfg = _payload(tmp_path)
        cfg["model"] = {"name": "resnet18", "num_targets": 2, "pretrained": False}
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/compare",
            json={"config": cfg, "model_names": ["resnet18", "resnet50"]},
        )
        assert resp.status_code == 422

    def test_compare_requires_two_models(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/compare",
            json={"config": _payload(tmp_path), "model_names": ["resnet18"]},
        )
        assert resp.status_code == 422


class TestRegressionSweep:
    def test_grid_sweep_dispatches_and_reports(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.sweep import SweepTrial

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        def fake_sweep(runner, base, space, *, mode, metric, n_trials, seed):  # type: ignore[no-untyped-def]
            return [
                SweepTrial(0, {"training.learning_rate": 0.05}, "success", {"r2": 0.9}),
                SweepTrial(1, {"training.learning_rate": 0.1}, "success", {"r2": 0.6}),
            ]

        orig = routes_mod.run_sweep
        routes_mod.run_sweep = fake_sweep
        try:
            client = TestClient(app, raise_server_exceptions=True)
            payload = {
                "config": _payload(tmp_path),
                "mode": "grid",
                "search_space": {"training.learning_rate": [0.05, 0.1]},
                "metric": "r2",
            }
            resp = client.post("/api/regression/sweep", json=payload)
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
            assert report["metric"] == "r2"
            assert report["total_trials"] == 2
            assert report["best_trial"]["overrides"]["training.learning_rate"] == 0.05
            assert (Path(report["report_dir"]) / "sweep_summary.json").exists()
        finally:
            routes_mod.run_sweep = orig
            routes_mod._current_run = None

    def test_sweep_rejects_unknown_path(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/sweep",
            json={
                "config": _payload(tmp_path),
                "mode": "grid",
                "search_space": {"training.nope": [1, 2]},
            },
        )
        assert resp.status_code == 422

    def test_sweep_conflict_when_already_running(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/sweep",
                json={
                    "config": _payload(tmp_path),
                    "search_space": {"training.learning_rate": [0.1]},
                },
            )
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None
