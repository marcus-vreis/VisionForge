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
        "name": "reg_cv",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(base), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }


class TestRegressionCv:
    def test_cv_dispatches_and_reports_fold_table(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.blocks.regression_cv import (
            CrossValidationReport,
            FoldResult,
        )

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}

        def fake_cv(config, *, n_folds, shuffle, seed, progress_callback=None):  # type: ignore[no-untyped-def]
            captured["n_folds"] = n_folds
            captured["shuffle"] = shuffle
            captured["seed"] = seed
            folds = [
                FoldResult(0, "success", 8, 2, {"r2": 0.8, "rmse": 1.0}),
                FoldResult(1, "success", 8, 2, {"r2": 0.9, "rmse": 0.8}),
            ]
            return CrossValidationReport(
                n_folds=n_folds,
                metric="r2",
                folds=folds,
                aggregate={"r2": {"mean": 0.85, "std": 0.05}},
            )

        orig = routes_mod.run_regression_cross_validation
        routes_mod.run_regression_cross_validation = fake_cv
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/cv",
                json={
                    "config": _payload(tmp_path),
                    "n_folds": 2,
                    "shuffle": True,
                    "fold_seed": 7,
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

            assert captured == {"n_folds": 2, "shuffle": True, "seed": 7}

            report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
            assert report["n_folds"] == 2
            assert report["metric"] == "r2"
            assert report["successful_folds"] == 2
            assert [f["fold"] for f in report["fold_results"]] == [0, 1]
            assert report["aggregate"]["r2"]["mean"] == 0.85
            # persisted, incl. the flat per-fold CSV (fold_results rows)
            assert (Path(report["report_dir"]) / "cv_summary.json").exists()
            assert (Path(report["report_dir"]) / "cv_ranking.csv").exists()
        finally:
            routes_mod.run_regression_cross_validation = orig
            routes_mod._current_run = None

    def test_conflict_when_already_running(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/regression/cv",
                json={"config": _payload(tmp_path), "n_folds": 3},
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
        cfg["model"] = {"name": "resnet18", "num_targets": 2, "pretrained": False}
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/regression/cv", json={"config": cfg, "n_folds": 3})
        assert resp.status_code == 422

    def test_rejects_single_fold(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/regression/cv", json={"config": _payload(tmp_path), "n_folds": 1}
        )
        assert resp.status_code == 422
