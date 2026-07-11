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
        "name": "seg_cv",
        "model": {"name": "unet", "num_classes": 2, "pretrained": False},
        "data": {"base_dir": str(base), "image_size": 32},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }


class TestSegmentationCv:
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

        def fake_cv(config, *, n_folds, shuffle, seed):  # type: ignore[no-untyped-def]
            captured["n_folds"] = n_folds
            captured["config_type"] = type(config).__name__
            folds = [
                FoldResult(0, "success", 4, 2, {"miou": 0.6, "dice": 0.7}),
                FoldResult(1, "success", 4, 2, {"miou": 0.8, "dice": 0.9}),
            ]
            return CrossValidationReport(
                n_folds=n_folds,
                metric="miou",
                folds=folds,
                aggregate={"miou": {"mean": 0.7, "std": 0.1}},
            )

        orig = routes_mod.run_segmentation_cross_validation
        routes_mod.run_segmentation_cross_validation = fake_cv
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/segmentation/cv",
                json={"config": _payload(tmp_path), "n_folds": 2},
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

            assert captured["n_folds"] == 2
            assert captured["config_type"] == "SegmentationConfig"

            report = client.get(f"/api/experiment/result/{run_id}").json()["report"]
            assert report["metric"] == "miou"
            assert report["successful_folds"] == 2
            assert report["aggregate"]["miou"]["mean"] == 0.7
            assert (Path(report["report_dir"]) / "cv_summary.json").exists()
        finally:
            routes_mod.run_segmentation_cross_validation = orig
            routes_mod._current_run = None

    def test_conflict_when_already_running(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/segmentation/cv",
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
        # ignore_index colliding with a class id → invalid SegmentationConfig
        cfg["data"]["ignore_index"] = 1
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/segmentation/cv", json={"config": cfg, "n_folds": 3})
        assert resp.status_code == 422
