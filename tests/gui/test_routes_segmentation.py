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
        "name": "seg_run",
        "model": {"name": "unet", "num_classes": 3, "pretrained": False},
        "data": {"base_dir": str(base)},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001},
        "output": {"models_dir": str(tmp_path / "models")},
    }


class TestSegmentationSchema:
    def test_schema_exposes_segmentation_config(self, app_and_routes: tuple) -> None:
        app, _ = app_and_routes
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/segmentation/schema")
        assert resp.status_code == 200
        schema = resp.json()
        assert "properties" in schema
        assert {"model", "data", "training"} <= set(schema["properties"])


class TestSegmentationRun:
    def test_run_dispatches_segmentation_block_with_callback(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.blocks.segmentation import SegmentationBlock

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}
        orig_setup = SegmentationBlock.setup
        orig_run = SegmentationBlock.run
        orig_report = SegmentationBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["class"] = type(self).__name__
            captured["model"] = config.model.name  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            captured["has_callback"] = self._progress_callback is not None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {"train": {"best_val_miou": 0.5, "run_dir": None}}

        SegmentationBlock.setup = fake_setup  # type: ignore[method-assign]
        SegmentationBlock.run = fake_run  # type: ignore[method-assign]
        SegmentationBlock.report = fake_report  # type: ignore[method-assign]

        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/segmentation/run", json=_payload(tmp_path))
            assert resp.status_code == 200, resp.text

            for _ in range(50):
                if "class" in captured:
                    break
                time.sleep(0.05)
            assert captured.get("class") == "SegmentationBlock"
            assert captured.get("model") == "unet"
            assert captured.get("has_callback") is True
        finally:
            SegmentationBlock.setup = orig_setup  # type: ignore[method-assign]
            SegmentationBlock.run = orig_run  # type: ignore[method-assign]
            SegmentationBlock.report = orig_report  # type: ignore[method-assign]
            routes_mod._current_run = None

    def test_run_conflict_when_already_running(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/segmentation/run", json=_payload(tmp_path))
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None

    def test_run_rejects_invalid_config(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        payload = _payload(tmp_path)
        # ignore_index (1) collides with a class id when num_classes=3 -> 422
        payload["data"] = {"base_dir": payload["data"]["base_dir"], "ignore_index": 1}
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/segmentation/run", json=payload)
        assert resp.status_code == 422


class TestSegmentationCompare:
    def test_compare_dispatches_and_reports(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.comparison import ComparisonTrial

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        def fake_compare(runner, config_dict, model_names, metric):  # type: ignore[no-untyped-def]
            return [
                ComparisonTrial("unet", "success", {"miou": 0.6}, training_time_s=2.0),
                ComparisonTrial("deeplabv3_resnet50", "failed", {}, error="boom"),
            ]

        orig = routes_mod.run_model_comparison
        routes_mod.run_model_comparison = fake_compare
        try:
            client = TestClient(app, raise_server_exceptions=True)
            payload = {
                "config": _payload(tmp_path),
                "model_names": ["unet", "deeplabv3_resnet50"],
            }
            resp = client.post("/api/segmentation/compare", json=payload)
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
            assert report["metric"] == "miou"  # defaulted from primary_metric
            assert report["total_ran"] == 2
            assert report["failed_count"] == 1
            assert report["top_3"][0]["model_arch"] == "unet"
        finally:
            routes_mod.run_model_comparison = orig
            routes_mod._current_run = None

    def test_compare_requires_two_models(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/segmentation/compare",
            json={"config": _payload(tmp_path), "model_names": ["unet"]},
        )
        assert resp.status_code == 422


class TestSegmentationSweep:
    def test_random_sweep_dispatches_and_reports(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.sweep import SweepTrial

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        def fake_sweep(runner, base, space, *, mode, metric, n_trials, seed):  # type: ignore[no-untyped-def]
            return [
                SweepTrial(
                    0, {"training.learning_rate": 0.02}, "success", {"miou": 0.7}
                )
            ]

        orig = routes_mod.run_sweep
        routes_mod.run_sweep = fake_sweep
        try:
            client = TestClient(app, raise_server_exceptions=True)
            payload = {
                "config": _payload(tmp_path),
                "mode": "random",
                "search_space": {
                    "training.learning_rate": {
                        "type": "uniform",
                        "low": 0.001,
                        "high": 0.1,
                    }
                },
                "n_trials": 1,
            }
            resp = client.post("/api/segmentation/sweep", json=payload)
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
            assert report["mode"] == "random"
            assert report["metric"] == "miou"  # defaulted from primary_metric
            assert report["successful_trials"] == 1
        finally:
            routes_mod.run_sweep = orig
            routes_mod._current_run = None
