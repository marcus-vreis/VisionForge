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
    for split in ("train", "val"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
    return {
        "name": "det_run",
        "model": {"backend": "ultralytics", "name": "yolo11n", "num_classes": 2},
        "data": {"base_dir": str(base), "image_size": 640},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.01},
        "output": {"models_dir": str(tmp_path / "models")},
    }


class TestDetectionSchema:
    def test_schema_exposes_detection_config(self, app_and_routes: tuple) -> None:
        app, _ = app_and_routes
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/detection/schema")
        assert resp.status_code == 200
        schema = resp.json()
        assert "properties" in schema
        assert {"model", "data", "training"} <= set(schema["properties"])


class TestDetectionRun:
    def test_run_dispatches_detection_block_with_callback(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.blocks.detection import DetectionBlock

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}
        orig_setup = DetectionBlock.setup
        orig_run = DetectionBlock.run
        orig_report = DetectionBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["class"] = type(self).__name__
            captured["model"] = config.model.name  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            captured["has_callback"] = self._progress_callback is not None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {"detection": {"best_map50_95": 0.4, "run_dir": None}}

        DetectionBlock.setup = fake_setup  # type: ignore[method-assign]
        DetectionBlock.run = fake_run  # type: ignore[method-assign]
        DetectionBlock.report = fake_report  # type: ignore[method-assign]

        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/detection/run", json=_payload(tmp_path))
            assert resp.status_code == 200, resp.text

            for _ in range(50):
                if "class" in captured:
                    break
                time.sleep(0.05)
            assert captured.get("class") == "DetectionBlock"
            assert captured.get("model") == "yolo11n"
            assert captured.get("has_callback") is True
        finally:
            DetectionBlock.setup = orig_setup  # type: ignore[method-assign]
            DetectionBlock.run = orig_run  # type: ignore[method-assign]
            DetectionBlock.report = orig_report  # type: ignore[method-assign]
            routes_mod._current_run = None

    def test_run_conflict_when_already_running(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = {"run_id": "x", "status": "running"}
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/detection/run", json=_payload(tmp_path))
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None

    def test_run_rejects_invalid_config(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        payload = _payload(tmp_path)
        payload["model"] = {"backend": "ultralytics", "name": "not_a_model"}
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/detection/run", json=payload)
        assert resp.status_code == 422
