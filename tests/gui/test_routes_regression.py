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
        "output": {"models_dir": str(tmp_path / "models")},
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
