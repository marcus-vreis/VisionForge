"""Tests for /api/device/info, /api/runs/{id}, and /api/runs/{id}/test."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api.routes import _find_run_dir
from visionforge.utils.cuda import CUDAInfo, GPUDevice

_TS = "2026-05-22T10:00:00"


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


def _write_run(tmp: Path, run_id: str = "20260522_100000_000000") -> Path:
    run_dir = tmp / "exp1" / run_id
    run_dir.mkdir(parents=True)
    data = {
        "id": f"exp1_{run_id}",
        "experiment": "exp1",
        "timestamp": _TS,
        "status": "completed",
        "device_used": "cpu",
        "run_dir": str(run_dir.resolve()),
        "config": {
            "name": "exp1",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 4,
                "seed": 0,
            },
            "data": {"base_dir": str(tmp)},
        },
        "metrics": {
            "best_val_loss": 0.5,
            "best_epoch": 1,
            "total_epochs": 1,
            "test_accuracy": 0.8,
        },
        "history": [],
        "artifacts": {"model": str(run_dir / "best_model.pth"), "graphics": []},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


class TestDeviceInfoEndpoint:
    def test_returns_no_gpu_when_cuda_unavailable(self, app_and_routes: tuple) -> None:
        """GET /api/device/info must return cuda_available=False without GPUs."""
        app, _ = app_and_routes
        with patch(
            "visionforge.gui.api.routes.check_cuda",
            return_value=CUDAInfo(available=False),
        ):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/device/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cuda_available"] is False
        assert body["gpus"] == []

    def test_returns_gpu_list_when_available(self, app_and_routes: tuple) -> None:
        """GET /api/device/info must expose every detected GPU."""
        app, _ = app_and_routes
        fake = CUDAInfo(
            available=True,
            device_count=2,
            current_device=0,
            device_name="NVIDIA RTX 4090",
            cuda_version="12.4",
            devices=(
                GPUDevice(
                    index=0,
                    name="RTX 4090",
                    total_memory_mb=24576,
                    compute_capability="8.9",
                ),
                GPUDevice(
                    index=1,
                    name="RTX 4090",
                    total_memory_mb=24576,
                    compute_capability="8.9",
                ),
            ),
        )
        with patch("visionforge.gui.api.routes.check_cuda", return_value=fake):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/device/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cuda_available"] is True
        assert body["cuda_version"] == "12.4"
        assert len(body["gpus"]) == 2
        assert body["gpus"][0]["index"] == 0
        assert body["gpus"][0]["total_memory_mb"] == 24576


class TestRunDetailEndpoint:
    def test_unknown_run_returns_404(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/runs/does_not_exist")
        assert resp.status_code == 404

    def test_known_run_returns_full_detail(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get(f"/api/runs/{run_dir.name}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == run_dir.name
        assert body["experiment_name"] == "exp1"
        assert body["device_used"] == "cpu"
        assert body["metrics"]["test_accuracy"] == 0.8
        assert body["started_at"] == datetime.fromisoformat(_TS).isoformat()
        assert body["tests"] == []


class TestFindRunDir:
    def test_by_folder_name(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        with patch("visionforge.gui.api.routes._MODELS_DIR", tmp_path):
            found = _find_run_dir(run_dir.name)
        assert found == run_dir

    def test_by_run_id_field(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path, run_id="20260522_100001_000000")
        with patch("visionforge.gui.api.routes._MODELS_DIR", tmp_path):
            found = _find_run_dir("exp1_20260522_100001_000000")
        assert found == run_dir

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        with patch("visionforge.gui.api.routes._MODELS_DIR", tmp_path):
            found = _find_run_dir("nothing-here")
        assert found is None


class TestExportRunMarkdown:
    def test_returns_markdown_with_filename(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get(f"/api/runs/{run_dir.name}/export_md")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert "attachment" in resp.headers.get("content-disposition", "")
        body = resp.text
        # The markdown must include the run identity + at least one metric.
        assert "exp1" in body
        assert "Run ID" in body or "run_id" in body.lower()

    def test_404_for_unknown_run(self, app_and_routes: tuple, tmp_path: Path) -> None:
        app, routes_mod = app_and_routes
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/runs/missing/export_md")
        assert resp.status_code == 404
