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


class TestDeleteRun:
    def test_unknown_run_returns_404(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """DELETE on a missing run must be a hard 404, not a silent no-op."""
        app, routes_mod = app_and_routes
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.delete("/api/runs/does_not_exist")
        assert resp.status_code == 404

    def test_known_run_removes_directory(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """DELETE removes the run_dir from disk and reports the run_id back."""
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.delete(f"/api/runs/{run_dir.name}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == run_dir.name
        assert body["status"] == "deleted"
        assert not run_dir.exists()

    def test_refuses_to_delete_running_run(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """The currently executing run must not be deletable — its files are
        open, and removing them would race the trainer."""
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        routes_mod._current_run = {
            "run_id": run_dir.name,
            "status": "running",
            "error": None,
        }
        try:
            with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
                client = TestClient(app, raise_server_exceptions=True)
                resp = client.delete(f"/api/runs/{run_dir.name}")
            assert resp.status_code == 409
            assert run_dir.exists()
        finally:
            routes_mod._current_run = None


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

    def test_markdown_includes_preprocessing_and_augmentation(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """Model card must surface preprocessing + augmentation so the run is reproducible."""
        app, routes_mod = app_and_routes
        run_id = "20260522_110000_000000"
        run_dir = tmp_path / "exp1" / run_id
        run_dir.mkdir(parents=True)
        data = {
            "id": f"exp1_{run_id}",
            "experiment": "exp1",
            "timestamp": _TS,
            "status": "completed",
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
                "data": {
                    "base_dir": str(tmp_path),
                    "preprocessing": {
                        "steps": [
                            {"kind": "gaussian_blur", "radius": 1.5},
                            {"kind": "wavelet", "band": "LL"},
                        ]
                    },
                    "transforms": {
                        "image_size": 224,
                        "horizontal_flip": True,
                        "rotation_degrees": 15,
                    },
                },
            },
            "metrics": {"best_val_loss": 0.4, "best_epoch": 1, "total_epochs": 1},
            "history": [],
            "artifacts": {"model": str(run_dir / "best_model.pth"), "graphics": []},
            "tests": [],
        }
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get(f"/api/runs/{run_id}/export_md")
        assert resp.status_code == 200
        body = resp.text
        assert "Preprocessing pipeline" in body
        assert "gaussian_blur" in body
        assert "wavelet" in body
        assert "Augmentation" in body
        assert "horizontal_flip" in body
        assert "image_size" in body


class TestExportRunToOnnx:
    def test_unknown_run_returns_404(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/runs/missing/export_onnx", json={})
        assert resp.status_code == 404

    def test_missing_checkpoint_returns_400(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """run.json without artifacts.model must surface as a 400 explanation."""
        app, routes_mod = app_and_routes
        run_id = "20260523_120000_000000"
        run_dir = tmp_path / "exp1" / run_id
        run_dir.mkdir(parents=True)
        data = {
            "id": f"exp1_{run_id}",
            "experiment": "exp1",
            "timestamp": _TS,
            "status": "completed",
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
                "data": {"base_dir": str(tmp_path)},
            },
            "metrics": {},
            "history": [],
            "artifacts": {},
            "tests": [],
        }
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(f"/api/runs/{run_id}/export_onnx", json={})
        assert resp.status_code == 400
        assert "checkpoint" in resp.text.lower()

    def test_export_invokes_block_with_request_params(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """The endpoint must build an ExperimentConfig where export_onnx is
        populated from the request and run the block in a worker thread."""
        from visionforge.blocks.export_onnx import ExportONNXBlock

        app, routes_mod = app_and_routes

        run_id = "20260523_130000_000000"
        run_dir = tmp_path / "exp_export" / run_id
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "best_model.pth"
        checkpoint.write_bytes(b"fake-checkpoint")
        data = {
            "id": f"exp_export_{run_id}",
            "experiment": "exp_export",
            "timestamp": _TS,
            "status": "completed",
            "config": {
                "name": "exp_export",
                "task": "binary",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 4,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
            },
            "metrics": {},
            "history": [],
            "artifacts": {"model": str(checkpoint)},
            "tests": [],
        }
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")

        captured: dict[str, object] = {}
        original_setup = ExportONNXBlock.setup
        original_run = ExportONNXBlock.run
        original_report = ExportONNXBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["opset"] = config.export_onnx.opset_version  # type: ignore[attr-defined]
            captured["validate"] = config.export_onnx.run_validate  # type: ignore[attr-defined]
            captured["checkpoint"] = str(config.export_onnx.checkpoint_path)  # type: ignore[attr-defined]
            # Pretend the export wrote a 100-byte file at the configured path.
            cfg = config.export_onnx  # type: ignore[attr-defined]
            cfg.output_onnx.parent.mkdir(parents=True, exist_ok=True)
            cfg.output_onnx.write_bytes(b"x" * 100)

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            return None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {
                "file_size_bytes": 100,
                "validation": {
                    "max_diff": 1e-6,
                    "within_tolerance": True,
                    "tolerance": 1e-4,
                },
                "benchmark": {"mean_ms": 2.5, "std_ms": 0.1, "n_runs": 50},
            }

        ExportONNXBlock.setup = fake_setup  # type: ignore[method-assign]
        ExportONNXBlock.run = fake_run  # type: ignore[method-assign]
        ExportONNXBlock.report = fake_report  # type: ignore[method-assign]

        try:
            with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
                client = TestClient(app, raise_server_exceptions=True)
                resp = client.post(
                    f"/api/runs/{run_id}/export_onnx",
                    json={
                        "opset_version": 18,
                        "validate": True,
                        "benchmark": True,
                        "benchmark_runs": 50,
                    },
                )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["file_size_bytes"] == 100
            assert body["validation"]["within_tolerance"] is True
            assert body["benchmark"]["mean_ms"] == 2.5
            assert body["output_onnx"].endswith("best_model.onnx")
            assert captured.get("opset") == 18
            assert captured.get("validate") is True
            assert captured.get("checkpoint") == str(checkpoint)
        finally:
            ExportONNXBlock.setup = original_setup  # type: ignore[method-assign]
            ExportONNXBlock.run = original_run  # type: ignore[method-assign]
            ExportONNXBlock.report = original_report  # type: ignore[method-assign]


class TestBatchPredictRun:
    def test_unknown_run_returns_404(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/runs/missing/batch_predict",
                json={"input_dir": str(tmp_path)},
            )
        assert resp.status_code == 404

    def test_invalid_input_dir_returns_400(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """A non-existent input_dir surfaces as a 400, not a 500."""
        app, routes_mod = app_and_routes
        run_id = "20260523_140000_000000"
        run_dir = tmp_path / "exp_batch" / run_id
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "best_model.pth"
        checkpoint.write_bytes(b"fake")
        data = {
            "id": f"exp_batch_{run_id}",
            "experiment": "exp_batch",
            "timestamp": _TS,
            "status": "completed",
            "config": {
                "name": "exp_batch",
                "task": "binary",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 4,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
            },
            "metrics": {},
            "history": [],
            "artifacts": {"model": str(checkpoint)},
            "tests": [],
        }
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                f"/api/runs/{run_id}/batch_predict",
                json={"input_dir": str(tmp_path / "does_not_exist")},
            )
        assert resp.status_code == 400
        assert "input_dir" in resp.text

    def test_dispatch_writes_csv_and_returns_counts(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """The endpoint builds an ExperimentConfig with batch_prediction populated
        and surfaces the block's counts in the response."""
        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        app, routes_mod = app_and_routes

        run_id = "20260523_150000_000000"
        run_dir = tmp_path / "exp_b" / run_id
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "best_model.pth"
        checkpoint.write_bytes(b"fake-checkpoint")

        # Create an input directory that exists.
        input_dir = tmp_path / "inbox"
        input_dir.mkdir()

        data = {
            "id": f"exp_b_{run_id}",
            "experiment": "exp_b",
            "timestamp": _TS,
            "status": "completed",
            "config": {
                "name": "exp_b",
                "task": "binary",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 4,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
            },
            "metrics": {},
            "history": [],
            "artifacts": {"model": str(checkpoint)},
            "tests": [],
        }
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")

        captured: dict[str, object] = {}
        original_setup = BatchPredictionBlock.setup
        original_run = BatchPredictionBlock.run
        original_report = BatchPredictionBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["input_dir"] = str(config.batch_prediction.input_dir)  # type: ignore[attr-defined]
            captured["recursive"] = config.batch_prediction.recursive  # type: ignore[attr-defined]
            captured["output_csv"] = str(config.batch_prediction.output_csv)  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            return None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {
                "total_processed": 42,
                "failed_files": ["broken.png"],
                "output_csv": str(run_dir / "predictions.csv"),
            }

        BatchPredictionBlock.setup = fake_setup  # type: ignore[method-assign]
        BatchPredictionBlock.run = fake_run  # type: ignore[method-assign]
        BatchPredictionBlock.report = fake_report  # type: ignore[method-assign]

        try:
            with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
                client = TestClient(app, raise_server_exceptions=True)
                resp = client.post(
                    f"/api/runs/{run_id}/batch_predict",
                    json={"input_dir": str(input_dir), "recursive": True},
                )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["total_processed"] == 42
            assert body["failed_count"] == 1
            assert body["failed_files"] == ["broken.png"]
            # output_csv defaulted to run_dir/predictions/<ts>.csv
            assert "predictions" in body["output_csv"]
            assert captured.get("recursive") is True
            assert captured.get("input_dir") == str(input_dir.resolve())
        finally:
            BatchPredictionBlock.setup = original_setup  # type: ignore[method-assign]
            BatchPredictionBlock.run = original_run  # type: ignore[method-assign]
            BatchPredictionBlock.report = original_report  # type: ignore[method-assign]
