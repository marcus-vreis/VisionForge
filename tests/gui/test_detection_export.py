"""Tests for the detection ONNX export path (Ultralytics-native)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from visionforge.gui.api.detection_export import export_detection_run
from visionforge.gui.api.schemas import ExportOnnxRequest


def _make_run(run_dir: Path, *, backend: str = "ultralytics", checkpoint: str) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": f"det_{run_dir.name}",
        "experiment": "det",
        "timestamp": "2026-06-03T12:00:00",
        "status": "completed",
        "config": {
            "task": "detection",
            "model": {"backend": backend, "name": "yolo11n", "num_classes": 2},
        },
        "artifacts": {"model": checkpoint, "graphics": []},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return data


class TestDetectionOnnxExport:
    def test_ultralytics_export_records_artifact(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "models" / "det" / "run0"
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"fake")
        onnx = tmp_path / "best.onnx"
        onnx.write_bytes(b"x" * 123)
        data = _make_run(run_dir, checkpoint=str(ckpt))

        fake_model = MagicMock()
        fake_model.export.return_value = str(onnx)
        fake_yolo = MagicMock(return_value=fake_model)

        with patch("visionforge.core.detection_trainer.YOLO", fake_yolo):
            resp = export_detection_run(run_dir, ExportOnnxRequest(), data)

        assert resp.file_size_bytes == 123
        assert Path(resp.output_onnx) == onnx.resolve()
        fake_model.export.assert_called_once()
        on_disk = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert Path(on_disk["artifacts"]["onnx"]) == onnx.resolve()

    def test_torchvision_export_unsupported(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "models" / "det" / "run1"
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"fake")
        data = _make_run(run_dir, backend="torchvision", checkpoint=str(ckpt))
        with pytest.raises(ValueError, match="torchvision"):
            export_detection_run(run_dir, ExportOnnxRequest(), data)

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "models" / "det" / "run2"
        data = _make_run(run_dir, checkpoint=str(tmp_path / "missing.pt"))
        with pytest.raises(FileNotFoundError):
            export_detection_run(run_dir, ExportOnnxRequest(), data)

    def test_ultralytics_absent_raises(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "models" / "det" / "run3"
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"fake")
        data = _make_run(run_dir, checkpoint=str(ckpt))
        with patch("visionforge.core.detection_trainer.YOLO", None):
            with pytest.raises(RuntimeError, match="ultralytics"):
                export_detection_run(run_dir, ExportOnnxRequest(), data)
