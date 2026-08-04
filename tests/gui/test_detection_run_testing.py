"""Tests for the detection per-model test path (evaluate checkpoint → mAP)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from visionforge.gui.api.detection_testing import evaluate_detection_run
from visionforge.gui.api.schemas import RunTestRequest


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)


def _make_yolo_val(base: Path, *, with_labels: bool = True) -> Path:
    """A minimal YOLO set: images/val + (optionally) labels/val."""
    _write_image(base / "images" / "val" / "img1.jpg")
    if with_labels:
        lbl = base / "labels" / "val" / "img1.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    return base


def _make_run_json(
    run_dir: Path, *, backend: str, model_name: str, num_classes: int, checkpoint: str
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": f"det_{run_dir.name}",
        "experiment": "det",
        "timestamp": "2026-06-03T12:00:00",
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": {
            "name": "det",
            "task": "detection",
            "model": {
                "backend": backend,
                "name": model_name,
                "num_classes": num_classes,
                "pretrained": False,
            },
            "data": {"base_dir": str(run_dir)},  # overwritten per test via base_dir
            "training": {"epochs": 1, "batch_size": 2, "workers": 0},
            "device": {"kind": "cpu"},
        },
        "metrics": {"map50": 0.4, "total_epochs": 1},
        "history": [],
        "artifacts": {"model": checkpoint, "graphics": []},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return data


class TestTorchvisionDetectionTest:
    def test_appends_map_record(self, tmp_path: Path) -> None:
        from visionforge.models.detection_factory import build_torchvision_detector

        ds = _make_yolo_val(tmp_path / "ds")
        run_dir = tmp_path / "models" / "det" / "20260603_120000_000000"
        # Save a state_dict matching the rebuilt architecture.
        model = build_torchvision_detector(
            "ssdlite320_mobilenet_v3_large", num_classes=1, pretrained=False
        )
        ckpt = tmp_path / "best.pt"
        torch.save(model.state_dict(), ckpt)
        data = _make_run_json(
            run_dir,
            backend="torchvision",
            model_name="ssdlite320_mobilenet_v3_large",
            num_classes=1,
            checkpoint=str(ckpt),
        )

        resp = evaluate_detection_run(
            run_dir, RunTestRequest(data_dir=str(ds), label="smoke"), data
        )

        assert "map50" in resp.metrics
        assert isinstance(resp.metrics["map50"], float)
        on_disk = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert len(on_disk["tests"]) == 1
        assert on_disk["tests"][0]["label"] == "smoke"

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        ds = _make_yolo_val(tmp_path / "ds")
        run_dir = tmp_path / "models" / "det" / "20260603_120000_000001"
        data = _make_run_json(
            run_dir,
            backend="torchvision",
            model_name="ssdlite320_mobilenet_v3_large",
            num_classes=1,
            checkpoint=str(tmp_path / "missing.pt"),
        )
        with pytest.raises(FileNotFoundError):
            evaluate_detection_run(run_dir, RunTestRequest(data_dir=str(ds)), data)

    def test_no_yolo_split_raises(self, tmp_path: Path) -> None:
        from visionforge.models.detection_factory import build_torchvision_detector

        empty = tmp_path / "empty"
        empty.mkdir()
        run_dir = tmp_path / "models" / "det" / "20260603_120000_000002"
        model = build_torchvision_detector(
            "ssdlite320_mobilenet_v3_large", num_classes=1, pretrained=False
        )
        ckpt = tmp_path / "best.pt"
        torch.save(model.state_dict(), ckpt)
        data = _make_run_json(
            run_dir,
            backend="torchvision",
            model_name="ssdlite320_mobilenet_v3_large",
            num_classes=1,
            checkpoint=str(ckpt),
        )
        with pytest.raises(ValueError, match="split"):
            evaluate_detection_run(run_dir, RunTestRequest(data_dir=str(empty)), data)


class TestUltralyticsDetectionTest:
    def test_mocked_val_records_map(self, tmp_path: Path) -> None:
        # Ultralytics needs train + val image dirs to synthesize data.yaml.
        ds = tmp_path / "ds"
        _write_image(ds / "images" / "train" / "t.jpg")
        _write_image(ds / "images" / "val" / "v.jpg")
        run_dir = tmp_path / "models" / "det" / "20260603_120000_000003"
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"fake")  # YOLO is mocked; file only needs to exist
        data = _make_run_json(
            run_dir,
            backend="ultralytics",
            model_name="yolo11n",
            num_classes=2,
            checkpoint=str(ckpt),
        )

        fake_results = SimpleNamespace(box=SimpleNamespace(map50=0.6, map=0.42))
        fake_model = MagicMock()
        fake_model.val.return_value = fake_results
        fake_yolo = MagicMock(return_value=fake_model)

        with patch("visionforge.core.detection_trainer.YOLO", fake_yolo):
            resp = evaluate_detection_run(
                run_dir, RunTestRequest(data_dir=str(ds)), data
            )

        assert resp.metrics == {
            "map50": pytest.approx(0.6),
            "map50_95": pytest.approx(0.42),
        }
        fake_model.val.assert_called_once()
        on_disk = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert len(on_disk["tests"]) == 1

    def test_raises_when_ultralytics_absent(self, tmp_path: Path) -> None:
        ds = tmp_path / "ds"
        _write_image(ds / "images" / "train" / "t.jpg")
        _write_image(ds / "images" / "val" / "v.jpg")
        run_dir = tmp_path / "models" / "det" / "20260603_120000_000004"
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"fake")
        data = _make_run_json(
            run_dir,
            backend="ultralytics",
            model_name="yolo11n",
            num_classes=2,
            checkpoint=str(ckpt),
        )
        with patch("visionforge.core.detection_trainer.YOLO", None):
            with pytest.raises(RuntimeError, match="ultralytics"):
                evaluate_detection_run(run_dir, RunTestRequest(data_dir=str(ds)), data)
