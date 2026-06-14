from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visionforge.gui.api.routes import _execute_run_gradcam
from visionforge.gui.api.schemas import GradCamRequest


def _make_run_dir(tmp: Path, *, task: str = "binary", num_classes: int = 2) -> Path:
    run_dir = tmp / "models" / "exp1" / "20260604_120000_000000"
    run_dir.mkdir(parents=True)

    from visionforge.models.factory import ModelFactory
    from visionforge.utils.config import ModelConfig

    model = ModelFactory.create(
        ModelConfig(name="resnet18", num_classes=num_classes, pretrained=False)
    )
    ckpt_path = run_dir / "best_model.pth"
    torch.save(model.state_dict(), ckpt_path)

    run_json = {
        "id": "exp1_20260604_120000_000000",
        "experiment": "exp1",
        "status": "completed",
        "config": {
            "name": "exp1",
            "task": task,
            "model": {
                "name": "resnet18",
                "num_classes": num_classes,
                "pretrained": False,
            },
            "data": {"base_dir": str(tmp), "transforms": {"image_size": 32}},
        },
        "artifacts": {"model": str(ckpt_path), "graphics": []},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return run_dir


def _make_input_dir(tmp: Path, n: int = 3) -> Path:
    d = tmp / "samples"
    d.mkdir(parents=True)
    for i in range(n):
        arr = np.random.default_rng(i).integers(0, 255, (40, 40, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(d / f"img{i}.png")
    return d


class TestExecuteRunGradcam:
    def test_generates_overlays_for_samples(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        input_dir = _make_input_dir(tmp_path, n=3)
        resp = _execute_run_gradcam(
            run_dir, GradCamRequest(input_dir=str(input_dir), num_samples=3)
        )
        assert resp.count == 3
        assert len(resp.items) == 3
        for item in resp.items:
            assert Path(item.overlay).is_file()
            assert item.overlay.endswith(".png")

    def test_respects_num_samples_cap(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        input_dir = _make_input_dir(tmp_path, n=5)
        resp = _execute_run_gradcam(
            run_dir, GradCamRequest(input_dir=str(input_dir), num_samples=2)
        )
        assert resp.count == 2

    def test_target_class_is_predicted_when_unset(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, num_classes=2)
        input_dir = _make_input_dir(tmp_path, n=1)
        resp = _execute_run_gradcam(
            run_dir, GradCamRequest(input_dir=str(input_dir), num_samples=1)
        )
        assert resp.items[0].predicted_class in (0, 1)

    def test_detection_run_rejected(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, task="detection")
        input_dir = _make_input_dir(tmp_path, n=1)
        with pytest.raises(ValueError, match="classification"):
            _execute_run_gradcam(
                run_dir, GradCamRequest(input_dir=str(input_dir), num_samples=1)
            )

    def test_empty_input_dir_raises(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="[Nn]o image"):
            _execute_run_gradcam(
                run_dir, GradCamRequest(input_dir=str(empty), num_samples=3)
            )

    def test_anomaly_run_rejected(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, task="anomaly")
        input_dir = _make_input_dir(tmp_path, n=1)
        with pytest.raises(ValueError, match="not available"):
            _execute_run_gradcam(
                run_dir, GradCamRequest(input_dir=str(input_dir), num_samples=1)
            )


def _make_regression_run_dir(tmp: Path) -> Path:
    from visionforge.models.regression_factory import RegressionModelFactory
    from visionforge.utils.regression_config import RegressionModelConfig

    run_dir = tmp / "models" / "reg1" / "20260604_120000_000000"
    run_dir.mkdir(parents=True)
    model = RegressionModelFactory.create(
        RegressionModelConfig(name="resnet18", num_targets=2, pretrained=False)
    )
    ckpt = run_dir / "best_model.pth"
    torch.save(model.state_dict(), ckpt)
    run_json = {
        "id": "reg1_20260604_120000_000000",
        "experiment": "reg1",
        "status": "completed",
        "config": {
            "name": "reg1",
            "task": "regression",
            "model": {"name": "resnet18", "num_targets": 2, "pretrained": False},
            "data": {"base_dir": str(tmp), "transforms": {"image_size": 32}},
        },
        "artifacts": {"model": str(ckpt), "graphics": []},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return run_dir


def _make_segmentation_run_dir(tmp: Path) -> Path:
    from visionforge.models.segmentation_factory import SegmentationModelFactory
    from visionforge.utils.segmentation_config import SegmentationModelConfig

    run_dir = tmp / "models" / "seg1" / "20260604_120000_000000"
    run_dir.mkdir(parents=True)
    model = SegmentationModelFactory.create(
        SegmentationModelConfig(
            name="lraspp_mobilenet_v3_large", num_classes=3, pretrained=False
        )
    )
    ckpt = run_dir / "best_model.pth"
    torch.save(model.state_dict(), ckpt)
    run_json = {
        "id": "seg1_20260604_120000_000000",
        "experiment": "seg1",
        "status": "completed",
        "config": {
            "name": "seg1",
            "task": "segmentation",
            "model": {
                "name": "lraspp_mobilenet_v3_large",
                "num_classes": 3,
                "pretrained": False,
            },
            "data": {"base_dir": str(tmp), "image_size": 64},
        },
        "artifacts": {"model": str(ckpt), "graphics": []},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return run_dir


class TestRegressionGradcam:
    def test_saliency_overlays_with_value_label(self, tmp_path: Path) -> None:
        run_dir = _make_regression_run_dir(tmp_path)
        input_dir = _make_input_dir(tmp_path, n=2)
        resp = _execute_run_gradcam(
            run_dir, GradCamRequest(input_dir=str(input_dir), num_samples=2)
        )
        assert resp.count == 2
        for item in resp.items:
            assert Path(item.overlay).is_file()
            assert item.predicted_class is None
            assert item.prediction is not None and item.prediction.startswith("pred=")


class TestSegmentationGradcam:
    def test_per_class_cam_overlays(self, tmp_path: Path) -> None:
        run_dir = _make_segmentation_run_dir(tmp_path)
        input_dir = _make_input_dir(tmp_path, n=2)
        resp = _execute_run_gradcam(
            run_dir,
            GradCamRequest(input_dir=str(input_dir), num_samples=2, target_class=1),
        )
        assert resp.count == 2
        for item in resp.items:
            assert Path(item.overlay).is_file()
            assert item.predicted_class == 1  # the requested target class
            assert item.prediction is not None and "class 1" in item.prediction
