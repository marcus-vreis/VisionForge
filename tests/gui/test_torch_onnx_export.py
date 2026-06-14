from __future__ import annotations

import json
from pathlib import Path

import torch

from visionforge.gui.api.schemas import ExportOnnxRequest
from visionforge.gui.api.torch_onnx_export import (
    export_regression_run,
    export_segmentation_run,
)
from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.models.segmentation_factory import SegmentationModelFactory
from visionforge.utils.regression_config import RegressionModelConfig
from visionforge.utils.segmentation_config import SegmentationModelConfig


def _write_run(run_dir: Path, config: dict, model: torch.nn.Module) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best.pt"
    torch.save(model.state_dict(), ckpt)
    data = {"config": config, "artifacts": {"model": str(ckpt)}}
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def test_regression_export_creates_validated_onnx(tmp_path: Path) -> None:
    model = RegressionModelFactory.create(
        RegressionModelConfig(name="resnet18", num_targets=1, pretrained=False)
    )
    config = {
        "name": "reg",
        "task": "regression",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {
            "base_dir": str(tmp_path),
            "target_columns": ["t"],
            "transforms": {"image_size": 64},
        },
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
        "output": {"models_dir": str(tmp_path / "models")},
    }
    run_dir = tmp_path / "reg_run"
    data = _write_run(run_dir, config, model)

    resp = export_regression_run(run_dir, ExportOnnxRequest(benchmark=False), data)
    assert Path(resp.output_onnx).exists()
    assert resp.file_size_bytes > 0
    assert resp.validation is not None and resp.validation["passed"] is True


def test_segmentation_export_handles_dict_output(tmp_path: Path) -> None:
    model = SegmentationModelFactory.create(
        SegmentationModelConfig(name="unet", num_classes=2, pretrained=False)
    )
    config = {
        "name": "seg",
        "task": "segmentation",
        "model": {"name": "unet", "num_classes": 2, "pretrained": False},
        "data": {"base_dir": str(tmp_path), "image_size": 64},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001},
        "output": {"models_dir": str(tmp_path / "models")},
    }
    run_dir = tmp_path / "seg_run"
    data = _write_run(run_dir, config, model)

    resp = export_segmentation_run(run_dir, ExportOnnxRequest(benchmark=False), data)
    assert Path(resp.output_onnx).exists()
    assert resp.file_size_bytes > 0
    assert resp.validation is not None and resp.validation["passed"] is True
