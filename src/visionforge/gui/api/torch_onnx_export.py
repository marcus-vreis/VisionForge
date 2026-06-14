"""ONNX export for regression and segmentation runs (ADR-041 / shared core helper).

Detection exports via Ultralytics (``detection_export``) and classification via
``ExportONNXBlock``; these two reuse ``core.onnx_export`` directly, building the
model from each task's factory + the run's checkpoint. The architecture is built
with ``pretrained=False`` (the checkpoint supplies the weights), so export never
triggers an ImageNet download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from visionforge.core.onnx_export import export_to_onnx
from visionforge.gui.api.schemas import ExportOnnxRequest, ExportOnnxResponse
from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.models.segmentation_factory import (
    SegmentationModelFactory,
    segmentation_logits,
)
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig


def _checkpoint(data: dict[str, Any], run_name: str) -> str:
    path = data.get("artifacts", {}).get("model")
    if not path:
        raise FileNotFoundError(
            f"Run '{run_name}' has no checkpoint recorded in artifacts.model."
        )
    return str(path)


def _load_state(model: nn.Module, checkpoint: str) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # type: ignore[arg-type]
    model.eval()


def _build_response(
    run_dir: Path, req: ExportOnnxRequest, model: nn.Module, image_size: int
) -> ExportOnnxResponse:
    output_onnx = req.output_onnx or str((run_dir / "best_model.onnx").resolve())
    report = export_to_onnx(
        model,
        image_size,
        Path(output_onnx),
        opset_version=req.opset_version,
        dynamic_axes=req.dynamic_axes,
        validate=req.run_validate,
        benchmark=req.benchmark,
        benchmark_runs=req.benchmark_runs,
    )
    return ExportOnnxResponse(
        output_onnx=output_onnx,
        file_size_bytes=report["file_size_bytes"],
        validation=report["validation"],
        benchmark=report["benchmark"],
    )


def export_regression_run(
    run_dir: Path, req: ExportOnnxRequest, data: dict[str, Any]
) -> ExportOnnxResponse:
    """Export a regression run's checkpoint to ONNX."""
    config = RegressionConfig.model_validate(data["config"])
    model_cfg = config.model.model_copy(
        update={"pretrained": False, "weights_path": None}
    )
    model = RegressionModelFactory.create(model_cfg)
    _load_state(model, _checkpoint(data, run_dir.name))
    return _build_response(run_dir, req, model, config.data.transforms.image_size)


class _SegLogits(nn.Module):
    """Wrap a segmentation model so ONNX export/validation see a plain tensor.

    torchvision segmentation models return an ``OrderedDict`` (``{"out": ...}``);
    ``segmentation_logits`` normalizes that (and U-Net's tensor) to one tensor.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return segmentation_logits(self.model(x))


def export_segmentation_run(
    run_dir: Path, req: ExportOnnxRequest, data: dict[str, Any]
) -> ExportOnnxResponse:
    """Export a segmentation run's checkpoint to ONNX (logits-wrapped)."""
    config = SegmentationConfig.model_validate(data["config"])
    model_cfg = config.model.model_copy(
        update={"pretrained": False, "weights_path": None}
    )
    base = SegmentationModelFactory.create(model_cfg)
    _load_state(base, _checkpoint(data, run_dir.name))
    return _build_response(run_dir, req, _SegLogits(base), config.data.image_size)


__all__ = ["export_regression_run", "export_segmentation_run"]
