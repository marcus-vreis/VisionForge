from __future__ import annotations

from typing import Any

import torch
from torch import nn

from visionforge.blocks.base import ExperimentBlock
from visionforge.core.onnx_export import export_to_onnx
from visionforge.models.factory import ModelFactory
from visionforge.utils.config import ExperimentConfig


class ExportONNXBlock(ExperimentBlock):
    """Export a trained checkpoint to ONNX, validate outputs, and benchmark latency."""

    def setup(self, config: ExperimentConfig) -> None:
        """Prepare model and capture image size from data config.

        Raises:
            ValueError: if export_onnx config is absent.
        """
        if config.export_onnx is None:
            raise ValueError(
                "ExportONNXBlock requires export_onnx to be set in ExperimentConfig."
            )
        self._config = config
        self._onnx_cfg = config.export_onnx
        self._image_size = config.data.transforms.image_size

        model = ModelFactory.create(config.model)
        state_dict = torch.load(
            str(self._onnx_cfg.checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)  # type: ignore[arg-type]
        model.eval()
        self._model: nn.Module = model

        self._report_data: dict[str, Any] = {}

    def run(self) -> None:
        """Export to ONNX, optionally validate and benchmark (shared core helper)."""
        cfg = self._onnx_cfg
        self._report_data = export_to_onnx(
            self._model,
            self._image_size,
            cfg.output_onnx,
            opset_version=cfg.opset_version,
            dynamic_axes=cfg.dynamic_axes,
            validate=cfg.run_validate,
            benchmark=cfg.benchmark,
            benchmark_runs=cfg.benchmark_runs,
            tolerance=cfg.validation_tolerance,
        )

    def report(self) -> dict[str, Any]:
        """Return export summary with file size, validation, and benchmark results."""
        return self._report_data


__all__ = ["ExportONNXBlock"]
