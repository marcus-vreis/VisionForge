"""Per-task Grad-CAM setup (ADR-053).

Builds the model + eval transform + backward target for an already-trained run,
dispatching on its task. Classification targets the predicted (or chosen) class
logit; regression targets a continuous output (saliency of a target column);
segmentation targets the mean logit of a chosen class channel. The architecture is
rebuilt with ``pretrained=False`` (the checkpoint supplies the weights), read
leniently from the run config so a moved dataset path doesn't block explainability.
Mirrors ``torch_onnx_export`` / ``torch_batch_predict`` (per-task function, shared
``core.gradcam``). Detection (Ultralytics) and anomaly (no class/conv target) are
unsupported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from visionforge.core.data import _build_transforms
from visionforge.models.factory import ModelFactory
from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.models.segmentation_factory import (
    SegmentationModelFactory,
    segmentation_logits,
)
from visionforge.utils.config import ModelConfig, PreprocessingConfig, TransformConfig
from visionforge.utils.regression_config import RegressionModelConfig
from visionforge.utils.segmentation_config import SegmentationModelConfig

_CLASSIFICATION_TASKS = ("binary", "multiclass", "classification", None)


@dataclass
class GradcamSetup:
    """Everything the Grad-CAM run loop needs for one task."""

    model: nn.Module
    transform: Any
    target_fn: Callable[[Any], torch.Tensor]
    describe: Callable[[torch.Tensor], tuple[int | None, str | None]]


class _SegLogits(nn.Module):
    """Wrap a segmentation model so its output is a plain ``[N, C, H, W]`` tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return segmentation_logits(self.model(x))


def _load_checkpoint(model: nn.Module, checkpoint: str) -> None:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # type: ignore[arg-type]
    model.eval()


def build_gradcam(
    data: dict[str, Any], *, target_index: int | None, checkpoint: str
) -> GradcamSetup:
    """Build the task-appropriate Grad-CAM model/transform/target from a run.json.

    Raises:
        ValueError: if the run's task does not support Grad-CAM (detection/anomaly).
    """
    config = data.get("config", {})
    task = config.get("task")
    model_dict = {**config.get("model", {}), "pretrained": False}
    model_dict.pop("weights_path", None)
    data_dict = config.get("data", {})
    preprocessing = PreprocessingConfig.model_validate(
        data_dict.get("preprocessing", {})
    )

    def _transform(transforms_dict: dict[str, Any]) -> Any:
        return _build_transforms(
            TransformConfig.model_validate(transforms_dict),
            is_train=False,
            preprocessing=preprocessing,
        )

    if task in _CLASSIFICATION_TASKS:
        model = ModelFactory.create(ModelConfig.model_validate(model_dict))
        _load_checkpoint(model, checkpoint)

        def cls_target(out: torch.Tensor) -> torch.Tensor:
            idx = (
                target_index
                if target_index is not None
                else int(out.argmax(dim=1).item())
            )
            return out[:, idx].sum()

        def cls_describe(out: torch.Tensor) -> tuple[int | None, str | None]:
            return int(out.argmax(dim=1).item()), None

        return GradcamSetup(
            model, _transform(data_dict.get("transforms", {})), cls_target, cls_describe
        )

    if task == "regression":
        model = RegressionModelFactory.create(
            RegressionModelConfig.model_validate(model_dict)
        )
        _load_checkpoint(model, checkpoint)
        idx = target_index or 0

        def reg_target(out: torch.Tensor) -> torch.Tensor:
            return out[:, idx].sum()

        def reg_describe(out: torch.Tensor) -> tuple[int | None, str | None]:
            values = [round(float(v), 4) for v in out[0].tolist()]
            return None, f"pred={values}"

        return GradcamSetup(
            model, _transform(data_dict.get("transforms", {})), reg_target, reg_describe
        )

    if task == "segmentation":
        base = SegmentationModelFactory.create(
            SegmentationModelConfig.model_validate(model_dict)
        )
        _load_checkpoint(base, checkpoint)
        model = _SegLogits(base)
        cls = target_index or 0
        # Segmentation keeps the resize on data.image_size, synced into transforms.
        tc = TransformConfig.model_validate(data_dict.get("transforms", {})).model_copy(
            update={"image_size": data_dict.get("image_size", 512)}
        )
        transform = _build_transforms(tc, is_train=False, preprocessing=preprocessing)

        def seg_target(out: torch.Tensor) -> torch.Tensor:
            return out[:, cls].mean()

        def seg_describe(out: torch.Tensor) -> tuple[int | None, str | None]:
            dominant = int(out[0].mean(dim=(1, 2)).argmax().item())
            return cls, f"target class {cls} · dominant {dominant}"

        return GradcamSetup(model, transform, seg_target, seg_describe)

    raise ValueError(
        f"Grad-CAM is not available for task '{task}' "
        "(supported: classification, regression, segmentation)."
    )


__all__ = ["GradcamSetup", "build_gradcam"]
