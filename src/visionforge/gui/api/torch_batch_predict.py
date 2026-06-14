"""Batch prediction for regression runs (ADR-041 slice 3 / shared core helper).

Classification batch-predict goes through ``BatchPredictionBlock``; regression
reuses the task-agnostic ``core.batch_predict`` helper, building the model from
the regression factory + the run's checkpoint (``pretrained=False`` so export
never triggers an ImageNet download) and writing one CSV row of continuous
targets per image. Mirrors ``torch_onnx_export`` (per-task function, shared core).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from torch import nn

from visionforge.core.batch_predict import (
    BatchPredictionSummary,
    predict_folder_to_csv,
)
from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.utils.config import TransformConfig
from visionforge.utils.cuda import check_cuda
from visionforge.utils.regression_config import RegressionConfig


def _inference_transform(tc: TransformConfig) -> T.Compose:
    """Resize + centre-crop + normalise — no augmentation for inference."""
    return T.Compose(
        [
            T.Resize(tc.image_size),
            T.CenterCrop(tc.image_size),
            T.ToTensor(),
            T.Normalize(mean=tc.normalize_mean, std=tc.normalize_std),
        ]
    )


def batch_predict_regression_run(
    *,
    run_name: str,
    data: dict[str, Any],
    input_dir: Path,
    output_csv: Path,
    recursive: bool,
) -> BatchPredictionSummary:
    """Run a regression checkpoint over a folder, writing target predictions to CSV.

    The CSV columns are ``filename`` followed by the run's ``target_columns``.
    """
    config = RegressionConfig.model_validate(data["config"])
    checkpoint = data.get("artifacts", {}).get("model")
    if not checkpoint:
        raise FileNotFoundError(
            f"Run '{run_name}' has no checkpoint recorded in artifacts.model."
        )

    model_cfg = config.model.model_copy(
        update={"pretrained": False, "weights_path": None}
    )
    model: nn.Module = RegressionModelFactory.create(model_cfg)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # type: ignore[arg-type]
    model.eval()

    device = torch.device("cuda" if check_cuda().available else "cpu")
    model.to(device)

    targets = config.data.target_columns

    def infer(tensor: torch.Tensor) -> dict[str, Any]:
        values = model(tensor).squeeze(0).tolist()
        if isinstance(values, float):  # single-target → scalar after squeeze
            values = [values]
        return {col: f"{val:.6f}" for col, val in zip(targets, values, strict=True)}

    return predict_folder_to_csv(
        input_dir=input_dir,
        output_csv=output_csv,
        transform=_inference_transform(config.data.transforms),
        infer=infer,
        fieldnames=["filename", *targets],
        device=device,
        recursive=recursive,
    )


__all__ = ["batch_predict_regression_run"]
