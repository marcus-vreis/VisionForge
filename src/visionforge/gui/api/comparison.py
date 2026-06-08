"""Generic model-comparison endpoints for regression and segmentation (ADR-041 slice 2).

Reuses the task-agnostic `GenericComparisonRunner` with the per-task `TaskRunner`
adapters. No existing task code is touched: per-arch configs are rebuilt from the
request's base config by swapping ``model.name``, then ranked. ``model_names`` and
``metric`` travel in the request body, so neither config grows a comparison field
(no config-surface change — ADR-041 already covers this slice).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from visionforge.blocks.regression_runner import RegressionRunner
from visionforge.blocks.segmentation_runner import SegmentationRunner
from visionforge.core.comparison import GenericComparisonRunner
from visionforge.utils.regression_config import _REGRESSION_BACKBONES, RegressionConfig
from visionforge.utils.segmentation_config import (
    _SEGMENTATION_MODELS,
    SegmentationConfig,
)

# Recorded metric columns per task (all higher-is-better, so descending rank holds).
_REGRESSION_METRICS = ("mse", "rmse", "mae", "r2")
_SEGMENTATION_METRICS = ("miou", "dice", "pixel_acc")


class RegressionCompareRequest(BaseModel):
    """Rank ≥2 regression backbones on the same dataset by R² (higher is better)."""

    config: RegressionConfig
    model_names: list[str] = Field(min_length=2)
    metric: Literal["r2"] = "r2"

    @field_validator("model_names")
    @classmethod
    def known_backbones(cls, v: list[str]) -> list[str]:
        unknown = [n for n in v if n not in _REGRESSION_BACKBONES]
        if unknown:
            raise ValueError(
                f"unknown regression backbone(s): {unknown}; "
                f"valid: {list(_REGRESSION_BACKBONES)}"
            )
        return v


class SegmentationCompareRequest(BaseModel):
    """Rank ≥2 segmentation architectures on the same dataset (higher is better)."""

    config: SegmentationConfig
    model_names: list[str] = Field(min_length=2)
    metric: Literal["miou", "dice", "pixel_acc"] = "miou"

    @field_validator("model_names")
    @classmethod
    def known_models(cls, v: list[str]) -> list[str]:
        unknown = [n for n in v if n not in _SEGMENTATION_MODELS]
        if unknown:
            raise ValueError(
                f"unknown segmentation model(s): {unknown}; "
                f"valid: {list(_SEGMENTATION_MODELS)}"
            )
        return v


def run_regression_comparison(req: RegressionCompareRequest) -> dict[str, Any]:
    """Run a regression model comparison and return the ranked summary report."""
    raw = req.config.model_dump(mode="json")
    trials = [
        (
            name,
            RegressionConfig.model_validate(
                {**raw, "model": {**raw["model"], "name": name}}
            ),
        )
        for name in req.model_names
    ]
    runner = GenericComparisonRunner(RegressionRunner(), _REGRESSION_METRICS)
    out_dir = req.config.output.reports_dir / req.config.name
    return runner.compare(trials, rank_by=req.metric, out_dir=out_dir).summary()


def run_segmentation_comparison(req: SegmentationCompareRequest) -> dict[str, Any]:
    """Run a segmentation model comparison and return the ranked summary report."""
    raw = req.config.model_dump(mode="json")
    trials = [
        (
            name,
            SegmentationConfig.model_validate(
                {**raw, "model": {**raw["model"], "name": name}}
            ),
        )
        for name in req.model_names
    ]
    runner = GenericComparisonRunner(SegmentationRunner(), _SEGMENTATION_METRICS)
    out_dir = req.config.output.reports_dir / req.config.name
    return runner.compare(trials, rank_by=req.metric, out_dir=out_dir).summary()


__all__ = [
    "RegressionCompareRequest",
    "SegmentationCompareRequest",
    "run_regression_comparison",
    "run_segmentation_comparison",
]
