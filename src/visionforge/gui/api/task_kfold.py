"""The K-fold fronts of regression and segmentation, light to import (GUI-11, #202).

The trainers import torch, so each front imports its task's module only when a
run starts. ``strategies.py`` re-exports them, and a test replaces them there.
They live here, not in ``strategies.py``, so a server job's child can import
the one it was named (ADR-252) without importing the server's routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visionforge.blocks.regression_cv import CrossValidationReport
    from visionforge.utils.regression_config import RegressionConfig
    from visionforge.utils.segmentation_config import SegmentationConfig


def run_regression_cross_validation(
    config: RegressionConfig, **kwargs: Any
) -> CrossValidationReport:
    """K-fold CV for regression, importing its trainer only when a run starts."""
    from visionforge.blocks.regression_cv import run_regression_cross_validation as run

    return run(config, **kwargs)


def run_segmentation_cross_validation(
    config: SegmentationConfig, **kwargs: Any
) -> CrossValidationReport:
    """K-fold CV for segmentation, importing its trainer only when a run starts."""
    from visionforge.blocks.segmentation_cv import (
        run_segmentation_cross_validation as run,
    )

    return run(config, **kwargs)


__all__ = ["run_regression_cross_validation", "run_segmentation_cross_validation"]
