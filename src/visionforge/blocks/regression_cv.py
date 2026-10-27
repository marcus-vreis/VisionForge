"""K-fold cross-validation for the image-regression task (ADR-050).

Mirrors the classification ``CrossValidationBlock`` but over the CSV-manifest
regression dataset: the pooled training rows are split into K folds (sklearn
``KFold``), each fold trains a fresh model on K-1 parts and is scored on the
held-out part, and the per-fold regression metrics (MSE/RMSE/MAE/R²) are
aggregated to mean ± std.

Reuses ``RegressionCsvDataset`` (two transform variants + ``Subset``), the
``RegressionModelFactory`` and the ``RegressionTrainer`` — no dataset refactor.
The train fold gets the augmented transform; the validation fold gets the clean
eval transform. A failed fold is recorded and skipped, not fatal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from visionforge.blocks import _cv_common
from visionforge.core.cancellation import CancellationToken
from visionforge.core.data import _build_transforms
from visionforge.core.regression_data import RegressionCsvDataset
from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.utils.regression_config import RegressionConfig

CrossValidationReport = _cv_common.CrossValidationReport
FoldResult = _cv_common.FoldResult
_emit_fold_start = _cv_common._emit_fold_start
_emit_untrained_fold_end = _cv_common._emit_untrained_fold_end
_FoldDataModule = _cv_common._FoldDataModule
aggregate_folds = _cv_common.aggregate_folds
run_folds = _cv_common.run_folds

# Regression metric names; the first is the primary ranking metric.
_METRIC_NAMES = ("r2", "rmse", "mae", "mse")


def run_regression_cross_validation(
    config: RegressionConfig,
    *,
    n_folds: int,
    shuffle: bool = True,
    seed: int = 42,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancellationToken | None = None,
) -> CrossValidationReport:
    """Run K-fold CV over the regression training manifest and aggregate metrics."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")

    data = config.data
    images_root = data.base_dir / data.images_dir
    train_csv = data.base_dir / data.train_csv
    targets = list(data.target_columns)

    # One dataset with the train-time augmentation, one with the clean eval
    # transform; folds select rows from each via Subset so the val fold is never
    # augmented.
    train_ds = RegressionCsvDataset(
        train_csv,
        images_root,
        data.image_column,
        targets,
        _build_transforms(
            data.transforms, is_train=True, preprocessing=data.preprocessing
        ),
    )
    val_ds = RegressionCsvDataset(
        train_csv,
        images_root,
        data.image_column,
        targets,
        _build_transforms(
            data.transforms, is_train=False, preprocessing=data.preprocessing
        ),
    )
    _cv_common.validate_fold_count(n_folds, len(train_ds), "training rows")

    return run_folds(
        _cv_common.FoldContext(
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            split=_cv_common.FoldSplit(n_folds, shuffle, seed),
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        ),
        _cv_common.FoldTask(
            create_model=RegressionModelFactory.create,
            create_trainer=RegressionTrainer,
            metrics=_cv_common.FoldMetrics(
                primary="r2",
                names=("mse", "rmse", "mae", "r2"),
                aggregate_names=_METRIC_NAMES,
                log_format="Fold {}/{}: r2={:.4f} rmse={:.4f}",
                log_fields=("r2", "rmse"),
            ),
        ),
    )


__all__ = [
    "FoldResult",
    "CrossValidationReport",
    "aggregate_folds",
    "run_regression_cross_validation",
]
