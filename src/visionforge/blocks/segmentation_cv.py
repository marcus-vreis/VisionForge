"""K-fold cross-validation for the semantic-segmentation task (ADR-050 parity).

Mirrors ``regression_cv`` over the paired image/mask dataset: the train split's
pairs are divided into K folds (sklearn ``KFold``), each fold trains a fresh
model on K-1 parts and is scored on the held-out part, and the per-fold
segmentation metrics (mIoU/Dice/pixel-acc) are aggregated to mean ± std.

Two ``SegmentationDataset`` instances are built over the same train split — one
with the train-time joint augmentation, one with the clean eval transform — and
folds select rows from each via ``Subset`` so the validation fold is never
augmented. The report shapes (``FoldResult``/``CrossValidationReport``) are the
shared ones from ``regression_cv``. A failed fold is recorded and skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from visionforge.blocks import _cv_common
from visionforge.core.cancellation import CancellationToken
from visionforge.core.segmentation_data import SegmentationDataset
from visionforge.core.segmentation_trainer import SegmentationTrainer
from visionforge.models.segmentation_factory import SegmentationModelFactory
from visionforge.utils.segmentation_config import SegmentationConfig

CrossValidationReport = _cv_common.CrossValidationReport
FoldResult = _cv_common.FoldResult
_emit_fold_start = _cv_common._emit_fold_start
_emit_untrained_fold_end = _cv_common._emit_untrained_fold_end
_FoldDataModule = _cv_common._FoldDataModule
aggregate_folds = _cv_common.aggregate_folds
run_folds = _cv_common.run_folds

# Segmentation metric names; the first is the primary ranking metric.
_METRIC_NAMES = ("miou", "dice", "pixel_acc")


def run_segmentation_cross_validation(
    config: SegmentationConfig,
    *,
    n_folds: int,
    shuffle: bool = True,
    seed: int = 42,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancellationToken | None = None,
) -> CrossValidationReport:
    """Run K-fold CV over the segmentation train split and aggregate metrics."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")

    data = config.data
    images_dir = data.base_dir / data.train_dir / data.images_subdir
    masks_dir = data.base_dir / data.train_dir / data.masks_subdir

    train_ds = SegmentationDataset(
        images_dir,
        masks_dir,
        image_size=data.image_size,
        ignore_index=data.ignore_index,
        transform_cfg=data.transforms,
        preprocessing_cfg=data.preprocessing,
        is_train=True,
    )
    val_ds = SegmentationDataset(
        images_dir,
        masks_dir,
        image_size=data.image_size,
        ignore_index=data.ignore_index,
        transform_cfg=data.transforms,
        preprocessing_cfg=data.preprocessing,
        is_train=False,
    )

    _cv_common.validate_fold_count(n_folds, len(train_ds), "train pairs")

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
            create_model=SegmentationModelFactory.create,
            create_trainer=SegmentationTrainer,
            metrics=_cv_common.FoldMetrics(
                primary="miou",
                names=_METRIC_NAMES,
                aggregate_names=_METRIC_NAMES,
                log_format="Fold {}/{}: miou={:.4f} dice={:.4f}",
                log_fields=("miou", "dice"),
            ),
        ),
    )


__all__ = ["run_segmentation_cross_validation"]
