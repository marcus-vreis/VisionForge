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

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from visionforge.blocks.regression_cv import CrossValidationReport, FoldResult
from visionforge.core.segmentation_data import SegmentationDataset
from visionforge.core.segmentation_trainer import SegmentationTrainer
from visionforge.models.segmentation_factory import SegmentationModelFactory
from visionforge.utils.segmentation_config import SegmentationConfig

# Segmentation metric names; the first is the primary ranking metric.
_METRIC_NAMES = ("miou", "dice", "pixel_acc")


class _FoldDataModule:
    """train/val loaders over ``Subset``s of the augmented / clean datasets."""

    def __init__(
        self,
        train_subset: Subset,  # type: ignore[type-arg]
        val_subset: Subset,  # type: ignore[type-arg]
        batch_size: int,
    ) -> None:
        self._train = train_subset
        self._val = val_subset
        self._batch_size = batch_size

    def train_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the fold's training split (shuffled, single-process).

        ``drop_last`` is enabled when the fold has more than one batch so a size-1
        trailing batch can't break BatchNorm; it's left off when there's only a
        single batch so a tiny fold still trains.
        """
        drop_last = len(self._train) > self._batch_size  # type: ignore[arg-type]
        return DataLoader(
            self._train,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=drop_last,
        )

    def val_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the fold's held-out validation split."""
        return DataLoader(
            self._val, batch_size=self._batch_size, shuffle=False, num_workers=0
        )


def run_segmentation_cross_validation(
    config: SegmentationConfig,
    *,
    n_folds: int,
    shuffle: bool = True,
    seed: int = 42,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> CrossValidationReport:
    """Run K-fold CV over the segmentation train split and aggregate metrics.

    Raises:
        ValueError: if ``n_folds`` < 2 or exceeds the number of train pairs.
    """
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

    n_samples = len(train_ds)
    if n_folds > n_samples:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed the number of train pairs "
            f"({n_samples})."
        )

    splitter = KFold(
        n_splits=n_folds, shuffle=shuffle, random_state=seed if shuffle else None
    )
    base_name = config.name
    folds: list[FoldResult] = []

    if progress_callback is not None:
        progress_callback({"event": "start", "total_folds": n_folds})

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(range(n_samples))):
        record = FoldResult(
            fold=fold_idx,
            status="failed",
            train_size=len(train_idx),
            val_size=len(val_idx),
        )
        try:
            fold_config = config.model_copy(
                update={"name": f"{base_name}_fold{fold_idx}"}
            )
            fold_data = _FoldDataModule(
                Subset(train_ds, train_idx.tolist()),
                Subset(val_ds, val_idx.tolist()),
                config.training.batch_size,
            )
            model = SegmentationModelFactory.create(fold_config.model)
            trainer = SegmentationTrainer(fold_config)
            result = trainer.fit(model, fold_data)

            state = torch.load(
                str(result.model_path), map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)  # type: ignore[arg-type]
            miou, dice, pixel_acc = trainer.evaluate(model, fold_data.val_loader())

            record.status = "success"
            record.metrics = {
                "miou": float(miou),
                "dice": float(dice),
                "pixel_acc": float(pixel_acc),
            }
            logger.info(
                "Fold {}/{}: miou={:.4f} dice={:.4f}",
                fold_idx + 1,
                n_folds,
                miou,
                dice,
            )
        except Exception as exc:  # noqa: BLE001 — one bad fold must not abort the sweep
            record.error = str(exc)
            logger.warning("Fold {}/{} failed: {}", fold_idx + 1, n_folds, exc)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        folds.append(record)
        if progress_callback is not None:
            progress_callback(
                {"event": "fold_end", "fold": fold_idx, "total_folds": n_folds}
            )

    return CrossValidationReport(
        n_folds=n_folds,
        metric="miou",
        folds=folds,
        aggregate=_aggregate(folds),
    )


def _aggregate(folds: list[FoldResult]) -> dict[str, dict[str, float]]:
    """Mean ± std of each metric over the successful folds."""
    successful = [f for f in folds if f.status == "success"]
    aggregate: dict[str, dict[str, float]] = {}
    for name in _METRIC_NAMES:
        values = [f.metrics[name] for f in successful if name in f.metrics]
        if values:
            aggregate[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return aggregate


__all__ = ["run_segmentation_cross_validation"]
