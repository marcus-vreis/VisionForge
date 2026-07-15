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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from visionforge.blocks._search_utils import make_trial_progress_wrapper
from visionforge.core.data import _build_transforms
from visionforge.core.regression_data import RegressionCsvDataset
from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.utils.regression_config import RegressionConfig

# Regression metric names; the first is the primary ranking metric.
_METRIC_NAMES = ("r2", "rmse", "mae", "mse")


@dataclass
class FoldResult:
    """Outcome of one cross-validation fold."""

    fold: int
    status: str
    train_size: int
    val_size: int
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""


@dataclass
class CrossValidationReport:
    """Aggregated K-fold result: per-fold metrics + mean ± std."""

    n_folds: int
    metric: str
    folds: list[FoldResult]
    aggregate: dict[str, dict[str, float]]  # metric -> {"mean": …, "std": …}


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
        trailing batch can't break BatchNorm (CV folds are small by construction);
        it's left off when there's only a single batch so a tiny fold still trains.
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


def run_regression_cross_validation(
    config: RegressionConfig,
    *,
    n_folds: int,
    shuffle: bool = True,
    seed: int = 42,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> CrossValidationReport:
    """Run K-fold CV over the regression training manifest and aggregate metrics.

    Raises:
        ValueError: if ``n_folds`` < 2 or exceeds the number of training rows.
    """
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
    n_samples = len(train_ds)
    if n_folds > n_samples:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed the number of training rows "
            f"({n_samples})."
        )

    splitter = KFold(
        n_splits=n_folds, shuffle=shuffle, random_state=seed if shuffle else None
    )
    base_name = config.name
    folds: list[FoldResult] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(range(n_samples))):
        # trial_start/trial_end is the vocabulary the GUI overlay tracks; the
        # wrapped inner callback streams each fold's epochs with trial context.
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "trial_start",
                    "trial_index": fold_idx,
                    "total_trials": n_folds,
                    "overrides": {"fold": fold_idx + 1},
                }
            )
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
            model = RegressionModelFactory.create(fold_config.model)
            trainer = RegressionTrainer(fold_config)
            result = trainer.fit(
                model,
                fold_data,
                progress_callback=make_trial_progress_wrapper(
                    progress_callback, fold_idx, n_folds
                ),
            )

            state = torch.load(
                str(result.model_path), map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)  # type: ignore[arg-type]
            mse, rmse, mae, r2 = trainer.evaluate(model, fold_data.val_loader())

            record.status = "success"
            record.metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
            }
            logger.info(
                "Fold {}/{}: r2={:.4f} rmse={:.4f}", fold_idx + 1, n_folds, r2, rmse
            )
        except Exception as exc:  # noqa: BLE001 — one bad fold must not abort the sweep
            record.error = str(exc)
            logger.warning("Fold {}/{} failed: {}", fold_idx + 1, n_folds, exc)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        folds.append(record)
        # On success the wrapped trainer's terminal "end" was already rewritten
        # to trial_end; emit one here only when the fold died before finishing.
        if progress_callback is not None and record.status != "success":
            progress_callback(
                {
                    "event": "trial_end",
                    "trial_index": fold_idx,
                    "total_trials": n_folds,
                    "status": record.status,
                }
            )

    return CrossValidationReport(
        n_folds=n_folds,
        metric="r2",
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


__all__ = [
    "FoldResult",
    "CrossValidationReport",
    "run_regression_cross_validation",
]
