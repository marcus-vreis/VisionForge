"""Shared fold loaders, reporting and execution for standalone task CV."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from visionforge.blocks._search_utils import make_trial_progress_wrapper
from visionforge.core.cancellation import (
    STOPPED,
    STOPPED_NOTE,
    CancellationToken,
    is_cancelled,
)
from visionforge.core.regression_data import RegressionCsvDataset
from visionforge.core.replicates import sample_std
from visionforge.core.resume import discard_resume_state
from visionforge.core.segmentation_data import SegmentationDataset
from visionforge.core.trainer import seed_before_building
from visionforge.utils.config import check_select_on_last
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig


@dataclass
class FoldResult:
    """Outcome of one cross-validation fold."""

    fold: int
    status: str
    train_size: int
    val_size: int
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""
    # The fold's own run folder; None when it failed before training.
    run_dir: str | None = None
    # The rule that picked the fold's epoch (ADR-242) and the epoch it picked;
    # None when the fold scored nothing.
    select_on: str = "best"
    selected_epoch: int | None = None


@dataclass
class CrossValidationReport:
    """Aggregated K-fold result: per-fold metrics + mean ± std."""

    n_folds: int
    metric: str
    folds: list[FoldResult]
    # metric -> {"mean": …, "std": …, "n": …}; std is None for a single fold.
    aggregate: dict[str, dict[str, float | int | None]]
    # The rule that picked each fold's epoch (ADR-242).
    select_on: str = "best"


def fold_drop_last(n_train: int, batch_size: int) -> bool:
    """Whether a K-fold train loader drops its trailing partial batch (ADR-050).

    A trailing size-one batch breaks BatchNorm in train mode, and folds are
    small. A fold that fits in one batch keeps it, or it would train on nothing.
    Shared by every K-fold (task K-fold here, classification in
    ``blocks/cross_validation.py``).
    """
    return n_train > batch_size


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
        """Return a shuffled, single-process loader for the training fold."""
        return DataLoader(
            self._train,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=fold_drop_last(len(self._train), self._batch_size),  # type: ignore[arg-type]
        )

    def val_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the fold's held-out validation split."""
        return DataLoader(
            self._val, batch_size=self._batch_size, shuffle=False, num_workers=0
        )


class FoldTrainingResult(Protocol):
    """Expose the training outcome required to score and stop a fold."""

    model_path: Path
    best_epoch: int
    total_epochs: int
    stopped: bool


class FoldTrainer(Protocol):
    """Describe the native training and evaluation operations used by CV."""

    def fit(
        self,
        model: nn.Module,
        data_module: _FoldDataModule,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        cancel_token: CancellationToken | None,
    ) -> FoldTrainingResult:
        """Train one fold and return its checkpoint and completion state."""
        ...

    def evaluate(self, model: nn.Module, loader: DataLoader[Any]) -> tuple[float, ...]:
        """Score a fold's held-out loader with the restored best checkpoint."""
        ...


@dataclass(frozen=True)
class FoldSplit:
    """Specify the reproducible K-fold partition without changing task config."""

    n_folds: int
    shuffle: bool
    seed: int


@dataclass(frozen=True)
class FoldContext:
    """Bind task configuration, paired dataset views and CV lifecycle controls."""

    config: RegressionConfig | SegmentationConfig
    train_ds: RegressionCsvDataset | SegmentationDataset
    val_ds: RegressionCsvDataset | SegmentationDataset
    split: FoldSplit
    progress_callback: Callable[[dict[str, Any]], None] | None
    cancel_token: CancellationToken | None
    select_on: Literal["best", "last"] = "best"


@dataclass(frozen=True)
class FoldMetrics:
    """Preserve the task's evaluation, aggregation and logging metric order."""

    primary: str
    names: tuple[str, ...]
    aggregate_names: tuple[str, ...]
    log_format: str
    log_fields: tuple[str, ...]


@dataclass(frozen=True)
class FoldTask:
    """Provide fold-local model and trainer factories with their metric contract."""

    create_model: Callable[[Any], nn.Module]
    create_trainer: Callable[[Any], FoldTrainer]
    metrics: FoldMetrics


def _emit_fold_start(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    fold_idx: int,
    n_folds: int,
) -> None:
    """Announce a fold before its training attempt."""
    if progress_callback is not None:
        progress_callback(
            {
                "event": "trial_start",
                "trial_index": fold_idx,
                "total_trials": n_folds,
                "overrides": {"fold": fold_idx + 1},
            }
        )


def _emit_untrained_fold_end(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    trained: bool,
    fold_idx: int,
    n_folds: int,
    record: FoldResult,
) -> None:
    """Close folds whose trainer did not emit its own end event."""
    if progress_callback is not None and not trained:
        progress_callback(
            {
                "event": "trial_end",
                "trial_index": fold_idx,
                "total_trials": n_folds,
                "status": record.status,
            }
        )


def fold_metric_stats(reported: list[float]) -> dict[str, float | int | None]:
    """Mean, sample std, finite count and excluded count of one fold metric.

    A NaN or infinite value is not a measurement (ADR-061): it stays out of the
    mean and std, and ``n_non_finite`` counts it. No finite value gives ``n = 0``
    and null statistics.
    """
    values = [v for v in reported if math.isfinite(v)]
    return {
        "mean": float(np.mean(values)) if values else None,
        "std": sample_std(values),
        "n": len(values),
        "n_non_finite": len(reported) - len(values),
        # Runs before ADR-111 wrote the population std (ddof=0).
        "std_ddof": 1,
    }


def aggregate_folds(
    folds: list[FoldResult], names: tuple[str, ...]
) -> dict[str, dict[str, float | int | None]]:
    """Mean, sample std and n of each metric over the folds that finished.

    A fold whose metric is NaN or infinite (an R² on a constant target) is not
    a measurement: it stays out of the mean and std, and ``n_non_finite`` counts
    it, as the replicate aggregate does (ADR-061). A metric with no finite value
    keeps its entry with ``n = 0`` and null statistics.
    """
    successful = [f for f in folds if f.status == "success"]
    aggregate: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        reported = [f.metrics[name] for f in successful if name in f.metrics]
        if reported:
            aggregate[name] = fold_metric_stats(reported)
    return aggregate


def _score_fold(
    result: Any,
    model: Any,
    trainer: Any,
    fold_data: _FoldDataModule,
    record: FoldResult,
    metric_names: tuple[str, ...],
    n_folds: int,
    log_format: str,
    log_fields: tuple[str, ...],
    select_on: str = "best",
) -> None:
    """Score completed folds at the epoch ``select_on`` names (ADR-242).

    "best" reloads the checkpoint the fold's own validation split chose; "last"
    scores the weights the model holds after its final epoch. A fold trains
    from epoch 1 and never resumes, so its last epoch is the epochs it ran.
    """
    if not result.total_epochs:
        return
    if select_on == "best":
        state = torch.load(
            str(result.model_path), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
    record.selected_epoch = (
        result.best_epoch if select_on == "best" else result.total_epochs
    )
    values = trainer.evaluate(model, fold_data.val_loader())
    record.status = "success"
    record.metrics = {
        name: float(value) for name, value in zip(metric_names, values, strict=True)
    }
    logger.info(
        log_format,
        record.fold + 1,
        n_folds,
        *(record.metrics[name] for name in log_fields),
    )


def _run_fold(
    context: FoldContext,
    task: FoldTask,
    train_idx: Any,
    val_idx: Any,
    record: FoldResult,
) -> tuple[bool, bool, FoldTrainingResult | None]:
    """Train one fold while isolating failures and preserving stop outcomes."""
    config = context.config
    n_folds = context.split.n_folds
    trained = False
    cut = False
    result = None
    try:
        fold_config = config.model_copy(
            update={"name": f"{config.name}_fold{record.fold}"}
        )
        fold_data = _FoldDataModule(
            Subset(context.train_ds, train_idx.tolist()),
            Subset(context.val_ds, val_idx.tolist()),
            config.training.batch_size,
        )
        seed_before_building(fold_config.training)
        model = task.create_model(fold_config.model)
        trainer = task.create_trainer(fold_config)
        result = trainer.fit(
            model,
            fold_data,
            progress_callback=make_trial_progress_wrapper(
                context.progress_callback, record.fold, n_folds
            ),
            cancel_token=context.cancel_token,
        )
        trained = True
        record.run_dir = str(result.model_path.parent)
        cut = result.stopped
        _score_fold(
            result,
            model,
            trainer,
            fold_data,
            record,
            task.metrics.names,
            n_folds,
            task.metrics.log_format,
            task.metrics.log_fields,
            context.select_on,
        )
    except Exception as exc:  # noqa: BLE001
        record.error = str(exc)
        # An evaluation failure remains a failure even inside the stop window.
        cut = False
        logger.warning("Fold {}/{} failed: {}", record.fold + 1, n_folds, exc)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return trained, cut, result


def run_folds(context: FoldContext, task: FoldTask) -> CrossValidationReport:
    """Execute K folds with shared failure, progress and cancellation handling."""
    training = context.config.training
    check_select_on_last(
        context.select_on,
        patience=training.early_stopping_patience,
        scheduler_kind=training.scheduler.kind,
    )
    n_folds = context.split.n_folds
    progress_callback = context.progress_callback
    cancel_token = context.cancel_token
    splitter = KFold(
        n_splits=n_folds,
        shuffle=context.split.shuffle,
        random_state=context.split.seed if context.split.shuffle else None,
    )
    folds: list[FoldResult] = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(range(len(context.train_ds)))
    ):
        _emit_fold_start(progress_callback, fold_idx, n_folds)
        record = FoldResult(
            fold=fold_idx,
            status="failed",
            select_on=context.select_on,
            train_size=len(train_idx),
            val_size=len(val_idx),
        )
        trained, cut, result = _run_fold(context, task, train_idx, val_idx, record)
        if cut:
            record.status = STOPPED
            record.error = STOPPED_NOTE
            # A stopped fold cannot resume independently of its parent K-fold job.
            assert result is not None
            discard_resume_state(result.model_path.parent)
        folds.append(record)
        _emit_untrained_fold_end(progress_callback, trained, fold_idx, n_folds, record)
        if is_cancelled(cancel_token):
            logger.info("K-fold stopped after {} of {} folds.", len(folds), n_folds)
            break
    return CrossValidationReport(
        n_folds=n_folds,
        metric=task.metrics.primary,
        folds=folds,
        aggregate=aggregate_folds(folds, task.metrics.aggregate_names),
        select_on=context.select_on,
    )


def validate_fold_count(n_folds: int, n_samples: int, sample_unit: str) -> None:
    """Reject folds that would exceed the task's available training samples."""
    if n_folds > n_samples:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed the number of {sample_unit} "
            f"({n_samples})."
        )
