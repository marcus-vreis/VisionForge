"""Shared fold loaders, reporting and execution for standalone task CV."""

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
from visionforge.core.cancellation import (
    STOPPED,
    STOPPED_NOTE,
    CancellationToken,
    is_cancelled,
)
from visionforge.core.replicates import sample_std
from visionforge.core.resume import discard_resume_state
from visionforge.core.trainer import seed_before_building


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


@dataclass
class CrossValidationReport:
    """Aggregated K-fold result: per-fold metrics + mean ± std."""

    n_folds: int
    metric: str
    folds: list[FoldResult]
    # metric -> {"mean": …, "std": …, "n": …}; std is None for a single fold.
    aggregate: dict[str, dict[str, float | int | None]]


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
        # A trailing size-one batch breaks BatchNorm unless the entire fold is tiny.
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


def aggregate_folds(
    folds: list[FoldResult], names: tuple[str, ...]
) -> dict[str, dict[str, float | int | None]]:
    """Mean, sample std and n of each metric over the folds that finished."""
    successful = [f for f in folds if f.status == "success"]
    aggregate: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        values = [f.metrics[name] for f in successful if name in f.metrics]
        if values:
            aggregate[name] = {
                "mean": float(np.mean(values)),
                "std": sample_std(values),
                "n": len(values),
                # Runs before ADR-111 wrote the population std (ddof=0).
                "std_ddof": 1,
            }
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
) -> None:
    """Score completed folds with their best weights and original metric order."""
    if not result.total_epochs:
        return
    state = torch.load(str(result.model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
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
    config: Any,
    train_ds: Any,
    val_ds: Any,
    train_idx: Any,
    val_idx: Any,
    record: FoldResult,
    n_folds: int,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_token: CancellationToken | None,
    create_model: Callable[..., Any],
    create_trainer: Callable[..., Any],
    metric_names: tuple[str, ...],
    log_format: str,
    log_fields: tuple[str, ...],
) -> tuple[bool, bool, Any]:
    """Train one fold while isolating failures and preserving stop outcomes."""
    trained = False
    cut = False
    result = None
    try:
        fold_config = config.model_copy(
            update={"name": f"{config.name}_fold{record.fold}"}
        )
        fold_data = _FoldDataModule(
            Subset(train_ds, train_idx.tolist()),
            Subset(val_ds, val_idx.tolist()),
            config.training.batch_size,
        )
        seed_before_building(fold_config.training)
        model = create_model(fold_config.model)
        trainer = create_trainer(fold_config)
        result = trainer.fit(
            model,
            fold_data,
            progress_callback=make_trial_progress_wrapper(
                progress_callback, record.fold, n_folds
            ),
            cancel_token=cancel_token,
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
            metric_names,
            n_folds,
            log_format,
            log_fields,
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


def run_folds(
    config: Any,
    train_ds: Any,
    val_ds: Any,
    *,
    n_folds: int,
    shuffle: bool,
    seed: int,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_token: CancellationToken | None,
    create_model: Callable[..., Any],
    create_trainer: Callable[..., Any],
    metric: str,
    metric_names: tuple[str, ...],
    aggregate_names: tuple[str, ...],
    log_format: str,
    log_fields: tuple[str, ...],
) -> CrossValidationReport:
    """Execute K folds with shared failure, progress and cancellation handling."""
    splitter = KFold(
        n_splits=n_folds, shuffle=shuffle, random_state=seed if shuffle else None
    )
    folds: list[FoldResult] = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(range(len(train_ds)))
    ):
        _emit_fold_start(progress_callback, fold_idx, n_folds)
        record = FoldResult(
            fold=fold_idx,
            status="failed",
            train_size=len(train_idx),
            val_size=len(val_idx),
        )
        trained, cut, result = _run_fold(
            config,
            train_ds,
            val_ds,
            train_idx,
            val_idx,
            record,
            n_folds,
            progress_callback,
            cancel_token,
            create_model,
            create_trainer,
            metric_names,
            log_format,
            log_fields,
        )
        if cut:
            record.status = STOPPED
            record.error = STOPPED_NOTE
            # A stopped fold cannot resume independently of its parent K-fold job.
            discard_resume_state(result.model_path.parent)
        folds.append(record)
        _emit_untrained_fold_end(progress_callback, trained, fold_idx, n_folds, record)
        if is_cancelled(cancel_token):
            logger.info("K-fold stopped after {} of {} folds.", len(folds), n_folds)
            break
    return CrossValidationReport(
        n_folds=n_folds,
        metric=metric,
        folds=folds,
        aggregate=aggregate_folds(folds, aggregate_names),
    )


def validate_fold_count(n_folds: int, n_samples: int, sample_unit: str) -> None:
    """Reject folds that would exceed the task's available training samples."""
    if n_folds > n_samples:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed the number of {sample_unit} "
            f"({n_samples})."
        )
