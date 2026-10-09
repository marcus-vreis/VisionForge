from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from visionforge.blocks._search_utils import make_trial_progress_wrapper
from visionforge.blocks.base import ExperimentBlock
from visionforge.core.cancellation import (
    STOPPED,
    STOPPED_NOTE,
    is_cancelled,
    job_was_stopped,
)
from visionforge.core.data import resolve_num_workers
from visionforge.core.evaluator import Evaluator
from visionforge.core.replicates import sample_std
from visionforge.core.trainer import Trainer
from visionforge.models.factory import ModelFactory
from visionforge.utils.config import ExperimentConfig


class _FoldDataModule:
    """Thin data module adapter for a single CV fold."""

    def __init__(
        self,
        train_subset: Subset,  # type: ignore[type-arg]
        val_subset: Subset,  # type: ignore[type-arg]
        fold_mean: list[float],
        fold_std: list[float],
        config: ExperimentConfig,
        num_workers: int = 0,
    ) -> None:
        tc = config.data.transforms
        train_transform = T.Compose(
            [
                T.Resize(tc.image_size),
                T.CenterCrop(tc.image_size),
                *([T.RandomHorizontalFlip()] if tc.horizontal_flip else []),
                *(
                    [T.RandomRotation(tc.rotation_degrees)]
                    if tc.rotation_degrees > 0
                    else []
                ),
                *(
                    [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
                    if tc.color_jitter
                    else []
                ),
                T.ToTensor(),
                T.Normalize(mean=fold_mean, std=fold_std),
            ]
        )
        val_transform = T.Compose(
            [
                T.Resize(tc.image_size),
                T.CenterCrop(tc.image_size),
                T.ToTensor(),
                T.Normalize(mean=fold_mean, std=fold_std),
            ]
        )

        # Rebind transforms on the underlying ImageFolder via index remapping.
        # Subset doesn't expose transform directly, so we wrap with a transformed copy.
        import copy

        train_ds: ImageFolder = train_subset.dataset  # type: ignore[assignment]
        val_ds: ImageFolder = val_subset.dataset  # type: ignore[assignment]

        self._train_ds = copy.copy(train_ds)
        self._train_ds.transform = train_transform
        self._train_ds.samples = [train_ds.samples[i] for i in train_subset.indices]
        self._train_ds.targets = [train_ds.targets[i] for i in train_subset.indices]

        self._val_ds = copy.copy(val_ds)
        self._val_ds.transform = val_transform
        self._val_ds.samples = [val_ds.samples[i] for i in val_subset.indices]
        self._val_ds.targets = [val_ds.targets[i] for i in val_subset.indices]

        self._batch_size = config.training.batch_size
        # Already resolved by the caller (`resolve_num_workers`): the raw
        # config value can be -1, which DataLoader refuses.
        self._num_workers = num_workers
        self._pin_memory = config.data.pin_memory
        self._class_names: list[str] = list(train_ds.classes)

    def train_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the fold's training split (shuffled)."""
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def val_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the fold's validation split."""
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def test_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """Alias for val_loader — CV has no separate test set per fold."""
        return self.val_loader()

    @property
    def class_names(self) -> list[str]:
        """Ordered class names from the training dataset."""
        return self._class_names


def _compute_fold_stats(
    dataset: ImageFolder, train_indices: list[int], num_workers: int
) -> tuple[list[float], list[float]]:
    """Compute per-channel mean and std from train-fold images only.

    Args:
        dataset: full ImageFolder loaded with T.ToTensor() only.
        train_indices: indices belonging to this fold's training set.

    Returns:
        Tuple of (mean, std) each as a 3-element list of floats.
    """
    subset = Subset(dataset, train_indices)
    loader = DataLoader(
        subset,
        batch_size=256,
        shuffle=False,  # order doesn't matter for stats, but must be deterministic
        num_workers=num_workers,
    )

    # Welford online algorithm for numerical stability.
    count = 0
    mean = torch.zeros(3)
    M2 = torch.zeros(3)

    for imgs, _ in loader:
        # imgs: [B, C, H, W]
        pixels = imgs.permute(1, 0, 2, 3).reshape(3, -1)  # [3, B*H*W]
        batch_mean = pixels.mean(dim=1)
        batch_var = pixels.var(dim=1, unbiased=False)
        batch_count = pixels.size(1)

        # Parallel Welford update.
        delta = batch_mean - mean
        new_count = count + batch_count
        mean = mean + delta * batch_count / new_count
        M2 = M2 + batch_var * batch_count + delta**2 * count * batch_count / new_count
        count = new_count

    std = (M2 / count).sqrt()

    # Guard against zero std (e.g. all-black synthetic images).
    std = torch.clamp(std, min=1e-6)

    return mean.tolist(), std.tolist()


class CrossValidationBlock(ExperimentBlock):
    """K-Fold and Stratified K-Fold cross-validation experiment block."""

    def setup(self, config: ExperimentConfig) -> None:
        """Validate config and initialise fold state.

        Raises:
            ValueError: if cross_validation config is missing.
        """
        if config.cross_validation is None:
            raise ValueError(
                "CrossValidationBlock requires cross_validation to be set in ExperimentConfig."
            )
        self._config = config
        self._fold_results: list[dict[str, Any]] = []
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None

    def run(self) -> None:
        """Execute all folds and write cv_summary.json."""
        cv = self._config.cross_validation
        assert cv is not None

        data_cfg = self._config.data
        pool_dir = data_cfg.base_dir / data_cfg.train_dir

        # Load without normalization — only used to get targets and for stat computation.
        raw_dataset = ImageFolder(str(pool_dir), transform=T.ToTensor())
        targets = np.array(raw_dataset.targets)
        n_samples = len(raw_dataset)
        indices = np.arange(n_samples)

        splitter: KFold | StratifiedKFold
        if cv.stratified:
            splitter = StratifiedKFold(
                n_splits=cv.n_folds,
                shuffle=cv.shuffle,
                random_state=cv.fold_seed if cv.shuffle else None,
            )
            split_iter = splitter.split(indices, targets)
        else:
            splitter = KFold(
                n_splits=cv.n_folds,
                shuffle=cv.shuffle,
                random_state=cv.fold_seed if cv.shuffle else None,
            )
            split_iter = splitter.split(indices)

        base_name = self._config.name

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            train_indices = train_idx.tolist()
            val_indices = val_idx.tolist()

            # trial_start/trial_end is the vocabulary the GUI overlay tracks
            # (same contract as grid/random search); the wrapped callback below
            # streams each fold's epochs annotated with trial context.
            if self._progress_callback is not None:
                self._progress_callback(
                    {
                        "event": "trial_start",
                        "trial_index": fold_idx,
                        "total_trials": cv.n_folds,
                        "overrides": {"fold": fold_idx + 1},
                    }
                )

            fold_record: dict[str, Any] = {
                "fold": fold_idx,
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "status": "failed",
                "error": "",
                # Epochs the fold's trainer ran (early stopping and a user stop
                # both cut it short); 0 for a fold that died before training.
                "epochs_completed": 0,
                "best_val_loss": None,
                "accuracy": None,
                "f1": None,
            }

            # Whether the fold's trainer got as far as emitting its own "end",
            # which the wrapper turns into this fold's trial_end, and whether
            # the stop cut it short.
            trained = False
            cut = False
            try:
                # The same policy as DataModule: -1 ("automático") becomes what
                # the machine affords, and a small fold loads in-process.
                workers = resolve_num_workers(data_cfg.num_workers, len(train_indices))
                fold_mean, fold_std = _compute_fold_stats(
                    raw_dataset, train_indices, workers
                )

                fold_raw: dict[str, Any] = self._config.model_dump(mode="json")
                fold_raw["name"] = f"{base_name}_fold{fold_idx}"
                fold_raw["data"]["transforms"]["normalize_mean"] = fold_mean
                fold_raw["data"]["transforms"]["normalize_std"] = fold_std
                fold_raw["output"]["models_dir"] = str(
                    self._config.output.models_dir / base_name / f"fold{fold_idx}"
                )
                fold_raw["output"]["graphics_dir"] = str(
                    self._config.output.graphics_dir / base_name / f"fold{fold_idx}"
                )
                fold_raw["output"]["logs_dir"] = str(
                    self._config.output.logs_dir / base_name / f"fold{fold_idx}"
                )
                fold_raw["output"]["reports_dir"] = str(
                    self._config.output.reports_dir / base_name / f"fold{fold_idx}"
                )

                fold_config = ExperimentConfig.model_validate(fold_raw)

                train_subset = Subset(raw_dataset, train_indices)
                val_subset = Subset(raw_dataset, val_indices)
                fold_data = _FoldDataModule(
                    train_subset,
                    val_subset,
                    fold_mean,
                    fold_std,
                    fold_config,
                    num_workers=workers,
                )

                model = ModelFactory.create(fold_config.model)
                train_result = Trainer(fold_config).fit(
                    model,
                    fold_data,
                    progress_callback=make_trial_progress_wrapper(
                        self._progress_callback, fold_idx, cv.n_folds
                    ),
                    cancel_token=self._cancel_token,
                )
                trained = True
                fold_record["epochs_completed"] = train_result.total_epochs
                cut = train_result.stopped
                # 0 epochs only when stopped before the first: nothing to score.
                if train_result.total_epochs:
                    state_dict = torch.load(
                        str(train_result.model_path),
                        map_location="cpu",
                        weights_only=True,
                    )
                    model.load_state_dict(state_dict)  # type: ignore[arg-type]

                    eval_result = Evaluator(fold_config).evaluate(
                        model, fold_data.val_loader()
                    )

                    fold_record["status"] = "success"
                    fold_record["best_val_loss"] = train_result.best_val_loss
                    fold_record["accuracy"] = eval_result.accuracy
                    fold_record["f1"] = eval_result.f1

                    logger.info(
                        "Fold {}/{} succeeded: val_loss={:.4f} accuracy={:.4f}",
                        fold_idx + 1,
                        cv.n_folds,
                        train_result.best_val_loss,
                        eval_result.accuracy,
                    )

            except Exception as exc:  # noqa: BLE001
                fold_record["error"] = str(exc)
                # An error is a failure even inside the stop window (ADR-111).
                cut = False
                logger.warning("Fold {}/{} failed: {}", fold_idx + 1, cv.n_folds, exc)

            finally:
                torch.cuda.empty_cache()

            # A fold the stop cut keeps its record but is not a fold of this
            # K-fold any more (ADR-111). One whose stop landed in its last epoch,
            # or in the evaluation after it, finished and counts.
            if cut:
                fold_record["status"] = STOPPED
                fold_record["error"] = STOPPED_NOTE

            # A fold whose Trainer finished emitted its own end, rewritten to
            # trial_end; one that died first is closed here, exactly once.
            if not trained and self._progress_callback is not None:
                self._progress_callback(
                    {
                        "event": "trial_end",
                        "trial_index": fold_idx,
                        "total_trials": cv.n_folds,
                        "status": fold_record["status"],
                    }
                )

            self._fold_results.append(fold_record)

            if is_cancelled(self._cancel_token):
                logger.info(
                    "K-fold stopped after {} of {} folds.",
                    len(self._fold_results),
                    cv.n_folds,
                )
                break

        # Single terminal 'end' after all folds so the GUI closes the SSE
        # stream once — inner fold 'end's were rewritten to 'trial_end'.
        if self._progress_callback is not None:
            self._progress_callback(
                {
                    "event": "end",
                    "total_epochs": 0,
                    "total_trials": len(self._fold_results),
                }
            )

        self._write_summary(base_name)
        # Also emit a top-level run.json so /api/runs picks the CV experiment
        # up alongside classification runs. Without this, CV results never
        # surface in the HistoryOverlay despite producing real metrics.
        self._write_top_level_run_json(base_name)

    def report(self) -> dict[str, Any]:
        """Return aggregated cross-validation metrics.

        A K-fold stopped before any fold finished is reported with no mean
        rather than as a failure: the researcher asked for the stop (ADR-111).

        Raises:
            RuntimeError: if all folds failed and the stop cut nothing.
        """
        successful = [r for r in self._fold_results if r["status"] == "success"]
        stopped = self._stopped()
        if not successful and not stopped:
            raise RuntimeError(
                "CrossValidationBlock: all folds failed — no metrics available."
            )

        accuracies = [r["accuracy"] for r in successful]
        f1s = [r["f1"] for r in successful]

        return {
            "fold_results": self._fold_results,
            "n_folds_ok": len(successful),
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else None,
            "std_accuracy": sample_std(accuracies),
            "mean_f1": float(np.mean(f1s)) if f1s else None,
            "std_f1": sample_std(f1s),
            "std_ddof": 1,
            # The stop cut this job (ADR-111): a unit was cut, or units were
            # left unrun -- all of them, if it landed in the first.
            "stopped": stopped,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _stopped(self) -> bool:
        """Whether the stop cut this K-fold: a fold cut, or folds left unrun."""
        cv = self._config.cross_validation
        assert cv is not None
        return job_was_stopped(
            [r["status"] for r in self._fold_results], cv.n_folds, self._cancel_token
        )

    def _write_summary(self, base_name: str) -> None:
        """Write cv_summary.json to reports_dir / base_name."""
        out_dir = self._config.output.reports_dir / base_name
        out_dir.mkdir(parents=True, exist_ok=True)

        successful = [r for r in self._fold_results if r["status"] == "success"]
        accuracies = [r["accuracy"] for r in successful] if successful else []
        f1s = [r["f1"] for r in successful] if successful else []

        cv = self._config.cross_validation
        assert cv is not None

        summary: dict[str, Any] = {
            "experiment": base_name,
            "n_folds": cv.n_folds,
            "folds": self._fold_results,
            "aggregate": {
                "n": len(accuracies),
                # Runs before ADR-111 wrote the population std (ddof=0).
                "std_ddof": 1,
                "mean_accuracy": float(np.mean(accuracies)) if accuracies else None,
                "std_accuracy": sample_std(accuracies),
                "mean_f1": float(np.mean(f1s)) if f1s else None,
                "std_f1": sample_std(f1s),
            },
        }

        (out_dir / "cv_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def _write_top_level_run_json(self, base_name: str) -> None:
        """Write a parent-level run.json so /api/runs treats CV like other runs.

        Aggregate metrics are flattened into the same keys ClassificationBlock
        uses (test_accuracy, test_f1, best_val_loss, total_epochs) so the
        existing RunSummary parser works without special-casing. The full
        per-fold breakdown is preserved under ``metrics.fold_results`` and
        ``metrics.cv_aggregate`` for RunDetail consumers.
        """
        cv_dir = self._config.output.models_dir / f"{base_name}_cv"
        cv_dir.mkdir(parents=True, exist_ok=True)

        successful = [r for r in self._fold_results if r["status"] == "success"]
        accuracies = [r["accuracy"] for r in successful] if successful else []
        f1s = [r["f1"] for r in successful] if successful else []
        val_losses = [
            r["best_val_loss"] for r in successful if r["best_val_loss"] is not None
        ]

        cv = self._config.cross_validation
        assert cv is not None

        mean_acc = float(np.mean(accuracies)) if accuracies else None
        mean_f1 = float(np.mean(f1s)) if f1s else None
        mean_val_loss = float(np.mean(val_losses)) if val_losses else None
        std_acc = sample_std(accuracies)
        std_f1 = sample_std(f1s)

        run_json: dict[str, Any] = {
            "experiment": base_name,
            # A K-fold the researcher stopped did what was asked of it, even
            # with no fold finished; only genuine failures make it "failed".
            "status": "completed" if successful or self._stopped() else "failed",
            # The run-level stop marker (ADR-111), read by History and the GUI.
            "stopped": self._stopped(),
            # Naive local time like every other run.json writer — mixing aware
            # and naive timestamps breaks the history sort (see routes.py).
            "timestamp": datetime.now().isoformat(),
            "config": self._config.model_dump(mode="json"),
            "metrics": {
                # Mirror the keys the RunSummary parser already understands so
                # CV results show the aggregate metric in the history list.
                # No other multi-unit run writes one; the sum is the training
                # actually done, early-stopped and cut folds included.
                "total_epochs": sum(
                    r.get("epochs_completed", 0) or 0 for r in self._fold_results
                ),
                "test_accuracy": mean_acc,
                "test_f1": mean_f1,
                "best_val_loss": mean_val_loss,
                # CV-specific keys consumed by RunDetailPanel:
                "fold_results": self._fold_results,
                # Means and stds are over the n_folds_ok folds only; a stopped
                # run lists fewer folds than n_folds (ADR-111).
                "cv_aggregate": {
                    "n_folds": cv.n_folds,
                    "n_folds_ok": len(successful),
                    "n_folds_failed": sum(
                        1 for r in self._fold_results if r["status"] == "failed"
                    ),
                    "n_folds_stopped": sum(
                        1 for r in self._fold_results if r["status"] == STOPPED
                    ),
                    # Runs before ADR-111 wrote the population std (ddof=0).
                    "std_ddof": 1,
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "mean_f1": mean_f1,
                    "std_f1": std_f1,
                },
            },
            "history": [],
            "device_used": None,
            "block": "cross_validation",
        }

        (cv_dir / "run.json").write_text(
            json.dumps(run_json, indent=2), encoding="utf-8"
        )


__all__ = ["CrossValidationBlock"]
