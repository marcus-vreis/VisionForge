from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from visionforge.blocks.base import ExperimentBlock
from visionforge.core.evaluator import Evaluator
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
        self._num_workers = config.data.num_workers
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

            fold_record: dict[str, Any] = {
                "fold": fold_idx,
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "status": "failed",
                "error": "",
                "best_val_loss": None,
                "accuracy": None,
                "f1": None,
            }

            try:
                fold_mean, fold_std = _compute_fold_stats(
                    raw_dataset, train_indices, data_cfg.num_workers
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
                    train_subset, val_subset, fold_mean, fold_std, fold_config
                )

                model = ModelFactory.create(fold_config.model)
                train_result = Trainer(fold_config).fit(model, fold_data)

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
                logger.warning("Fold {}/{} failed: {}", fold_idx + 1, cv.n_folds, exc)

            finally:
                torch.cuda.empty_cache()

            self._fold_results.append(fold_record)

        self._write_summary(base_name)
        # Also emit a top-level run.json so /api/runs picks the CV experiment
        # up alongside classification runs. Without this, CV results never
        # surface in the HistoryOverlay despite producing real metrics.
        self._write_top_level_run_json(base_name)

    def report(self) -> dict[str, Any]:
        """Return aggregated cross-validation metrics.

        Raises:
            RuntimeError: if all folds failed.
        """
        successful = [r for r in self._fold_results if r["status"] == "success"]
        if not successful:
            raise RuntimeError(
                "CrossValidationBlock: all folds failed — no metrics available."
            )

        accuracies = [r["accuracy"] for r in successful]
        f1s = [r["f1"] for r in successful]

        return {
            "fold_results": self._fold_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
        }

    # ── private ───────────────────────────────────────────────────────────────

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
                "mean_accuracy": float(np.mean(accuracies)) if accuracies else None,
                "std_accuracy": float(np.std(accuracies)) if accuracies else None,
                "mean_f1": float(np.mean(f1s)) if f1s else None,
                "std_f1": float(np.std(f1s)) if f1s else None,
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
        std_acc = float(np.std(accuracies)) if accuracies else None
        std_f1 = float(np.std(f1s)) if f1s else None

        run_json: dict[str, Any] = {
            "experiment": base_name,
            "status": "completed" if successful else "failed",
            "timestamp": datetime.now(UTC).isoformat(),
            "config": self._config.model_dump(mode="json"),
            "metrics": {
                # Mirror the keys the RunSummary parser already understands so
                # CV results show the aggregate metric in the history list.
                "total_epochs": sum(
                    r.get("epochs_completed", 0) or 0 for r in self._fold_results
                ),
                "test_accuracy": mean_acc,
                "test_f1": mean_f1,
                "best_val_loss": mean_val_loss,
                # CV-specific keys consumed by RunDetailPanel:
                "fold_results": self._fold_results,
                "cv_aggregate": {
                    "n_folds": cv.n_folds,
                    "n_folds_ok": len(successful),
                    "n_folds_failed": len(self._fold_results) - len(successful),
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
