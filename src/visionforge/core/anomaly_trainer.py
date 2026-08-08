"""Training/scoring loop for the anomaly-detection task.

Handles both model families behind one interface:

- **ConvAutoencoder** — a real gradient loop minimizing reconstruction MSE on
  normal images; the per-image anomaly score is the mean reconstruction error.
  Best checkpoint by lowest train reconstruction loss; test AUROC is computed and
  streamed each epoch for monitoring.
- **PatchCore** — no gradient training: one pass builds the coreset memory bank
  from the normal train set, then the test set is scored. Emits a single fit
  ``epoch_end``.

The metric is image-level AUROC over the labelled test split (sklearn), plus a
decision threshold taken from a percentile of the normal-score distribution and
the resulting image-level F1. Reuses ``resolve_device``/``_seed_everything``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score, roc_auc_score

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_fingerprint import fingerprint_from_config
from visionforge.core.tracking import TensorBoardLogger
from visionforge.core.trainer import _seed_everything, resolve_device
from visionforge.models.anomaly_factory import ConvAutoencoder, PatchCore
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.environment import capture_environment


def compute_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Image-level AUROC; returns 0.5 when only one class is present."""
    y = labels.detach().cpu().numpy()
    s = scores.detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def compute_threshold(normal_scores: torch.Tensor, percentile: float) -> float:
    """Decision threshold = ``percentile`` of the normal-score distribution."""
    if normal_scores.numel() == 0:
        return 0.0
    return float(np.percentile(normal_scores.detach().cpu().numpy(), percentile))


def score_images(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Per-image anomaly score (reconstruction error or patch distance).

    Shared by the training loop and batch-prediction so both families score the
    same way: ConvAutoencoder = mean squared reconstruction error, PatchCore =
    max patch nearest-neighbor distance.
    """
    if isinstance(model, ConvAutoencoder):
        recon = model(inputs)
        err: torch.Tensor = ((recon - inputs) ** 2).mean(dim=(1, 2, 3))
        return err
    if isinstance(model, PatchCore):
        return model.score(inputs)
    raise TypeError(f"Unsupported anomaly model: {type(model).__name__}")


@dataclass
class AnomalyEpochResult:
    """Metrics for a single anomaly training epoch."""

    epoch: int
    train_loss: float
    val_auroc: float
    val_threshold: float
    val_image_f1: float


@dataclass
class AnomalyTrainResult:
    """Summary of a completed anomaly training run."""

    best_epoch: int
    best_auroc: float
    total_epochs: int
    device_used: str
    history: list[AnomalyEpochResult] = field(default_factory=list)
    model_path: Path = field(default_factory=lambda: Path("."))


class AnomalyTrainer:
    """Manages training/scoring for one anomaly-detection experiment."""

    def __init__(self, config: AnomalyConfig) -> None:
        self._config = config
        self._percentile = config.training.threshold_percentile
        self._device, self._dp_ids, self._device_label = resolve_device(config.device)

    @property
    def device_label(self) -> str:
        """Human-readable label of the device actually used (or fallback)."""
        return self._device_label

    def fit(
        self,
        model: nn.Module,
        data_module: Any,
        optimizer: torch.optim.Optimizer | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> AnomalyTrainResult:
        """Train (autoencoder) or fit the memory bank (PatchCore) and score test."""
        _seed_everything(
            self._config.training.seed,
            deterministic=self._config.training.deterministic,
        )
        logger.info("AnomalyTrainer using device: {}", self._device_label)
        model = model.to(self._device)
        run_dir = self._make_run_dir()
        model_path = run_dir / "best_model.pth"

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "total_epochs": (
                        1
                        if isinstance(model, PatchCore)
                        else self._config.training.epochs
                    ),
                    "device": self._device_label,
                }
            )

        tb = TensorBoardLogger(run_dir / "tensorboard")
        try:
            if isinstance(model, PatchCore):
                result = self._fit_patchcore(
                    model, data_module, model_path, progress_callback, tb
                )
            else:
                result = self._fit_autoencoder(
                    model,
                    data_module,
                    optimizer,
                    model_path,
                    progress_callback,
                    tb,
                    cancel_token,
                )
        finally:
            tb.close()

        self._write_run_json(run_dir, result)
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        if progress_callback is not None:
            progress_callback({"event": "end", "total_epochs": result.total_epochs})
        return result

    def evaluate(
        self, model: nn.Module, train_loader: Any, test_loader: Any
    ) -> tuple[float, float, float]:
        """Return (auroc, threshold, image_f1) for ``model`` on the test split.

        The threshold is derived from the normal (train) score distribution; the
        F1 uses that threshold against the binary test labels.
        """
        return self.evaluate_with_scores(model, train_loader, test_loader)[0]

    def evaluate_with_scores(
        self, model: nn.Module, train_loader: Any, test_loader: Any
    ) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
        """As ``evaluate``, plus the per-image test labels and anomaly scores.

        Those arrays are what the bootstrap intervals resample (ADR-076), and
        they come from the same pass that produced the aggregates, so a report
        cannot show a metric computed differently from its interval.
        """
        model = model.to(self._device)
        model.eval()
        normal_scores = self._collect_scores(model, train_loader)[0]
        test_scores, test_labels = self._collect_scores(model, test_loader)
        metrics = self._metrics(normal_scores, test_scores, test_labels)
        return (
            metrics,
            test_labels.detach().cpu().numpy().ravel(),
            test_scores.detach().cpu().numpy().ravel(),
        )

    # ── private ─────────────────────────────────────────────────────────────────

    def _fit_autoencoder(
        self,
        model: nn.Module,
        data_module: Any,
        optimizer: torch.optim.Optimizer | None,
        model_path: Path,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        tb: TensorBoardLogger,
        cancel_token: CancellationToken | None = None,
    ) -> AnomalyTrainResult:
        cfg = self._config.training
        optimizer = optimizer or self._build_optimizer(model)
        criterion = nn.MSELoss()
        train_loader = data_module.train_loader()
        test_loader = data_module.test_loader()

        history: list[AnomalyEpochResult] = []
        best_loss = float("inf")
        best_epoch = 0
        patience = 0
        t0 = time.monotonic()

        for epoch in range(1, cfg.epochs + 1):
            # The safe point: the previous epoch's checkpoint is written and its
            # metrics emitted, so stopping here leaves the run directory whole.
            if is_cancelled(cancel_token):
                logger.info(
                    "Run cancelled at epoch {}; keeping the best checkpoint so far.",
                    epoch,
                )
                break

            model.train()
            total = 0.0
            for inputs, _ in train_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                recon = model(inputs)
                loss = criterion(recon, inputs)
                loss.backward()
                optimizer.step()
                total += loss.item()
            n = len(train_loader)
            train_loss = total / n if n > 0 else 0.0

            normal_scores = self._collect_scores(model, train_loader)[0]
            test_scores, test_labels = self._collect_scores(model, test_loader)
            auroc, threshold, f1 = self._metrics(
                normal_scores, test_scores, test_labels
            )
            history.append(AnomalyEpochResult(epoch, train_loss, auroc, threshold, f1))
            tb.log_scalars(
                epoch,
                {
                    "loss/recon_train": train_loss,
                    "auroc/val": auroc,
                    "image_f1/val": f1,
                    "threshold/val": threshold,
                },
            )

            logger.info(
                "Epoch {}/{} | recon_loss={:.4f} auroc={:.4f} f1={:.4f}",
                epoch,
                cfg.epochs,
                train_loss,
                auroc,
                f1,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "total_epochs": cfg.epochs,
                        "train_loss": train_loss,
                        "val_auroc": auroc,
                        "val_image_f1": f1,
                        "val_threshold": threshold,
                        "elapsed_s": round(time.monotonic() - t0, 3),
                    }
                )

            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                patience = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch {}.", epoch)
                    break

        if not model_path.is_file():
            torch.save(model.state_dict(), model_path)
            best_epoch = best_epoch or len(history)
        best = next((r for r in history if r.epoch == best_epoch), history[-1])
        return AnomalyTrainResult(
            best_epoch=best_epoch,
            best_auroc=best.val_auroc,
            total_epochs=len(history),
            device_used=self._device_label,
            history=history,
            model_path=model_path,
        )

    def _fit_patchcore(
        self,
        model: PatchCore,
        data_module: Any,
        model_path: Path,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        tb: TensorBoardLogger,
    ) -> AnomalyTrainResult:
        model.eval()
        train_loader = data_module.train_loader()
        test_loader = data_module.test_loader()
        t0 = time.monotonic()

        with torch.no_grad():
            patches = []
            for inputs, _ in train_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                feats = model.extract(inputs).reshape(-1, model.feature_dim)
                patches.append(feats.cpu())
            model.fit(torch.cat(patches, dim=0).to(self._device))

        normal_scores = self._collect_scores(model, train_loader)[0]
        test_scores, test_labels = self._collect_scores(model, test_loader)
        auroc, threshold, f1 = self._metrics(normal_scores, test_scores, test_labels)
        torch.save(model.state_dict(), model_path)
        tb.log_scalars(
            1,
            {"auroc/val": auroc, "image_f1/val": f1, "threshold/val": threshold},
        )

        logger.info("PatchCore fit | auroc={:.4f} f1={:.4f}", auroc, f1)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_end",
                    "epoch": 1,
                    "total_epochs": 1,
                    "train_loss": 0.0,
                    "val_auroc": auroc,
                    "val_image_f1": f1,
                    "val_threshold": threshold,
                    "elapsed_s": round(time.monotonic() - t0, 3),
                }
            )
        return AnomalyTrainResult(
            best_epoch=1,
            best_auroc=auroc,
            total_epochs=1,
            device_used=self._device_label,
            history=[AnomalyEpochResult(1, 0.0, auroc, threshold, f1)],
            model_path=model_path,
        )

    def _collect_scores(
        self, model: nn.Module, loader: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scores, labels) over ``loader`` as 1-D CPU tensors."""
        model.eval()
        scores: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, label in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                scores.append(self._image_scores(model, inputs).cpu())
                labels.append(
                    label if isinstance(label, torch.Tensor) else torch.tensor(label)
                )
        if not scores:
            return torch.empty(0), torch.empty(0)
        return torch.cat(scores), torch.cat(labels)

    def _image_scores(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Per-image anomaly score (reconstruction error or patch distance)."""
        return score_images(model, inputs)

    def _metrics(
        self,
        normal_scores: torch.Tensor,
        test_scores: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> tuple[float, float, float]:
        auroc = compute_auroc(test_scores, test_labels)
        threshold = compute_threshold(normal_scores, self._percentile)
        preds = (test_scores >= threshold).long().cpu().numpy()
        y = test_labels.long().cpu().numpy()
        f1 = float(f1_score(y, preds, zero_division=0))
        return auroc, threshold, f1

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        cfg = self._config.training
        builders: dict[str, Callable[..., torch.optim.Optimizer]] = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
        }
        return builders[cfg.optimizer](
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self._config.output.models_dir / self._config.name / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _write_run_json(self, run_dir: Path, result: AnomalyTrainResult) -> None:
        """Write the ADR-013 run.json with anomaly metrics + history."""
        best = next((r for r in result.history if r.epoch == result.best_epoch), None)
        run_json: dict[str, Any] = {
            "id": f"{self._config.name}_{run_dir.name}",
            "experiment": self._config.name,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "device_used": result.device_used,
            "environment": capture_environment(),
            # Proves two runs saw the same data, not just the same path.
            "dataset_fingerprint": fingerprint_from_config(self._config),
            "run_dir": str(run_dir.resolve()),
            "config": self._config.model_dump(mode="json"),
            "metrics": {
                "best_auroc": result.best_auroc,
                "best_epoch": result.best_epoch,
                "total_epochs": result.total_epochs,
                "auroc": best.val_auroc if best else 0.0,
                "threshold": best.val_threshold if best else 0.0,
                "image_f1": best.val_image_f1 if best else 0.0,
            },
            "history": [
                {
                    "epoch": r.epoch,
                    "train_loss": r.train_loss,
                    "val_auroc": r.val_auroc,
                    "val_image_f1": r.val_image_f1,
                    "val_threshold": r.val_threshold,
                }
                for r in result.history
            ],
            "artifacts": {
                "model": str(result.model_path),
                "graphics": [],
                "report": None,
            },
            "tests": [],
        }
        (run_dir / "run.json").write_text(
            json.dumps(run_json, indent=2), encoding="utf-8"
        )


__all__ = [
    "AnomalyTrainer",
    "AnomalyTrainResult",
    "AnomalyEpochResult",
    "compute_auroc",
    "compute_threshold",
    "score_images",
]
