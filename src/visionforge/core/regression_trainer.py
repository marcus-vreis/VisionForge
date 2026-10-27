"""Training loop for the image-regression task.

Mirrors the classification ``Trainer`` (device resolution, per-epoch loop,
best-by-val-loss checkpoint, ADR-013 ``run.json``, SSE ``start``/``epoch_end``/
``end`` events) but with a regression criterion (MSE/MAE/Huber) and regression
metrics (MSE, RMSE, MAE, R²). Reuses ``resolve_device`` and ``_seed_everything``
from ``core.trainer`` so device handling stays identical across tasks.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from visionforge.core import trainer as trainer_common
from visionforge.core import training_loop
from visionforge.core.cancellation import CancellationToken
from visionforge.core.tracking import TensorBoardLogger
from visionforge.core.training_health import stagnant_loss, summarize
from visionforge.models.factory import final_linear
from visionforge.utils.regression_config import RegressionConfig

_seed_everything = trainer_common._seed_everything
resolve_device = trainer_common.resolve_device
save_best_checkpoint = trainer_common.save_best_checkpoint


@dataclass
class RegressionEpochResult:
    """Metrics for a single regression training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_mse: float
    val_rmse: float
    val_mae: float
    val_r2: float


@dataclass
class RegressionTrainResult:
    """Summary of a completed regression training run."""

    best_epoch: int
    best_val_loss: float
    total_epochs: int
    device_used: str
    history: list[RegressionEpochResult] = field(default_factory=list)
    model_path: Path = field(default_factory=lambda: Path("."))
    # Things about this run the researcher has to be told, because the
    # metrics alone would look like a result (ADR-099).
    warnings: list[dict[str, str]] = field(default_factory=list)
    # True only when the loop broke on the stop token (ADR-111): a run whose
    # stop arrived in its last epoch, or after it, finished and is not cut.
    stopped: bool = False


class _MetricAccumulator:
    """Streaming MSE/RMSE/MAE/R² over flattened (sample × target) elements.

    Pools across target columns; for single-target regression (the common case)
    this is exactly the per-target metric. Multi-target R² is a pooled estimate.
    """

    def __init__(self) -> None:
        self.n = 0
        self.sum_sq_err = 0.0
        self.sum_abs_err = 0.0
        self.sum_y = 0.0
        self.sum_y_sq = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate pooled regression errors and target moments."""
        diff = preds - targets
        self.n += targets.numel()
        self.sum_sq_err += float((diff * diff).sum().item())
        self.sum_abs_err += float(diff.abs().sum().item())
        self.sum_y += float(targets.sum().item())
        self.sum_y_sq += float((targets * targets).sum().item())

    def compute(self) -> tuple[float, float, float, float]:
        """Return (mse, rmse, mae, r2)."""
        if self.n == 0:
            return 0.0, 0.0, 0.0, 0.0
        mse = self.sum_sq_err / self.n
        rmse = math.sqrt(mse)
        mae = self.sum_abs_err / self.n
        ss_tot = self.sum_y_sq - (self.sum_y * self.sum_y) / self.n
        r2 = 1.0 - (self.sum_sq_err / ss_tot) if ss_tot > 1e-12 else 0.0
        return mse, rmse, mae, r2


class RegressionTrainer(training_loop.GradientTrainerMethods):
    """Manages the full training loop for one regression experiment."""

    _epoch_result_type = RegressionEpochResult
    _initial_best = float("inf")
    _epoch_fields = ("val_mse", "val_rmse", "val_mae", "val_r2")

    def __init__(self, config: RegressionConfig) -> None:
        self._config = config
        self._loader_workers: dict[str, int] = {}
        self._device, self._dp_ids, self._device_label = resolve_device(config.device)

    def fit(
        self,
        model: nn.Module,
        data_module: Any,
        optimizer: torch.optim.Optimizer | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: CancellationToken | None = None,
        resume_dir: Path | None = None,
    ) -> RegressionTrainResult:
        """Run the regression training loop."""
        return training_loop.fit_training(
            self,
            model,
            data_module,
            training_loop.FitOptions(
                optimizer, progress_callback, cancel_token, resume_dir
            ),
            training_loop.FitServices(
                _seed_everything, TensorBoardLogger, save_best_checkpoint
            ),
            training_loop.FitResultAdapter(
                result_type=RegressionTrainResult,
                metric=training_loop.MetricPolicy("val_loss"),
                best_field="best_val_loss",
                log_format="RegressionTrainer using device: {}",
            ),
        )

    def evaluate(
        self, model: nn.Module, loader: Any
    ) -> tuple[float, float, float, float]:
        """Compute (mse, rmse, mae, r2) for the model over the loader."""
        return self.evaluate_with_predictions(model, loader)[0]

    def evaluate_with_predictions(
        self, model: nn.Module, loader: Any
    ) -> tuple[tuple[float, float, float, float], np.ndarray, np.ndarray]:
        """Return regression metrics, per-sample targets and predictions."""
        model = model.to(self._device)
        model.eval()
        metrics = _MetricAccumulator()
        targets_seen: list[np.ndarray] = []
        preds_seen: list[np.ndarray] = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                outputs = model(inputs)
                metrics.update(outputs, targets)
                targets_seen.append(
                    targets.detach().cpu().numpy().reshape(len(targets), -1)
                )
                preds_seen.append(
                    outputs.detach().cpu().numpy().reshape(len(targets), -1)
                )
        empty = np.zeros((0, 1))
        return (
            metrics.compute(),
            np.concatenate(targets_seen) if targets_seen else empty,
            np.concatenate(preds_seen) if preds_seen else empty,
        )

    # ── private helpers ────────────────────────────────────────────────────────

    _report_epoch = staticmethod(
        partial(
            training_loop.report_epoch,
            scalar_fields={
                "loss/train": "train_loss",
                "loss/val": "val_loss",
                "r2/val": "val_r2",
                "rmse/val": "val_rmse",
                "mae/val": "val_mae",
                "mse/val": "val_mse",
            },
            event_fields=("train_loss", "val_loss", "val_rmse", "val_mae", "val_r2"),
            log_format="Epoch {}/{} | train_loss={:.4f} val_loss={:.4f} rmse={:.4f} mae={:.4f} r2={:.4f}",
            log_fields=("train_loss", "val_loss", "val_rmse", "val_mae", "val_r2"),
        )
    )

    @staticmethod
    def _training_warnings(
        history: list[RegressionEpochResult],
    ) -> list[dict[str, str]]:
        """Summarize and log training health warnings for the completed run."""
        return training_loop.training_warnings(history, summarize, stagnant_loss)

    @staticmethod
    def _split_named_params(
        model: nn.Module,
    ) -> training_loop.ParameterGroups:
        """Return (head_params, backbone_params) split at the final Linear.

        The head is the last ``nn.Linear``, not the last named child: on VGG and
        AlexNet that child is a `classifier` block holding three Linear layers,
        so the old split called 119M parameters "head" and froze almost nothing
        under feature extraction (ADR-101).

        Unwraps DataParallel so the head is the model's own final layer, not the
        ``module`` wrapper.
        """
        core = getattr(model, "module", model)
        head_module = final_linear(core)
        head_ids = (
            {id(p) for p in head_module.parameters()}
            if head_module is not None
            else set()
        )
        head: list[nn.Parameter] = []
        backbone: list[nn.Parameter] = []
        parameters = tuple(core.parameters())
        head.extend(param for param in parameters if id(param) in head_ids)
        backbone.extend(param for param in parameters if id(param) not in head_ids)
        return head, backbone

    def _build_criterion(self) -> nn.Module:
        cfg = self._config.training
        if cfg.loss == "mae":
            return nn.L1Loss()
        if cfg.loss == "huber":
            return nn.HuberLoss(delta=cfg.huber_delta)
        return nn.MSELoss()

    def _train_epoch(
        self,
        model: nn.Module,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train one epoch using the shared optimization loop."""
        return training_loop.train_epoch(
            model,
            loader,
            optimizer,
            criterion,
            self._device,
        )

    def _eval_epoch(
        self,
        model: nn.Module,
        loader: Any,
        criterion: nn.Module,
    ) -> tuple[float, float, float, float, float]:
        """Return (val_loss, mse, rmse, mae, r2)."""
        model.eval()
        total_loss = 0.0
        metrics = _MetricAccumulator()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                metrics.update(outputs, targets)
        n = len(loader)
        val_loss = total_loss / n if n > 0 else 0.0
        mse, rmse, mae, r2 = metrics.compute()
        return val_loss, mse, rmse, mae, r2

    def _write_run_json(self, run_dir: Path, result: RegressionTrainResult) -> None:
        """Write the ADR-013 run.json with regression metrics and history."""
        training_loop.write_task_run_json(
            self._config,
            run_dir,
            result,
            getattr(self, "_loader_workers", {}),
            best_metric="best_val_loss",
            epoch_metrics={
                "mse": "val_mse",
                "rmse": "val_rmse",
                "mae": "val_mae",
                "r2": "val_r2",
            },
            history_fields=(
                "epoch",
                "train_loss",
                "val_loss",
                "val_mse",
                "val_rmse",
                "val_mae",
                "val_r2",
            ),
        )


__all__ = [
    "RegressionTrainer",
    "RegressionTrainResult",
    "RegressionEpochResult",
]
