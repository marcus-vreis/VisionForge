"""Training loop for the image-regression task (Phase 6).

Mirrors the classification ``Trainer`` (device resolution, per-epoch loop,
best-by-val-loss checkpoint, ADR-013 ``run.json``, SSE ``start``/``epoch_end``/
``end`` events) but with a regression criterion (MSE/MAE/Huber) and regression
metrics (MSE, RMSE, MAE, R²). Reuses ``resolve_device`` and ``_seed_everything``
from ``core.trainer`` so device handling stays identical across tasks. See
documentation/PHASE6_REGRESSION_PLAN.md.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from visionforge.core.dataset_fingerprint import fingerprint_from_config
from visionforge.core.tracking import TensorBoardLogger
from visionforge.core.trainer import _seed_everything, resolve_device
from visionforge.utils.environment import capture_environment
from visionforge.utils.regression_config import RegressionConfig


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


class RegressionTrainer:
    """Manages the full training loop for one regression experiment."""

    def __init__(self, config: RegressionConfig) -> None:
        self._config = config
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
    ) -> RegressionTrainResult:
        """Run the regression training loop.

        Args:
            model: nn.Module emitting ``num_targets`` continuous outputs.
            data_module: object exposing train_loader() and val_loader().
            optimizer: pre-built optimizer; when None, one is built from config.
            progress_callback: called at start/end and after each epoch with a
                progress dict. Safe to call from a worker thread.
        """
        cfg = self._config.training
        _seed_everything(cfg.seed, deterministic=cfg.deterministic)
        logger.info("RegressionTrainer using device: {}", self._device_label)

        model = self._prepare_model(model)
        self._apply_transfer_learning(model)
        optimizer = optimizer if optimizer is not None else self._build_optimizer(model)
        criterion = self._build_criterion()
        scheduler = self._build_scheduler(optimizer)
        run_dir = self._make_run_dir()
        model_path = run_dir / "best_model.pth"
        tb = TensorBoardLogger(run_dir / "tensorboard")

        history: list[RegressionEpochResult] = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        t0 = time.monotonic()

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "total_epochs": cfg.epochs,
                    "device": self._device_label,
                }
            )

        train_loader = data_module.train_loader()
        val_loader = data_module.val_loader()
        save_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")
        save_future = None

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            val_loss, mse, rmse, mae, r2 = self._eval_epoch(
                model, val_loader, criterion
            )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            result = RegressionEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_mse=mse,
                val_rmse=rmse,
                val_mae=mae,
                val_r2=r2,
            )
            history.append(result)
            tb.log_scalars(
                epoch,
                {
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "r2/val": r2,
                    "rmse/val": rmse,
                    "mae/val": mae,
                    "mse/val": mse,
                },
            )

            logger.info(
                "Epoch {}/{} | train_loss={:.4f} val_loss={:.4f} "
                "rmse={:.4f} mae={:.4f} r2={:.4f}",
                epoch,
                cfg.epochs,
                train_loss,
                val_loss,
                rmse,
                mae,
                r2,
            )

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "total_epochs": cfg.epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_rmse": rmse,
                        "val_mae": mae,
                        "val_r2": r2,
                        "elapsed_s": round(time.monotonic() - t0, 3),
                    }
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                state_dict = (
                    model.module.state_dict()  # type: ignore[union-attr]
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                )
                if save_future is not None:
                    save_future.result()
                save_future = save_pool.submit(torch.save, state_dict, model_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch {}.", epoch)
                    break

        if save_future is not None:
            save_future.result()
        save_pool.shutdown(wait=False)

        train_result = RegressionTrainResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            total_epochs=len(history),
            device_used=self._device_label,
            history=history,
            model_path=model_path,
        )
        self._write_run_json(run_dir, train_result)
        tb.close()

        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        if progress_callback is not None:
            progress_callback({"event": "end", "total_epochs": len(history)})

        return train_result

    def evaluate(
        self, model: nn.Module, loader: Any
    ) -> tuple[float, float, float, float]:
        """Compute (mse, rmse, mae, r2) for ``model`` over ``loader``.

        Reusable by the block for test-set metrics and any future per-model test
        endpoint; moves the model to the resolved device and runs inference-only.
        """
        model = model.to(self._device)
        model.eval()
        metrics = _MetricAccumulator()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                metrics.update(model(inputs), targets)
        return metrics.compute()

    # ── private helpers ────────────────────────────────────────────────────────

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self._device)
        if self._dp_ids:
            model = nn.DataParallel(model, device_ids=self._dp_ids)
        return model

    def _apply_transfer_learning(self, model: nn.Module) -> None:
        """Freeze the backbone for feature extraction (no-op otherwise).

        ``feature_extraction`` freezes every child except the head (the last
        named child); ``fine_tuning`` and the unset case leave all params
        trainable — the LR split happens in ``_build_optimizer``.
        """
        tl = self._config.transfer_learning
        if tl is None or tl.mode != "feature_extraction":
            return
        head, backbone = self._split_named_params(model)
        for p in backbone:
            p.requires_grad = False
        for p in head:
            p.requires_grad = True

    @staticmethod
    def _split_named_params(
        model: nn.Module,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Return (head_params, backbone_params) split by the last named child.

        Unwraps DataParallel so the head is the model's own final layer, not the
        ``module`` wrapper.
        """
        core = getattr(model, "module", model)
        children = list(core.named_children())
        head_name = children[-1][0] if children else None
        head: list[nn.Parameter] = []
        backbone: list[nn.Parameter] = []
        for name, child in children:
            (head if name == head_name else backbone).extend(child.parameters())
        return head, backbone

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        cfg = self._config.training
        builders: dict[str, Callable[..., torch.optim.Optimizer]] = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
        }
        builder = builders[cfg.optimizer]
        tl = self._config.transfer_learning

        if tl is None:
            return builder(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

        if tl.mode == "feature_extraction":
            trainable = [p for p in model.parameters() if p.requires_grad]
            return builder(
                trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )

        # fine_tuning: backbone at a reduced LR, head at the full LR.
        head, backbone = self._split_named_params(model)
        groups = [
            {
                "params": backbone,
                "lr": cfg.learning_rate * tl.backbone_lr_multiplier,
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": head,
                "lr": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
            },
        ]
        return builder(groups)

    def _build_criterion(self) -> nn.Module:
        cfg = self._config.training
        if cfg.loss == "mae":
            return nn.L1Loss()
        if cfg.loss == "huber":
            return nn.HuberLoss(delta=cfg.huber_delta)
        return nn.MSELoss()

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        sched_cfg = self._config.training.scheduler
        kind = sched_cfg.kind
        if kind == "none":
            return None
        if kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self._config.training.epochs
            )
        if kind == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=sched_cfg.patience,
            factor=sched_cfg.factor,
            min_lr=sched_cfg.min_lr,
        )

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self._config.output.models_dir / self._config.name / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _train_epoch(
        self,
        model: nn.Module,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        n = len(loader)
        return total_loss / n if n > 0 else 0.0

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
        """Write the ADR-013 run.json with regression metrics + history."""
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
                "best_val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
                "total_epochs": result.total_epochs,
                "mse": best.val_mse if best else 0.0,
                "rmse": best.val_rmse if best else 0.0,
                "mae": best.val_mae if best else 0.0,
                "r2": best.val_r2 if best else 0.0,
            },
            "history": [
                {
                    "epoch": r.epoch,
                    "train_loss": r.train_loss,
                    "val_loss": r.val_loss,
                    "val_mse": r.val_mse,
                    "val_rmse": r.val_rmse,
                    "val_mae": r.val_mae,
                    "val_r2": r.val_r2,
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
    "RegressionTrainer",
    "RegressionTrainResult",
    "RegressionEpochResult",
]
