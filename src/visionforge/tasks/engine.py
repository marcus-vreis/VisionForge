"""Generic training engine for researcher-defined tasks (ADR-058, brick 2).

Drives a :class:`~visionforge.tasks.base.TaskSpec`'s four Level 1 hooks with
everything the built-in trainers provide — seeding (+ determinism flag), device
resolution, the epoch loop with optional AMP, LR scheduling, early stopping,
best-checkpoint selection by the task's declared primary metric (direction
aware), SSE ``start``/``epoch_end``/``end`` events, TensorBoard scalars, the
primary-metric curve plot and the ADR-013 ``run.json`` with environment capture.

A Level 2 task (custom ``run``) receives a :class:`TaskRunContext` instead and
owns the loop while still honouring the run-dir/SSE/run.json contracts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from loguru import logger
from torch import nn

from visionforge.core.plotter import MetricsPlotter
from visionforge.core.tracking import TensorBoardLogger
from visionforge.core.trainer import _seed_everything, resolve_device
from visionforge.tasks.base import BaseTaskConfig
from visionforge.tasks.registry import TaskInfo
from visionforge.utils.environment import capture_environment

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class TaskEpochResult:
    """One epoch of a custom-task run: train loss + the task's val metrics."""

    epoch: int
    train_loss: float
    val_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TaskRunResult:
    """Summary of a completed custom-task run."""

    metrics: dict[str, float]
    best_epoch: int
    total_epochs: int
    device_used: str
    run_dir: Path
    model_path: Path | None
    history: list[TaskEpochResult] = field(default_factory=list)


@dataclass
class TaskRunContext:
    """What a Level 2 ``run(cfg, ctx)`` gets to honour the contracts itself."""

    run_dir: Path
    device: torch.device
    device_label: str
    emit: ProgressCallback

    def save_checkpoint(self, model: nn.Module, name: str = "best_model.pth") -> Path:
        """Persist ``model``'s state dict under the run dir and return the path."""
        path = self.run_dir / name
        torch.save(model.state_dict(), path)
        return path


class GenericTaskEngine:
    """Runs one custom task config end to end (Level 1 hooks or Level 2 run)."""

    def __init__(self, info: TaskInfo, config: BaseTaskConfig) -> None:
        if info.spec_cls is None:
            raise ValueError(f"Task '{info.key}' has no spec class registered.")
        self._info = info
        self._config = config
        self._device, _, self._device_label = resolve_device(config.device)

    def run(self, progress_callback: ProgressCallback | None = None) -> TaskRunResult:
        """Train per the spec and return the aggregated result.

        Raises:
            ValueError: when a hook reports no value for the primary metric.
        """
        cfg = self._config
        info = self._info
        assert info.spec_cls is not None  # narrowed in __init__
        spec = info.spec_cls()
        emit: ProgressCallback = progress_callback or (lambda _e: None)

        _seed_everything(cfg.training.seed, deterministic=cfg.training.deterministic)
        run_dir = self._make_run_dir()
        logger.info("Custom task '{}' using device: {}", info.key, self._device_label)

        if spec.has_custom_run():
            ctx = TaskRunContext(
                run_dir=run_dir,
                device=self._device,
                device_label=self._device_label,
                emit=emit,
            )
            emit({"event": "start", "total_epochs": 1, "device": self._device_label})
            metrics = spec.run(cfg, ctx) or {}
            result = TaskRunResult(
                metrics=metrics,
                best_epoch=1,
                total_epochs=1,
                device_used=self._device_label,
                run_dir=run_dir,
                model_path=None,
            )
            self._write_run_json(result)
            emit({"event": "end", "total_epochs": 1})
            return result

        return self._run_level1(spec, cfg, run_dir, emit)

    # ── level 1 loop ─────────────────────────────────────────────────────────

    def _run_level1(
        self,
        spec: Any,
        cfg: BaseTaskConfig,
        run_dir: Path,
        emit: ProgressCallback,
    ) -> TaskRunResult:
        model = spec.build_model(cfg).to(self._device)
        train_loader, val_loader, test_loader = spec.build_loaders(cfg)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        primary = self._info.primary_metric
        higher_better = self._info.metrics.get(primary, "higher") == "higher"
        best_value = float("-inf") if higher_better else float("inf")
        best_epoch = 0
        patience = 0
        model_path = run_dir / "best_model.pth"
        history: list[TaskEpochResult] = []
        use_amp = cfg.training.mixed_precision and self._device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        tb = TensorBoardLogger(run_dir / "tensorboard")
        t0 = time.monotonic()

        emit(
            {
                "event": "start",
                "total_epochs": cfg.training.epochs,
                "device": self._device_label,
            }
        )
        try:
            for epoch in range(1, cfg.training.epochs + 1):
                model.train()
                total = 0.0
                batches = 0
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(self._device.type, enabled=use_amp):
                        loss = spec.compute_loss(model, batch, cfg)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total += float(loss.detach())
                    batches += 1
                train_loss = total / batches if batches else 0.0

                model.eval()
                val_metrics = spec.compute_metrics(model, val_loader, cfg)
                if primary not in val_metrics:
                    raise ValueError(
                        f"compute_metrics must return the declared primary metric "
                        f"'{primary}' (got: {sorted(val_metrics)})."
                    )
                value = float(val_metrics[primary])
                history.append(TaskEpochResult(epoch, train_loss, dict(val_metrics)))
                tb.log_scalars(
                    epoch,
                    {"loss/train": train_loss}
                    | {f"{k}/val": v for k, v in val_metrics.items()},
                )
                emit(
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "total_epochs": cfg.training.epochs,
                        "train_loss": train_loss,
                        # val_accuracy keeps the live monitor's chart populated
                        # for any task (it plots this compat field).
                        "val_accuracy": value,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "elapsed_s": round(time.monotonic() - t0, 3),
                    }
                )

                improved = value > best_value if higher_better else value < best_value
                if improved:
                    best_value = value
                    best_epoch = epoch
                    patience = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    patience += 1
                    if patience >= cfg.training.early_stopping_patience:
                        logger.info("Early stopping at epoch {}.", epoch)
                        break

                if scheduler is not None:
                    if isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        scheduler.step(value)
                    else:
                        scheduler.step()
        finally:
            tb.close()

        if not model_path.is_file():  # epochs=0-like guard; keep a checkpoint
            torch.save(model.state_dict(), model_path)
            best_epoch = best_epoch or len(history)

        # Reload the best checkpoint and compute the final (+ optional test) metrics.
        model.load_state_dict(
            torch.load(str(model_path), map_location="cpu", weights_only=True)
        )
        model = model.to(self._device)
        model.eval()
        final_metrics = {
            k: float(v) for k, v in spec.compute_metrics(model, val_loader, cfg).items()
        }
        if test_loader is not None:
            test_metrics = spec.compute_metrics(model, test_loader, cfg)
            final_metrics |= {f"test_{k}": float(v) for k, v in test_metrics.items()}

        self._render_primary_curve(run_dir, history)
        result = TaskRunResult(
            metrics=final_metrics,
            best_epoch=best_epoch,
            total_epochs=len(history),
            device_used=self._device_label,
            run_dir=run_dir,
            model_path=model_path,
            history=history,
        )
        self._write_run_json(result)
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        emit({"event": "end", "total_epochs": len(history)})
        return result

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self._config.output.models_dir / f"{self._config.name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        cfg = self._config.training
        params = [p for p in model.parameters() if p.requires_grad]
        if cfg.optimizer == "sgd":
            return torch.optim.SGD(
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        return torch.optim.Adam(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Any | None:
        sched = self._config.training.scheduler
        if sched.kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self._config.training.epochs
            )
        if sched.kind == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sched.step_size, gamma=sched.gamma
            )
        if sched.kind == "plateau":
            mode: Literal["min", "max"] = (
                "max"
                if self._info.metrics.get(self._info.primary_metric) == "higher"
                else "min"
            )
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                patience=sched.patience,
                factor=sched.factor,
                min_lr=sched.min_lr,
            )
        return None

    def _render_primary_curve(
        self, run_dir: Path, history: list[TaskEpochResult]
    ) -> None:
        primary = self._info.primary_metric
        values = [r.val_metrics[primary] for r in history if primary in r.val_metrics]
        if not values:
            return
        MetricsPlotter.metric_curve(
            [r.epoch for r in history if primary in r.val_metrics],
            values,
            run_dir / f"{primary}_curve.png",
            label=primary,
            ylabel=primary,
            title=f"{self._info.label} — {primary} (val)",
        )

    def _write_run_json(self, result: TaskRunResult) -> None:
        """Write the ADR-013 run.json; ``task`` is stamped ``custom:<key>``."""
        run_json: dict[str, Any] = {
            "id": f"{self._config.name}_{result.run_dir.name}",
            "experiment": self._config.name,
            "task": f"custom:{self._info.key}",
            "task_label": self._info.label,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "device_used": result.device_used,
            "environment": capture_environment(),
            "run_dir": str(result.run_dir.resolve()),
            "config": self._config.model_dump(mode="json"),
            "metrics": {
                "best_epoch": result.best_epoch,
                "total_epochs": result.total_epochs,
                **result.metrics,
            },
            "history": [
                {
                    "epoch": r.epoch,
                    "train_loss": r.train_loss,
                    **{f"val_{k}": v for k, v in r.val_metrics.items()},
                }
                for r in result.history
            ],
            "artifacts": {
                "model": str(result.model_path) if result.model_path else None,
                "graphics": [],
                "report": None,
            },
            "tests": [],
        }
        (result.run_dir / "run.json").write_text(
            json.dumps(run_json, indent=2), encoding="utf-8"
        )


__all__ = [
    "GenericTaskEngine",
    "TaskEpochResult",
    "TaskRunContext",
    "TaskRunResult",
]
