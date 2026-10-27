from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from visionforge.core import training_loop
from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.resume import (
    ResumeState,
    clear_resume_state,
)
from visionforge.core.tracking import TensorBoardLogger
from visionforge.core.training_health import (
    collapsed_predictions,
    stagnant_loss,
    summarize,
)
from visionforge.utils.config import DeviceConfig, ExperimentConfig
from visionforge.utils.cuda import check_cuda


@dataclass
class EpochResult:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


@dataclass
class TrainResult:
    """Summary of a completed training run."""

    best_epoch: int
    best_val_loss: float
    total_epochs: int
    device_used: str
    history: list[EpochResult] = field(default_factory=list)
    model_path: Path = field(default_factory=lambda: Path("."))
    # Things about this run the researcher has to be told, because the metrics
    # alone would look like a result (ADR-099).
    warnings: list[dict[str, str]] = field(default_factory=list)
    # True only when the loop broke on the stop token (ADR-111): a run whose
    # stop arrived in its last epoch, or after it, finished and is not cut.
    stopped: bool = False


def frozen_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Copy state tensors to independent CPU storage for asynchronous writes."""
    return deepcopy(
        state_dict,
        {
            id(value): value.detach().to("cpu", copy=True)
            for value in state_dict.values()
            if isinstance(value, torch.Tensor)
        },
    )


def save_best_checkpoint(
    pool: ThreadPoolExecutor,
    pending: Future[None] | None,
    model: nn.Module,
    path: Path,
) -> Future[None]:
    """Save frozen model weights after the pending checkpoint completes."""
    state = (
        model.module.state_dict()  # type: ignore[union-attr]
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    if pending is not None:
        pending.result()
    return pool.submit(torch.save, frozen_state_dict(state), path)


def _seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed every RNG and configure cuDNN performance mode.

    Args:
        seed: integer seed applied to stdlib, numpy, and PyTorch RNGs.
        deterministic: when True (the default), cuDNN uses deterministic
            algorithms and benchmark auto-tuning is off, so the same config
            and seed reproduce the same numbers. When False, ``cudnn.benchmark``
            lets cuDNN auto-select a convolution algorithm per input shape.

            The cost of determinism was measured rather than assumed (ADR-098),
            because this docstring used to claim "3-5× slower" and that is not
            what a RTX 5060 Ti shows on 2-epoch runs: resnet18@224 came out even,
            and resnet50@224 and resnet18@320 were both *faster* deterministic
            (-19% and -16%). Auto-tuning costs time up front and only pays for
            itself over many epochs with a stable shape, which is not the shape
            of an experiment. Long training may still favour ``False``; that is
            why it remains a knob.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_before_building(training: Any) -> None:
    """Seed model construction using the configured determinism settings."""
    _seed_everything(training.seed, deterministic=training.deterministic)


def resolve_device(device_cfg: DeviceConfig) -> tuple[torch.device, list[int], str]:
    """Resolve the requested device, DataParallel ids and display label."""
    info = check_cuda()

    if device_cfg.kind == "cpu":
        return torch.device("cpu"), [], "cpu"

    if not info.available:
        logger.warning(
            "Device requested '{}' but CUDA unavailable — falling back to CPU.",
            device_cfg.kind,
        )
        return torch.device("cpu"), [], "cpu (fallback: CUDA unavailable)"

    available_ids = [d.index for d in info.devices]

    if device_cfg.kind == "cuda":
        return _resolve_single_cuda(device_cfg, info, available_ids)

    # multi_cuda
    requested = device_cfg.gpu_ids if device_cfg.gpu_ids else available_ids
    valid = [i for i in requested if i in available_ids]
    if len(valid) < 2:
        logger.warning(
            "multi_cuda requested but fewer than 2 valid GPUs ({}); using GPU 0.",
            valid,
        )
        primary = valid[0] if valid else 0
        name = (
            info.devices[primary].name
            if primary < len(info.devices)
            else f"cuda:{primary}"
        )
        return torch.device(f"cuda:{primary}"), [], f"cuda:{primary} ({name})"

    primary = valid[0]
    names = ", ".join(info.devices[i].name for i in valid if i < len(info.devices))
    return torch.device(f"cuda:{primary}"), valid, f"multi_cuda{valid} ({names})"


def _resolve_single_cuda(
    device_cfg: DeviceConfig, info: Any, available_ids: list[int]
) -> tuple[torch.device, list[int], str]:
    """Select the requested CUDA device or retain the GPU 0 fallback."""
    if device_cfg.gpu_ids:
        primary = device_cfg.gpu_ids[0]
        if primary not in available_ids:
            logger.warning(
                "GPU {} not in available {}; using GPU 0.", primary, available_ids
            )
            primary = 0
    else:
        primary = 0
    name = (
        info.devices[primary].name if primary < len(info.devices) else f"cuda:{primary}"
    )
    return torch.device(f"cuda:{primary}"), [], f"cuda:{primary} ({name})"


class Trainer:
    """Manages the full training loop for one classification experiment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._loader_workers: dict[str, int] = {}
        self._device, self._dp_ids, self._device_label = resolve_device(config.device)
        # Cache last-epoch val probs/labels so callers (block, evaluator) can plot ROC.
        self._last_val_labels: list[int] = []
        self._last_val_probs: list[float] = []
        # Kept so the run can be checked for collapse: a model predicting one
        # class still reports an accuracy, and it is the class prior (ADR-099).
        self._last_val_preds: list[int] = []

    @property
    def device_label(self) -> str:
        """Human-readable label of the device actually used (or fallback message)."""
        return self._device_label

    def fit(
        self,
        model: nn.Module,
        data_module: Any,
        optimizer: torch.optim.Optimizer | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: CancellationToken | None = None,
        resume_dir: Path | None = None,
    ) -> TrainResult:
        """Train the model and return its best checkpoint and epoch history."""
        resumed = training_loop.load_resume_state(resume_dir, self._config)
        cfg = self._config.training
        _seed_everything(cfg.seed, deterministic=cfg.deterministic)

        logger.info("Trainer using device: {}", self._device_label)

        model, optimizer, criterion, scheduler, scaler, run_dir, tb = (
            self._initialize_training(model, optimizer, resume_dir, resumed)
        )
        model_path = run_dir / "best_model.pth"

        history, best_val_loss, best_epoch, patience_counter, start_epoch = (
            self._restore_training(
                model, optimizer, scheduler, scaler, resumed, run_dir
            )
        )
        t0 = time.monotonic()

        train_loader, val_loader, self._loader_workers = training_loop.start_training(
            data_module, progress_callback, cfg.epochs, self._device_label
        )
        state = training_loop.EpochState(
            history, best_val_loss, best_epoch, patience_counter, start_epoch
        )
        writes, stopped = training_loop.run_epochs(
            state,
            training_loop.EpochRuntime(
                train_loader, val_loader, cfg.epochs, resumed, cancel_token
            ),
            training_loop.CheckpointContext(
                model,
                model_path,
                save_best_checkpoint,
                lambda: self._checkpoint_components(
                    model, optimizer, scheduler, scaler
                ),
            ),
            training_loop.EpochCallbacks(
                compute=lambda epoch: self._compute_epoch_result(
                    epoch,
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    scheduler,
                    scaler=scaler,
                ),
                report=lambda result: self._report_epoch(
                    result, tb, progress_callback, t0
                ),
                early_stop=self._early_stopping_reached,
            ),
            training_loop.MetricPolicy("val_loss"),
        )
        best_val_loss, best_epoch = state.best_metric, state.best_epoch

        if not is_cancelled(cancel_token):
            clear_resume_state(run_dir)

        train_result = TrainResult(
            stopped=stopped,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            total_epochs=len(history),
            device_used=self._device_label,
            history=history,
            model_path=model_path,
        )
        return self._finish_training(
            train_result,
            run_dir,
            data_module,
            tb,
            writes.pool,
            writes.future,
            progress_callback,
        )

    # ── private helpers ────────────────────────────────────────────────────────

    def _initialize_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        resume_dir: Path | None,
        resumed: ResumeState | None,
    ) -> tuple[
        nn.Module,
        torch.optim.Optimizer,
        nn.Module,
        Any,
        Any,
        Path,
        TensorBoardLogger,
    ]:
        """Prepare the model, optimization tools and output directory."""
        cfg = self._config.training
        model = self._prepare_model(model)
        optimizer = optimizer if optimizer is not None else self._build_optimizer(model)
        criterion = self._build_criterion()
        scheduler = self._build_scheduler(optimizer)
        # AMP only meaningful on CUDA; on CPU AMP is a no-op + warning.
        use_amp = cfg.mixed_precision and self._device.type == "cuda"
        # torch.amp.* is the non-deprecated API (the torch.cuda.amp.* aliases emit
        # a FutureWarning since torch 2.4). The generic namespace exists since 2.3,
        # which is the project's floor (pyproject torch>=2.3), so this is safe.
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
        # Resolved before anything is created: a resumed run must continue in its
        # own directory, or it leaves an empty one behind and writes its
        # TensorBoard events somewhere nothing will look.
        run_dir = (
            resume_dir
            if resumed is not None and resume_dir is not None
            else self._make_run_dir()
        )
        tb = TensorBoardLogger(run_dir / "tensorboard")

        return model, optimizer, criterion, scheduler, scaler, run_dir, tb

    def _restore_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        resumed: ResumeState | None,
        run_dir: Path,
    ) -> tuple[list[EpochResult], float, int, int, int]:
        """Restore optimization state and classification epoch history."""
        return training_loop.restore_training_state(
            resumed,
            model,
            optimizer,
            scheduler,
            run_dir,
            self._config.training.epochs,
            EpochResult,
            float("inf"),
            scaler,
        )

    def _step_scheduler(self, scheduler: Any, val_loss: float) -> None:
        """Advance the scheduler using validation loss for plateau mode."""
        training_loop.step_scheduler(scheduler, val_loss)

    def _report_epoch(
        self,
        result: EpochResult,
        tb: TensorBoardLogger,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        t0: float,
    ) -> None:
        """Report the task-specific epoch metrics and progress event."""
        training_loop.report_epoch(
            result,
            tb,
            progress_callback,
            t0,
            self._config.training.epochs,
            {
                "loss/train": "train_loss",
                "loss/val": "val_loss",
                "accuracy/train": "train_accuracy",
                "accuracy/val": "val_accuracy",
            },
            ("train_loss", "train_accuracy", "val_loss", "val_accuracy"),
            "Epoch {}/{} | train_loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f}",
            ("train_loss", "train_accuracy", "val_loss", "val_accuracy"),
        )

    def _checkpoint_components(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
    ) -> dict[str, Any]:
        """Snapshot model and optimization components for exact continuation."""
        return training_loop.checkpoint_components(
            model, optimizer, scheduler, scaler, unwrap_model=True
        )

    def _early_stopping_reached(self, patience_counter: int) -> bool:
        """Keep zero patience disabled at the stopping boundary."""
        return training_loop.early_stopping_reached(
            patience_counter, self._config.training.early_stopping_patience
        )

    def _finish_training(
        self,
        train_result: TrainResult,
        run_dir: Path,
        data_module: Any,
        tb: TensorBoardLogger,
        save_pool: ThreadPoolExecutor,
        save_future: Future[None] | None,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> TrainResult:
        """Summarize health, finish checkpoint writes and publish the run."""
        # A run that collapsed still reports an accuracy, so say so out loud
        # rather than letting the number stand on its own (ADR-099).
        health = summarize(
            [
                collapsed_predictions(
                    self._last_val_preds,
                    2
                    if self._config.task == "binary"
                    else self._config.model.num_classes,
                ),
                stagnant_loss([h.train_loss for h in train_result.history]),
            ]
        )
        train_result.warnings = health
        for warning in health:
            logger.warning("{}", warning["message"])

        # Ensure the last checkpoint is fully written before proceeding.
        if save_future is not None:
            save_future.result()
        save_pool.shutdown(wait=False)

        self._write_run_json(
            run_dir, train_result, getattr(data_module, "class_names", None)
        )
        tb.close()

        # Release VRAM held by activations / optimizer state so back-to-back
        # runs (grid search, model comparison) don't accumulate fragmentation.
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        if progress_callback is not None:
            # Deliberately the epochs that ran, not the configured total that
            # `epoch_end` carries: on a stopped run the two differ, and the
            # difference is how the overlay tells "stopped" from "finished"
            # (lib/run-control.ts, ADR-111).
            progress_callback(
                {"event": "end", "total_epochs": len(train_result.history)}
            )

        return train_result

    def _compute_epoch_result(
        self,
        epoch: int,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Any,
        scaler: torch.amp.GradScaler | None = None,
    ) -> EpochResult:
        """Compute the task-specific epoch metrics and advance its scheduler."""
        train_loss, train_acc = self._train_epoch(
            model, train_loader, optimizer, criterion, scaler=scaler
        )
        val_loss, val_acc = self._eval_epoch(model, val_loader, criterion)

        self._step_scheduler(scheduler, val_loss)

        return EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
        )

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Move the model to its device and apply DataParallel when configured."""
        return training_loop.prepare_model(model, self._device, self._dp_ids)

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        cfg = self._config.training
        builders: dict[str, Callable[..., torch.optim.Optimizer]] = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
        }
        return builders[cfg.optimizer](
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def _build_criterion(self) -> nn.Module:
        if self._config.task == "binary":
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Build the configured learning-rate scheduler."""
        return training_loop.build_scheduler(optimizer, self._config.training)

    def _make_run_dir(self) -> Path:
        """Create the configured timestamped output directory."""
        return training_loop.make_run_dir(
            self._config.output.models_dir, self._config.name
        )

    def _train_epoch(
        self,
        model: nn.Module,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        *,
        scaler: torch.amp.GradScaler | None = None,
    ) -> tuple[float, float]:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        use_amp = scaler is not None
        for inputs, labels in loader:
            inputs = inputs.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)
            if self._config.task == "binary":
                target = labels.float().unsqueeze(1)
            else:
                target = labels
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                scaler.scale(loss).backward()  # type: ignore[union-attr]
                scaler.step(optimizer)  # type: ignore[union-attr]
                scaler.update()  # type: ignore[union-attr]
            else:
                outputs = model(inputs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                if self._config.task == "binary":
                    preds = (outputs.sigmoid() > 0.5).squeeze(1).long()
                else:
                    preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        n = len(loader)
        loss = total_loss / n if n > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return loss, acc

    def _eval_epoch(
        self,
        model: nn.Module,
        loader: Any,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels: list[int] = []
        all_probs: list[float] = []
        all_preds: list[int] = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                if self._config.task == "binary":
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().unsqueeze(1))
                    probs = outputs.sigmoid().squeeze(1)
                    preds = (probs > 0.5).long()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    probs = outputs.softmax(dim=1).max(dim=1).values
                    preds = outputs.argmax(dim=1)
                total_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
        # Cache for downstream ROC plotting and the health check.
        self._last_val_labels = all_labels
        self._last_val_probs = all_probs
        self._last_val_preds = all_preds
        n = len(loader)
        acc = correct / total if total > 0 else 0.0
        return (total_loss / n if n > 0 else 0.0), acc

    def _write_run_json(
        self, run_dir: Path, result: TrainResult, class_names: list[str] | None = None
    ) -> None:
        """Write the run.json file with full run metadata."""
        run_json: dict[str, Any] = {
            **training_loop.run_json_header(
                self._config,
                run_dir,
                result.stopped,
                result.device_used,
                getattr(self, "_loader_workers", {}),
            ),
            # The index->name mapping the checkpoint was trained with. Recorded
            # because it used to be recovered by re-reading the training folder,
            # which silently stopped working the moment a dataset was renamed or
            # moved: Grad-CAM fell back to bare indices and dropped the ground
            # truth without saying why.
            "class_names": class_names or [],
            "config": self._config.model_dump(mode="json"),
            "metrics": {
                # A run stopped before its first epoch has no best loss; the
                # inf sentinel would be written as `Infinity`, which no JSON
                # reader accepts and which turned the result endpoint into a 500.
                "best_val_loss": result.best_val_loss if result.total_epochs else None,
                "best_epoch": result.best_epoch,
                "total_epochs": result.total_epochs,
            },
            "history": [
                {
                    "epoch": r.epoch,
                    "train_loss": r.train_loss,
                    "train_accuracy": r.train_accuracy,
                    "val_loss": r.val_loss,
                    "val_accuracy": r.val_accuracy,
                }
                for r in result.history
            ],
            **training_loop.run_json_tail(result),
        }
        (run_dir / "run.json").write_text(
            json.dumps(run_json, indent=2), encoding="utf-8"
        )


__all__ = ["Trainer", "TrainResult", "EpochResult", "resolve_device"]
