from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_fingerprint import fingerprint_from_config
from visionforge.core.tracking import TensorBoardLogger
from visionforge.utils.config import DeviceConfig, ExperimentConfig
from visionforge.utils.cuda import check_cuda
from visionforge.utils.environment import capture_environment


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


def _seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed every RNG and configure cuDNN performance mode.

    Args:
        seed: integer seed applied to stdlib, numpy, and PyTorch RNGs.
        deterministic: when True, forces cuDNN to use deterministic
            algorithms and disables benchmark auto-tuning.  This
            guarantees bitwise reproducibility but **significantly**
            reduces GPU throughput (often 3-5× slower on CNNs).
            When False (default), ``cudnn.benchmark`` is enabled so
            cuDNN auto-selects the fastest convolution algorithm for
            each input shape — the single largest factor in GPU
            utilization for CNN workloads.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(device_cfg: DeviceConfig) -> tuple[torch.device, list[int], str]:
    """Resolve a DeviceConfig into a torch.device + (optional) DataParallel ids.

    Returns:
        Tuple of (torch.device, gpu_ids_list, human_label).
        gpu_ids_list is empty when kind=='cpu' or single-GPU. The human label is
        a short string used in logs and run.json so the user can verify after the
        fact which device was *actually* used.
    """
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
            info.devices[primary].name
            if primary < len(info.devices)
            else f"cuda:{primary}"
        )
        return torch.device(f"cuda:{primary}"), [], f"cuda:{primary} ({name})"

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


class Trainer:
    """Manages the full training loop for one classification experiment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._device, self._dp_ids, self._device_label = resolve_device(config.device)
        # Cache last-epoch val probs/labels so callers (block, evaluator) can plot ROC.
        self._last_val_labels: list[int] = []
        self._last_val_probs: list[float] = []

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
    ) -> TrainResult:
        """Run the training loop.

        Args:
            model: nn.Module with the correct final layer.
            data_module: object exposing train_loader() and val_loader().
            optimizer: pre-built optimizer; when None, one is built from config.
            progress_callback: called after each epoch and at start/end with a
                progress dict. Safe to call from a worker thread.

        Returns:
            TrainResult with best epoch, loss, history, and saved model path.
        """
        cfg = self._config.training
        _seed_everything(cfg.seed, deterministic=cfg.deterministic)

        logger.info("Trainer using device: {}", self._device_label)

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
        run_dir = self._make_run_dir()
        model_path = run_dir / "best_model.pth"
        tb = TensorBoardLogger(run_dir / "tensorboard")

        history: list[EpochResult] = []
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

        # Build loaders ONCE — reusing them across epochs avoids re-creating
        # DataLoader objects (and re-spawning persistent workers) every epoch.
        train_loader = data_module.train_loader()
        val_loader = data_module.val_loader()

        # Background thread pool for non-blocking checkpoint writes.
        save_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")
        save_future = None

        for epoch in range(1, cfg.epochs + 1):
            # The safe point: the previous epoch's checkpoint is written and its
            # metrics emitted, so stopping here leaves the run directory whole.
            if is_cancelled(cancel_token):
                logger.info(
                    "Run cancelled at epoch {}; keeping the best checkpoint so far.",
                    epoch,
                )
                break

            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion, scaler=scaler
            )
            val_loss, val_acc = self._eval_epoch(model, val_loader, criterion)

            # Plateau scheduler is reactive — step it after we have val loss.
            # The other schedulers step monotonically by epoch.
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
            history.append(result)
            tb.log_scalars(
                epoch,
                {
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "accuracy/train": train_acc,
                    "accuracy/val": val_acc,
                },
            )

            logger.info(
                "Epoch {}/{} | train_loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f}",
                epoch,
                cfg.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "total_epochs": cfg.epochs,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "elapsed_s": round(time.monotonic() - t0, 3),
                    }
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save module.state_dict() to avoid 'module.' key prefix from DataParallel.
                state_dict = (
                    model.module.state_dict()  # type: ignore[union-attr]
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                )
                # Wait for any previous checkpoint write to finish before
                # overwriting the dict, then launch a new async write.
                if save_future is not None:
                    save_future.result()
                save_future = save_pool.submit(torch.save, state_dict, model_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch {}.", epoch)
                    break

        # Ensure the last checkpoint is fully written before proceeding.
        if save_future is not None:
            save_future.result()
        save_pool.shutdown(wait=False)

        train_result = TrainResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            total_epochs=len(history),
            device_used=self._device_label,
            history=history,
            model_path=model_path,
        )
        self._write_run_json(
            run_dir, train_result, getattr(data_module, "class_names", None)
        )
        tb.close()

        # Release VRAM held by activations / optimizer state so back-to-back
        # runs (grid search, model comparison) don't accumulate fragmentation.
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        if progress_callback is not None:
            progress_callback({"event": "end", "total_epochs": len(history)})

        return train_result

    # ── private helpers ────────────────────────────────────────────────────────

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Move model to device; wrap with DataParallel when multi_cuda is requested."""
        model = model.to(self._device)
        if self._dp_ids:
            model = nn.DataParallel(model, device_ids=self._dp_ids)
        return model

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
        """Build a torch scheduler from the SchedulerConfig, or None for 'none'."""
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
        # plateau — reactive on validation loss; stepped manually with val_loss.
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
        # Cache for downstream ROC plotting.
        self._last_val_labels = all_labels
        self._last_val_probs = all_probs
        n = len(loader)
        acc = correct / total if total > 0 else 0.0
        return (total_loss / n if n > 0 else 0.0), acc

    def _write_run_json(
        self, run_dir: Path, result: TrainResult, class_names: list[str] | None = None
    ) -> None:
        """Write the run.json file with full run metadata."""
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
            # The index->name mapping the checkpoint was trained with. Recorded
            # because it used to be recovered by re-reading the training folder,
            # which silently stopped working the moment a dataset was renamed or
            # moved: Grad-CAM fell back to bare indices and dropped the ground
            # truth without saying why.
            "class_names": class_names or [],
            "config": self._config.model_dump(mode="json"),
            "metrics": {
                "best_val_loss": result.best_val_loss,
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


__all__ = ["Trainer", "TrainResult", "EpochResult", "resolve_device"]
