from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from visionforge.utils.config import ExperimentConfig
from visionforge.utils.cuda import check_cuda


@dataclass
class EpochResult:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


@dataclass
class TrainResult:
    """Summary of a completed training run."""

    best_epoch: int
    best_val_loss: float
    total_epochs: int
    history: list[EpochResult] = field(default_factory=list)
    model_path: Path = field(default_factory=lambda: Path("."))


def _seed_everything(seed: int) -> None:
    """Apply seed to random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    """Manages the full training loop for one classification experiment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._cuda_info = check_cuda()
        self._device = torch.device("cuda" if self._cuda_info.available else "cpu")

    def fit(self, model: nn.Module, data_module: Any) -> TrainResult:
        """Run the training loop.

        Args:
            model: nn.Module with the correct final layer.
            data_module: object exposing train_loader() and val_loader().

        Returns:
            TrainResult with best epoch, loss, history, and saved model path.
        """
        cfg = self._config.training
        _seed_everything(cfg.seed)

        model = self._prepare_model(model)
        optimizer = self._build_optimizer(model)
        criterion = self._build_criterion()
        run_dir = self._make_run_dir()
        model_path = run_dir / "best_model.pth"

        history: list[EpochResult] = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_epoch(
                model, data_module.train_loader(), optimizer, criterion
            )
            val_loss, val_acc = self._eval_epoch(
                model, data_module.val_loader(), criterion
            )

            result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
            history.append(result)

            logger.info(
                "Epoch {}/{} | train_loss={:.4f} val_loss={:.4f} val_acc={:.4f}",
                epoch,
                cfg.epochs,
                train_loss,
                val_loss,
                val_acc,
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
                torch.save(state_dict, model_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch {}.", epoch)
                    break

        train_result = TrainResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            total_epochs=len(history),
            history=history,
            model_path=model_path,
        )
        self._write_run_json(run_dir, train_result)

        return train_result

    # ── private helpers ────────────────────────────────────────────────────────

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Move model to device; wrap with DataParallel for multi-GPU."""
        model = model.to(self._device)
        if self._cuda_info.available and self._cuda_info.device_count > 1:
            model = nn.DataParallel(model)
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
        for inputs, labels in loader:
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)
            if self._config.task == "binary":
                labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
    ) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                if self._config.task == "binary":
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().unsqueeze(1))
                    preds = (outputs.sigmoid() > 0.5).squeeze(1).long()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                total_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        n = len(loader)
        acc = correct / total if total > 0 else 0.0
        return (total_loss / n if n > 0 else 0.0), acc

    def _write_run_json(self, run_dir: Path, result: TrainResult) -> None:
        """Write the run.json file with full run metadata."""
        run_json: dict[str, Any] = {
            "id": f"{self._config.name}_{run_dir.name}",
            "experiment": self._config.name,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
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
        }
        (run_dir / "run.json").write_text(
            json.dumps(run_json, indent=2), encoding="utf-8"
        )


__all__ = ["Trainer", "TrainResult", "EpochResult"]
