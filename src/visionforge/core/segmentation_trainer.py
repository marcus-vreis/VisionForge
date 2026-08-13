"""Training loop for the semantic-segmentation task.

Mirrors the regression/classification trainers (device resolution, per-epoch
loop, scheduler, ADR-013 ``run.json``, SSE ``start``/``epoch_end``/``end``
events) but with a pixel-wise criterion (cross-entropy / Dice / combined) and
segmentation metrics (mean IoU, Dice, pixel accuracy via a streaming confusion
matrix). The best checkpoint is selected by **val mean IoU** (higher is better),
the segmentation standard, not by val loss. Reuses ``resolve_device`` and
``_seed_everything`` from ``core.trainer``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_fingerprint import fingerprint_from_config
from visionforge.core.resume import (
    ResumeState,
    clear_resume_state,
    load_resume_state,
    save_resume_state,
)
from visionforge.core.tracking import TensorBoardLogger
from visionforge.core.trainer import _seed_everything, resolve_device
from visionforge.models.segmentation_factory import segmentation_logits
from visionforge.utils.environment import capture_environment
from visionforge.utils.segmentation_config import SegmentationConfig


@dataclass
class SegmentationEpochResult:
    """Metrics for a single segmentation training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_miou: float
    val_dice: float
    val_pixel_acc: float


@dataclass
class SegmentationTrainResult:
    """Summary of a completed segmentation training run."""

    best_epoch: int
    best_val_miou: float
    total_epochs: int
    device_used: str
    history: list[SegmentationEpochResult] = field(default_factory=list)
    model_path: Path = field(default_factory=lambda: Path("."))


def per_image_confusion(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    """One KxK confusion matrix per image in the batch.

    Applies the same ignore-index and in-range filtering as
    ``SegmentationMetricAccumulator.update``, so the matrices sum to exactly the
    accumulator's — that identity is what lets a bootstrap interval (ADR-076)
    bracket the metric a run actually reports.
    """
    preds = logits.argmax(dim=1).reshape(logits.shape[0], -1)
    truth = target.reshape(target.shape[0], -1)
    out = np.zeros((truth.shape[0], num_classes, num_classes), dtype=np.int64)
    for i in range(truth.shape[0]):
        t, p = truth[i].to(torch.long), preds[i].to(torch.long)
        keep = (
            (t != ignore_index)
            & (t >= 0)
            & (t < num_classes)
            & (p >= 0)
            & (p < num_classes)
        )
        t, p = t[keep], p[keep]
        counts = torch.bincount(t * num_classes + p, minlength=num_classes**2)
        out[i] = counts.reshape(num_classes, num_classes).cpu().numpy()
    return out


class SegmentationMetricAccumulator:
    """Streaming mean-IoU / Dice / pixel-accuracy via a K×K confusion matrix.

    Rows are ground-truth class ids, columns are predicted class ids; pixels
    equal to ``ignore_index`` are excluded. IoU and Dice are averaged over the
    classes that actually appear (present in ground truth or prediction), so
    absent classes do not drag the mean to zero.
    """

    def __init__(self, num_classes: int, ignore_index: int) -> None:
        self._k = num_classes
        self._ignore = ignore_index
        self._cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        preds = logits.argmax(dim=1)
        valid = target != self._ignore
        t = target[valid].to(torch.long)
        p = preds[valid].to(torch.long)
        # In-range guard: a stray id outside [0, K) would corrupt the matrix.
        in_range = (t >= 0) & (t < self._k) & (p >= 0) & (p < self._k)
        t, p = t[in_range], p[in_range]
        idx = t * self._k + p
        binc = torch.bincount(idx, minlength=self._k * self._k)
        self._cm += binc.reshape(self._k, self._k).to(self._cm.device)

    def compute(self) -> tuple[float, float, float]:
        """Return (mean_iou, mean_dice, pixel_accuracy)."""
        cm = self._cm.to(torch.float64)
        total = cm.sum().item()
        if total == 0:
            return 0.0, 0.0, 0.0

        diag = cm.diag()
        row = cm.sum(dim=1)  # ground-truth totals per class
        col = cm.sum(dim=0)  # prediction totals per class
        union = row + col - diag

        present = (row + col) > 0
        if present.any():
            iou = diag[present] / union[present].clamp_min(1e-9)
            dice = (2.0 * diag[present]) / (row + col)[present].clamp_min(1e-9)
            mean_iou = float(iou.mean().item())
            mean_dice = float(dice.mean().item())
        else:
            mean_iou = mean_dice = 0.0

        pixel_acc = float(diag.sum().item() / total)
        return mean_iou, mean_dice, pixel_acc


def dice_loss(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """Soft multi-class Dice loss (1 - mean Dice over classes).

    Computes Dice over softmax probabilities against a one-hot target, masking
    out ``ignore_index`` pixels. Differentiable and safe when a class is absent.
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)

    valid = (target != ignore_index).unsqueeze(1)  # [N,1,H,W]
    safe_target = torch.where(target == ignore_index, 0, target)
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).to(probs.dtype)

    probs = probs * valid
    one_hot = one_hot * valid

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    cardinality = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
    return 1.0 - dice.mean()


class _CombinedLoss(nn.Module):
    """Cross-entropy + Dice, the common segmentation hybrid."""

    def __init__(self, ignore_index: int) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._ignore = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self._ce(logits, target) + dice_loss(
            logits, target, self._ignore
        )
        return loss


class _DiceCriterion(nn.Module):
    """Module wrapper around ``dice_loss`` so it composes like ``nn`` criteria."""

    def __init__(self, ignore_index: int) -> None:
        super().__init__()
        self._ignore = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(logits, target, self._ignore)


class SegmentationTrainer:
    """Manages the full training loop for one segmentation experiment."""

    def __init__(self, config: SegmentationConfig) -> None:
        self._config = config
        self._num_classes = config.model.num_classes
        self._ignore_index = config.data.ignore_index
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
        resume_dir: Path | None = None,
    ) -> SegmentationTrainResult:
        """Run the segmentation training loop (best checkpoint by val mIoU)."""
        cfg = self._config.training
        _seed_everything(cfg.seed, deterministic=cfg.deterministic)
        logger.info("SegmentationTrainer using device: {}", self._device_label)

        model = self._prepare_model(model)
        self._apply_transfer_learning(model)
        optimizer = optimizer if optimizer is not None else self._build_optimizer(model)
        criterion = self._build_criterion()
        scheduler = self._build_scheduler(optimizer)
        # Resolved before anything is created: a resumed run must continue in its
        # own directory, or it leaves an empty one behind and writes its
        # TensorBoard events somewhere nothing will look.
        resumed = load_resume_state(resume_dir) if resume_dir is not None else None
        run_dir = (
            resume_dir
            if resumed is not None and resume_dir is not None
            else self._make_run_dir()
        )
        model_path = run_dir / "best_model.pth"
        tb = TensorBoardLogger(run_dir / "tensorboard")

        history: list[SegmentationEpochResult] = []
        best_val_miou = -1.0
        best_epoch = 0
        patience_counter = 0
        start_epoch = 1

        # Weights alone cannot continue: the optimizer and scheduler come back
        # too, or the search restarts where the loss curve never was.
        if resumed is not None:
            model.load_state_dict(resumed.model)
            optimizer.load_state_dict(resumed.optimizer)
            if scheduler is not None and resumed.scheduler is not None:
                scheduler.load_state_dict(resumed.scheduler)
            best_val_miou = resumed.best_metric
            best_epoch = resumed.best_epoch
            patience_counter = resumed.patience_counter
            history = [SegmentationEpochResult(**h) for h in resumed.history]
            start_epoch = resumed.epoch + 1
            logger.info(
                "Resuming {} at epoch {} of {}.",
                run_dir.name,
                start_epoch,
                cfg.epochs,
            )
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

        for epoch in range(start_epoch, cfg.epochs + 1):
            # The safe point: the previous epoch's checkpoint is written and its
            # metrics emitted, so stopping here leaves the run directory whole.
            if is_cancelled(cancel_token):
                logger.info(
                    "Run cancelled at epoch {}; keeping the best checkpoint so far.",
                    epoch,
                )
                break

            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            val_loss, miou, dice, pixel_acc = self._eval_epoch(
                model, val_loader, criterion
            )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            history.append(
                SegmentationEpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_miou=miou,
                    val_dice=dice,
                    val_pixel_acc=pixel_acc,
                )
            )
            tb.log_scalars(
                epoch,
                {
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "miou/val": miou,
                    "dice/val": dice,
                    "pixel_acc/val": pixel_acc,
                },
            )

            logger.info(
                "Epoch {}/{} | train_loss={:.4f} val_loss={:.4f} "
                "mIoU={:.4f} dice={:.4f} pixel_acc={:.4f}",
                epoch,
                cfg.epochs,
                train_loss,
                val_loss,
                miou,
                dice,
                pixel_acc,
            )

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "total_epochs": cfg.epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_miou": miou,
                        "val_dice": dice,
                        "val_pixel_acc": pixel_acc,
                        "elapsed_s": round(time.monotonic() - t0, 3),
                    }
                )

            # Every epoch, not only on improvement: a run dies on its worst
            # epochs too, and stale state resumes into a past already left.
            save_resume_state(
                run_dir,
                ResumeState(
                    epoch=epoch,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict() if scheduler is not None else None,
                    scaler=None,
                    best_metric=best_val_miou,
                    best_epoch=best_epoch,
                    patience_counter=patience_counter,
                    history=[asdict(h) for h in history],
                ),
            )

            if miou > best_val_miou:
                best_val_miou = miou
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

        if not is_cancelled(cancel_token):
            clear_resume_state(run_dir)

        if save_future is not None:
            save_future.result()
        save_pool.shutdown(wait=False)

        # Guard: if mIoU never improved past the -1 sentinel (degenerate run),
        # still persist the final weights so the block can reload something.
        if not model_path.is_file():
            torch.save(
                model.module.state_dict()  # type: ignore[union-attr]
                if isinstance(model, nn.DataParallel)
                else model.state_dict(),
                model_path,
            )
            best_epoch = best_epoch or len(history)
            best_val_miou = max(best_val_miou, 0.0)

        train_result = SegmentationTrainResult(
            best_epoch=best_epoch,
            best_val_miou=best_val_miou,
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

    def evaluate(self, model: nn.Module, loader: Any) -> tuple[float, float, float]:
        """Compute (mean_iou, dice, pixel_acc) for ``model`` over ``loader``."""
        return self.evaluate_with_confusion(model, loader)[0]

    def evaluate_with_confusion(
        self, model: nn.Module, loader: Any
    ) -> tuple[tuple[float, float, float], np.ndarray]:
        """As ``evaluate``, plus one KxK confusion matrix **per image**.

        Per-image rather than per-pixel because the image is the sampling unit
        the bootstrap intervals resample (ADR-076) — pixels inside one image are
        not independent draws. Summing the returned matrices reproduces the
        accumulator's single matrix exactly, so the aggregates and the intervals
        cannot describe different numbers.
        """
        model = model.to(self._device)
        model.eval()
        metrics = SegmentationMetricAccumulator(self._num_classes, self._ignore_index)
        per_image: list[np.ndarray] = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                logits = segmentation_logits(model(inputs))
                metrics.update(logits, targets)
                per_image.append(
                    per_image_confusion(
                        logits, targets, self._num_classes, self._ignore_index
                    )
                )
        stacked = (
            np.concatenate(per_image)
            if per_image
            else np.zeros((0, self._num_classes, self._num_classes))
        )
        return metrics.compute(), stacked

    # ── private helpers ────────────────────────────────────────────────────────

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self._device)
        if self._dp_ids:
            model = nn.DataParallel(model, device_ids=self._dp_ids)
        return model

    def _apply_transfer_learning(self, model: nn.Module) -> None:
        """Freeze the backbone for feature extraction (no-op otherwise).

        ``feature_extraction`` freezes every child except the dense head (the last
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

        Unwraps DataParallel so the head is the model's own final layer. The dense
        head is the last named child across all families: ``classifier`` for the
        torchvision DeepLab/FCN/LR-ASPP models and ``outc`` for U-Net.
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
        loss = self._config.training.loss
        if loss == "dice":
            return _DiceCriterion(self._ignore_index)
        if loss == "combined":
            return _CombinedLoss(self._ignore_index)
        return nn.CrossEntropyLoss(ignore_index=self._ignore_index)

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
            logits = segmentation_logits(model(inputs))
            loss = criterion(logits, targets)
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
    ) -> tuple[float, float, float, float]:
        """Return (val_loss, mean_iou, dice, pixel_acc)."""
        model.eval()
        total_loss = 0.0
        metrics = SegmentationMetricAccumulator(self._num_classes, self._ignore_index)
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                logits = segmentation_logits(model(inputs))
                total_loss += criterion(logits, targets).item()
                metrics.update(logits, targets)
        n = len(loader)
        val_loss = total_loss / n if n > 0 else 0.0
        miou, dice, pixel_acc = metrics.compute()
        return val_loss, miou, dice, pixel_acc

    def _write_run_json(self, run_dir: Path, result: SegmentationTrainResult) -> None:
        """Write the ADR-013 run.json with segmentation metrics + history."""
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
                "best_val_miou": result.best_val_miou,
                "best_epoch": result.best_epoch,
                "total_epochs": result.total_epochs,
                "miou": best.val_miou if best else 0.0,
                "dice": best.val_dice if best else 0.0,
                "pixel_acc": best.val_pixel_acc if best else 0.0,
            },
            "history": [
                {
                    "epoch": r.epoch,
                    "train_loss": r.train_loss,
                    "val_loss": r.val_loss,
                    "val_miou": r.val_miou,
                    "val_dice": r.val_dice,
                    "val_pixel_acc": r.val_pixel_acc,
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
    "SegmentationTrainer",
    "SegmentationTrainResult",
    "SegmentationEpochResult",
    "SegmentationMetricAccumulator",
    "dice_loss",
    "per_image_confusion",
]
