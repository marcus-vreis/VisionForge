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

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visionforge.core import trainer as trainer_common
from visionforge.core import training_loop
from visionforge.core.cancellation import CancellationToken
from visionforge.core.tracking import TensorBoardLogger
from visionforge.core.training_health import (
    collapsed_segmentation,
    stagnant_loss,
    summarize,
)
from visionforge.models.segmentation_factory import segmentation_logits
from visionforge.utils.messages import tr
from visionforge.utils.segmentation_config import SegmentationConfig

_seed_everything = trainer_common._seed_everything
resolve_device = trainer_common.resolve_device
save_best_checkpoint = trainer_common.save_best_checkpoint


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
    # Things about this run the researcher has to be told, because the
    # metrics alone would look like a result (ADR-099).
    warnings: list[dict[str, str]] = field(default_factory=list)
    # True only when the loop broke on the stop token (ADR-111): a run whose
    # stop arrived in its last epoch, or after it, finished and is not cut.
    stopped: bool = False


def per_image_confusion(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    """Return one KxK confusion matrix per image using valid target pixels."""
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
        self.per_class_iou: list[float] = []
        self.present_classes: int = 0
        self._k = num_classes
        self._ignore = ignore_index
        self._cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate confusion counts for valid target classes."""
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
            # Kept for the collapse check: a uniform mask shows up as one class
            # with a real IoU and the rest at zero (ADR-101).
            self.per_class_iou = [float(v) for v in iou.tolist()]
            self.present_classes = int(present.sum().item())
        else:
            mean_iou = mean_dice = 0.0
            self.per_class_iou = []
            self.present_classes = 0

        pixel_acc = float(diag.sum().item() / total)
        return mean_iou, mean_dice, pixel_acc


def check_mask_ids(masks: torch.Tensor, num_classes: int, ignore_index: int) -> None:
    """Reject out-of-range mask ids on CPU before they poison CUDA state."""
    valid = masks[masks != ignore_index]
    if valid.numel() == 0:
        return
    found = int(valid.max())
    if found >= num_classes or int(valid.min()) < 0:
        found = found if found >= num_classes else int(valid.min())
        raise ValueError(
            tr(
                "segmentation.mask_id_out_of_range",
                found=found,
                num_classes=num_classes,
                last=num_classes - 1,
                needed=max(found + 1, num_classes),
                ignore=ignore_index,
            )
        )


def dice_loss(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """Compute soft multi-class Dice loss while masking ignored targets."""
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
        """Compute the configured segmentation criterion."""
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
        """Compute the configured segmentation criterion."""
        return dice_loss(logits, target, self._ignore)


class SegmentationTrainer(training_loop.GradientTrainerMethods):
    """Manages the full training loop for one segmentation experiment."""

    _epoch_result_type = SegmentationEpochResult
    _initial_best = -1.0
    _epoch_fields = ("val_miou", "val_dice", "val_pixel_acc")

    def __init__(self, config: SegmentationConfig) -> None:
        self._config = config
        self._loader_workers: dict[str, int] = {}
        self._num_classes = config.model.num_classes
        self._ignore_index = config.data.ignore_index
        self._device, self._dp_ids, self._device_label = resolve_device(config.device)

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
        return training_loop.fit_training(
            self,
            model,
            data_module,
            optimizer,
            progress_callback,
            cancel_token,
            resume_dir,
            result_type=SegmentationTrainResult,
            seed=_seed_everything,
            tensorboard_factory=TensorBoardLogger,
            save_best=save_best_checkpoint,
            metric_name="val_miou",
            best_field="best_val_miou",
            log_format="SegmentationTrainer using device: {}",
            higher_is_better=True,
            ensure_checkpoint=self._ensure_checkpoint,
        )

    def evaluate(self, model: nn.Module, loader: Any) -> tuple[float, float, float]:
        """Compute (mean_iou, dice, pixel_acc) for ``model`` over ``loader``."""
        return self.evaluate_with_confusion(model, loader)[0]

    def evaluate_with_confusion(
        self, model: nn.Module, loader: Any
    ) -> tuple[tuple[float, float, float], np.ndarray]:
        """Return segmentation metrics and one KxK confusion matrix per image."""
        model = model.to(self._device)
        model.eval()
        metrics = SegmentationMetricAccumulator(self._num_classes, self._ignore_index)
        self._last_metrics = metrics
        per_image: list[np.ndarray] = []
        with torch.no_grad():
            for inputs, targets in loader:
                check_mask_ids(targets, self._num_classes, self._ignore_index)
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

    _report_epoch = staticmethod(
        partial(
            training_loop.report_epoch,
            scalar_fields={
                "loss/train": "train_loss",
                "loss/val": "val_loss",
                "miou/val": "val_miou",
                "dice/val": "val_dice",
                "pixel_acc/val": "val_pixel_acc",
            },
            event_fields=(
                "train_loss",
                "val_loss",
                "val_miou",
                "val_dice",
                "val_pixel_acc",
            ),
            log_format="Epoch {}/{} | train_loss={:.4f} val_loss={:.4f} mIoU={:.4f} dice={:.4f} pixel_acc={:.4f}",
            log_fields=(
                "train_loss",
                "val_loss",
                "val_miou",
                "val_dice",
                "val_pixel_acc",
            ),
        )
    )

    def _training_warnings(
        self, history: list[SegmentationEpochResult]
    ) -> list[dict[str, str]]:
        """Summarize and log training health warnings for the completed run."""
        return training_loop.training_warnings(
            history,
            summarize,
            stagnant_loss,
            additional=lambda: [
                collapsed_segmentation(
                    getattr(getattr(self, "_last_metrics", None), "per_class_iou", []),
                    present_classes=getattr(
                        getattr(self, "_last_metrics", None), "present_classes", 0
                    ),
                )
            ],
        )

    @staticmethod
    def _ensure_checkpoint(
        model: nn.Module,
        model_path: Path,
        history: list[SegmentationEpochResult],
        best_epoch: int,
        best_val_miou: float,
    ) -> tuple[int, float]:
        """Save final weights when completed epochs produced no best checkpoint."""
        # Guard: if mIoU never improved past the -1 sentinel (degenerate run),
        # still persist the final weights so the block can reload something.
        # Not for a run stopped before its first epoch: those weights are
        # untrained, and the block would score them as a result (ADR-111).
        if history and not model_path.is_file():
            torch.save(
                model.module.state_dict()  # type: ignore[union-attr]
                if isinstance(model, nn.DataParallel)
                else model.state_dict(),
                model_path,
            )
            best_epoch = best_epoch or len(history)
            best_val_miou = max(best_val_miou, 0.0)

        return best_epoch, best_val_miou

    @staticmethod
    def _split_named_params(
        model: nn.Module,
    ) -> training_loop.ParameterGroups:
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

    def _build_criterion(self) -> nn.Module:
        loss = self._config.training.loss
        if loss == "dice":
            return _DiceCriterion(self._ignore_index)
        if loss == "combined":
            return _CombinedLoss(self._ignore_index)
        return nn.CrossEntropyLoss(ignore_index=self._ignore_index)

    def _train_epoch(
        self,
        model: nn.Module,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train one epoch using the shared optimization loop."""
        return training_loop.train_epoch(
            outputs_transform=segmentation_logits,
            validate_targets=lambda targets: check_mask_ids(
                targets, self._num_classes, self._ignore_index
            ),
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self._device,
        )

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
                check_mask_ids(targets, self._num_classes, self._ignore_index)
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
        """Write the ADR-013 run.json with segmentation metrics and history."""
        training_loop.write_task_run_json(
            self._config,
            run_dir,
            result,
            getattr(self, "_loader_workers", {}),
            best_metric="best_val_miou",
            epoch_metrics={
                "val_miou": "val_miou",
                "val_dice": "val_dice",
                "val_pixel_acc": "val_pixel_acc",
            },
            history_fields=(
                "epoch",
                "train_loss",
                "val_loss",
                "val_miou",
                "val_dice",
                "val_pixel_acc",
            ),
        )


__all__ = [
    "SegmentationTrainer",
    "SegmentationTrainResult",
    "SegmentationEpochResult",
    "SegmentationMetricAccumulator",
    "dice_loss",
    "per_image_confusion",
]
