"""Drive detection training for a `DetectionConfig` (Ultralytics or torchvision).

The Ultralytics path wraps ``YOLO.train``; the torchvision path runs a
hand-written loss-dict loop (``_fit_torchvision``, ADR-035) over any model the
factory supports (Faster R-CNN, SSD / SSDLite, RetinaNet). Both hook the same
SSE event shape the classification Trainer uses (ADR-032) and write a
``run.json`` compatible with the ADR-013 contract so detection runs appear in
``/api/runs``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader

from visionforge.core.detection_data import DetectionDataModule, resolve_yolo_split
from visionforge.core.detection_dataset import DetectionDataset, detection_collate
from visionforge.core.detection_metrics import mean_average_precision_50
from visionforge.core.plotter import MetricsPlotter
from visionforge.core.tracking import TensorBoardLogger
from visionforge.models.detection_factory import build_torchvision_detector
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.environment import capture_environment

try:  # ultralytics is an optional extra ([detection]); bound lazily.
    from ultralytics import YOLO as _YOLO  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _YOLO = None

# Module global so tests can patch it without ultralytics installed.
YOLO: Any = _YOLO

# Ultralytics exposes its per-epoch metrics on ``trainer.metrics`` with these
# namespaced keys. Validation losses come as ``val/<component>_loss`` and the
# training losses as ``train/<component>_loss``.
_MAP50_KEY = "metrics/mAP50(B)"
_MAP5095_KEY = "metrics/mAP50-95(B)"
_PRECISION_KEY = "metrics/precision(B)"
_RECALL_KEY = "metrics/recall(B)"
_BOXLOSS_KEY = "val/box_loss"
_VAL_LOSS_KEYS = {
    "val_box_loss": "val/box_loss",
    "val_cls_loss": "val/cls_loss",
    "val_dfl_loss": "val/dfl_loss",
}
_TRAIN_LOSS_KEYS = {
    "train_box_loss": "train/box_loss",
    "train_cls_loss": "train/cls_loss",
    "train_dfl_loss": "train/dfl_loss",
}


@dataclass
class DetectionEpochResult:
    """One epoch's detection metrics.

    ``box_loss`` is the validation box loss kept for best-epoch fallback and
    backward compatibility; the namespaced loss components and precision/recall
    are populated by the Ultralytics backend (``None`` for torchvision).
    """

    epoch: int
    map50: float | None
    map50_95: float | None
    box_loss: float | None
    precision: float | None = None
    recall: float | None = None
    train_box_loss: float | None = None
    train_cls_loss: float | None = None
    train_dfl_loss: float | None = None
    val_box_loss: float | None = None
    val_cls_loss: float | None = None
    val_dfl_loss: float | None = None


@dataclass
class DetectionTrainResult:
    """Outcome of a detection training run."""

    best_epoch: int
    best_map50_95: float | None
    total_epochs: int
    device_used: str
    history: list[DetectionEpochResult]
    model_path: Path
    run_dir: Path


def _extract(metrics: dict[str, Any], key: str) -> float | None:
    """Read a float metric defensively; Ultralytics omits keys early in training."""
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: float | None, default: float) -> float:
    """Return the first non-None value, else ``default``."""
    for value in values:
        if value is not None:
            return value
    return default


def _epoch_event(
    result: DetectionEpochResult,
    total_epochs: int,
    *,
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> dict[str, Any]:
    """Build the SSE ``epoch_end`` payload for one detection epoch.

    Carries the detection-native metrics plus the generic
    train_loss/val_loss/val_accuracy the shared overlay reads, so the live
    monitor degrades cleanly when a field is absent (torchvision, early epochs).
    """
    return {
        "event": "epoch_end",
        "epoch": result.epoch,
        "total_epochs": total_epochs,
        "map50": result.map50,
        "map50_95": result.map50_95,
        "precision": result.precision,
        "recall": result.recall,
        "box_loss": result.box_loss,
        "train_box_loss": result.train_box_loss,
        "train_cls_loss": result.train_cls_loss,
        "train_dfl_loss": result.train_dfl_loss,
        "val_box_loss": result.val_box_loss,
        "val_cls_loss": result.val_cls_loss,
        "val_dfl_loss": result.val_dfl_loss,
        "train_loss": _first_present(
            train_loss, result.train_box_loss, result.box_loss, default=0.0
        ),
        "val_loss": _first_present(
            val_loss, result.val_box_loss, result.box_loss, default=0.0
        ),
        "val_accuracy": _first_present(result.map50_95, result.map50, default=0.0),
    }


def _is_worker_spawn_crash(message: str) -> bool:
    """True when an exception message matches a DataLoader worker dying at
    startup — on Windows the near-certain cause is the page file being too
    small for N spawned workers to each reload torch's CUDA DLLs
    (WinError 1455 loading shm.dll)."""
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "1455",
            "paging file",
            "paginação",
            "shm.dll",
            "dataloader worker",
            "ran out of input",
        )
    )


class DetectionTrainer:
    """Wraps Ultralytics training for a `DetectionConfig`."""

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._device_label = self._resolve_device_label()

    def fit(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> DetectionTrainResult:
        """Run training for the configured backend and return the result.

        Raises:
            RuntimeError: with an actionable message when DataLoader workers
                crash at startup on Windows (page-file exhaustion).
        """
        try:
            if self._config.model.backend == "torchvision":
                return self._fit_torchvision(progress_callback)
            return self._fit_ultralytics(progress_callback)
        except (OSError, RuntimeError, EOFError) as exc:
            if sys.platform == "win32" and _is_worker_spawn_crash(str(exc)):
                raise RuntimeError(
                    f"Os workers do DataLoader morreram ao iniciar — no Windows "
                    f"isso quase sempre é o arquivo de paginação pequeno demais "
                    f"para {self._config.training.workers} workers recarregarem "
                    f"as DLLs CUDA do torch (WinError 1455). Reduza "
                    f"training.workers para 0–2, ou aumente a memória virtual "
                    f"do Windows (Sistema → Configurações avançadas → "
                    f"Desempenho → Memória virtual)."
                ) from exc
            raise

    def _fit_ultralytics(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> DetectionTrainResult:
        """Drive Ultralytics YOLO/RT-DETR training.

        Raises:
            RuntimeError: if the ultralytics extra is not installed.
        """
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install the detection extra: "
                "pip install 'visionforge[detection]'."
            )

        cfg = self._config.training
        run_dir = self._make_run_dir()
        data_yaml = DetectionDataModule(self._config).resolve_data_yaml(out_dir=run_dir)

        model = YOLO(self._resolve_weights())
        history: list[DetectionEpochResult] = []

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "total_epochs": cfg.epochs,
                    "device": self._device_label,
                }
            )

        def _on_epoch_end(trainer: Any) -> None:
            epoch = int(getattr(trainer, "epoch", len(history))) + 1
            metrics: dict[str, Any] = getattr(trainer, "metrics", {}) or {}
            map50 = _extract(metrics, _MAP50_KEY)
            map95 = _extract(metrics, _MAP5095_KEY)
            box_loss = _extract(metrics, _BOXLOSS_KEY)
            train_losses = {
                k: _extract(metrics, v) for k, v in _TRAIN_LOSS_KEYS.items()
            }
            val_losses = {k: _extract(metrics, v) for k, v in _VAL_LOSS_KEYS.items()}
            result = DetectionEpochResult(
                epoch=epoch,
                map50=map50,
                map50_95=map95,
                box_loss=box_loss,
                precision=_extract(metrics, _PRECISION_KEY),
                recall=_extract(metrics, _RECALL_KEY),
                **train_losses,
                **val_losses,
            )
            history.append(result)
            if progress_callback is not None:
                progress_callback(_epoch_event(result, cfg.epochs))

        model.add_callback("on_fit_epoch_end", _on_epoch_end)
        logger.info(
            "Detection training: {} ({} epochs) on {}",
            self._config.model.name,
            cfg.epochs,
            self._device_label,
        )
        model.train(
            data=str(data_yaml),
            imgsz=self._config.data.image_size,
            device=self._device_arg(),
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            verbose=False,
            **self._ultralytics_train_kwargs(),
        )

        result = self._build_result(run_dir, history)
        self._write_run_json(run_dir, result)
        if progress_callback is not None:
            progress_callback({"event": "end", "total_epochs": len(history)})
        return result

    # ── torchvision backend ─────────────────────────────────────────────────────

    def _fit_torchvision(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> DetectionTrainResult:
        """Train a torchvision detector with a loss-dict loop.

        Each epoch trains on the loss dict and selects the best checkpoint by
        validation mAP@50 (`_eval_torchvision_map`); validation loss is also
        logged (frozen-BN detectors are safe to forward in train mode under
        no_grad). See ADR-035.

        Raises:
            ValueError: if data.base_dir is not set (YOLO layout is required).
        """
        cfg = self._config.training
        if self._config.data.base_dir is None:
            raise ValueError(
                "torchvision backend requires data.base_dir (YOLO layout); "
                "data_yaml-only is not supported yet."
            )

        run_dir = self._make_run_dir()
        device = self._resolve_torch_device()

        # Build the model before the data so an unsupported model (SSD/RetinaNet)
        # fails fast with the factory's NotImplementedError.
        model = build_torchvision_detector(
            self._config.model.name,
            self._config.model.num_classes,
            self._config.model.pretrained,
        )
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_torchvision_optimizer(params)
        model_path = run_dir / "weights" / "best.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader = self._build_loaders(self._config.data.base_dir)

        history: list[DetectionEpochResult] = []
        train_losses: list[float | None] = []
        best_map = -1.0
        best_epoch = 0

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "total_epochs": cfg.epochs,
                    "device": self._device_label,
                }
            )

        logger.info(
            "Detection (torchvision) training: {} ({} epochs) on {}",
            self._config.model.name,
            cfg.epochs,
            self._device_label,
        )
        tb = TensorBoardLogger(run_dir / "tensorboard")
        try:
            for epoch in range(1, cfg.epochs + 1):
                train_loss = self._run_torchvision_epoch(
                    model, train_loader, device, optimizer
                )
                val_loss = self._eval_torchvision_loss(model, val_loader, device)
                map50 = self._eval_torchvision_map(model, val_loader, device)

                epoch_result = DetectionEpochResult(
                    epoch=epoch,
                    map50=map50,
                    map50_95=None,
                    box_loss=val_loss,
                    val_box_loss=val_loss,
                )
                history.append(epoch_result)
                train_losses.append(train_loss)
                tb.log_scalars(
                    epoch,
                    {
                        "loss/train": train_loss,
                        "loss/val": val_loss,
                        "map50/val": map50,
                    },
                )
                # Select by validation mAP@50 (higher is better).
                if map50 > best_map:
                    best_map = map50
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_path)

                if progress_callback is not None:
                    progress_callback(
                        _epoch_event(
                            epoch_result,
                            cfg.epochs,
                            train_loss=train_loss,
                            val_loss=val_loss,
                        )
                    )
        finally:
            tb.close()

        if not model_path.exists():  # epochs=0 guard
            torch.save(model.state_dict(), model_path)

        # torchvision has no Ultralytics results.png, so synthesize the same
        # loss + mAP history chart the GUI shows for YOLO runs.
        self._render_torchvision_plot(run_dir, history, train_losses)

        result = DetectionTrainResult(
            best_epoch=best_epoch,
            best_map50_95=None,
            total_epochs=len(history),
            device_used=self._device_label,
            history=history,
            model_path=model_path,
            run_dir=run_dir,
        )
        self._write_run_json(run_dir, result)
        if progress_callback is not None:
            progress_callback({"event": "end", "total_epochs": len(history)})
        return result

    def _run_torchvision_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        total = 0.0
        n = 0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1
        return total / n if n else 0.0

    def _eval_torchvision_loss(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        # torchvision detectors only return losses in train mode with targets;
        # their default BN is frozen, so a no_grad train-mode pass is safe.
        model.train()
        total = 0.0
        n = 0
        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                total += float(sum(loss_dict.values()).item())
                n += 1
        return total / n if n else 0.0

    def _eval_torchvision_map(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        """Validation mAP@0.5 — model.eval() predictions vs ground truth."""
        model.eval()
        preds: list[dict[str, torch.Tensor]] = []
        gts: list[dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for images, targets in loader:
                outputs = model([img.to(device) for img in images])
                for out in outputs:
                    preds.append({k: v.detach().cpu() for k, v in out.items()})
                for t in targets:
                    gts.append({"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()})
        return mean_average_precision_50(preds, gts).map50

    def _build_loaders(self, base_dir: Path) -> tuple[DataLoader, DataLoader]:
        train_ds = DetectionDataset(*self._resolve_split_dirs(base_dir, "train"))
        val_ds = DetectionDataset(*self._resolve_split_dirs(base_dir, "val"))
        workers = self._config.training.workers
        train_loader = DataLoader(
            train_ds,
            batch_size=self._config.training.batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=detection_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self._config.training.batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=detection_collate,
        )
        return train_loader, val_loader

    def _resolve_split_dirs(self, base: Path, split: str) -> tuple[Path, Path]:
        """Return (images_dir, labels_dir) for a split, supporting both YOLO layouts.

        Raises:
            ValueError: if the split's image folder is not found.
        """
        resolved = resolve_yolo_split(base, split)
        if resolved is None:
            raise ValueError(
                f"Could not find the '{split}' images under '{base}' "
                f"(expected 'images/{split}' or '{split}/images')."
            )
        return resolved

    def _resolve_torch_device(self) -> torch.device:
        d = self._config.device
        if d.kind == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        idx = d.gpu_ids[0] if d.gpu_ids else 0
        return torch.device(f"cuda:{idx}")

    # ── private ───────────────────────────────────────────────────────────────

    def _build_result(
        self, run_dir: Path, history: list[DetectionEpochResult]
    ) -> DetectionTrainResult:
        scored = [h for h in history if h.map50_95 is not None]
        best = max(scored, key=lambda h: h.map50_95 or 0.0, default=None)
        return DetectionTrainResult(
            best_epoch=best.epoch if best else 0,
            best_map50_95=best.map50_95 if best else None,
            total_epochs=len(history),
            device_used=self._device_label,
            history=history,
            model_path=run_dir / "weights" / "best.pt",
            run_dir=run_dir,
        )

    def _ultralytics_train_kwargs(self) -> dict[str, Any]:
        """Translate the training config into Ultralytics ``train`` arguments.

        Every key here is a documented Ultralytics ``train`` hyperparameter; the
        config defaults mirror Ultralytics' own, so omitting a field trains
        identically to a bare ``YOLO.train`` call.
        """
        cfg = self._config.training
        aug = cfg.augmentation
        return {
            "epochs": cfg.epochs,
            "batch": cfg.batch_size,
            "lr0": cfg.learning_rate,
            "patience": cfg.patience,
            "seed": cfg.seed,
            "workers": cfg.workers,
            # optimizer
            "optimizer": cfg.optimizer,
            "momentum": cfg.momentum,
            "weight_decay": cfg.weight_decay,
            # schedule
            "lrf": cfg.lrf,
            "cos_lr": cfg.cos_lr,
            "warmup_epochs": cfg.warmup_epochs,
            "warmup_momentum": cfg.warmup_momentum,
            "warmup_bias_lr": cfg.warmup_bias_lr,
            # loss gains
            "box": cfg.box,
            "cls": cfg.cls,
            "dfl": cfg.dfl,
            # regularization / mechanics
            "label_smoothing": cfg.label_smoothing,
            "dropout": cfg.dropout,
            "nbs": cfg.nbs,
            "freeze": cfg.freeze,
            "amp": cfg.amp,
            "close_mosaic": cfg.close_mosaic,
            "single_cls": cfg.single_cls,
            "rect": cfg.rect,
            "multi_scale": cfg.multi_scale,
            # augmentation
            "hsv_h": aug.hsv_h,
            "hsv_s": aug.hsv_s,
            "hsv_v": aug.hsv_v,
            "degrees": aug.degrees,
            "translate": aug.translate,
            "scale": aug.scale,
            "shear": aug.shear,
            "perspective": aug.perspective,
            "flipud": aug.flipud,
            "fliplr": aug.fliplr,
            "bgr": aug.bgr,
            "mosaic": aug.mosaic,
            "mixup": aug.mixup,
            "copy_paste": aug.copy_paste,
            "auto_augment": aug.auto_augment,
            "erasing": aug.erasing,
        }

    def _build_torchvision_optimizer(
        self, params: list[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        """Build the torchvision optimizer from the shared optimizer config.

        ``auto`` maps to SGD (the torchvision detection reference recipe). Adam-
        family choices use ``momentum`` as beta1; SGD uses it as momentum.
        """
        cfg = self._config.training
        name = cfg.optimizer
        if name in ("Adam", "AdamW", "Adamax", "NAdam", "RAdam"):
            builder = getattr(torch.optim, name)
            adam: torch.optim.Optimizer = builder(
                params,
                lr=cfg.learning_rate,
                betas=(cfg.momentum, 0.999),
                weight_decay=cfg.weight_decay,
            )
            return adam
        if name == "RMSProp":
            return torch.optim.RMSprop(
                params,
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    def _resolve_weights(self) -> str:
        """Checkpoint path, else a pretrained ``.pt`` or a scratch ``.yaml``."""
        m = self._config.model
        if m.weights_path is not None:
            return str(m.weights_path)
        return f"{m.name}{'.pt' if m.pretrained else '.yaml'}"

    def _device_arg(self) -> Any:
        d = self._config.device
        if d.kind == "cpu":
            return "cpu"
        if d.kind == "multi_cuda":
            return d.gpu_ids if d.gpu_ids else 0
        return d.gpu_ids[0] if d.gpu_ids else 0

    def _resolve_device_label(self) -> str:
        d = self._config.device
        if d.kind == "cpu":
            return "cpu"
        if d.kind == "multi_cuda":
            return f"multi_cuda:{d.gpu_ids or 'all'}"
        return f"cuda:{d.gpu_ids[0] if d.gpu_ids else 0}"

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self._config.output.models_dir / self._config.name / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _render_torchvision_plot(
        self,
        run_dir: Path,
        history: list[DetectionEpochResult],
        train_losses: list[float | None],
    ) -> None:
        """Write a results.png (loss + mAP@50 over epochs) for a torchvision run."""
        if not history:
            return
        MetricsPlotter.detection_results(
            epochs=[h.epoch for h in history],
            train_losses=train_losses,
            val_losses=[h.box_loss for h in history],
            map50s=[h.map50 for h in history],
            save_path=run_dir / "results.png",
        )

    def _write_run_json(self, run_dir: Path, result: DetectionTrainResult) -> None:
        graphics = [
            str(run_dir / p)
            for p in (
                "results.png",
                "confusion_matrix.png",
                "BoxPR_curve.png",
                "BoxF1_curve.png",
            )
            if (run_dir / p).exists()
        ]
        best = next((h for h in result.history if h.epoch == result.best_epoch), None)
        run_json: dict[str, Any] = {
            "id": f"{self._config.name}_{run_dir.name}",
            "experiment": self._config.name,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "device_used": result.device_used,
            "environment": capture_environment(),
            "run_dir": str(run_dir.resolve()),
            "config": self._config.model_dump(mode="json"),
            "metrics": {
                "map50_95": result.best_map50_95,
                "map50": best.map50 if best else None,
                "precision": best.precision if best else None,
                "recall": best.recall if best else None,
                "box_loss": best.box_loss if best else None,
                "best_epoch": result.best_epoch,
                "total_epochs": result.total_epochs,
            },
            "history": [
                {
                    "epoch": h.epoch,
                    "map50": h.map50,
                    "map50_95": h.map50_95,
                    "precision": h.precision,
                    "recall": h.recall,
                    "box_loss": h.box_loss,
                    "train_box_loss": h.train_box_loss,
                    "train_cls_loss": h.train_cls_loss,
                    "train_dfl_loss": h.train_dfl_loss,
                    "val_box_loss": h.val_box_loss,
                    "val_cls_loss": h.val_cls_loss,
                    "val_dfl_loss": h.val_dfl_loss,
                }
                for h in result.history
            ],
            "artifacts": {
                "model": str(result.model_path),
                "graphics": graphics,
                "report": None,
            },
            "tests": [],
        }
        (run_dir / "run.json").write_text(
            json.dumps(run_json, indent=2), encoding="utf-8"
        )


__all__ = [
    "DetectionTrainer",
    "DetectionTrainResult",
    "DetectionEpochResult",
]
