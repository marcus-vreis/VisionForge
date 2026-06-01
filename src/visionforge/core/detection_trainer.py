"""Drive detection training for a `DetectionConfig` (Ultralytics or torchvision).

The Ultralytics path wraps ``YOLO.train``; the torchvision path runs a
hand-written loss-dict loop (``_fit_torchvision``, ADR-035) over the Faster R-CNN
family. Both hook the same SSE event shape the classification Trainer uses
(ADR-032) and write a ``run.json`` compatible with the ADR-013 contract so
detection runs appear in ``/api/runs``. torchvision SSD/RetinaNet are not wired
yet — the model factory raises ``NotImplementedError`` for them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader

from visionforge.core.detection_data import DetectionDataModule
from visionforge.core.detection_dataset import DetectionDataset, detection_collate
from visionforge.core.detection_metrics import mean_average_precision_50
from visionforge.models.detection_factory import build_torchvision_detector
from visionforge.utils.detection_config import DetectionConfig

try:  # ultralytics is an optional extra ([detection]); bound lazily.
    from ultralytics import YOLO as _YOLO  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _YOLO = None

# Module global so tests can patch it without ultralytics installed.
YOLO: Any = _YOLO

# Ultralytics metric keys, with the leading "metrics/" namespace it uses.
_MAP50_KEY = "metrics/mAP50(B)"
_MAP5095_KEY = "metrics/mAP50-95(B)"
_BOXLOSS_KEY = "val/box_loss"


@dataclass
class DetectionEpochResult:
    """One epoch's detection metrics."""

    epoch: int
    map50: float | None
    map50_95: float | None
    box_loss: float | None


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


class DetectionTrainer:
    """Wraps Ultralytics training for a `DetectionConfig`."""

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._device_label = self._resolve_device_label()

    def fit(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> DetectionTrainResult:
        """Run training for the configured backend and return the result."""
        if self._config.model.backend == "torchvision":
            return self._fit_torchvision(progress_callback)
        return self._fit_ultralytics(progress_callback)

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
            history.append(DetectionEpochResult(epoch, map50, map95, box_loss))
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "total_epochs": cfg.epochs,
                        "map50": map50,
                        "map50_95": map95,
                        "box_loss": box_loss,
                        # Mirror the classification overlay's required fields so
                        # the live monitor renders detection runs unchanged until
                        # the GUI brick adds detection-specific display.
                        "train_loss": box_loss if box_loss is not None else 0.0,
                        "val_loss": box_loss if box_loss is not None else 0.0,
                        "val_accuracy": map95 if map95 is not None else 0.0,
                    }
                )

        model.add_callback("on_fit_epoch_end", _on_epoch_end)
        logger.info(
            "Detection training: {} ({} epochs) on {}",
            self._config.model.name,
            cfg.epochs,
            self._device_label,
        )
        model.train(
            data=str(data_yaml),
            epochs=cfg.epochs,
            batch=cfg.batch_size,
            imgsz=self._config.data.image_size,
            lr0=cfg.learning_rate,
            patience=cfg.patience,
            seed=cfg.seed,
            workers=cfg.workers,
            device=self._device_arg(),
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            verbose=False,
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
        optimizer = torch.optim.SGD(
            params, lr=cfg.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        model_path = run_dir / "weights" / "best.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader = self._build_loaders(self._config.data.base_dir)

        history: list[DetectionEpochResult] = []
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
        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._run_torchvision_epoch(
                model, train_loader, device, optimizer
            )
            val_loss = self._eval_torchvision_loss(model, val_loader, device)
            map50 = self._eval_torchvision_map(model, val_loader, device)

            history.append(
                DetectionEpochResult(
                    epoch=epoch, map50=map50, map50_95=None, box_loss=val_loss
                )
            )
            # Select by validation mAP@50 (higher is better).
            if map50 > best_map:
                best_map = map50
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)

            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "epoch_end",
                        "epoch": epoch,
                        "total_epochs": cfg.epochs,
                        "map50": map50,
                        "map50_95": None,
                        "box_loss": val_loss,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": map50,
                    }
                )

        if not model_path.exists():  # epochs=0 guard
            torch.save(model.state_dict(), model_path)

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
        candidates = [
            (base / "images" / split, base / "labels" / split),
            (base / split / "images", base / split / "labels"),
        ]
        for images_dir, labels_dir in candidates:
            if images_dir.is_dir():
                return images_dir, labels_dir
        raise ValueError(
            f"Could not find the '{split}' images under '{base}' "
            f"(expected 'images/{split}' or '{split}/images')."
        )

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
        run_json: dict[str, Any] = {
            "id": f"{self._config.name}_{run_dir.name}",
            "experiment": self._config.name,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "device_used": result.device_used,
            "run_dir": str(run_dir.resolve()),
            "config": self._config.model_dump(mode="json"),
            "metrics": {
                "map50_95": result.best_map50_95,
                "map50": next(
                    (h.map50 for h in result.history if h.epoch == result.best_epoch),
                    None,
                ),
                "box_loss": next(
                    (
                        h.box_loss
                        for h in result.history
                        if h.epoch == result.best_epoch
                    ),
                    None,
                ),
                "best_epoch": result.best_epoch,
                "total_epochs": result.total_epochs,
            },
            "history": [
                {
                    "epoch": h.epoch,
                    "map50": h.map50,
                    "map50_95": h.map50_95,
                    "box_loss": h.box_loss,
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
