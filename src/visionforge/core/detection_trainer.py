"""Drive Ultralytics YOLO / RT-DETR training for a `DetectionConfig`.

Ultralytics owns its training loop; this wrapper translates our config into
``YOLO.train`` arguments, hooks an epoch callback to stream progress through the
same SSE event shape the classification Trainer uses (ADR-032), and writes a
``run.json`` compatible with the ADR-013 contract so detection runs appear in
``/api/runs``. The torchvision backend (Faster R-CNN / SSD) is scaffolded and
raises ``NotImplementedError`` until a follow-up brick.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from visionforge.core.detection_data import DetectionDataModule
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
        """Run Ultralytics training and return the result.

        Raises:
            NotImplementedError: if the torchvision backend is selected.
            RuntimeError: if the ultralytics extra is not installed.
        """
        if self._config.model.backend == "torchvision":
            raise NotImplementedError(
                "torchvision detection backend is not implemented yet; "
                "use backend='ultralytics'."
            )
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
