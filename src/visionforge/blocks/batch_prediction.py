from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as T
from loguru import logger
from PIL import Image

from visionforge.blocks.base import ExperimentBlock
from visionforge.models.factory import ModelFactory
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.cuda import check_cuda


class BatchPredictionBlock(ExperimentBlock):
    """Loads a trained checkpoint and runs inference over a folder of images, writing CSV output."""

    def setup(self, config: ExperimentConfig) -> None:
        """Build and load the model from checkpoint.

        Raises:
            ValueError: if config.batch_prediction is None.
        """
        if config.batch_prediction is None:
            raise ValueError(
                "batch_prediction config is required for BatchPredictionBlock."
            )

        self._config = config
        self._bp = config.batch_prediction

        cuda_info = check_cuda()
        self._device = torch.device("cuda" if cuda_info.available else "cpu")

        model = ModelFactory.create(config.model)
        state_dict = torch.load(
            str(self._bp.checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)  # type: ignore[arg-type]
        model.eval()
        self._model: nn.Module = model.to(self._device)

        self._transform = self._build_inference_transform()
        self._has_run = False
        self._total_processed = 0
        self._failed_files: list[str] = []

    def run(self) -> None:
        """Walk input_dir, run inference on each image, write predictions to CSV."""
        paths = self._collect_image_paths()
        is_binary = self._config.model.num_classes == 1
        fieldnames = self._build_fieldnames(is_binary)

        self._bp.output_csv.parent.mkdir(parents=True, exist_ok=True)

        self._has_run = True

        with self._bp.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for path in paths:
                row = self._process_image(path, is_binary, fieldnames)
                if row is not None:
                    writer.writerow(row)
                    self._total_processed += 1

    def report(self) -> dict[str, Any]:
        """Return summary: processed count, failed files, output CSV path."""
        if not getattr(self, "_has_run", False):
            return {}
        return {
            "total_processed": self._total_processed,
            "failed_files": list(self._failed_files),
            "output_csv": str(self._bp.output_csv),
        }

    # ── private ────────────────────────────────────────────────────────────────

    def _build_inference_transform(self) -> T.Compose:
        """Centre-crop + normalise transform — no augmentation for inference."""
        tc = self._config.data.transforms
        return T.Compose(
            [
                T.Resize(tc.image_size),
                T.CenterCrop(tc.image_size),
                T.ToTensor(),
                T.Normalize(mean=tc.normalize_mean, std=tc.normalize_std),
            ]
        )

    def _collect_image_paths(self) -> list[Path]:
        """Collect image file paths according to recursive flag and extensions."""
        exts = {e.lower() for e in self._bp.image_extensions}
        root = self._bp.input_dir

        if self._bp.recursive:
            candidates = root.rglob("*")
        else:
            candidates = root.iterdir()

        return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in exts)

    def _build_fieldnames(self, is_binary: bool) -> list[str]:
        if is_binary:
            return ["filename", "predicted_class", "prob_positive"]
        return ["filename", "predicted_class"] + [
            f"prob_class_{i}" for i in range(self._config.model.num_classes)
        ]

    def _process_image(
        self,
        path: Path,
        is_binary: bool,
        fieldnames: list[str],
    ) -> dict[str, Any] | None:
        """Load, transform, and run inference on a single image.

        Returns None on load failure — the failure is logged and recorded.
        """
        try:
            image = Image.open(path).convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(self._device)
        except Exception as exc:
            logger.warning("Skipping {}: {}", path, exc)
            self._failed_files.append(str(path))
            return None

        with torch.no_grad():
            logits = self._model(tensor)

        if is_binary:
            prob = logits.sigmoid().item()
            pred_idx = int(prob >= 0.5)
            row: dict[str, Any] = {
                "filename": str(path),
                "predicted_class": self._resolve_class_name(pred_idx),
                "prob_positive": f"{prob:.6f}",
            }
        else:
            probs = logits.softmax(dim=1).squeeze(0).tolist()
            pred_idx = int(logits.argmax(dim=1).item())
            row = {
                "filename": str(path),
                "predicted_class": self._resolve_class_name(pred_idx),
            }
            for i, p in enumerate(probs):
                row[f"prob_class_{i}"] = f"{p:.6f}"

        return row

    def _resolve_class_name(self, idx: int) -> str:
        """Return human-readable label or string index when class_names is absent."""
        names = self._bp.class_names
        if names is not None and idx < len(names):
            return names[idx]
        return str(idx)


__all__ = ["BatchPredictionBlock"]
