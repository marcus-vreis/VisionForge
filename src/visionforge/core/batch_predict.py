"""Task-agnostic folder batch-prediction (ADR-041 slice 3).

Walks a folder of images, applies a task-supplied transform, runs a task-supplied
inference callback per image, and writes one CSV row per image. The per-task
inference head (softmax probabilities, continuous targets, anomaly score) lives
in the caller — this helper only owns the file-walk, image loading, error
recording and CSV writing, so every task batch-predicts through the same code.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from PIL import Image

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


@dataclass
class BatchPredictionSummary:
    """Outcome of a folder batch-prediction run."""

    output_csv: str
    total_processed: int = 0
    failed_files: list[str] = field(default_factory=list)


def collect_image_paths(
    input_dir: Path,
    *,
    recursive: bool,
    image_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> list[Path]:
    """Return sorted image files under ``input_dir`` filtered by extension."""
    exts = {e.lower() for e in image_extensions}
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in exts)


def predict_folder_to_csv(
    *,
    input_dir: Path,
    output_csv: Path,
    transform: Callable[[Image.Image], torch.Tensor],
    infer: Callable[[torch.Tensor], dict[str, Any]],
    fieldnames: list[str],
    device: torch.device,
    recursive: bool = False,
    image_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> BatchPredictionSummary:
    """Run per-image inference over a folder and write one CSV row per image.

    ``transform`` maps a PIL image to an un-batched tensor; ``infer`` maps the
    batched ``(1, C, H, W)`` tensor to the row values keyed by ``fieldnames``
    minus ``filename`` (which the helper fills with the source path). Images that
    fail to load are skipped, logged and recorded in ``failed_files``.
    """
    paths = collect_image_paths(
        input_dir, recursive=recursive, image_extensions=image_extensions
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    processed = 0

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            try:
                image = Image.open(path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)
            except Exception as exc:  # noqa: BLE001 — a bad image must not abort the batch
                logger.warning("Skipping {}: {}", path, exc)
                failed.append(str(path))
                continue

            with torch.no_grad():
                values = infer(tensor)
            writer.writerow({"filename": str(path), **values})
            processed += 1

    return BatchPredictionSummary(
        output_csv=str(output_csv),
        total_processed=processed,
        failed_files=failed,
    )


__all__ = [
    "BatchPredictionSummary",
    "DEFAULT_IMAGE_EXTENSIONS",
    "collect_image_paths",
    "predict_folder_to_csv",
]
