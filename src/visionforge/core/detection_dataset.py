"""YOLO-format dataset adapter for torchvision detectors.

Reads a YOLO split (an images folder + a parallel labels folder with one
``<stem>.txt`` per image, each line ``class cx cy w h`` normalized to [0,1])
and yields ``(image, target)`` where ``target`` has ``boxes`` (xyxy, absolute
pixels) and ``labels`` (int64). torchvision detectors reserve label 0 for
background, so YOLO class ``c`` is emitted as ``c + 1``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]:
    """Collate (image, target) pairs into parallel tuples (torchvision API)."""
    images, targets = zip(*batch, strict=True)
    return images, targets


class DetectionDataset(Dataset):
    """Image + YOLO-label pairs as torchvision (image, target) samples."""

    def __init__(self, images_dir: Path, labels_dir: Path) -> None:
        self._images_dir = Path(images_dir)
        self._labels_dir = Path(labels_dir)
        self._images = sorted(
            p
            for p in self._images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path = self._images[idx]
        with Image.open(img_path) as im:
            image = im.convert("RGB")
            width, height = image.size
            tensor = to_tensor(image)

        boxes, labels = self._read_labels(img_path, width, height)
        target: dict[str, torch.Tensor] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return tensor, target

    # ── private ───────────────────────────────────────────────────────────────

    def _read_labels(
        self, img_path: Path, width: int, height: int
    ) -> tuple[list[list[float]], list[int]]:
        label_path = self._labels_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            return [], []

        boxes: list[list[float]] = []
        labels: list[int] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = (float(p) for p in parts)
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            x2 = (cx + w / 2) * width
            y2 = (cy + h / 2) * height
            # Clamp to the image and skip degenerate boxes.
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)  # 0 is background for torchvision
        return boxes, labels


__all__ = ["DetectionDataset", "detection_collate"]
