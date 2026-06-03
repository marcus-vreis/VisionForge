from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from visionforge.core.detection_dataset import DetectionDataset, detection_collate


def _make_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (120, 120, 120)).save(path)


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    return images, labels


class TestDetectionDataset:
    def test_len_counts_only_images(self, tmp_path: Path) -> None:
        images, labels = _dataset(tmp_path)
        _make_image(images / "a.jpg", 20, 10)
        _make_image(images / "b.png", 20, 10)
        (images / "notes.txt").write_text("ignore me", "utf-8")
        ds = DetectionDataset(images, labels)
        assert len(ds) == 2

    def test_yolo_box_converted_to_absolute_xyxy(self, tmp_path: Path) -> None:
        images, labels = _dataset(tmp_path)
        _make_image(images / "a.jpg", 20, 10)
        # class 0, centered, half width, full height.
        (labels / "a.txt").write_text("0 0.5 0.5 0.5 1.0\n", "utf-8")

        image, target = DetectionDataset(images, labels)[0]
        assert image.shape == (3, 10, 20)  # C,H,W
        assert target["boxes"].tolist() == [[5.0, 0.0, 15.0, 10.0]]
        # YOLO class 0 → label 1 (0 is background for torchvision).
        assert target["labels"].tolist() == [1]

    def test_missing_label_file_yields_empty_target(self, tmp_path: Path) -> None:
        images, labels = _dataset(tmp_path)
        _make_image(images / "a.jpg", 16, 16)
        _, target = DetectionDataset(images, labels)[0]
        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)

    def test_multiple_boxes(self, tmp_path: Path) -> None:
        images, labels = _dataset(tmp_path)
        _make_image(images / "a.jpg", 100, 100)
        (labels / "a.txt").write_text(
            "0 0.25 0.25 0.2 0.2\n2 0.75 0.75 0.5 0.5\n", "utf-8"
        )
        _, target = DetectionDataset(images, labels)[0]
        assert target["boxes"].shape == (2, 4)
        assert target["labels"].tolist() == [1, 3]

    def test_degenerate_box_skipped(self, tmp_path: Path) -> None:
        images, labels = _dataset(tmp_path)
        _make_image(images / "a.jpg", 20, 20)
        # zero width box → skipped.
        (labels / "a.txt").write_text("0 0.5 0.5 0.0 0.5\n", "utf-8")
        _, target = DetectionDataset(images, labels)[0]
        assert target["boxes"].shape == (0, 4)

    def test_collate_groups_into_tuples(self, tmp_path: Path) -> None:
        images, labels = _dataset(tmp_path)
        _make_image(images / "a.jpg", 16, 16)
        _make_image(images / "b.jpg", 16, 16)
        ds = DetectionDataset(images, labels)
        imgs, targets = detection_collate([ds[0], ds[1]])
        assert len(imgs) == 2 and len(targets) == 2
        assert all(isinstance(t, torch.Tensor) for t in imgs)
