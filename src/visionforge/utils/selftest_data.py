"""Tiny synthetic datasets for the end-to-end self-test (ADR-060).

Every builder writes the exact on-disk layout its task consumes in production —
no fixtures, no mocks — but sized so a one-epoch run finishes in seconds on CPU:
32-64px images, a handful per split. Signal is deliberately class-correlated so
a model can actually learn something and metrics are not degenerate.

These live in ``src/`` (not ``tests/``) because ``visionforge selftest`` ships
them: a researcher who just installed VisionForge can verify the whole pipeline
without owning a dataset yet.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

# Deterministic across runs so a failing self-test is reproducible.
_SEED = 20260715


def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _tinted_image(rng: np.random.Generator, channel: int, size: int) -> Image.Image:
    """Noise with one channel pushed bright — a learnable per-class signal."""
    arr = rng.integers(0, 80, (size, size, 3), dtype=np.uint8)
    arr[..., channel] = rng.integers(160, 255, (size, size), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def build_classification_dataset(
    base: Path, *, size: int = 32, per_class: int = 6
) -> Path:
    """ImageFolder layout: ``<base>/{train,val,test}/<class>/*.png`` (2 classes)."""
    rng = _rng()
    for split in ("train", "val", "test"):
        for channel, cls in enumerate(("class_a", "class_b")):
            cdir = base / split / cls
            cdir.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                _tinted_image(rng, channel, size).save(cdir / f"img{i}.png")
    return base


def build_regression_dataset(base: Path, *, size: int = 32, rows: int = 12) -> Path:
    """CSV-manifest layout: ``<base>/{train,val,test}.csv`` + ``<base>/images/``.

    The target is a linear function of the image's red level, so R² is
    meaningful rather than noise.
    """
    rng = _rng()
    images = base / "images"
    images.mkdir(parents=True, exist_ok=True)

    for split, n in (("train", rows), ("val", max(rows // 2, 4)), ("test", 4)):
        manifest: list[tuple[str, float]] = []
        for i in range(n):
            level = int(rng.integers(30, 240))
            name = f"{split}_{i}.png"
            arr = np.full((size, size, 3), 40, dtype=np.uint8)
            arr[..., 0] = level
            Image.fromarray(arr, "RGB").save(images / name)
            manifest.append((name, level / 255.0))
        with (base / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "target"])
            writer.writerows(manifest)
    return base


def build_segmentation_dataset(base: Path, *, size: int = 64, pairs: int = 6) -> Path:
    """Paired layout: ``<base>/<split>/{images,masks}/*.png`` with 3 classes.

    Each mask marks a rectangle whose position mirrors the image's bright
    block, so mIoU responds to learning instead of staying at chance.
    """
    rng = _rng()
    for split, n in (("train", pairs), ("val", max(pairs // 2, 2)), ("test", 2)):
        img_dir = base / split / "images"
        mask_dir = base / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cls = i % 2  # alternate which class the foreground belongs to
            arr = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
            mask = np.zeros((size, size), dtype=np.uint8)
            y0, x0 = size // 4, size // 4
            y1, x1 = y0 + size // 2, x0 + size // 2
            arr[y0:y1, x0:x1, cls] = 220
            mask[y0:y1, x0:x1] = cls + 1  # 0 = background, 1/2 = foreground classes
            Image.fromarray(arr, "RGB").save(img_dir / f"img{i}.png")
            Image.fromarray(mask, "L").save(mask_dir / f"img{i}.png")
    return base


def build_anomaly_dataset(base: Path, *, size: int = 64, normals: int = 8) -> Path:
    """MVTec-style layout: ``<base>/train/good`` (normal only) + ``<base>/test/{good,defect}``.

    Defective images carry a bright square the normal ones never have, so
    reconstruction error separates the two and AUROC is not 0.5.
    """
    rng = _rng()
    train_good = base / "train" / "good"
    test_good = base / "test" / "good"
    test_bad = base / "test" / "defect"
    for d in (train_good, test_good, test_bad):
        d.mkdir(parents=True, exist_ok=True)

    def _normal(i: int) -> np.ndarray:
        return rng.integers(90, 130, (size, size, 3), dtype=np.uint8)

    for i in range(normals):
        Image.fromarray(_normal(i), "RGB").save(train_good / f"n{i}.png")
    for i in range(4):
        Image.fromarray(_normal(i), "RGB").save(test_good / f"n{i}.png")
        bad = _normal(i)
        bad[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 255
        Image.fromarray(bad, "RGB").save(test_bad / f"d{i}.png")
    return base


def build_detection_dataset(base: Path, *, size: int = 128, per_split: int = 4) -> Path:
    """YOLO layout: ``<base>/images/<split>`` + ``<base>/labels/<split>`` (1 class).

    Each image holds one bright box whose normalized coordinates match its
    label line, so the detector has a consistent target to fit.
    """
    rng = _rng()
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_split):
            arr = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
            # Box occupies the central 40% — matches the label below.
            lo, hi = int(size * 0.3), int(size * 0.7)
            arr[lo:hi, lo:hi] = 230
            Image.fromarray(arr, "RGB").save(img_dir / f"img{i}.jpg")
            (lbl_dir / f"img{i}.txt").write_text(
                "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
            )
    return base


__all__ = [
    "build_anomaly_dataset",
    "build_classification_dataset",
    "build_detection_dataset",
    "build_regression_dataset",
    "build_segmentation_dataset",
]
