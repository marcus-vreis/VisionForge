"""Datasets and DataLoaders for the anomaly-detection task (Phase 9).

MVTec-AD-style layout: ``train/<normal_dir>`` holds normal images only
(unsupervised training, all label 0); ``test/`` holds the ``<normal_dir>``
subfolder (label 0) plus one or more defect subfolders (any other subdir →
label 1). Reuses ``core.data._build_transforms`` so augmentation/preprocessing
match the other tasks. See documentation/PHASE9_ANOMALY_PLAN.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from visionforge.core.data import _build_transforms
from visionforge.utils.anomaly_config import AnomalyConfig

_SMALL_DATASET_THRESHOLD = 500
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


class AnomalyImageDataset(Dataset):  # type: ignore[type-arg]
    """Image dataset over an explicit ``(path, label)`` list.

    Each item yields ``(image_tensor, label)`` where label is 0 (normal) or 1
    (anomalous). Picklable for Windows-spawn DataLoader workers.
    """

    def __init__(
        self,
        items: list[tuple[Path, int]],
        *,
        image_size: int,
        transform: T.Compose | None = None,
        transform_cfg: Any = None,
        preprocessing_cfg: Any = None,
        is_train: bool = False,
    ) -> None:
        self._items = items
        if transform is not None:
            self._transform = transform
        else:
            from visionforge.utils.config import PreprocessingConfig, TransformConfig

            tc = transform_cfg or TransformConfig(image_size=image_size)
            pp = preprocessing_cfg or PreprocessingConfig()
            self._transform = _build_transforms(tc, is_train=is_train, preprocessing=pp)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self._items[idx]
        with Image.open(path) as img:
            tensor = self._transform(img.convert("RGB"))
        return tensor, label


class AnomalyDataModule:
    """Builds normal-only train and labelled test datasets + DataLoaders."""

    def __init__(self, config: AnomalyConfig) -> None:
        cfg = config.data
        base = cfg.base_dir
        # Sync the resize resolution: data.image_size is the anomaly-task knob,
        # but _build_transforms resizes to TransformConfig.image_size.
        tc = cfg.transforms.model_copy(update={"image_size": cfg.image_size})
        pp = cfg.preprocessing

        self._batch_size = config.training.batch_size
        self._num_workers = cfg.num_workers
        self._pin_memory = cfg.pin_memory

        train_normal = base / cfg.train_dir / cfg.normal_dir
        train_items = [(p, 0) for p in _list_images(train_normal)]
        if not train_items:
            raise ValueError(
                f"No normal training images in {train_normal} (train_dir/normal_dir)."
            )

        test_root = base / cfg.test_dir
        if not test_root.is_dir():
            raise ValueError(f"test directory not found: {test_root}")
        test_items: list[tuple[Path, int]] = []
        for sub in sorted(p for p in test_root.iterdir() if p.is_dir()):
            label = 0 if sub.name == cfg.normal_dir else 1
            test_items.extend((p, label) for p in _list_images(sub))
        if not test_items:
            raise ValueError(f"No test images found under {test_root}.")

        self.train_size = len(train_items)
        self.test_size = len(test_items)

        self._train = AnomalyImageDataset(
            train_items,
            image_size=cfg.image_size,
            transform_cfg=tc,
            preprocessing_cfg=pp,
            is_train=True,
        )
        self._test = AnomalyImageDataset(
            test_items,
            image_size=cfg.image_size,
            transform_cfg=tc,
            preprocessing_cfg=pp,
            is_train=False,
        )

        if self._num_workers > 0 and self.train_size < _SMALL_DATASET_THRESHOLD:
            from loguru import logger

            logger.info(
                "Anomaly train set has {} images (< {}); setting num_workers=0.",
                self.train_size,
                _SMALL_DATASET_THRESHOLD,
            )
            self._num_workers = 0

    def _loader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self._batch_size,
            "num_workers": self._num_workers,
            "pin_memory": self._pin_memory,
        }
        if self._num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def train_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the normal-only training split (shuffled)."""
        return DataLoader(self._train, shuffle=True, **self._loader_kwargs())

    def test_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the labelled test split (normal=0, anomalous=1)."""
        return DataLoader(self._test, shuffle=False, **self._loader_kwargs())


__all__ = ["AnomalyDataModule", "AnomalyImageDataset"]
