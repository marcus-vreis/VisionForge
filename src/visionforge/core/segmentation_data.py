"""Paired image/mask dataset and DataLoaders for the segmentation task.

A segmentation dataset is, per split, an ``images`` directory and a ``masks``
directory whose files pair by filename stem (``img001.jpg`` ↔ ``img001.png``).
Masks are single-channel images whose pixel value is the class id; an
``ignore_index`` marks void pixels excluded from loss/metrics.

Image and mask must transform in lockstep — the same crop/flip/rotation must
apply to both — so this module does **not** reuse ``core.data._build_transforms``
(which builds an image-only ``T.Compose``). Instead the geometric ops are applied
jointly in ``__getitem__`` via the functional API: the image is resized with
bilinear interpolation and normalized; the mask is resized with nearest-neighbor
(preserving integer class ids) and left as a ``long`` tensor. The preprocessing
filter pipeline runs on the image only.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from visionforge.core.data import _PreprocessingTransform
from visionforge.core.loader_lifecycle import LoaderCache
from visionforge.utils.config import PreprocessingConfig, TransformConfig
from visionforge.utils.segmentation_config import SegmentationConfig
from visionforge.utils.workers import suggested_workers

# Disable DataLoader workers below this many images: spawn/serialisation overhead
# (high on Windows) exceeds the load time on small sets. Mirrors core.data.
_SMALL_DATASET_THRESHOLD = 500

# Image extensions accepted for the image branch (mask is always read by stem).
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_MASK_EXTS = (".png", ".bmp", ".tif", ".tiff")


_LOADER_POOLS = 3


class SegmentationDataset(Dataset):  # type: ignore[type-arg]
    """Paired image/mask segmentation dataset.

    Pairs each image under ``images_dir`` with a same-stem mask under
    ``masks_dir`` at construction. Each item yields ``(image_tensor, mask_long)``
    where the image is a normalized ``float32`` ``[3, H, W]`` tensor and the mask
    is a ``long`` ``[H, W]`` tensor of class ids.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        *,
        image_size: int,
        ignore_index: int,
        transform_cfg: TransformConfig,
        preprocessing_cfg: PreprocessingConfig,
        is_train: bool,
    ) -> None:
        self._image_size = image_size
        self._ignore_index = ignore_index
        self._is_train = is_train

        self._hflip = transform_cfg.horizontal_flip and is_train
        self._rotation = transform_cfg.rotation_degrees if is_train else 0
        self._normalize_mean = list(transform_cfg.normalize_mean)
        self._normalize_std = list(transform_cfg.normalize_std)
        self._jitter: T.ColorJitter | None = (
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            if (transform_cfg.color_jitter and is_train)
            else None
        )
        self._preprocess = _PreprocessingTransform(
            [s.model_dump() for s in preprocessing_cfg.steps]
        )

        self._pairs = self._pair_files(images_dir, masks_dir)
        if not self._pairs:
            raise ValueError(
                f"No image/mask pairs found under {images_dir} / {masks_dir}; "
                f"expected paired files by stem (no images?)."
            )

    @staticmethod
    def _pair_files(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
        images = sorted(
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if not images:
            raise ValueError(f"No image files in {images_dir}.")
        masks_by_stem: dict[str, Path] = {
            p.stem: p
            for p in masks_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _MASK_EXTS
        }
        pairs: list[tuple[Path, Path]] = []
        for img in images:
            mask = masks_by_stem.get(img.stem)
            if mask is None:
                raise FileNotFoundError(
                    f"No mask for image '{img.name}' (looked for stem "
                    f"'{img.stem}' in {masks_dir})."
                )
            pairs.append((img, mask))
        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self._pairs[idx]
        with Image.open(img_path) as im:
            image = self._preprocess(im.convert("RGB"))
        with Image.open(mask_path) as mk:
            mask = mk.copy()
        if mask.mode in ("RGB", "RGBA"):
            raise ValueError(
                f"Mask '{mask_path.name}' is {mask.mode}; segmentation expects a "
                f"single-channel integer-id mask (mode 'L' or 'P'). RGB/palette-"
                f"color masks need a colour→id map, which is not implemented."
            )

        size = [self._image_size, self._image_size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        if self._hflip and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if self._rotation > 0:
            angle = random.uniform(-self._rotation, self._rotation)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=self._ignore_index,
            )

        if self._jitter is not None:
            image = self._jitter(image)

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, self._normalize_mean, self._normalize_std)
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image_t, mask_t


class SegmentationDataModule:
    """Builds paired image/mask datasets and their DataLoaders.

    train and val are required; test is optional (``test_loader`` returns None
    when the test split directory is absent).
    """

    def __init__(self, config: SegmentationConfig) -> None:
        cfg = config.data
        base = cfg.base_dir

        self._batch_size = config.training.batch_size
        self._num_workers = cfg.num_workers
        self._cache = LoaderCache()
        self._pin_memory = cfg.pin_memory

        def _dataset(split: str, *, is_train: bool) -> SegmentationDataset:
            split_dir = base / split
            return SegmentationDataset(
                split_dir / cfg.images_subdir,
                split_dir / cfg.masks_subdir,
                image_size=cfg.image_size,
                ignore_index=cfg.ignore_index,
                transform_cfg=cfg.transforms,
                preprocessing_cfg=cfg.preprocessing,
                is_train=is_train,
            )

        self._train = _dataset(cfg.train_dir, is_train=True)
        self._val = _dataset(cfg.val_dir, is_train=False)
        self._test: SegmentationDataset | None = None
        if (base / cfg.test_dir / cfg.images_subdir).is_dir():
            self._test = _dataset(cfg.test_dir, is_train=False)

        if self._num_workers > 0 and len(self._train) < _SMALL_DATASET_THRESHOLD:
            from loguru import logger

            logger.info(
                "Segmentation set has {} images (< {}); setting num_workers=0.",
                len(self._train),
                _SMALL_DATASET_THRESHOLD,
            )
            self._num_workers = 0

        # Cap by what this machine can commit, not by what the config asked for:
        # each spawned worker re-imports torch and its CUDA DLLs, and a request
        # for more than the budget is how a run dies with WinError 1455 before
        # its first epoch (ADR-081/098). Lowering silently would hide the
        # machine's limit, so it says so.
        affordable = suggested_workers(loader_pools=_LOADER_POOLS)
        if self._num_workers > affordable:
            from loguru import logger

            logger.warning(
                "num_workers={} exceeds what this machine can commit for {} "
                "loader pools; using {}.",
                self._num_workers,
                _LOADER_POOLS,
                affordable,
            )
            self._num_workers = affordable

    def _loader_kwargs(self, *, persistent: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self._batch_size,
            "num_workers": self._num_workers,
            "pin_memory": self._pin_memory,
        }
        if self._num_workers > 0:
            kwargs["persistent_workers"] = persistent
            kwargs["prefetch_factor"] = 2
        return kwargs

    def train_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the training split (shuffled)."""
        return self._cache.cached(
            "train",
            lambda: DataLoader(
                self._train, shuffle=True, **self._loader_kwargs(persistent=True)
            ),
        )

    def val_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the validation split."""
        return self._cache.cached(
            "val",
            lambda: DataLoader(
                self._val, shuffle=False, **self._loader_kwargs(persistent=True)
            ),
        )

    def test_loader(self) -> DataLoader | None:  # type: ignore[type-arg]
        """DataLoader for the test split, or None when no test split exists."""
        test = self._test
        if test is None:
            return None
        return self._cache.cached(
            "test",
            lambda: DataLoader(
                test, shuffle=False, **self._loader_kwargs(persistent=False)
            ),
        )

    def close(self) -> None:
        """Stop every worker pool this module started.

        Callers run inside a long-lived server process, where a failed run's
        traceback keeps the loaders alive and their workers with them.
        """
        self._cache.close()


__all__ = ["SegmentationDataModule", "SegmentationDataset"]
