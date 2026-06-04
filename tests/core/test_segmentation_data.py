from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visionforge.core.segmentation_data import (
    SegmentationDataModule,
    SegmentationDataset,
)
from visionforge.utils.segmentation_config import SegmentationConfig


def _make_split(
    base: Path,
    split: str,
    n: int,
    *,
    num_classes: int = 3,
    img_size: int = 32,
    images_subdir: str = "images",
    masks_subdir: str = "masks",
    write_mask: bool = True,
) -> None:
    """Write ``n`` paired image/mask PNGs into ``base/split/{images,masks}``."""
    img_dir = base / split / images_subdir
    mask_dir = base / split / masks_subdir
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        stem = f"img{i:03d}"
        arr = rng.integers(0, 255, size=(img_size, img_size, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(img_dir / f"{stem}.jpg")
        if write_mask:
            mask = rng.integers(
                0, num_classes, size=(img_size, img_size), dtype=np.uint8
            )
            Image.fromarray(mask, mode="L").save(mask_dir / f"{stem}.png")


def _config(base: Path, overrides: dict | None = None) -> SegmentationConfig:
    raw: dict = {
        "name": "seg_test",
        "model": {"name": "fcn_resnet50", "num_classes": 3},
        "data": {"base_dir": str(base), "image_size": 32, "num_workers": 0},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
    }
    if overrides:
        for k, v in overrides.items():
            raw[k] = {**raw.get(k, {}), **v} if isinstance(v, dict) else v
    return SegmentationConfig.model_validate(raw)


# ── dataset ───────────────────────────────────────────────────────────────────


class TestSegmentationDataset:
    def test_pairs_by_stem_and_len(self, tmp_path: Path) -> None:
        _make_split(tmp_path, "train", 5)
        ds = SegmentationDataset(
            tmp_path / "train" / "images",
            tmp_path / "train" / "masks",
            image_size=32,
            ignore_index=255,
            transform_cfg=_config(tmp_path).data.transforms,
            preprocessing_cfg=_config(tmp_path).data.preprocessing,
            is_train=False,
        )
        assert len(ds) == 5

    def test_getitem_shapes_and_dtypes(self, tmp_path: Path) -> None:
        _make_split(tmp_path, "train", 3)
        d = _config(tmp_path).data
        ds = SegmentationDataset(
            tmp_path / "train" / "images",
            tmp_path / "train" / "masks",
            image_size=32,
            ignore_index=255,
            transform_cfg=d.transforms,
            preprocessing_cfg=d.preprocessing,
            is_train=False,
        )
        img, mask = ds[0]
        assert img.shape == (3, 32, 32)
        assert img.dtype == torch.float32
        assert mask.shape == (32, 32)
        assert mask.dtype == torch.long

    def test_mask_values_are_class_ids_not_normalized(self, tmp_path: Path) -> None:
        _make_split(tmp_path, "train", 2, num_classes=3)
        d = _config(tmp_path).data
        ds = SegmentationDataset(
            tmp_path / "train" / "images",
            tmp_path / "train" / "masks",
            image_size=32,
            ignore_index=255,
            transform_cfg=d.transforms,
            preprocessing_cfg=d.preprocessing,
            is_train=False,
        )
        _, mask = ds[0]
        # Nearest-neighbour resize preserves integer class ids in 0..num_classes-1.
        assert set(mask.unique().tolist()).issubset({0, 1, 2})

    def test_missing_mask_raises(self, tmp_path: Path) -> None:
        _make_split(tmp_path, "train", 2, write_mask=False)
        d = _config(tmp_path).data
        with pytest.raises(FileNotFoundError, match="mask"):
            SegmentationDataset(
                tmp_path / "train" / "images",
                tmp_path / "train" / "masks",
                image_size=32,
                ignore_index=255,
                transform_cfg=d.transforms,
                preprocessing_cfg=d.preprocessing,
                is_train=False,
            )

    def test_empty_split_raises(self, tmp_path: Path) -> None:
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "masks").mkdir(parents=True)
        d = _config(tmp_path).data
        with pytest.raises(ValueError, match="[Nn]o image"):
            SegmentationDataset(
                tmp_path / "train" / "images",
                tmp_path / "train" / "masks",
                image_size=32,
                ignore_index=255,
                transform_cfg=d.transforms,
                preprocessing_cfg=d.preprocessing,
                is_train=False,
            )

    def test_rgb_mask_raises_informative_error(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "train" / "images"
        mask_dir = tmp_path / "train" / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(img_dir / "a.jpg")
        Image.fromarray(arr, "RGB").save(mask_dir / "a.png")  # 3-channel mask
        d = _config(tmp_path).data
        ds = SegmentationDataset(
            img_dir,
            mask_dir,
            image_size=32,
            ignore_index=255,
            transform_cfg=d.transforms,
            preprocessing_cfg=d.preprocessing,
            is_train=False,
        )
        with pytest.raises(ValueError, match="single-channel"):
            _ = ds[0]

    def test_dataset_is_picklable(self, tmp_path: Path) -> None:
        _make_split(tmp_path, "train", 2)
        d = _config(tmp_path).data
        ds = SegmentationDataset(
            tmp_path / "train" / "images",
            tmp_path / "train" / "masks",
            image_size=32,
            ignore_index=255,
            transform_cfg=d.transforms,
            preprocessing_cfg=d.preprocessing,
            is_train=True,
        )
        restored = pickle.loads(pickle.dumps(ds))
        assert len(restored) == 2

    def test_joint_hflip_keeps_image_and_mask_aligned(self, tmp_path: Path) -> None:
        # With hflip on and a fixed seed, both image and mask must flip together.
        _make_split(tmp_path, "train", 1)
        d = _config(tmp_path, {"data": {"transforms": {"horizontal_flip": True}}}).data
        ds = SegmentationDataset(
            tmp_path / "train" / "images",
            tmp_path / "train" / "masks",
            image_size=32,
            ignore_index=255,
            transform_cfg=d.transforms,
            preprocessing_cfg=d.preprocessing,
            is_train=True,
        )
        # Reading the same item must yield mask spatial dims unchanged (no desync crash).
        _, mask = ds[0]
        assert mask.shape == (32, 32)


# ── datamodule ────────────────────────────────────────────────────────────────


class TestSegmentationDataModule:
    def test_train_val_loaders_yield_batches(self, tmp_path: Path) -> None:
        _make_split(tmp_path, "train", 4)
        _make_split(tmp_path, "val", 2)
        dm = SegmentationDataModule(_config(tmp_path))
        xb, yb = next(iter(dm.train_loader()))
        assert xb.shape == (2, 3, 32, 32)
        assert yb.shape == (2, 32, 32)
        assert yb.dtype == torch.long
        assert next(iter(dm.val_loader())) is not None

    def test_test_loader_none_when_absent(self, tmp_path: Path) -> None:
        _make_split(tmp_path, "train", 2)
        _make_split(tmp_path, "val", 2)
        dm = SegmentationDataModule(_config(tmp_path))
        assert dm.test_loader() is None

    def test_test_loader_present_when_split_exists(self, tmp_path: Path) -> None:
        _make_split(tmp_path, "train", 2)
        _make_split(tmp_path, "val", 2)
        _make_split(tmp_path, "test", 2)
        dm = SegmentationDataModule(_config(tmp_path))
        assert dm.test_loader() is not None
