from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visionforge.core.anomaly_data import AnomalyDataModule, AnomalyImageDataset
from visionforge.utils.anomaly_config import AnomalyConfig


def _img(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _make_mvtec(
    base: Path, n_train: int = 4, n_normal: int = 2, n_defect: int = 3
) -> None:
    for i in range(n_train):
        _img(base / "train" / "good" / f"n{i}.png")
    for i in range(n_normal):
        _img(base / "test" / "good" / f"g{i}.png")
    for i in range(n_defect):
        _img(base / "test" / "broken" / f"b{i}.png")


def _config(base: Path) -> AnomalyConfig:
    return AnomalyConfig.model_validate(
        {
            "name": "anom",
            "model": {"name": "autoencoder"},
            "data": {"base_dir": str(base), "image_size": 32, "num_workers": 0},
            "training": {"epochs": 1, "batch_size": 2},
        }
    )


class TestAnomalyImageDataset:
    def test_yields_image_and_label(self, tmp_path: Path) -> None:
        _img(tmp_path / "a.png")
        ds = AnomalyImageDataset([(tmp_path / "a.png", 1)], image_size=32)
        img, label = ds[0]
        assert img.shape == (3, 32, 32)
        assert img.dtype == torch.float32
        assert label == 1

    def test_picklable(self, tmp_path: Path) -> None:
        _img(tmp_path / "a.png")
        ds = AnomalyImageDataset([(tmp_path / "a.png", 0)], image_size=32)
        restored = pickle.loads(pickle.dumps(ds))
        assert len(restored) == 1


class TestAnomalyDataModule:
    def test_train_is_normal_only_label_zero(self, tmp_path: Path) -> None:
        _make_mvtec(tmp_path)
        dm = AnomalyDataModule(_config(tmp_path))
        labels = [int(y) for _, yb in dm.train_loader() for y in yb]
        assert labels == [0, 0, 0, 0]

    def test_train_count_matches(self, tmp_path: Path) -> None:
        _make_mvtec(tmp_path, n_train=4)
        dm = AnomalyDataModule(_config(tmp_path))
        assert dm.train_size == 4

    def test_test_labels_normal_zero_defect_one(self, tmp_path: Path) -> None:
        _make_mvtec(tmp_path, n_normal=2, n_defect=3)
        dm = AnomalyDataModule(_config(tmp_path))
        labels = sorted(int(y) for _, yb in dm.test_loader() for y in yb)
        assert labels == [0, 0, 1, 1, 1]

    def test_batch_shapes(self, tmp_path: Path) -> None:
        _make_mvtec(tmp_path)
        dm = AnomalyDataModule(_config(tmp_path))
        xb, yb = next(iter(dm.train_loader()))
        assert xb.shape[1:] == (3, 32, 32)
        assert yb.dtype == torch.long

    def test_empty_train_normal_raises(self, tmp_path: Path) -> None:
        # test/ present but train/good/ empty
        _img(tmp_path / "test" / "good" / "g.png")
        (tmp_path / "train" / "good").mkdir(parents=True)
        with pytest.raises(ValueError, match="[Nn]o .*normal"):
            AnomalyDataModule(_config(tmp_path))

    def test_missing_test_raises(self, tmp_path: Path) -> None:
        for i in range(2):
            _img(tmp_path / "train" / "good" / f"n{i}.png")
        with pytest.raises(ValueError, match="test"):
            AnomalyDataModule(_config(tmp_path))
