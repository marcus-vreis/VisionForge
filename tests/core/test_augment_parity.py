"""``transforms.augment`` is honoured by every training path (STAT-2, ADR-083).

Each path builds its own transform pipeline, and two of them (segmentation and
the K-fold folds) once ignored the flag. One parametrized suite reads the same
training index under different RNG seeds: with ``augment=False`` the tensors
must be bit-identical even though flip, rotation and jitter are configured; with
``augment=True`` some seed must change them.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder

from visionforge.blocks import regression_cv, segmentation_cv
from visionforge.blocks.cross_validation import _FoldDataModule
from visionforge.core.anomaly_data import AnomalyDataModule
from visionforge.core.data import DataModule
from visionforge.core.detection_trainer import (
    _NEUTRAL_AUGMENTATION,
    _augmentation_kwargs,
)
from visionforge.core.regression_data import RegressionDataModule
from visionforge.core.segmentation_data import SegmentationDataModule
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

_SIZE = 32
_SEEDS = tuple(range(8))
# Every random step on at once: flip, rotation and colour jitter.
_TRANSFORMS = {
    "image_size": _SIZE,
    "horizontal_flip": True,
    "rotation_degrees": 30,
    "color_jitter": True,
}

Reader = Callable[[int], tuple[torch.Tensor, ...]]


def _noise(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(
        0, 255, size=(_SIZE, _SIZE, 3), dtype=np.uint8
    )
    Image.fromarray(arr, mode="RGB").save(path)


def _transforms(augment: bool) -> dict[str, Any]:
    return {**_TRANSFORMS, "augment": augment}


def _reader(dataset: Dataset) -> Reader:  # type: ignore[type-arg]
    """Read ``dataset[0]`` with torch, random and numpy seeded the same way."""

    def read(seed: int) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        item = dataset[0]
        return tuple(t for t in item if isinstance(t, torch.Tensor))

    return read


# ── one builder per path: (tmp_path, augment) -> the training dataset ────────


def _classification(tmp_path: Path, augment: bool) -> Dataset:  # type: ignore[type-arg]
    for split in ("train", "val", "test"):
        for cls in ("a", "b"):
            for i in range(2):
                _noise(tmp_path / split / cls / f"{i}.png", i)
    config = ExperimentConfig.model_validate(
        {
            "name": "parity",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {"epochs": 1, "batch_size": 2, "seed": 0},
            "data": {
                "base_dir": str(tmp_path),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": _transforms(augment),
            },
        }
    )
    return DataModule(config).train_loader().dataset


def _classification_fold(tmp_path: Path, augment: bool) -> Dataset:  # type: ignore[type-arg]
    for cls in ("a", "b"):
        for i in range(4):
            _noise(tmp_path / cls / f"{i}.png", i)
    config = ExperimentConfig.model_validate(
        {
            "name": "parity_cv",
            "task": "binary",
            "block": "cross_validation",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {"epochs": 1, "batch_size": 2, "seed": 0},
            "data": {
                "base_dir": str(tmp_path),
                "train_dir": "",
                "num_workers": 0,
                "pin_memory": False,
                "transforms": _transforms(augment),
            },
        }
    )
    pool = ImageFolder(str(tmp_path))
    fold = _FoldDataModule(
        Subset(pool, [0, 1, 4, 5]),
        Subset(pool, [2, 6]),
        [0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2],
        config,
    )
    return fold.train_loader().dataset


def _regression_config(tmp_path: Path, augment: bool) -> RegressionConfig:
    lines = ["image,target"]
    for i, name in enumerate(("a", "b", "c", "d")):
        _noise(tmp_path / "images" / f"{name}.png", i)
        lines.append(f"{name}.png,{float(i)}")
    for split in ("train", "val"):
        (tmp_path / f"{split}.csv").write_text("\n".join(lines), encoding="utf-8")
    return RegressionConfig.model_validate(
        {
            "name": "parity_reg",
            "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
            "data": {
                "base_dir": str(tmp_path),
                "target_columns": ["target"],
                "num_workers": 0,
                "pin_memory": False,
                "transforms": _transforms(augment),
            },
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
        }
    )


def _regression(tmp_path: Path, augment: bool) -> Dataset:  # type: ignore[type-arg]
    config = _regression_config(tmp_path, augment)
    return RegressionDataModule(config).train_loader().dataset


def _regression_fold(
    tmp_path: Path, augment: bool, monkeypatch: pytest.MonkeyPatch
) -> Dataset:  # type: ignore[type-arg]
    config = _regression_config(tmp_path, augment)
    return _captured_fold_train_ds(
        monkeypatch,
        regression_cv,
        lambda: regression_cv.run_regression_cross_validation(config, n_folds=2),
    )


def _segmentation_config(tmp_path: Path, augment: bool) -> SegmentationConfig:
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for i in range(4):
            _noise(tmp_path / split / "images" / f"img{i}.png", i)
            mask = rng.integers(0, 3, size=(_SIZE, _SIZE), dtype=np.uint8)
            masks = tmp_path / split / "masks"
            masks.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(masks / f"img{i}.png")
    return SegmentationConfig.model_validate(
        {
            "name": "parity_seg",
            "model": {"name": "fcn_resnet50", "num_classes": 3},
            "data": {
                "base_dir": str(tmp_path),
                "image_size": _SIZE,
                "num_workers": 0,
                "pin_memory": False,
                "transforms": _transforms(augment),
            },
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
        }
    )


def _segmentation(tmp_path: Path, augment: bool) -> Dataset:  # type: ignore[type-arg]
    config = _segmentation_config(tmp_path, augment)
    return SegmentationDataModule(config).train_loader().dataset


def _segmentation_fold(
    tmp_path: Path, augment: bool, monkeypatch: pytest.MonkeyPatch
) -> Dataset:  # type: ignore[type-arg]
    config = _segmentation_config(tmp_path, augment)
    return _captured_fold_train_ds(
        monkeypatch,
        segmentation_cv,
        lambda: segmentation_cv.run_segmentation_cross_validation(config, n_folds=2),
    )


def _anomaly(tmp_path: Path, augment: bool) -> Dataset:  # type: ignore[type-arg]
    for i in range(4):
        _noise(tmp_path / "train" / "good" / f"n{i}.png", i)
    _noise(tmp_path / "test" / "good" / "g0.png", 10)
    _noise(tmp_path / "test" / "broken" / "b0.png", 11)
    config = AnomalyConfig.model_validate(
        {
            "name": "parity_anom",
            "model": {"name": "autoencoder"},
            "data": {
                "base_dir": str(tmp_path),
                "image_size": _SIZE,
                "num_workers": 0,
                "pin_memory": False,
                "transforms": _transforms(augment),
            },
            "training": {"epochs": 1, "batch_size": 2},
        }
    )
    return AnomalyDataModule(config).train_loader().dataset


def _captured_fold_train_ds(
    monkeypatch: pytest.MonkeyPatch, module: Any, run: Callable[[], object]
) -> Dataset:  # type: ignore[type-arg]
    """The training dataset a K-fold hands to ``run_folds``, without training."""
    seen: list[Any] = []

    def stub(context: Any, task: Any) -> None:
        seen.append(context.train_ds)

    monkeypatch.setattr(module, "run_folds", stub)
    run()
    assert len(seen) == 1
    return seen[0]


_PATHS: dict[str, Callable[..., Dataset]] = {  # type: ignore[type-arg]
    "classification": _classification,
    "classification_fold": _classification_fold,
    "regression": _regression,
    "regression_fold": _regression_fold,
    "segmentation": _segmentation,
    "segmentation_fold": _segmentation_fold,
    "anomaly": _anomaly,
}
_NEEDS_MONKEYPATCH = {"regression_fold", "segmentation_fold"}


def _train_reader(
    path: str, tmp_path: Path, augment: bool, monkeypatch: pytest.MonkeyPatch
) -> Reader:
    build = _PATHS[path]
    if path in _NEEDS_MONKEYPATCH:
        dataset = build(tmp_path, augment, monkeypatch)
    else:
        dataset = build(tmp_path, augment)
    return _reader(dataset)


@pytest.mark.parametrize("path", list(_PATHS))
def test_augment_false_reads_the_same_tensors_under_any_seed(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read = _train_reader(path, tmp_path, False, monkeypatch)

    reference = read(_SEEDS[0])
    assert reference  # at least the image tensor
    for seed in _SEEDS[1:]:
        other = read(seed)
        assert len(other) == len(reference)
        for got, want in zip(other, reference, strict=True):
            assert torch.equal(got, want), f"{path}: seed {seed} changed the tensor"


@pytest.mark.parametrize("path", list(_PATHS))
def test_augment_true_varies_the_tensors_across_seeds(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read = _train_reader(path, tmp_path, True, monkeypatch)

    reference = read(_SEEDS[0])
    varied = any(
        not all(torch.equal(g, w) for g, w in zip(read(seed), reference, strict=True))
        for seed in _SEEDS[1:]
    )
    assert varied, f"{path}: augment=True produced the same tensors for every seed"


# ── detection (Ultralytics): arguments, not tensors ───────────────────────────


class _Aug:
    """The detection config's augmentation fields, set to non-neutral values."""

    def __init__(self, *, augment: bool) -> None:
        self.augment = augment
        for key, neutral in _NEUTRAL_AUGMENTATION.items():
            setattr(self, key, "randaugment" if neutral is None else 0.5)


def test_detection_augment_false_sends_every_neutral_value() -> None:
    kwargs = _augmentation_kwargs(_Aug(augment=False))

    assert kwargs == _NEUTRAL_AUGMENTATION
    assert kwargs is not _NEUTRAL_AUGMENTATION  # a copy: callers may mutate it


def test_detection_augment_true_sends_the_configured_values() -> None:
    aug = _Aug(augment=True)

    kwargs = _augmentation_kwargs(aug)

    assert set(kwargs) == set(_NEUTRAL_AUGMENTATION)
    assert all(kwargs[key] != _NEUTRAL_AUGMENTATION[key] for key in kwargs)
    assert kwargs == {key: getattr(aug, key) for key in _NEUTRAL_AUGMENTATION}
