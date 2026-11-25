"""The classification K-fold drops the trailing partial batch, like the task K-fold.

With N = 82 and 5 folds the train folds hold 65 or 66 images. At batch 32 a fold
of 65 ends in a batch of one; a stride-32 backbone at image_size 32 hands
BatchNorm a [1, C, 1, 1] tensor, which it refuses in train mode. Those folds
failed and the K-fold averaged only the survivors. ``_cv_common`` (regression
and segmentation K-fold, ADR-050) already dropped the partial batch whenever the
fold holds more than one; both now share ``fold_drop_last``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

from visionforge.blocks._cv_common import fold_drop_last
from visionforge.blocks.cross_validation import _FoldDataModule
from visionforge.utils.config import ExperimentConfig


def _dataset(root: Path, per_class: int) -> Path:
    for cls in ("a", "b"):
        folder = root / "train" / cls
        folder.mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (8, 8), (i, 0, 0)).save(folder / f"{i}.png")
    return root


def _config(base: Path, tmp_path: Path, batch_size: int) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "kfold_last_batch",
            "block": "cross_validation",
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "training": {"epochs": 1, "batch_size": batch_size, "seed": 0},
            "data": {
                "base_dir": str(base),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32, "augment": False},
            },
            "output": {
                "models_dir": str(tmp_path / "models"),
                "reports_dir": str(tmp_path / "reports"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
            },
            "device": {"kind": "cpu"},
            "cross_validation": {"n_folds": 5},
        }
    )


def _fold(tmp_path: Path, n_train: int, batch_size: int) -> _FoldDataModule:
    base = _dataset(tmp_path / "ds", per_class=41)  # 82 images, as in the finding
    dataset = ImageFolder(str(base / "train"))
    return _FoldDataModule(
        Subset(dataset, list(range(n_train))),
        Subset(dataset, list(range(n_train, 82))),
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        _config(base, tmp_path, batch_size),
    )


class TestFoldTrainLoader:
    def test_trailing_batch_of_one_is_dropped(self, tmp_path: Path) -> None:
        fold = _fold(tmp_path, n_train=65, batch_size=32)

        sizes = [images.shape[0] for images, _ in fold.train_loader()]

        assert sizes == [32, 32]

    def test_resnet18_at_32px_trains_through_every_batch(self, tmp_path: Path) -> None:
        """The real failure: BatchNorm on the [1, 512, 1, 1] output of layer4."""
        fold = _fold(tmp_path, n_train=65, batch_size=32)
        model = torchvision.models.resnet18(weights=None, num_classes=2)
        model.train()

        with torch.no_grad():
            for images, _ in fold.train_loader():
                model(images)

    def test_a_fold_smaller_than_one_batch_keeps_its_images(
        self, tmp_path: Path
    ) -> None:
        fold = _fold(tmp_path, n_train=5, batch_size=32)

        sizes = [images.shape[0] for images, _ in fold.train_loader()]

        assert sizes == [5]

    def test_validation_loader_keeps_every_image(self, tmp_path: Path) -> None:
        fold = _fold(tmp_path, n_train=65, batch_size=32)

        assert sum(images.shape[0] for images, _ in fold.val_loader()) == 17


class TestFoldDropLast:
    @pytest.mark.parametrize(
        ("n_train", "batch_size", "expected"),
        [(65, 32, True), (66, 32, True), (32, 32, False), (5, 32, False)],
    )
    def test_rule(self, n_train: int, batch_size: int, expected: bool) -> None:
        assert fold_drop_last(n_train, batch_size) is expected
