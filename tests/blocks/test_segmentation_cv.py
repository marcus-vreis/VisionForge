from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionforge.blocks.segmentation_cv import run_segmentation_cross_validation
from visionforge.utils.segmentation_config import SegmentationConfig

_N_PAIRS = 6


def _make_dataset(tmp_path: Path) -> Path:
    base = tmp_path / "ds"
    images = base / "train" / "images"
    masks = base / "train" / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for i in range(_N_PAIRS):
        Image.new("RGB", (32, 32), color=(i * 10, 20, 30)).save(images / f"s{i}.png")
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1
        Image.fromarray(mask, "L").save(masks / f"s{i}.png")
    return base


def _config(tmp_path: Path) -> SegmentationConfig:
    base = _make_dataset(tmp_path)
    return SegmentationConfig.model_validate(
        {
            "name": "seg_cv",
            "model": {"name": "unet", "num_classes": 2, "pretrained": False},
            "data": {"base_dir": str(base), "image_size": 32},
            "training": {"epochs": 1, "batch_size": 2},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


class TestSegmentationCrossValidation:
    def test_two_fold_cv_runs_and_aggregates(self, tmp_path: Path) -> None:
        report = run_segmentation_cross_validation(
            _config(tmp_path), n_folds=2, shuffle=True, seed=42
        )

        assert report.n_folds == 2
        assert report.metric == "miou"
        assert len(report.folds) == 2
        assert all(f.status == "success" for f in report.folds), [
            f.error for f in report.folds
        ]

        # the folds partition all pairs with no leakage
        assert sum(f.val_size for f in report.folds) == _N_PAIRS
        assert all(f.train_size + f.val_size == _N_PAIRS for f in report.folds)

        # every segmentation metric is aggregated to mean ± std
        for name in ("miou", "dice", "pixel_acc"):
            assert name in report.aggregate
            assert {"mean", "std"} <= report.aggregate[name].keys()

    def test_rejects_too_few_folds(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            run_segmentation_cross_validation(_config(tmp_path), n_folds=1)

    def test_rejects_more_folds_than_pairs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exceed"):
            run_segmentation_cross_validation(_config(tmp_path), n_folds=_N_PAIRS + 1)
