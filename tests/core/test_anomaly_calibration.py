"""The decision threshold must be calibrated where the model is applied.

Training on augmented normals is deliberate. Calibrating the threshold on them
is not: rotation pads corners and flips move structure, so reconstruction error
there is higher than on clean images, the percentile lands above every test
score, and a model with AUROC 0.79 reports F1 = 0.00 (ADR-098).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visionforge.core.anomaly_data import AnomalyDataModule
from visionforge.core.anomaly_trainer import _calibration_loader
from visionforge.utils.anomaly_config import AnomalyConfig


def _dataset(tmp_path: Path) -> Path:
    """Structured images, so a rotation visibly changes the pixels."""
    rng = np.random.default_rng(0)
    for split, classes in (("train", ("good",)), ("test", ("good", "defect"))):
        for cls in classes:
            d = tmp_path / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                arr = np.zeros((64, 64, 3), dtype=np.uint8)
                arr[16:48, 16:48] = 220  # a centred square: rotation moves it
                arr += rng.integers(0, 20, arr.shape, dtype=np.uint8)
                Image.fromarray(arr, "RGB").save(d / f"{i}.png")
    return tmp_path


def _config(base: Path, *, augment: bool) -> AnomalyConfig:
    return AnomalyConfig.model_validate(
        {
            "name": "calib",
            "model": {"name": "autoencoder", "latent_dim": 8},
            "data": {
                "base_dir": str(base),
                "image_size": 64,
                "num_workers": 0,
                "transforms": {
                    "image_size": 64,
                    "augment": augment,
                    "rotation_degrees": 30,
                    "horizontal_flip": True,
                },
            },
            "training": {"epochs": 1, "batch_size": 2, "seed": 0},
            "output": {"models_dir": str(base / "models")},
            "device": {"kind": "cpu"},
        }
    )


class TestCalibrationLoader:
    def test_it_reads_the_same_images_as_training(self, tmp_path: Path) -> None:
        dm = AnomalyDataModule(_config(_dataset(tmp_path), augment=True))

        assert dm.calibration_loader().dataset is not dm.train_loader().dataset
        assert len(dm.calibration_loader().dataset) == len(dm.train_loader().dataset)

    def test_it_does_not_augment(self, tmp_path: Path) -> None:
        """Two passes over the calibration loader must be identical; the
        augmented training loader is what makes that not true of itself."""
        dm = AnomalyDataModule(_config(_dataset(tmp_path), augment=True))

        first = torch.cat([x for x, _ in dm.calibration_loader()])
        second = torch.cat([x for x, _ in dm.calibration_loader()])

        assert torch.equal(first, second)

    def test_the_training_view_really_is_augmented(self, tmp_path: Path) -> None:
        """Guards the premise: without this the test above proves nothing."""
        dm = AnomalyDataModule(_config(_dataset(tmp_path), augment=True))
        loader = dm.train_loader()

        passes = [torch.cat([x for x, _ in loader]).sum() for _ in range(6)]

        assert len({float(p) for p in passes}) > 1

    def test_the_helper_falls_back_for_a_module_without_one(self) -> None:
        class Old:
            def train_loader(self) -> str:
                return "train"

        assert _calibration_loader(Old()) == "train"

    def test_the_helper_prefers_the_calibration_view(self) -> None:
        class New:
            def train_loader(self) -> str:
                return "train"

            def calibration_loader(self) -> str:
                return "calibration"

        assert _calibration_loader(New()) == "calibration"
