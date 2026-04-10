from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionforge.core.data import DataModule, _build_transforms
from visionforge.utils.config import ExperimentConfig, TransformConfig


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Minimal ImageFolder structure with tiny RGB images."""
    for split in ["train", "val", "test"]:
        for cls in ["class_a", "class_b"]:
            folder = tmp_path / split / cls
            folder.mkdir(parents=True)
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(folder / "image.png")
    return tmp_path


@pytest.fixture
def binary_config(dataset_root: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "test",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "seed": 42,
            },
            "data": {
                "base_dir": str(dataset_root),
                "num_workers": 0,
                "pin_memory": False,
            },
        }
    )


class TestBuildTransforms:
    def test_train_includes_augmentation_by_default(self) -> None:
        """Default TransformConfig must include RandomHorizontalFlip in train."""
        tc = TransformConfig()
        transform = _build_transforms(tc, is_train=True)
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomHorizontalFlip" in names

    def test_val_excludes_augmentation(self) -> None:
        """Val transform must not include any augmentation."""
        tc = TransformConfig()
        transform = _build_transforms(tc, is_train=False)
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomHorizontalFlip" not in names
        assert "RandomRotation" not in names

    def test_color_jitter_disabled_by_default(self) -> None:
        """ColorJitter must be absent when color_jitter=False."""
        tc = TransformConfig(color_jitter=False)
        transform = _build_transforms(tc, is_train=True)
        names = [type(t).__name__ for t in transform.transforms]
        assert "ColorJitter" not in names

    def test_color_jitter_enabled(self) -> None:
        """ColorJitter must be present when color_jitter=True."""
        tc = TransformConfig(color_jitter=True)
        transform = _build_transforms(tc, is_train=True)
        names = [type(t).__name__ for t in transform.transforms]
        assert "ColorJitter" in names

    def test_no_rotation_when_degrees_zero(self) -> None:
        """RandomRotation must be absent when rotation_degrees=0."""
        tc = TransformConfig(rotation_degrees=0)
        transform = _build_transforms(tc, is_train=True)
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomRotation" not in names


class TestDataModule:
    def test_loaders_are_created(self, binary_config: ExperimentConfig) -> None:
        """DataModule must expose train, val and test loaders."""
        dm = DataModule(binary_config)
        assert dm.train_loader() is not None
        assert dm.val_loader() is not None
        assert dm.test_loader() is not None

    def test_train_loader_batch_size(self, binary_config: ExperimentConfig) -> None:
        """Batch size from config must be respected in train loader."""
        dm = DataModule(binary_config)
        batch = next(iter(dm.train_loader()))
        inputs, _ = batch
        assert inputs.shape[0] <= binary_config.training.batch_size

    def test_val_loader_not_shuffled(self, binary_config: ExperimentConfig) -> None:
        """Val loader must have shuffle=False."""
        dm = DataModule(binary_config)
        loader = dm.val_loader()
        assert loader.sampler.__class__.__name__ == "SequentialSampler"

    def test_test_loader_not_shuffled(self, binary_config: ExperimentConfig) -> None:
        """Test loader must have shuffle=False."""
        dm = DataModule(binary_config)
        loader = dm.test_loader()
        assert loader.sampler.__class__.__name__ == "SequentialSampler"

    def test_class_names_returns_sorted_classes(
        self, binary_config: ExperimentConfig
    ) -> None:
        """class_names must return sorted class names from the training dataset."""
        dm = DataModule(binary_config)
        names = dm.class_names
        assert names == ["class_a", "class_b"]

    def test_train_loader_is_shuffled(self, binary_config: ExperimentConfig) -> None:
        """Train loader must use RandomSampler (shuffle=True)."""
        dm = DataModule(binary_config)
        loader = dm.train_loader()
        assert loader.sampler.__class__.__name__ == "RandomSampler"
