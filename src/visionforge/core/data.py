from __future__ import annotations

from collections.abc import Callable

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from visionforge.utils.config import ExperimentConfig, TransformConfig


def _build_transforms(config: TransformConfig, *, is_train: bool) -> T.Compose:
    """Build a transform pipeline from TransformConfig.

    Args:
        config: transform settings.
        is_train: whether to include augmentation steps.

    Returns:
        A composed torchvision transform.
    """
    steps: list[Callable] = [
        T.Resize(config.image_size),
        T.CenterCrop(config.image_size),
    ]

    if is_train:
        if config.horizontal_flip:
            steps.append(T.RandomHorizontalFlip())
        if config.rotation_degrees > 0:
            steps.append(T.RandomRotation(config.rotation_degrees))
        if config.color_jitter:
            steps.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    steps += [
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ]
    return T.Compose(steps)


class DataModule:
    """Wraps ImageFolder datasets and exposes DataLoaders for train/val/test splits."""

    def __init__(self, config: ExperimentConfig) -> None:
        cfg = config.data
        tc = cfg.transforms

        train_path = cfg.base_dir / cfg.train_dir
        val_path = cfg.base_dir / cfg.val_dir
        test_path = cfg.base_dir / cfg.test_dir

        self._batch_size = config.training.batch_size
        self._num_workers = cfg.num_workers
        self._pin_memory = cfg.pin_memory

        self._train = ImageFolder(
            str(train_path), transform=_build_transforms(tc, is_train=True)
        )
        self._val = ImageFolder(
            str(val_path), transform=_build_transforms(tc, is_train=False)
        )
        self._test = ImageFolder(
            str(test_path), transform=_build_transforms(tc, is_train=False)
        )

    def train_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the training split (shuffled)."""
        return DataLoader(
            self._train,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def val_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the validation split."""
        return DataLoader(
            self._val,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def test_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the test split."""
        return DataLoader(
            self._test,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )


__all__ = ["DataModule"]
