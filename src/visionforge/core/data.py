from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from visionforge.core.loader_lifecycle import LoaderCache
from visionforge.core.preprocessing import apply_pipeline
from visionforge.utils.config import (
    ExperimentConfig,
    PreprocessingConfig,
    TransformConfig,
)
from visionforge.utils.workers import suggested_workers

_LOADER_POOLS = 3


class _PreprocessingTransform:
    """Picklable callable that runs a preprocessing pipeline on a PIL image.

    Must be a top-level class (not a closure) so DataLoader workers can pickle
    the dataset transform under the Windows 'spawn' start method. An empty step
    list makes it the identity, so it is safe to use unconditionally.
    """

    def __init__(self, steps: list[dict[str, Any]]) -> None:
        self._steps = steps

    def __call__(self, img: Image.Image) -> Image.Image:
        final, _ = apply_pipeline(img, self._steps)
        return final


def _make_preprocessing_transform(
    config: PreprocessingConfig,
) -> Callable[[Image.Image], Image.Image]:
    """Bind a PreprocessingConfig into a single torchvision-compatible callable.

    Returns an identity transform when the pipeline is empty so the standard
    transforms run untouched in the default case.
    """
    return _PreprocessingTransform([s.model_dump() for s in config.steps])


def _build_transforms(
    config: TransformConfig,
    *,
    is_train: bool,
    preprocessing: PreprocessingConfig | None = None,
) -> T.Compose:
    """Build a transform pipeline.

    Order: preprocessing pipeline → resize/crop → augmentation → ToTensor →
    Normalize. The preprocessing runs first so subsequent random augmentations
    (flip, rotation, jitter) operate on filtered images.

    Args:
        config: transform/augmentation settings.
        is_train: whether to include random augmentation steps.
        preprocessing: optional filter pipeline applied before everything else.

    Returns:
        A composed torchvision transform.
    """
    steps: list[Callable] = []
    if preprocessing is not None and preprocessing.steps:
        steps.append(_make_preprocessing_transform(preprocessing))

    steps += [
        T.Resize(config.image_size),
        T.CenterCrop(config.image_size),
    ]

    if is_train and config.augment:
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
        pp = cfg.preprocessing

        train_path = cfg.base_dir / cfg.train_dir
        val_path = cfg.base_dir / cfg.val_dir
        test_path = cfg.base_dir / cfg.test_dir

        self._batch_size = config.training.batch_size
        self._num_workers = cfg.num_workers
        self._pin_memory = cfg.pin_memory
        self._cache = LoaderCache()

        self._train = ImageFolder(
            str(train_path),
            transform=_build_transforms(tc, is_train=True, preprocessing=pp),
        )
        self._val = ImageFolder(
            str(val_path),
            transform=_build_transforms(tc, is_train=False, preprocessing=pp),
        )
        self._test = ImageFolder(
            str(test_path),
            transform=_build_transforms(tc, is_train=False, preprocessing=pp),
        )

        # Auto-downgrade: multiprocessing workers have a fixed startup +
        # serialisation cost that exceeds the data-loading time on small
        # datasets.  On Windows this cost is especially high because Python
        # uses the "spawn" method.  Disable workers when there are fewer
        # images than a reasonable threshold.
        _SMALL_DATASET_THRESHOLD = 500
        if self._num_workers > 0 and len(self._train) < _SMALL_DATASET_THRESHOLD:
            from loguru import logger

            logger.info(
                "Dataset has {} images (< {}); setting num_workers=0 "
                "to avoid multiprocessing overhead.",
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
        """Common DataLoader kwargs with GPU-friendly defaults.

        When ``num_workers > 0``:
        - ``persistent_workers`` keeps worker processes alive between epochs,
          avoiding expensive re-spawn overhead (especially on Windows). Only the
          loaders that are iterated once per epoch earn it; the test split is
          read once, so a pool that outlives it is pure cost.
        - ``prefetch_factor=2`` pre-loads the next 2×num_workers batches while
          the GPU processes the current batch, eliminating data-loading stalls.
        """
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

    def test_loader(self) -> DataLoader:  # type: ignore[type-arg]
        """DataLoader for the test split."""
        return self._cache.cached(
            "test",
            lambda: DataLoader(
                self._test, shuffle=False, **self._loader_kwargs(persistent=False)
            ),
        )

    def close(self) -> None:
        """Stop every worker pool this module started.

        Callers run inside a long-lived server process, where a failed run's
        traceback keeps the loaders alive and their workers with them.
        """
        self._cache.close()

    @property
    def class_names(self) -> list[str]:
        """Ordered class names from the training dataset."""
        return list(self._train.classes)  # type: ignore[arg-type]


__all__ = ["DataModule"]
