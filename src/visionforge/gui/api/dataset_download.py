"""One-shot dataset download to a local folder (ADR-055).

Provider-based: each provider fetches a dataset into a local directory the existing
data flow can consume, then the user points a task at it. Local-first and
user-initiated — nothing in the core training path touches the network.

torchvision built-ins are materialized into an ImageFolder layout
(``<out>/<split>/<class>/*.png``) so the classification DataModule trains on them
directly. Heavier providers (Roboflow / Kaggle / Hugging Face) land in later slices
as optional lazy extras. Each provider raises a clear error when its extra or
credentials are missing.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# name -> torchvision.datasets class name. All take a ``train`` kwarg and expose
# ``.classes``; items are ``(PIL.Image, int)``.
_TORCHVISION_DATASETS = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "mnist": "MNIST",
    "fashion_mnist": "FashionMNIST",
    "kmnist": "KMNIST",
}


@dataclass
class DatasetDownloadResult:
    """Outcome of a one-shot dataset download."""

    provider: str
    dataset: str
    out_dir: str
    total_images: int
    splits: dict[str, int] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)


def torchvision_datasets() -> list[str]:
    """Names of the supported torchvision built-in datasets."""
    return sorted(_TORCHVISION_DATASETS)


def _materialize(ds: Any, split_dir: Path, class_names: Any, limit: int | None) -> int:
    """Write ``(PIL image, label)`` items into ``split_dir/<class>/<idx>.png``."""
    per_class: dict[str, int] = defaultdict(int)
    written = 0
    for idx in range(len(ds)):
        image, label = ds[idx]
        if class_names and label < len(class_names):
            name = str(class_names[label])
        else:
            name = str(label)
        name = name.replace("/", "_").replace(" ", "_")
        if limit is not None and per_class[name] >= limit:
            continue
        class_dir = split_dir / name
        class_dir.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(class_dir / f"{idx}.png")
        per_class[name] += 1
        written += 1
    return written


def download_torchvision(
    name: str,
    out_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "test"),
    limit: int | None = None,
) -> DatasetDownloadResult:
    """Download a torchvision built-in dataset and materialize it as an ImageFolder.

    ``limit`` caps images per class per split (None = all). The raw download goes to
    a temp dir; only the materialized PNGs are kept under ``out_dir``.

    Raises:
        ValueError: if ``name`` is not a supported torchvision dataset.
    """
    key = name.lower()
    if key not in _TORCHVISION_DATASETS:
        raise ValueError(
            f"Unknown torchvision dataset '{name}'. "
            f"Available: {', '.join(torchvision_datasets())}."
        )
    import torchvision.datasets as tvd

    ds_cls = getattr(tvd, _TORCHVISION_DATASETS[key])
    out = Path(out_dir)
    split_counts: dict[str, int] = {}
    classes: list[str] = []
    total = 0

    with tempfile.TemporaryDirectory() as raw_root:
        for split in splits:
            ds = ds_cls(root=raw_root, train=(split == "train"), download=True)
            ds_classes = list(getattr(ds, "classes", []) or [])
            classes = ds_classes or classes
            count = _materialize(ds, out / split, ds_classes, limit)
            split_counts[split] = count
            total += count
            logger.info("torchvision {} {}: {} images", key, split, count)

    return DatasetDownloadResult(
        provider="torchvision",
        dataset=key,
        out_dir=str(out.resolve()),
        total_images=total,
        splits=split_counts,
        classes=classes,
    )


def download_dataset(
    provider: str,
    *,
    dataset: str,
    out_dir: str,
    splits: tuple[str, ...] = ("train", "test"),
    limit: int | None = None,
    **provider_kwargs: Any,
) -> DatasetDownloadResult:
    """Dispatch a one-shot download to the named provider.

    Raises:
        ValueError: for an unknown or not-yet-implemented provider.
    """
    if provider == "torchvision":
        return download_torchvision(dataset, out_dir, splits=splits, limit=limit)
    if provider in ("roboflow", "kaggle", "huggingface"):
        raise ValueError(
            f"Dataset provider '{provider}' is not implemented yet (coming soon)."
        )
    raise ValueError(
        f"Unknown dataset provider '{provider}' "
        "(use torchvision / roboflow / kaggle / huggingface)."
    )


__all__ = [
    "DatasetDownloadResult",
    "download_dataset",
    "download_torchvision",
    "torchvision_datasets",
]
