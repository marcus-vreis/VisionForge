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


def _carve_validation(train_dir: Path, val_dir: Path, fraction: float) -> int:
    """Move a stratified slice of ``train_dir`` into ``val_dir``.

    Per class, so a rare class keeps representation in both splits, and by
    sorted filename rather than at random, so re-running the download twice
    produces the same split.
    """
    moved = 0
    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        n = int(len(files) * fraction)
        if n == 0:
            continue
        target = val_dir / class_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for path in files[:n]:
            path.rename(target / path.name)
            moved += 1
    return moved


def download_torchvision(
    name: str,
    out_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "test"),
    limit: int | None = None,
    val_fraction: float = 0.2,
) -> DatasetDownloadResult:
    """Download a torchvision built-in dataset and materialize it as an ImageFolder.

    ``limit`` caps images per class per split (None = all). The raw download goes to
    a temp dir; only the materialized PNGs are kept under ``out_dir``.

    torchvision ships these datasets as train/test only, but every VisionForge
    task expects train/val/test — so a downloaded dataset used to land one
    split short and the picker reported "Faltando: validação" on what should be
    the smoothest possible first run. ``val_fraction`` carves the missing split
    out of train; set it to 0 to keep torchvision's original two.

    Raises:
        ValueError: if ``name`` is not a supported torchvision dataset.
    """
    key = name.lower()
    if key not in _TORCHVISION_DATASETS:
        raise ValueError(
            f"Unknown torchvision dataset '{name}'. "
            f"Available: {', '.join(torchvision_datasets())}."
        )
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
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

    if val_fraction > 0 and "train" in split_counts and "val" not in split_counts:
        moved = _carve_validation(out / "train", out / "val", val_fraction)
        if moved:
            split_counts["train"] -= moved
            split_counts["val"] = moved
            logger.info("torchvision {}: carved {} images into val", key, moved)

    return DatasetDownloadResult(
        provider="torchvision",
        dataset=key,
        out_dir=str(out.resolve()),
        total_images=total,
        splits=split_counts,
        classes=classes,
    )


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _count_images(root: Path) -> tuple[int, dict[str, int]]:
    """Count image files under ``root``, grouped by top-level subdir (split)."""
    total = 0
    splits: dict[str, int] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        n = sum(1 for f in sub.rglob("*") if f.suffix.lower() in _IMAGE_EXTS)
        if n:
            splits[sub.name] = n
            total += n
    loose = sum(
        1 for f in root.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
    )
    total += loose
    return total, splits


def download_roboflow(
    dataset: str,
    out_dir: str | Path,
    *,
    api_key: str | None,
    version: int | None = None,
    dataset_format: str = "folder",
) -> DatasetDownloadResult:
    """Download a Roboflow dataset export into ``out_dir``.

    ``dataset`` is ``"workspace/project"``; ``dataset_format`` is the Roboflow export
    format ("folder" gives an ImageFolder-style classification layout, "yolov8" a
    detection layout, etc.). Counts the downloaded images per split.

    Raises:
        ValueError: if api_key/version are missing or ``dataset`` is malformed.
        ImportError: if the optional ``roboflow`` extra is not installed.
    """
    if not api_key:
        raise ValueError("Roboflow requires an api_key.")
    if version is None:
        raise ValueError("Roboflow requires a version number.")
    if "/" not in dataset:
        raise ValueError("Roboflow dataset must be 'workspace/project'.")
    try:
        from roboflow import Roboflow
    except ImportError as exc:  # pragma: no cover - only without the roboflow extra
        raise ImportError(
            'roboflow is not installed. Add the optional extra: pip install -e ".[roboflow]".'
        ) from exc

    workspace, project_name = dataset.split("/", 1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    project.version(version).download(dataset_format, location=str(out))

    total, splits = _count_images(out)
    logger.info("roboflow {} v{}: {} images", dataset, version, total)
    return DatasetDownloadResult(
        provider="roboflow",
        dataset=f"{dataset}:v{version}",
        out_dir=str(out.resolve()),
        total_images=total,
        splits=splits,
        classes=[],
    )


def download_kaggle(dataset: str, out_dir: str | Path) -> DatasetDownloadResult:
    """Download and unzip a Kaggle dataset into ``out_dir``.

    ``dataset`` is ``"owner/dataset-slug"``. Authenticates via ``kaggle.json`` or the
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` env vars. Counts the extracted images.

    Raises:
        ValueError: if ``dataset`` is malformed or credentials are missing.
        ImportError: if the optional ``kaggle`` extra is not installed.
    """
    if "/" not in dataset:
        raise ValueError("Kaggle dataset must be 'owner/dataset-slug'.")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - only without the kaggle extra
        raise ImportError(
            'kaggle is not installed. Add the optional extra: pip install -e ".[kaggle]".'
        ) from exc
    except OSError as exc:  # kaggle auto-authenticates on import; missing creds raise
        raise ValueError(
            "Kaggle credentials not found (set kaggle.json or "
            f"KAGGLE_USERNAME/KAGGLE_KEY): {exc}"
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    try:
        api.authenticate()
    except OSError as exc:
        raise ValueError(
            "Kaggle credentials not found (set kaggle.json or "
            f"KAGGLE_USERNAME/KAGGLE_KEY): {exc}"
        ) from exc
    api.dataset_download_files(dataset, path=str(out), unzip=True)

    total, splits = _count_images(out)
    logger.info("kaggle {}: {} images", dataset, total)
    return DatasetDownloadResult(
        provider="kaggle",
        dataset=dataset,
        out_dir=str(out.resolve()),
        total_images=total,
        splits=splits,
        classes=[],
    )


def _hf_image_label_cols(features: Any) -> tuple[str | None, str | None, list[str]]:
    """Find the image + label columns of a Hugging Face dataset's features.

    Detects feature types by class name (``Image`` / ``ClassLabel``) to avoid a hard
    import of ``datasets`` types, with a fallback to the conventional column names.
    """
    image_col: str | None = None
    label_col: str | None = None
    class_names: list[str] = []
    for name, feat in features.items():
        type_name = type(feat).__name__
        if image_col is None and type_name == "Image":
            image_col = name
        if label_col is None and type_name == "ClassLabel":
            label_col = name
            class_names = list(getattr(feat, "names", []))
    if image_col is None and "image" in features:
        image_col = "image"
    if label_col is None and "label" in features:
        label_col = "label"
    return image_col, label_col, class_names


def _materialize_hf(
    split_ds: Any,
    split_dir: Path,
    image_col: str,
    label_col: str,
    class_names: list[str],
) -> int:
    """Write a Hugging Face split's ``(image, label)`` rows into an ImageFolder."""
    written = 0
    for idx, example in enumerate(split_ds):
        image = example[image_col]
        label = example[label_col]
        if class_names and isinstance(label, int) and label < len(class_names):
            name = class_names[label]
        else:
            name = str(label)
        name = str(name).replace("/", "_").replace(" ", "_")
        class_dir = split_dir / name
        class_dir.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(class_dir / f"{idx}.png")
        written += 1
    return written


def download_huggingface(
    dataset: str, out_dir: str | Path, *, token: str | None = None
) -> DatasetDownloadResult:
    """Download a Hugging Face image dataset and materialize it as an ImageFolder.

    All splits the dataset provides are written to ``<out>/<split>/<class>/*.png``.

    Raises:
        ValueError: if the dataset has no image+label features to materialize.
        ImportError: if the optional ``huggingface`` extra (``datasets``) is missing.
    """
    try:
        from datasets import load_dataset  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            'datasets is not installed. Add the optional extra: pip install -e ".[huggingface]".'
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded = load_dataset(dataset, token=token)
    # DatasetDict is a dict of split->Dataset; a bare Dataset has no .items().
    split_map = dict(loaded.items()) if hasattr(loaded, "items") else {"train": loaded}

    first = next(iter(split_map.values()))
    image_col, label_col, class_names = _hf_image_label_cols(first.features)
    if image_col is None or label_col is None:
        raise ValueError(
            f"Hugging Face dataset '{dataset}' has no image+label features to "
            "materialize into an ImageFolder."
        )

    total = 0
    split_counts: dict[str, int] = {}
    for split, split_ds in split_map.items():
        count = _materialize_hf(
            split_ds, out / split, image_col, label_col, class_names
        )
        split_counts[split] = count
        total += count
        logger.info("huggingface {} {}: {} images", dataset, split, count)

    return DatasetDownloadResult(
        provider="huggingface",
        dataset=dataset,
        out_dir=str(out.resolve()),
        total_images=total,
        splits=split_counts,
        classes=class_names,
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
    if provider == "roboflow":
        return download_roboflow(
            dataset,
            out_dir,
            api_key=provider_kwargs.get("api_key"),
            version=provider_kwargs.get("version"),
            dataset_format=provider_kwargs.get("dataset_format") or "folder",
        )
    if provider == "kaggle":
        return download_kaggle(dataset, out_dir)
    if provider == "huggingface":
        return download_huggingface(
            dataset, out_dir, token=provider_kwargs.get("token")
        )
    raise ValueError(
        f"Unknown dataset provider '{provider}' "
        "(use torchvision / roboflow / kaggle / huggingface)."
    )


__all__ = [
    "DatasetDownloadResult",
    "download_dataset",
    "download_huggingface",
    "download_kaggle",
    "download_roboflow",
    "download_torchvision",
    "torchvision_datasets",
]
