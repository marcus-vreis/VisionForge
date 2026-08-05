"""CSV-manifest dataset and DataLoaders for the image-regression task (Phase 6).

A regression dataset is a per-split CSV (``image,target[,target2,…]``) plus an
images root. Each row maps an image to one or more continuous targets. This
module reuses the classification transform/preprocessing pipeline
(``core.data._build_transforms``) so augmentation and the filter pipeline behave
identically across tasks; only the label changes (a float vector instead of a
class index). See documentation/PHASE6_REGRESSION_PLAN.md.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from visionforge.core.data import _build_transforms
from visionforge.core.loader_lifecycle import LoaderCache
from visionforge.utils.regression_config import RegressionConfig

# Disable DataLoader workers below this many rows: spawn/serialisation overhead
# (high on Windows) exceeds the load time on small sets. Mirrors core.data.
_SMALL_DATASET_THRESHOLD = 500


class RegressionCsvDataset(Dataset):  # type: ignore[type-arg]
    """Image-regression dataset backed by a CSV manifest.

    Reads ``csv_path`` once into memory; each row yields ``(image_tensor,
    target_tensor)`` where the target is a float32 vector of ``target_columns``.
    """

    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        image_column: str,
        target_columns: list[str],
        transform: T.Compose,
    ) -> None:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Regression manifest not found: {csv_path}")

        self._images_root = images_root
        self._transform = transform
        self._target_columns = target_columns
        self._rows: list[tuple[str, list[float]]] = []

        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            missing = [c for c in [image_column, *target_columns] if c not in header]
            if missing:
                raise ValueError(
                    f"Manifest '{csv_path}' is missing column(s): "
                    f"{', '.join(missing)}. Found: {', '.join(header)}."
                )
            for line_no, row in enumerate(reader, start=2):
                image = (row.get(image_column) or "").strip()
                if not image:
                    raise ValueError(
                        f"Manifest '{csv_path}' row {line_no}: empty "
                        f"'{image_column}' value."
                    )
                try:
                    targets = [float(row[c]) for c in target_columns]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Manifest '{csv_path}' row {line_no}: non-numeric "
                        f"target ({exc})."
                    ) from exc
                self._rows.append((image, targets))

        if not self._rows:
            raise ValueError(f"Regression manifest '{csv_path}' has no rows.")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_rel, targets = self._rows[idx]
        image_path = self._images_root / image_rel
        with Image.open(image_path) as img:
            tensor = self._transform(img.convert("RGB"))
        target = torch.tensor(targets, dtype=torch.float32)
        return tensor, target


class RegressionDataModule:
    """Builds CSV-backed regression datasets and their DataLoaders.

    train and val are required; test is optional (``test_loader`` returns None
    when the test CSV is absent).
    """

    def __init__(self, config: RegressionConfig) -> None:
        cfg = config.data
        tc = cfg.transforms
        pp = cfg.preprocessing
        base = cfg.base_dir
        images_root = base / cfg.images_dir

        self._batch_size = config.training.batch_size
        self._num_workers = cfg.num_workers
        self._cache = LoaderCache()
        self._pin_memory = cfg.pin_memory

        def _dataset(csv_name: str, *, is_train: bool) -> RegressionCsvDataset:
            return RegressionCsvDataset(
                base / csv_name,
                images_root,
                cfg.image_column,
                list(cfg.target_columns),
                _build_transforms(tc, is_train=is_train, preprocessing=pp),
            )

        self._train = _dataset(cfg.train_csv, is_train=True)
        self._val = _dataset(cfg.val_csv, is_train=False)
        self._test: RegressionCsvDataset | None = None
        if (base / cfg.test_csv).is_file():
            self._test = _dataset(cfg.test_csv, is_train=False)

        if self._num_workers > 0 and len(self._train) < _SMALL_DATASET_THRESHOLD:
            from loguru import logger

            logger.info(
                "Regression set has {} rows (< {}); setting num_workers=0.",
                len(self._train),
                _SMALL_DATASET_THRESHOLD,
            )
            self._num_workers = 0

    def _loader_kwargs(self, *, persistent: bool) -> dict[str, Any]:
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

    def test_loader(self) -> DataLoader | None:  # type: ignore[type-arg]
        """DataLoader for the test split, or None when no test CSV exists."""
        test = self._test
        if test is None:
            return None
        return self._cache.cached(
            "test",
            lambda: DataLoader(
                test, shuffle=False, **self._loader_kwargs(persistent=False)
            ),
        )

    def close(self) -> None:
        """Stop every worker pool this module started.

        Callers run inside a long-lived server process, where a failed run's
        traceback keeps the loaders alive and their workers with them.
        """
        self._cache.close()


__all__ = ["RegressionDataModule", "RegressionCsvDataset"]
