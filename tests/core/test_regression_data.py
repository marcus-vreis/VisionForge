from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch
from PIL import Image

from visionforge.core.data import _build_transforms
from visionforge.core.regression_data import (
    RegressionCsvDataset,
    RegressionDataModule,
)
from visionforge.utils.config import TransformConfig
from visionforge.utils.regression_config import RegressionConfig


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (120, 120, 120)).save(path)


@pytest.fixture
def regression_root(tmp_path: Path) -> Path:
    """A CSV-manifest regression dataset with tiny RGB images."""
    images = tmp_path / "images"
    rows = {"train": ["a", "b", "c"], "val": ["d", "e"], "test": ["f"]}
    for split, names in rows.items():
        lines = ["image,target"]
        for i, name in enumerate(names):
            rel = f"{split}/{name}.png"
            _make_image(images / rel)
            lines.append(f"{rel},{float(i)}")
        (tmp_path / f"{split}.csv").write_text("\n".join(lines), encoding="utf-8")
    return tmp_path


def _config(root: Path, overrides: dict | None = None) -> RegressionConfig:
    raw: dict = {
        "name": "reg",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {
            "base_dir": str(root),
            "target_columns": ["target"],
            "num_workers": 0,
            "transforms": {"image_size": 32},
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
    }
    if overrides:
        raw["data"].update(overrides)
    return RegressionConfig.model_validate(raw)


def _transform() -> object:
    return _build_transforms(TransformConfig(image_size=32), is_train=False)


# ── RegressionCsvDataset ──────────────────────────────────────────────────────


class TestRegressionCsvDataset:
    def test_yields_image_and_target_tensors(self, regression_root: Path) -> None:
        ds = RegressionCsvDataset(
            regression_root / "train.csv",
            regression_root / "images",
            "image",
            ["target"],
            _transform(),  # type: ignore[arg-type]
        )
        assert len(ds) == 3
        image, target = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)
        assert target.dtype == torch.float32
        assert target.shape == (1,)

    def test_multi_target_vector(self, tmp_path: Path) -> None:
        _make_image(tmp_path / "images" / "x.png")
        (tmp_path / "train.csv").write_text(
            "image,a,b\nx.png,1.5,2.5\n", encoding="utf-8"
        )
        ds = RegressionCsvDataset(
            tmp_path / "train.csv",
            tmp_path / "images",
            "image",
            ["a", "b"],
            _transform(),  # type: ignore[arg-type]
        )
        _, target = ds[0]
        assert target.tolist() == [1.5, 2.5]

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="manifest not found"):
            RegressionCsvDataset(
                tmp_path / "nope.csv",
                tmp_path,
                "image",
                ["target"],
                _transform(),  # type: ignore[arg-type]
            )

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        (tmp_path / "m.csv").write_text("image,wrong\nx.png,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing column"):
            RegressionCsvDataset(
                tmp_path / "m.csv",
                tmp_path,
                "image",
                ["target"],
                _transform(),  # type: ignore[arg-type]
            )

    def test_non_numeric_target_raises(self, tmp_path: Path) -> None:
        (tmp_path / "m.csv").write_text(
            "image,target\nx.png,notanumber\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="non-numeric target"):
            RegressionCsvDataset(
                tmp_path / "m.csv",
                tmp_path,
                "image",
                ["target"],
                _transform(),  # type: ignore[arg-type]
            )

    def test_empty_manifest_raises(self, tmp_path: Path) -> None:
        (tmp_path / "m.csv").write_text("image,target\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no rows"):
            RegressionCsvDataset(
                tmp_path / "m.csv",
                tmp_path,
                "image",
                ["target"],
                _transform(),  # type: ignore[arg-type]
            )


# ── RegressionDataModule ──────────────────────────────────────────────────────


class TestRegressionDataModule:
    def test_loaders_yield_batches(self, regression_root: Path) -> None:
        dm = RegressionDataModule(_config(regression_root))
        images, targets = next(iter(dm.train_loader()))
        assert images.shape[1:] == (3, 32, 32)
        assert targets.shape[1:] == (1,)
        assert images.shape[0] == targets.shape[0]

    def test_val_loader_present(self, regression_root: Path) -> None:
        dm = RegressionDataModule(_config(regression_root))
        assert sum(len(t) for _, t in dm.val_loader()) == 2

    def test_test_loader_present_when_csv_exists(self, regression_root: Path) -> None:
        dm = RegressionDataModule(_config(regression_root))
        loader = dm.test_loader()
        assert loader is not None
        assert sum(len(t) for _, t in loader) == 1

    def test_test_loader_none_when_absent(self, regression_root: Path) -> None:
        (regression_root / "test.csv").unlink()
        dm = RegressionDataModule(_config(regression_root))
        assert dm.test_loader() is None

    def test_small_dataset_downgrades_workers(self, regression_root: Path) -> None:
        dm = RegressionDataModule(_config(regression_root, {"num_workers": 4}))
        assert dm._num_workers == 0

    def test_dataset_is_picklable(self, regression_root: Path) -> None:
        # DataLoader workers must pickle the dataset under Windows 'spawn'.
        ds = RegressionCsvDataset(
            regression_root / "train.csv",
            regression_root / "images",
            "image",
            ["target"],
            _transform(),  # type: ignore[arg-type]
        )
        restored = pickle.loads(pickle.dumps(ds))
        assert len(restored) == len(ds)
