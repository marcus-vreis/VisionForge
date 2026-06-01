from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visionforge.core.detection_data import DetectionDataModule
from visionforge.utils.detection_config import DetectionConfig


def _config(tmp_path: Path, data: dict, num_classes: int = 2) -> DetectionConfig:
    return DetectionConfig.model_validate(
        {
            "name": "det",
            "model": {"name": "yolo11n", "num_classes": num_classes},
            "data": data,
            "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.01},
            "output": {"models_dir": str(tmp_path / "models")},
        }
    )


def _make_layout(base: Path, style: str, splits: tuple[str, ...]) -> None:
    """Create a minimal YOLO layout. style='images_split' → images/<split>;
    style='split_images' → <split>/images."""
    for split in splits:
        if style == "images_split":
            (base / "images" / split).mkdir(parents=True, exist_ok=True)
            (base / "labels" / split).mkdir(parents=True, exist_ok=True)
        else:
            (base / split / "images").mkdir(parents=True, exist_ok=True)
            (base / split / "labels").mkdir(parents=True, exist_ok=True)


# ── passthrough ───────────────────────────────────────────────────────────────


class TestPassthrough:
    def test_explicit_data_yaml_is_returned_as_is(self, tmp_path: Path) -> None:
        dy = tmp_path / "data.yaml"
        dy.write_text("path: .\ntrain: images\nval: images\nnames: [a]\n", "utf-8")
        cfg = _config(tmp_path, {"data_yaml": str(dy)})
        out = DetectionDataModule(cfg).resolve_data_yaml()
        assert out == dy


# ── synthesis ─────────────────────────────────────────────────────────────────


class TestSynthesis:
    def test_images_split_layout(self, tmp_path: Path) -> None:
        base = tmp_path / "ds"
        _make_layout(base, "images_split", ("train", "val"))
        cfg = _config(tmp_path, {"base_dir": str(base)})
        out = DetectionDataModule(cfg).resolve_data_yaml()

        assert out.exists()
        spec = yaml.safe_load(out.read_text("utf-8"))
        assert spec["train"] == "images/train"
        assert spec["val"] == "images/val"
        assert Path(spec["path"]) == base.resolve()

    def test_split_images_layout(self, tmp_path: Path) -> None:
        base = tmp_path / "ds"
        _make_layout(base, "split_images", ("train", "val"))
        cfg = _config(tmp_path, {"base_dir": str(base)})
        spec = yaml.safe_load(
            DetectionDataModule(cfg).resolve_data_yaml().read_text("utf-8")
        )
        assert spec["train"] == "train/images"
        assert spec["val"] == "val/images"

    def test_test_split_included_when_present(self, tmp_path: Path) -> None:
        base = tmp_path / "ds"
        _make_layout(base, "images_split", ("train", "val", "test"))
        cfg = _config(tmp_path, {"base_dir": str(base)})
        spec = yaml.safe_load(
            DetectionDataModule(cfg).resolve_data_yaml().read_text("utf-8")
        )
        assert spec["test"] == "images/test"

    def test_test_split_omitted_when_absent(self, tmp_path: Path) -> None:
        base = tmp_path / "ds"
        _make_layout(base, "images_split", ("train", "val"))
        cfg = _config(tmp_path, {"base_dir": str(base)})
        spec = yaml.safe_load(
            DetectionDataModule(cfg).resolve_data_yaml().read_text("utf-8")
        )
        assert "test" not in spec

    def test_missing_val_raises(self, tmp_path: Path) -> None:
        base = tmp_path / "ds"
        _make_layout(base, "images_split", ("train",))
        cfg = _config(tmp_path, {"base_dir": str(base)})
        with pytest.raises(ValueError, match="val"):
            DetectionDataModule(cfg).resolve_data_yaml()


# ── class names ───────────────────────────────────────────────────────────────


class TestClassNames:
    def test_names_from_config(self, tmp_path: Path) -> None:
        base = tmp_path / "ds"
        _make_layout(base, "images_split", ("train", "val"))
        cfg = _config(
            tmp_path,
            {"base_dir": str(base), "class_names": ["cat", "dog"]},
            num_classes=2,
        )
        spec = yaml.safe_load(
            DetectionDataModule(cfg).resolve_data_yaml().read_text("utf-8")
        )
        assert list(spec["names"]) == ["cat", "dog"]
        assert spec["nc"] == 2

    def test_names_from_classes_txt(self, tmp_path: Path) -> None:
        base = tmp_path / "ds"
        _make_layout(base, "images_split", ("train", "val"))
        (base / "classes.txt").write_text("car\nbike\nbus\n", "utf-8")
        cfg = _config(tmp_path, {"base_dir": str(base)}, num_classes=3)
        spec = yaml.safe_load(
            DetectionDataModule(cfg).resolve_data_yaml().read_text("utf-8")
        )
        assert list(spec["names"]) == ["car", "bike", "bus"]

    def test_generated_names_from_num_classes(self, tmp_path: Path) -> None:
        base = tmp_path / "ds"
        _make_layout(base, "images_split", ("train", "val"))
        cfg = _config(tmp_path, {"base_dir": str(base)}, num_classes=2)
        spec = yaml.safe_load(
            DetectionDataModule(cfg).resolve_data_yaml().read_text("utf-8")
        )
        assert list(spec["names"]) == ["class_0", "class_1"]


# ── output location ───────────────────────────────────────────────────────────


class TestOutputLocation:
    def test_writes_into_explicit_out_dir(self, tmp_path: Path) -> None:
        base = tmp_path / "ds"
        _make_layout(base, "images_split", ("train", "val"))
        cfg = _config(tmp_path, {"base_dir": str(base)})
        out_dir = tmp_path / "elsewhere"
        out = DetectionDataModule(cfg).resolve_data_yaml(out_dir=out_dir)
        assert out.parent == out_dir
        assert out.exists()
