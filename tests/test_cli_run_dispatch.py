from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visionforge.__main__ import build_task_block


def _write(tmp: Path, raw: dict) -> Path:
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def _cls_dataset(tmp: Path) -> str:
    for split in ("train", "val", "test"):
        for cls in ("a", "b"):
            (tmp / split / cls).mkdir(parents=True, exist_ok=True)
    return str(tmp)


class TestBuildTaskBlock:
    def test_classification_dispatches_to_classification_block(
        self, tmp_path: Path
    ) -> None:
        ds = _cls_dataset(tmp_path)
        cfg_path = _write(
            tmp_path,
            {
                "name": "c",
                "task": "multiclass",
                "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
                "data": {"base_dir": ds, "num_workers": 0},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "ClassificationBlock"

    def test_grid_search_block_still_dispatches(self, tmp_path: Path) -> None:
        ds = _cls_dataset(tmp_path)
        cfg_path = _write(
            tmp_path,
            {
                "name": "g",
                "task": "multiclass",
                "block": "grid_search",
                "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
                "data": {"base_dir": ds, "num_workers": 0},
                "grid_search": {"hyperparameters": {"training.learning_rate": [0.1]}},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "GridSearchBlock"

    def test_detection_dispatches_to_detection_block(self, tmp_path: Path) -> None:
        cfg_path = _write(
            tmp_path,
            {
                "name": "d",
                "task": "detection",
                "model": {"backend": "ultralytics", "name": "yolo11n"},
                "data": {"base_dir": str(tmp_path)},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "DetectionBlock"

    def test_regression_dispatches_to_regression_block(self, tmp_path: Path) -> None:
        cfg_path = _write(
            tmp_path,
            {
                "name": "r",
                "task": "regression",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {"base_dir": str(tmp_path), "target_columns": ["target"]},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "RegressionBlock"

    def test_segmentation_dispatches_to_segmentation_block(
        self, tmp_path: Path
    ) -> None:
        cfg_path = _write(
            tmp_path,
            {
                "name": "s",
                "task": "segmentation",
                "model": {"name": "fcn_resnet50", "num_classes": 3},
                "data": {"base_dir": str(tmp_path)},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "SegmentationBlock"

    def test_anomaly_dispatches_to_anomaly_block(self, tmp_path: Path) -> None:
        cfg_path = _write(
            tmp_path,
            {
                "name": "a",
                "task": "anomaly",
                "model": {"name": "autoencoder"},
                "data": {"base_dir": str(tmp_path)},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "AnomalyBlock"

    def test_unknown_task_raises(self, tmp_path: Path) -> None:
        cfg_path = _write(tmp_path, {"name": "x", "task": "teleportation"})
        with pytest.raises(ValueError, match="Unknown task"):
            build_task_block(cfg_path)
