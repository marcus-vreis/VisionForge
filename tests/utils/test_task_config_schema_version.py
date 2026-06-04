from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from visionforge.utils.anomaly_config import AnomalyConfig, load_anomaly_config
from visionforge.utils.config import CURRENT_SCHEMA_VERSION
from visionforge.utils.detection_config import DetectionConfig, load_detection_config
from visionforge.utils.regression_config import (
    RegressionConfig,
    load_regression_config,
)
from visionforge.utils.segmentation_config import (
    SegmentationConfig,
    load_segmentation_config,
)


def _regression(tmp: Path) -> dict[str, Any]:
    return {
        "model": {"name": "resnet18", "num_targets": 1},
        "data": {"base_dir": str(tmp), "target_columns": ["target"]},
    }


def _segmentation(tmp: Path) -> dict[str, Any]:
    return {
        "model": {"name": "fcn_resnet50", "num_classes": 3},
        "data": {"base_dir": str(tmp)},
    }


def _anomaly(tmp: Path) -> dict[str, Any]:
    return {"model": {"name": "autoencoder"}, "data": {"base_dir": str(tmp)}}


def _detection(tmp: Path) -> dict[str, Any]:
    return {
        "model": {"backend": "ultralytics", "name": "yolo11n"},
        "data": {"base_dir": str(tmp)},
    }


# (config class, loader, raw-builder)
_CASES: list[
    tuple[type[BaseModel], Callable[[Path | str], Any], Callable[[Path], dict]]
] = [
    (RegressionConfig, load_regression_config, _regression),
    (SegmentationConfig, load_segmentation_config, _segmentation),
    (AnomalyConfig, load_anomaly_config, _anomaly),
    (DetectionConfig, load_detection_config, _detection),
]

_IDS = ["regression", "segmentation", "anomaly", "detection"]


@pytest.mark.parametrize(("cls", "loader", "build"), _CASES, ids=_IDS)
class TestTaskConfigSchemaVersion:
    def test_defaults_to_current_version(
        self, cls: type[BaseModel], loader: Callable, build: Callable, tmp_path: Path
    ) -> None:
        cfg = cls.model_validate(build(tmp_path))
        assert cfg.model_dump()["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_future_version_rejected(
        self, cls: type[BaseModel], loader: Callable, build: Callable, tmp_path: Path
    ) -> None:
        raw = {**build(tmp_path), "schema_version": CURRENT_SCHEMA_VERSION + 1}
        with pytest.raises(ValidationError, match="newer version"):
            cls.model_validate(raw)

    def test_legacy_yaml_without_version_migrates(
        self, cls: type[BaseModel], loader: Callable, build: Callable, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(build(tmp_path)), encoding="utf-8")
        cfg = loader(cfg_path)
        assert cfg.model_dump()["schema_version"] == CURRENT_SCHEMA_VERSION
