"""The shipped example configs must stay structurally valid.

Each `configs/<task>.yaml` is a copy-me template; its dataset paths don't exist
on disk, so we override `data.base_dir` to a real tmp dir and validate the rest
through the task's config model. Catches schema drift in the examples (a broken
template is a bad first impression and breaks `visionforge run`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _load_with_base_dir(name: str, base_dir: Path) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load((_CONFIGS / name).read_text(encoding="utf-8"))
    raw["data"]["base_dir"] = str(base_dir)
    return raw


@pytest.mark.parametrize(
    ("filename", "cls", "task"),
    [
        ("detection.yaml", DetectionConfig, "detection"),
        ("regression.yaml", RegressionConfig, "regression"),
        ("segmentation.yaml", SegmentationConfig, "segmentation"),
        ("anomaly.yaml", AnomalyConfig, "anomaly"),
    ],
)
def test_example_config_is_valid(
    filename: str, cls: type[BaseModel], task: str, tmp_path: Path
) -> None:
    raw = _load_with_base_dir(filename, tmp_path)
    cfg = cls.model_validate(raw)
    assert cfg.model_dump()["task"] == task
