"""The self-test payloads must put the image size where each task's config reads it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig
from visionforge.utils.selftest import build_cases

# Each task keeps the size in a different field, so a payload that sends it to
# the wrong one validates fine and trains at the schema default instead.
_SIZE_GETTERS: dict[str, tuple[type[BaseModel], Callable[[Any], int], int]] = {
    "classification": (ExperimentConfig, lambda c: c.data.transforms.image_size, 32),
    "regression": (RegressionConfig, lambda c: c.data.transforms.image_size, 32),
    "segmentation": (SegmentationConfig, lambda c: c.data.image_size, 64),
    "anomaly": (AnomalyConfig, lambda c: c.data.image_size, 64),
    "detection": (DetectionConfig, lambda c: c.data.image_size, 128),
}


@pytest.mark.parametrize("task", sorted(_SIZE_GETTERS))
def test_simple_payload_trains_at_the_intended_small_size(
    task: str, tmp_path: Path
) -> None:
    config_cls, effective_size, expected = _SIZE_GETTERS[task]
    case = next(c for c in build_cases(tmp_path, (task,), ("simple",)))
    config = config_cls.model_validate(case.payload)
    assert effective_size(config) == expected
