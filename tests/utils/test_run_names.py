"""Run names are portable directory components in every built-in and custom task."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from visionforge.tasks.base import BaseTaskConfig
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

_CONFIGS = [
    ExperimentConfig,
    DetectionConfig,
    RegressionConfig,
    SegmentationConfig,
    AnomalyConfig,
    BaseTaskConfig,
]


@pytest.mark.parametrize("config_type", _CONFIGS)
@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        ".",
        "..",
        "../escape",
        "a/b",
        "a\\b",
        "C:\\escape",
        "C:escape",
        "bad:name",
        "nul",
        "CON.txt",
        "LPT1",
        "COM1.txt",
        "trailing.",
        "trailing ",
        "bad\x00name",
        "bad\nname",
        "bad?name",
    ],
)
def test_unsafe_names_fail_at_config_boundary(
    tmp_path: Path, config_type: type[BaseModel], name: str
) -> None:
    """Unsafe names fail validation before a task can prepare models or outputs."""
    with pytest.raises(ValidationError) as error:
        config_type.model_validate({"name": name, "data": {"base_dir": str(tmp_path)}})
    assert any(item["loc"] == ("name",) for item in error.value.errors())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("config_type", _CONFIGS)
@pytest.mark.parametrize(
    "name", ["run_01", "Café trial 1", "resnet-18.v2", "model..variant"]
)
def test_valid_names_are_preserved(
    tmp_path: Path, config_type: type[BaseModel], name: str
) -> None:
    """Unicode, interior spaces and ordinary dots do not change stored names."""
    config = config_type.model_validate(
        {"name": name, "data": {"base_dir": str(tmp_path)}}
    )
    assert config.model_dump()["name"] == name


@pytest.mark.parametrize("config_type", _CONFIGS)
def test_default_names_remain_valid(
    tmp_path: Path, config_type: type[BaseModel]
) -> None:
    """Every existing task default remains a valid single directory component."""
    config = config_type.model_validate({"data": {"base_dir": str(tmp_path)}})
    assert config.model_dump()["name"]
