"""`-1` means "decide for me", and the decision is about memory.

Scaling workers with the GPU or the model is the intuitive heuristic and the
wrong one: a worker never holds the model — it loads images while the model
sits on the GPU — so what it costs is commit, about a gigabyte each on Windows
(ADR-098/103).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from visionforge.utils.anomaly_config import AnomalyDataConfig
from visionforge.utils.config import DataConfig
from visionforge.utils.detection_config import DetectionTrainingConfig
from visionforge.utils.regression_config import RegressionDataConfig
from visionforge.utils.segmentation_config import SegmentationDataConfig

WORKER_FIELDS = [
    pytest.param(DataConfig, "num_workers", id="classification"),
    pytest.param(RegressionDataConfig, "num_workers", id="regression"),
    pytest.param(SegmentationDataConfig, "num_workers", id="segmentation"),
    pytest.param(AnomalyDataConfig, "num_workers", id="anomaly"),
    pytest.param(DetectionTrainingConfig, "workers", id="detection"),
]


def _minimal(
    config_cls: type[BaseModel], extra: dict[str, object]
) -> dict[str, object]:
    """The payload plus whatever else this config demands.

    The data configs require a `base_dir` (and regression a target column); the
    detection training config requires neither. Filling only what is mandatory
    keeps each test about the worker field.
    """
    payload = dict(extra)
    if "base_dir" in config_cls.model_fields:
        payload["base_dir"] = "."
    if "target_columns" in config_cls.model_fields:
        payload["target_columns"] = ["target"]
    return payload


class TestAutomaticIsTheDefault:
    @pytest.mark.parametrize(("config_cls", "field"), WORKER_FIELDS)
    def test_every_task_defaults_to_automatic(
        self, config_cls: type[BaseModel], field: str
    ) -> None:
        assert config_cls.model_fields[field].default == -1

    @pytest.mark.parametrize(("config_cls", "field"), WORKER_FIELDS)
    def test_zero_is_still_valid(self, config_cls: type[BaseModel], field: str) -> None:
        """Zero has its own meaning: load in the main process."""
        model = config_cls.model_validate(_minimal(config_cls, {field: 0}))

        assert getattr(model, field) == 0

    @pytest.mark.parametrize(("config_cls", "field"), WORKER_FIELDS)
    def test_below_minus_one_is_refused(
        self, config_cls: type[BaseModel], field: str
    ) -> None:
        with pytest.raises(ValueError):
            config_cls.model_validate(_minimal(config_cls, {field: -2}))

    @pytest.mark.parametrize(("config_cls", "field"), WORKER_FIELDS)
    def test_the_field_says_what_minus_one_does(
        self, config_cls: type[BaseModel], field: str
    ) -> None:
        description = config_cls.model_fields[field].description

        assert description is not None
        assert "-1" in description


class TestResolution:
    def test_the_automatic_value_comes_from_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.utils import workers as w

        monkeypatch.setattr(w, "available_commit_bytes", lambda: 16 * 1024**3)
        monkeypatch.setattr(w.os, "cpu_count", lambda: 32)

        # Half of 16 GB at 1 GB a worker, split across three loader pools.
        assert w.suggested_workers(loader_pools=3) == 2

    def test_a_starved_machine_resolves_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.utils import workers as w

        monkeypatch.setattr(w, "available_commit_bytes", lambda: 1 * 1024**3)
        monkeypatch.setattr(w.os, "cpu_count", lambda: 32)

        assert w.suggested_workers(loader_pools=3) == 0
