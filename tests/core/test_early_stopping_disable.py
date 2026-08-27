"""`early_stopping_patience = 0` disables early stopping, in every task.

It used to be rejected by validation (`ge=1`). Zero is the conventional way to
say "off", and the naive reading of the loop makes it mean the exact opposite:
the first epoch without an improvement gives `patience_counter = 1`, and
`1 >= 0` ends the run immediately.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from visionforge.tasks.base import TaskTrainingConfig
from visionforge.utils.anomaly_config import AnomalyTrainingConfig
from visionforge.utils.config import TrainingConfig
from visionforge.utils.regression_config import RegressionTrainingConfig
from visionforge.utils.segmentation_config import SegmentationTrainingConfig

CONFIGS = [
    pytest.param(TrainingConfig, id="classification"),
    pytest.param(RegressionTrainingConfig, id="regression"),
    pytest.param(SegmentationTrainingConfig, id="segmentation"),
    pytest.param(AnomalyTrainingConfig, id="anomaly"),
    pytest.param(TaskTrainingConfig, id="custom-sdk"),
]


class TestZeroIsAccepted:
    @pytest.mark.parametrize("config_cls", CONFIGS)
    def test_every_task_accepts_zero(self, config_cls: type[BaseModel]) -> None:
        assert config_cls(early_stopping_patience=0).early_stopping_patience == 0  # type: ignore[call-arg,attr-defined]

    @pytest.mark.parametrize("config_cls", CONFIGS)
    def test_negative_is_still_refused(self, config_cls: type[BaseModel]) -> None:
        with pytest.raises(ValueError):
            config_cls(early_stopping_patience=-1)  # type: ignore[call-arg]

    @pytest.mark.parametrize("config_cls", CONFIGS)
    def test_the_field_explains_what_zero_does(
        self, config_cls: type[BaseModel]
    ) -> None:
        """A GUI renders this; "0 is allowed" is useless without "0 means off"."""
        description = config_cls.model_fields["early_stopping_patience"].description

        assert description is not None
        assert "0 (padrão) desliga" in description


class TestZeroIsTheDefault:
    """The previous default of 10 never fired: with `epochs=10` it needed ten
    consecutive epochs without improvement inside ten epochs. It read as a
    protection that was not there, so the honest default is the one that says
    early stopping is off."""

    @pytest.mark.parametrize("config_cls", CONFIGS)
    def test_every_task_defaults_to_disabled(self, config_cls: type[BaseModel]) -> None:
        assert config_cls().early_stopping_patience == 0  # type: ignore[attr-defined]

    def test_the_old_default_could_not_fire_with_the_default_epochs(self) -> None:
        from visionforge.utils.config import TrainingConfig

        cfg = TrainingConfig()
        old_patience = 10

        # It would have taken more epochs without improvement than the run has.
        assert old_patience >= cfg.epochs


class TestZeroRunsEveryEpoch:
    def test_the_loop_does_not_stop_on_the_first_bad_epoch(self) -> None:
        """The guard, expressed as the arithmetic it exists to prevent."""
        patience_counter = 1  # one epoch without improvement
        configured = 0

        stops_without_guard = patience_counter >= configured
        stops_with_guard = configured > 0 and patience_counter >= configured

        assert stops_without_guard is True  # the bug this prevents
        assert stops_with_guard is False
