"""`select_on` picks the epoch that scores each task K-fold fold (ADR-242).

Regression and segmentation share ``_cv_common``. The trainer and the model are
stubbed (no weights, no training): ``fit`` leaves the model at the *last* epoch's
value, the checkpoint on disk holds the *best* epoch's value, and ``evaluate``
returns whatever the model holds, so a metric says which weights scored the fold.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tests.blocks.test_regression_cv import _config as regression_config
from tests.blocks.test_segmentation_cv import _config as segmentation_config
from visionforge.blocks import _cv_common
from visionforge.blocks.regression_cv import run_regression_cross_validation
from visionforge.blocks.segmentation_cv import run_segmentation_cross_validation
from visionforge.utils.segmentation_config import SegmentationConfig

_BEST_VALUE = 1.0
_LAST_VALUE = 9.0
_BEST_EPOCH = 2
_EPOCHS = 5


def _model() -> nn.Module:
    return nn.Linear(1, 1, bias=False)


class _FakeTrainer:
    """Trains to ``_LAST_VALUE``; its best epoch is ``_BEST_EPOCH`` of ``_EPOCHS``."""

    def __init__(self, config: Any) -> None:
        self._models_dir = Path(config.output.models_dir)
        # regression scores mse/rmse/mae/r2, segmentation miou/dice/pixel_acc
        self._n_metrics = 3 if isinstance(config, SegmentationConfig) else 4

    def fit(self, model: Any, data: Any, **_: Any) -> SimpleNamespace:
        with torch.no_grad():
            model.weight.fill_(_LAST_VALUE)
        return SimpleNamespace(
            model_path=self._models_dir / "best_model.pth",
            best_epoch=_BEST_EPOCH,
            total_epochs=_EPOCHS,
            stopped=False,
        )

    def evaluate(self, model: Any, loader: Any) -> tuple[float, ...]:
        return (float(model.weight.item()),) * self._n_metrics


@pytest.fixture
def stub_trainers() -> Any:
    with (
        patch("visionforge.blocks.regression_cv.RegressionTrainer", _FakeTrainer),
        patch("visionforge.blocks.segmentation_cv.SegmentationTrainer", _FakeTrainer),
    ):
        yield


_TASKS = [
    pytest.param(
        run_regression_cross_validation,
        "regression_cv",
        "RegressionModelFactory",
        regression_config,
        "r2",
        id="regression",
    ),
    pytest.param(
        run_segmentation_cross_validation,
        "segmentation_cv",
        "SegmentationModelFactory",
        segmentation_config,
        "miou",
        id="segmentation",
    ),
]


@pytest.mark.usefixtures("stub_trainers")
@pytest.mark.parametrize(("runner", "module", "factory", "make", "primary"), _TASKS)
class TestWhichEpochScoresTheFold:
    def _kfold(
        self,
        tmp_path: Path,
        runner: Callable[..., Any],
        module: str,
        factory: str,
        make: Callable[[Path], Any],
        rule: str | None,
    ) -> tuple[Any, Any]:
        kwargs: dict[str, Any] = {} if rule is None else {"select_on": rule}
        best_state = {"weight": torch.full((1, 1), _BEST_VALUE)}
        with (
            patch(
                f"visionforge.blocks.{module}.{factory}.create",
                side_effect=lambda _model_cfg: _model(),
            ),
            patch.object(_cv_common.torch, "load", return_value=best_state) as load,
        ):
            report = runner(make(tmp_path), n_folds=2, **kwargs)
        return report, load

    @pytest.mark.parametrize(
        ("rule", "epoch", "score", "loads"),
        [
            ("best", _BEST_EPOCH, _BEST_VALUE, 2),
            ("last", _EPOCHS, _LAST_VALUE, 0),
        ],
    )
    def test_the_rule_decides_the_weights_and_the_epoch(
        self,
        tmp_path: Path,
        runner: Callable[..., Any],
        module: str,
        factory: str,
        make: Callable[[Path], Any],
        primary: str,
        rule: str,
        epoch: int,
        score: float,
        loads: int,
    ) -> None:
        report, load = self._kfold(tmp_path, runner, module, factory, make, rule)

        assert report.select_on == rule
        assert [f.select_on for f in report.folds] == [rule, rule]
        assert [f.selected_epoch for f in report.folds] == [epoch, epoch]
        assert [f.metrics[primary] for f in report.folds] == [score, score]
        assert load.call_count == loads

    def test_the_default_keeps_scoring_the_best_epoch(
        self,
        tmp_path: Path,
        runner: Callable[..., Any],
        module: str,
        factory: str,
        make: Callable[[Path], Any],
        primary: str,
    ) -> None:
        report, load = self._kfold(tmp_path, runner, module, factory, make, None)

        assert report.select_on == "best"
        assert [f.selected_epoch for f in report.folds] == [_BEST_EPOCH, _BEST_EPOCH]
        assert report.folds[0].metrics[primary] == _BEST_VALUE
        assert load.call_count == 2


@pytest.mark.usefixtures("stub_trainers")
def test_a_fold_that_scored_nothing_has_no_selected_epoch(tmp_path: Path) -> None:
    def boom(self: _FakeTrainer, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    with (
        patch.object(_FakeTrainer, "fit", boom),
        patch(
            "visionforge.blocks.regression_cv.RegressionModelFactory.create",
            side_effect=lambda _model_cfg: _model(),
        ),
    ):
        report = run_regression_cross_validation(
            regression_config(tmp_path), n_folds=2, select_on="last"
        )

    assert [f.status for f in report.folds] == ["failed", "failed"]
    assert [f.selected_epoch for f in report.folds] == [None, None]
    assert report.select_on == "last"
