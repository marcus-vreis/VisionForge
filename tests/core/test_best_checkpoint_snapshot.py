"""The best checkpoint is written from a frozen copy of the weights.

The trainers save the best model in a background thread while training goes
on. ``model.state_dict()`` shares storage with the parameters, so handing it to
that thread lets the next optimizer step change what ends up in
``best_model.pth``. The classification trainer copies first; the regression and
segmentation trainers did not.
"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.core.test_regression_trainer import (
    DummyRegressor,
    FakeRegressionDataModule,
)
from tests.core.test_regression_trainer import _config as _regression_config
from tests.core.test_segmentation_trainer import FakeSegDataModule, TinySegModel
from tests.core.test_segmentation_trainer import _config as _segmentation_config
from visionforge.core import regression_trainer, segmentation_trainer
from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.core.segmentation_trainer import SegmentationTrainer


class _RecordingPool:
    """Runs each job at once and keeps what it was handed, plus the live model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.jobs: list[tuple[Any, ...]] = []
        _POOLS.append(self)

    def submit(self, fn: Any, *args: Any) -> Future[Any]:
        self.jobs.append(args)
        future: Future[Any] = Future()
        future.set_result(fn(*args))
        return future

    def shutdown(self, wait: bool = True) -> None:
        del wait


_POOLS: list[_RecordingPool] = []


def _shares_storage(saved: dict[str, Any], model: nn.Module) -> list[str]:
    """Names of saved tensors whose memory is a live parameter or buffer of ``model``."""
    live = {t.untyped_storage().data_ptr() for t in model.state_dict().values()}
    return [
        name
        for name, value in saved.items()
        if isinstance(value, torch.Tensor)
        and value.untyped_storage().data_ptr() in live
    ]


@pytest.mark.parametrize(
    ("module", "fit"),
    [
        (
            regression_trainer,
            lambda tmp, model: RegressionTrainer(_regression_config(tmp)).fit(
                model, FakeRegressionDataModule()
            ),
        ),
        (
            segmentation_trainer,
            lambda tmp, model: SegmentationTrainer(_segmentation_config(tmp)).fit(
                model, FakeSegDataModule()
            ),
        ),
    ],
    ids=["regression", "segmentation"],
)
def test_best_checkpoint_does_not_alias_the_training_weights(
    module: Any, fit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _POOLS.clear()
    monkeypatch.setattr(module, "ThreadPoolExecutor", _RecordingPool)
    model = DummyRegressor() if module is regression_trainer else TinySegModel(3)

    fit(tmp_path, model)

    saved = [args[0] for pool in _POOLS for args in pool.jobs]
    assert saved, "no best checkpoint was handed to the writer"
    for state in saved:
        assert _shares_storage(state, model) == []
