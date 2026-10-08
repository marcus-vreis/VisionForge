"""Each trainer says whether its loop was cut by the stop (ADR-111).

The orchestrators used to infer "this unit was cut" from the token alone, so a
unit whose stop landed in its last epoch, or in the test evaluation after it,
was labelled stopped and dropped from the aggregate although it had finished.
Only the loop knows whether it broke on the token, so each trainer reports it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.anomaly_trainer import AnomalyTrainer
from visionforge.core.cancellation import CancellationToken
from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.core.segmentation_trainer import SegmentationTrainer
from visionforge.core.trainer import Trainer
from visionforge.models.anomaly_factory import ConvAutoencoder

from .test_resume_trainers import (
    EPOCHS,
    FakeAnomalyData,
    FakeClassificationData,
    FakeRegressionData,
    FakeSegData,
    TinyClassifier,
    TinyRegressor,
    TinySegModel,
    _anomaly_config,
    _classification_config,
    _regression_config,
    _segmentation_config,
)


def _press_stop_after(token: CancellationToken, epoch: int) -> Any:
    def _callback(event: dict[str, Any]) -> None:
        if event.get("event") == "epoch_end" and event.get("epoch") == epoch:
            token.cancel()

    return _callback


def _fit(family: str, tmp_path: Path, stop_after: int) -> Any:
    token = CancellationToken()
    callback = _press_stop_after(token, stop_after)
    if family == "classification":
        return Trainer(_classification_config(tmp_path)).fit(
            TinyClassifier(),
            FakeClassificationData(),
            progress_callback=callback,
            cancel_token=token,
        )
    if family == "regression":
        return RegressionTrainer(_regression_config(tmp_path)).fit(
            TinyRegressor(),
            FakeRegressionData(),
            progress_callback=callback,
            cancel_token=token,
        )
    if family == "segmentation":
        return SegmentationTrainer(_segmentation_config(tmp_path)).fit(
            TinySegModel(),
            FakeSegData(),
            progress_callback=callback,
            cancel_token=token,
        )
    return AnomalyTrainer(_anomaly_config(tmp_path)).fit(
        ConvAutoencoder(latent_dim=8),
        FakeAnomalyData(),
        progress_callback=callback,
        cancel_token=token,
    )


FAMILIES = ["classification", "regression", "segmentation", "anomaly"]


def _run_json(result: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (result.model_path.parent / "run.json").read_text("utf-8")
    )
    return data


@pytest.mark.parametrize("family", FAMILIES)
def test_a_loop_that_broke_on_the_stop_says_so(family: str, tmp_path: Path) -> None:
    result = _fit(family, tmp_path, stop_after=1)

    assert result.total_epochs == 1
    assert result.stopped is True
    # The run-level marker the History and the GUI read (ADR-111).
    assert _run_json(result)["stopped"] is True


@pytest.mark.parametrize("family", FAMILIES)
def test_a_stop_in_the_last_epoch_does_not_cut_the_run(
    family: str, tmp_path: Path
) -> None:
    result = _fit(family, tmp_path, stop_after=EPOCHS)

    assert result.total_epochs == EPOCHS
    assert result.stopped is False
    assert _run_json(result)["stopped"] is False


@pytest.mark.parametrize("family", FAMILIES)
def test_without_a_stop_nothing_is_cut(family: str, tmp_path: Path) -> None:
    result = _fit(family, tmp_path, stop_after=EPOCHS + 1)

    assert result.stopped is False
