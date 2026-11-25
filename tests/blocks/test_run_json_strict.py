"""A task block's run.json stays valid JSON when a test metric is NaN.

Python's ``json`` writes NaN as the bare token ``NaN``, which a browser's
``JSON.parse`` and ``jq`` refuse; the block writers go through ``dumps_strict``
so an undefined metric lands as ``null`` (ADR-061).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from visionforge.blocks.anomaly import AnomalyBlock
from visionforge.blocks.regression import RegressionBlock
from visionforge.blocks.segmentation import SegmentationBlock


def _refuse_constant(token: str) -> Any:
    raise ValueError(f"non-JSON token {token} in run.json")


@pytest.mark.parametrize(
    ("block_cls", "test_metrics", "nan_key"),
    [
        (RegressionBlock, (0.1, 0.3, 0.2, math.nan), "test_r2"),
        (SegmentationBlock, (math.nan, 0.5, 0.9), "test_miou"),
        (AnomalyBlock, (math.nan, 0.5, 0.8), "test_auroc"),
    ],
)
def test_a_nan_test_metric_is_written_as_null(
    tmp_path: Path,
    block_cls: type[Any],
    test_metrics: tuple[float, ...],
    nan_key: str,
) -> None:
    (tmp_path / "run.json").write_text(
        json.dumps({"metrics": {"best_val_loss": 0.4}, "artifacts": {}}),
        encoding="utf-8",
    )
    # Only the attributes _update_run_json reads; building the block would
    # need a dataset and a model.
    block = object.__new__(block_cls)
    block._test_metrics = test_metrics
    block._metric_cis = {}
    block._test_scores = None

    block._update_run_json(tmp_path, [])

    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    data = json.loads(text, parse_constant=_refuse_constant)
    assert data["metrics"][nan_key] is None
    assert data["metrics"]["best_val_loss"] == pytest.approx(0.4)


def test_the_strict_parse_refuses_the_old_nan_token() -> None:
    """The check above is not vacuous: the raw ``json.dumps`` output fails it."""
    with pytest.raises(ValueError, match="NaN"):
        json.loads(json.dumps({"r2": math.nan}), parse_constant=_refuse_constant)
