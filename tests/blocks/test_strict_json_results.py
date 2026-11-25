"""Result files are JSON for any reader: a NaN metric is written as ``null``.

``json.dumps`` writes ``NaN`` by default, which a strict parser (the browser's
``JSON.parse``, ``jq``, most loaders) refuses. A seed or fold whose metric came
back non-finite is a known outcome (ADR-061), so the summaries that carry it
must stay readable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from visionforge.blocks.cross_validation import CrossValidationBlock
from visionforge.blocks.model_comparison import ModelComparisonBlock
from visionforge.core.run_groups import write_group_summary
from visionforge.utils.config import ExperimentConfig

NAN = float("nan")


def _reject(token: str) -> Any:
    raise ValueError(f"not valid JSON: {token}")


def _strict_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def _config(tmp_path: Path, extra: dict[str, Any], block: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "strict",
            "task": "binary",
            "block": block,
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {"learning_rate": 0.001, "epochs": 1, "batch_size": 2},
            "data": {"base_dir": str(tmp_path), "num_workers": 0},
            "output": {
                "models_dir": str(tmp_path / "models"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
            },
            **extra,
        }
    )


def test_comparison_summary_writes_null_for_a_nan_metric(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        {"model_comparison": {"model_names": ["resnet18", "resnet34"], "metric": "f1"}},
        "model_comparison",
    )
    block = ModelComparisonBlock()
    block.setup(config)
    block._trials = [
        {
            "status": "success",
            "model_arch": "resnet18",
            "effective_learning_rate": 0.001,
            "accuracy": 0.9,
            "f1": NAN,
            "auc_roc": 0.8,
            "training_time_s": 1.0,
        }
    ]
    block._write_artifacts()

    data = _strict_load(tmp_path / "reports" / "strict" / "comparison_summary.json")
    assert data[0]["f1"] is None
    assert data[0]["accuracy"] == 0.9
    # The in-memory trial is untouched: only the file is sanitised.
    assert math.isnan(block._trials[0]["f1"])


def test_cv_summary_writes_null_for_a_nan_fold_metric(tmp_path: Path) -> None:
    config = _config(tmp_path, {"cross_validation": {"n_folds": 2}}, "cross_validation")
    block = CrossValidationBlock()
    block.setup(config)
    block._fold_results = [
        {"status": "success", "accuracy": NAN, "f1": 0.5},
        {"status": "failed", "accuracy": None, "f1": None},
    ]
    block._write_summary("strict_cv")

    data = _strict_load(tmp_path / "reports" / "strict_cv" / "cv_summary.json")
    assert data["folds"][0]["accuracy"] is None
    assert data["aggregate"]["mean_accuracy"] is None


@pytest.mark.parametrize("kind", ["comparison", "replicates"])
def test_group_summary_writes_null_for_a_nan(tmp_path: Path, kind: str) -> None:
    report: dict[str, Any] = {
        "trials": [{"status": "success", "seed": 1, "metrics": {"f1": NAN}}],
        "aggregates": {"f1": {"n": 1, "mean": NAN, "std": NAN}},
        "seeds": [1],
        "comparisons": [{"label_a": "a", "label_b": "b", "p_value": NAN}],
        "non_finite_seeds": {"a": [1]},
    }
    config = {"name": "strict", "output": {"reports_dir": str(tmp_path)}}

    out_dir = Path(write_group_summary(config, kind, report))

    data = _strict_load(out_dir / f"{kind}_summary.json")
    assert data["trials"][0]["metrics"]["f1"] is None
    assert data["aggregates"]["f1"]["mean"] is None
    assert data["comparisons"][0]["p_value"] is None
    assert data["non_finite_seeds"] == {"a": [1]}
