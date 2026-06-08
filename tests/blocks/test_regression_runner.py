from __future__ import annotations

import ast
import pathlib
from typing import Any
from unittest.mock import patch

import pytest

from visionforge.core.task_runner import RunResult


def _mock_block_report(
    mse: float, rmse: float, mae: float, r2: float
) -> dict[str, Any]:
    return {
        "train": {"best_epoch": 1, "best_val_loss": 0.4, "total_epochs": 1},
        "test": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
    }


class TestRegressionRunnerHappyPath:
    def test_metrics_populated_from_test(self) -> None:
        """result.metrics must hold mse/rmse/mae/r2 from block.report()['test']."""
        from visionforge.blocks.regression_runner import RegressionRunner

        runner = RegressionRunner()
        with (
            patch("visionforge.blocks.regression_runner.RegressionBlock.setup"),
            patch("visionforge.blocks.regression_runner.RegressionBlock.run"),
            patch(
                "visionforge.blocks.regression_runner.RegressionBlock.report",
                lambda self: _mock_block_report(0.1, 0.3, 0.2, 0.88),
            ),
        ):
            result = runner.run(object())

        assert result.status == "success"
        assert result.metrics["r2"] == pytest.approx(0.88)
        assert result.metrics["mse"] == pytest.approx(0.1)
        assert result.training_time_s is not None and result.training_time_s >= 0.0

    def test_missing_test_block_yields_empty_metrics(self) -> None:
        """A run without a test split (no 'test' key) still succeeds, metrics empty."""
        from visionforge.blocks.regression_runner import RegressionRunner

        runner = RegressionRunner()
        with (
            patch("visionforge.blocks.regression_runner.RegressionBlock.setup"),
            patch("visionforge.blocks.regression_runner.RegressionBlock.run"),
            patch(
                "visionforge.blocks.regression_runner.RegressionBlock.report",
                lambda self: {"train": {"best_val_loss": 0.4}},
            ),
        ):
            result = runner.run(object())

        assert result.status == "success"
        assert result.metrics == {}


class TestRegressionRunnerFailurePath:
    def test_failure_when_run_raises(self) -> None:
        """run() must return a failed RunResult when the block raises."""
        from visionforge.blocks.regression_runner import RegressionRunner

        runner = RegressionRunner()
        with (
            patch("visionforge.blocks.regression_runner.RegressionBlock.setup"),
            patch(
                "visionforge.blocks.regression_runner.RegressionBlock.run",
                side_effect=RuntimeError("training exploded"),
            ),
        ):
            result = runner.run(object())

        assert result.status == "failed"
        assert "training exploded" in result.error
        assert result.training_time_s is None


class TestRegressionRunnerInterface:
    def test_primary_metric_is_r2(self) -> None:
        from visionforge.blocks.regression_runner import RegressionRunner

        assert RegressionRunner().primary_metric() == "r2"

    def test_metrics_returns_result_metrics(self) -> None:
        from visionforge.blocks.regression_runner import RegressionRunner

        r = RunResult(metrics={"r2": 0.9}, status="success", training_time_s=1.0)
        assert RegressionRunner().metrics(r) is r.metrics

    def test_no_gc_or_torch_import(self) -> None:
        """RegressionRunner must not import gc or torch (cleanup is the caller's job)."""
        import visionforge.blocks.regression_runner as rr_mod

        source = pathlib.Path(rr_mod.__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "gc" not in imported
        assert "torch" not in imported
