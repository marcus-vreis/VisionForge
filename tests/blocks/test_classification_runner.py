from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from visionforge.core.task_runner import RunResult


def _mock_block_report(accuracy: float, f1: float, auc_roc: float) -> dict[str, Any]:
    return {
        "train": {"best_epoch": 1, "best_val_loss": 0.4, "total_epochs": 1},
        "eval": {
            "accuracy": accuracy,
            "f1": f1,
            "precision": 0.85,
            "recall": 0.80,
            "auc_roc": auc_roc,
        },
    }


class TestClassificationRunnerHappyPath:
    def test_returns_run_result_on_success(self) -> None:
        """run() must return RunResult with status='success' when block succeeds."""
        from visionforge.blocks.classification_runner import ClassificationRunner

        runner = ClassificationRunner()
        fake_cfg = object()

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            return _mock_block_report(0.9, 0.85, 0.93)

        with (
            patch("visionforge.blocks.classification_runner.ClassificationBlock.setup"),
            patch(
                "visionforge.blocks.classification_runner.ClassificationBlock.run",
                mock_run,
            ),
            patch(
                "visionforge.blocks.classification_runner.ClassificationBlock.report",
                mock_report,
            ),
        ):
            result = runner.run(fake_cfg)

        assert isinstance(result, RunResult)
        assert result.status == "success"
        assert result.error == ""

    def test_metrics_populated_from_eval(self) -> None:
        """result.metrics must contain accuracy, f1, auc_roc from block.report()['eval']."""
        from visionforge.blocks.classification_runner import ClassificationRunner

        runner = ClassificationRunner()
        fake_cfg = object()

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            return _mock_block_report(0.91, 0.87, 0.94)

        with (
            patch("visionforge.blocks.classification_runner.ClassificationBlock.setup"),
            patch(
                "visionforge.blocks.classification_runner.ClassificationBlock.run",
                mock_run,
            ),
            patch(
                "visionforge.blocks.classification_runner.ClassificationBlock.report",
                mock_report,
            ),
        ):
            result = runner.run(fake_cfg)

        assert result.metrics["accuracy"] == pytest.approx(0.91)
        assert result.metrics["f1"] == pytest.approx(0.87)
        assert result.metrics["auc_roc"] == pytest.approx(0.94)

    def test_training_time_is_non_negative(self) -> None:
        """result.training_time_s must be a non-negative float on success."""
        from visionforge.blocks.classification_runner import ClassificationRunner

        runner = ClassificationRunner()
        fake_cfg = object()

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            return _mock_block_report(0.8, 0.75, 0.82)

        with (
            patch("visionforge.blocks.classification_runner.ClassificationBlock.setup"),
            patch(
                "visionforge.blocks.classification_runner.ClassificationBlock.run",
                mock_run,
            ),
            patch(
                "visionforge.blocks.classification_runner.ClassificationBlock.report",
                mock_report,
            ),
        ):
            result = runner.run(fake_cfg)

        assert result.training_time_s is not None
        assert result.training_time_s >= 0.0


class TestClassificationRunnerFailurePath:
    def test_returns_failed_result_when_block_raises(self) -> None:
        """run() must return RunResult with status='failed' when ClassificationBlock.run raises."""
        from visionforge.blocks.classification_runner import ClassificationRunner

        runner = ClassificationRunner()
        fake_cfg = object()

        def mock_run(self: Any) -> None:  # noqa: ANN001
            raise RuntimeError("training exploded")

        with (
            patch("visionforge.blocks.classification_runner.ClassificationBlock.setup"),
            patch(
                "visionforge.blocks.classification_runner.ClassificationBlock.run",
                mock_run,
            ),
        ):
            result = runner.run(fake_cfg)

        assert result.status == "failed"
        assert "training exploded" in result.error
        assert result.metrics == {}
        assert result.training_time_s is None

    def test_returns_failed_result_when_setup_raises(self) -> None:
        """run() must return RunResult with status='failed' when setup() raises."""
        from visionforge.blocks.classification_runner import ClassificationRunner

        runner = ClassificationRunner()
        fake_cfg = object()

        with patch(
            "visionforge.blocks.classification_runner.ClassificationBlock.setup",
            side_effect=ValueError("bad config"),
        ):
            result = runner.run(fake_cfg)

        assert result.status == "failed"
        assert "bad config" in result.error


class TestClassificationRunnerInterface:
    def test_primary_metric_returns_accuracy(self) -> None:
        """primary_metric() must return 'accuracy' as the classification default."""
        from visionforge.blocks.classification_runner import ClassificationRunner

        runner = ClassificationRunner()
        assert runner.primary_metric() == "accuracy"

    def test_metrics_returns_result_metrics(self) -> None:
        """metrics() must return result.metrics unchanged."""
        from visionforge.blocks.classification_runner import ClassificationRunner

        runner = ClassificationRunner()
        r = RunResult(
            metrics={"accuracy": 0.9, "f1": 0.85, "auc_roc": 0.92},
            status="success",
            training_time_s=5.0,
            error="",
        )
        assert runner.metrics(r) is r.metrics

    def test_no_gc_or_torch_import(self) -> None:
        """ClassificationRunner must not import gc or torch (cleanup is the caller's job)."""
        import ast
        import pathlib

        import visionforge.blocks.classification_runner as cr_mod

        source_file = pathlib.Path(cr_mod.__file__)  # type: ignore[arg-type]
        if source_file.suffix == ".pyc":
            source_file = source_file.parents[1] / (
                source_file.stem.split(".")[0] + ".py"
            )

        assert source_file.exists(), f"Cannot locate source: {source_file}"
        tree = ast.parse(source_file.read_text(encoding="utf-8"))

        # Collect every top-level module name that is actually imported.
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])

        assert "gc" not in imported, "classification_runner must not import gc"
        assert "torch" not in imported, "classification_runner must not import torch"
