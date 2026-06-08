from __future__ import annotations

import ast
import pathlib
from typing import Any
from unittest.mock import patch

import pytest


def _mock_block_report(miou: float, dice: float, pixel_acc: float) -> dict[str, Any]:
    return {
        "train": {"best_epoch": 1, "best_val_miou": 0.6, "total_epochs": 1},
        "test": {"miou": miou, "dice": dice, "pixel_acc": pixel_acc},
    }


class TestSegmentationRunnerHappyPath:
    def test_metrics_populated_from_test(self) -> None:
        """result.metrics must hold miou/dice/pixel_acc from report()['test']."""
        from visionforge.blocks.segmentation_runner import SegmentationRunner

        runner = SegmentationRunner()
        with (
            patch("visionforge.blocks.segmentation_runner.SegmentationBlock.setup"),
            patch("visionforge.blocks.segmentation_runner.SegmentationBlock.run"),
            patch(
                "visionforge.blocks.segmentation_runner.SegmentationBlock.report",
                lambda self: _mock_block_report(0.72, 0.83, 0.95),
            ),
        ):
            result = runner.run(object())

        assert result.status == "success"
        assert result.metrics["miou"] == pytest.approx(0.72)
        assert result.metrics["pixel_acc"] == pytest.approx(0.95)
        assert result.training_time_s is not None and result.training_time_s >= 0.0


class TestSegmentationRunnerFailurePath:
    def test_failure_when_setup_raises(self) -> None:
        """run() must return a failed RunResult when setup raises."""
        from visionforge.blocks.segmentation_runner import SegmentationRunner

        runner = SegmentationRunner()
        with patch(
            "visionforge.blocks.segmentation_runner.SegmentationBlock.setup",
            side_effect=ValueError("bad config"),
        ):
            result = runner.run(object())

        assert result.status == "failed"
        assert "bad config" in result.error


class TestSegmentationRunnerInterface:
    def test_primary_metric_is_miou(self) -> None:
        from visionforge.blocks.segmentation_runner import SegmentationRunner

        assert SegmentationRunner().primary_metric() == "miou"

    def test_no_gc_or_torch_import(self) -> None:
        """SegmentationRunner must not import gc or torch (cleanup is the caller's job)."""
        import visionforge.blocks.segmentation_runner as sr_mod

        source = pathlib.Path(sr_mod.__file__)
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
