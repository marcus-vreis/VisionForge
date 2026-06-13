from __future__ import annotations

from typing import Any

import pytest

from visionforge.blocks import segmentation_runner as sr_mod
from visionforge.blocks.segmentation_runner import SegmentationRunner


class _FakeBlock:
    def __init__(self, report: dict[str, Any]) -> None:
        self._report = report

    def setup(self, cfg: Any) -> None:
        pass

    def run(self) -> None:
        pass

    def report(self) -> dict[str, Any]:
        return self._report


class _RaisingBlock:
    def setup(self, cfg: Any) -> None:
        raise RuntimeError("boom")


class TestSegmentationRunner:
    def test_success_extracts_test_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = {"test": {"miou": 0.62, "dice": 0.71, "pixel_acc": 0.93}}
        monkeypatch.setattr(sr_mod, "SegmentationBlock", lambda: _FakeBlock(report))
        result = SegmentationRunner().run(object())
        assert result.status == "success"
        assert result.metrics["miou"] == pytest.approx(0.62)
        assert result.metrics["pixel_acc"] == pytest.approx(0.93)
        assert result.training_time_s is not None

    def test_no_test_set_yields_empty_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sr_mod, "SegmentationBlock", lambda: _FakeBlock({"train": {}})
        )
        result = SegmentationRunner().run(object())
        assert result.status == "success"
        assert result.metrics == {}

    def test_failure_is_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sr_mod, "SegmentationBlock", lambda: _RaisingBlock())
        result = SegmentationRunner().run(object())
        assert result.status == "failed"
        assert "boom" in result.error

    def test_primary_metric(self) -> None:
        assert SegmentationRunner().primary_metric() == "miou"
