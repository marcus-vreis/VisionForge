from __future__ import annotations

from typing import Any

import pytest

from visionforge.blocks import detection_runner as dr_mod
from visionforge.blocks.detection_runner import DetectionRunner


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


class TestDetectionRunner:
    def test_success_extracts_map_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = {"detection": {"best_map50_95": 0.42, "best_map50": 0.61}}
        monkeypatch.setattr(dr_mod, "DetectionBlock", lambda: _FakeBlock(report))
        result = DetectionRunner().run(object())
        assert result.status == "success"
        assert result.metrics["map50_95"] == pytest.approx(0.42)
        assert result.metrics["map50"] == pytest.approx(0.61)
        assert result.training_time_s is not None

    def test_torchvision_without_map50_95(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = {"detection": {"best_map50_95": None, "best_map50": 0.5}}
        monkeypatch.setattr(dr_mod, "DetectionBlock", lambda: _FakeBlock(report))
        result = DetectionRunner().run(object())
        assert "map50_95" not in result.metrics
        assert result.metrics["map50"] == pytest.approx(0.5)

    def test_failure_is_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dr_mod, "DetectionBlock", lambda: _RaisingBlock())
        result = DetectionRunner().run(object())
        assert result.status == "failed"
        assert "boom" in result.error

    def test_primary_metric(self) -> None:
        assert DetectionRunner().primary_metric() == "map50_95"
