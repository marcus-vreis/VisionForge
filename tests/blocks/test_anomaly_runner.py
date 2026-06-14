from __future__ import annotations

from typing import Any

import pytest

from visionforge.blocks import anomaly_runner as ar_mod
from visionforge.blocks.anomaly_runner import AnomalyRunner


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


class TestAnomalyRunner:
    def test_success_extracts_test_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = {"test": {"auroc": 0.93, "image_f1": 0.81, "threshold": 0.5}}
        monkeypatch.setattr(ar_mod, "AnomalyBlock", lambda: _FakeBlock(report))
        result = AnomalyRunner().run(object())
        assert result.status == "success"
        assert result.metrics["auroc"] == pytest.approx(0.93)
        assert result.metrics["image_f1"] == pytest.approx(0.81)
        # threshold is a decision cut-off, not a ranking metric — excluded.
        assert "threshold" not in result.metrics
        assert result.training_time_s is not None

    def test_no_test_set_yields_empty_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ar_mod, "AnomalyBlock", lambda: _FakeBlock({"train": {}}))
        result = AnomalyRunner().run(object())
        assert result.status == "success"
        assert result.metrics == {}

    def test_failure_is_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ar_mod, "AnomalyBlock", lambda: _RaisingBlock())
        result = AnomalyRunner().run(object())
        assert result.status == "failed"
        assert "boom" in result.error

    def test_primary_metric(self) -> None:
        assert AnomalyRunner().primary_metric() == "auroc"
