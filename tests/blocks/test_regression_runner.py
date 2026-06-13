from __future__ import annotations

from typing import Any

import pytest

from visionforge.blocks import regression_runner as rr_mod
from visionforge.blocks.regression_runner import RegressionRunner


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


class TestRegressionRunner:
    def test_success_extracts_test_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = {"test": {"mse": 0.5, "rmse": 0.7, "mae": 0.4, "r2": 0.85}}
        monkeypatch.setattr(rr_mod, "RegressionBlock", lambda: _FakeBlock(report))
        result = RegressionRunner().run(object())
        assert result.status == "success"
        assert result.metrics["r2"] == pytest.approx(0.85)
        assert result.metrics["rmse"] == pytest.approx(0.7)
        assert result.training_time_s is not None

    def test_no_test_set_yields_empty_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            rr_mod, "RegressionBlock", lambda: _FakeBlock({"train": {}})
        )
        result = RegressionRunner().run(object())
        assert result.status == "success"
        assert result.metrics == {}

    def test_failure_is_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rr_mod, "RegressionBlock", lambda: _RaisingBlock())
        result = RegressionRunner().run(object())
        assert result.status == "failed"
        assert "boom" in result.error

    def test_primary_metric(self) -> None:
        assert RegressionRunner().primary_metric() == "r2"
