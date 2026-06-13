from __future__ import annotations

from typing import Any

from visionforge.core.comparison import run_model_comparison
from visionforge.core.task_runner import RunResult


class _FakeConfig:
    """Stand-in config type whose model_validate just echoes the dict."""

    @classmethod
    def model_validate(cls, d: dict[str, Any]) -> dict[str, Any]:
        return d


class _FakeRunner:
    """Deterministic runner: maps an arch name to a fixed primary-metric value."""

    config_type = _FakeConfig

    def __init__(self, scores: dict[str, float], fail: set[str] | None = None) -> None:
        self._scores = scores
        self._fail = fail or set()
        self.seen: list[str] = []

    def run(self, cfg: dict[str, Any]) -> RunResult:
        arch = cfg["model"]["name"]
        self.seen.append(arch)
        if arch in self._fail:
            return RunResult(status="failed", error="boom")
        return RunResult(
            metrics={"r2": self._scores[arch]},
            status="success",
            training_time_s=1.0,
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return result.metrics

    def primary_metric(self) -> str:
        return "r2"


_BASE: dict[str, Any] = {"name": "exp", "model": {"name": "resnet18"}}


class TestRunModelComparison:
    def test_ranks_successful_by_metric_descending(self) -> None:
        runner = _FakeRunner({"a": 0.3, "b": 0.9, "c": 0.6})
        trials = run_model_comparison(runner, _BASE, ["a", "b", "c"], "r2")
        assert [t.model_arch for t in trials] == ["b", "c", "a"]
        assert all(t.status == "success" for t in trials)

    def test_overrides_model_name_per_arch(self) -> None:
        runner = _FakeRunner({"a": 0.1, "b": 0.2})
        run_model_comparison(runner, _BASE, ["a", "b"], "r2")
        assert runner.seen == ["a", "b"]

    def test_failures_sort_last_and_keep_error(self) -> None:
        runner = _FakeRunner({"a": 0.5, "c": 0.8}, fail={"b"})
        trials = run_model_comparison(runner, _BASE, ["a", "b", "c"], "r2")
        assert [t.model_arch for t in trials] == ["c", "a", "b"]
        assert trials[-1].status == "failed"
        assert trials[-1].error == "boom"

    def test_missing_metric_sorts_as_zero(self) -> None:
        runner = _FakeRunner({"a": -0.2, "b": 0.4})
        # 'a' has a negative r2; it should rank below 'b' but above any failure.
        trials = run_model_comparison(runner, _BASE, ["a", "b"], "r2")
        assert [t.model_arch for t in trials] == ["b", "a"]

    def test_does_not_mutate_base_config(self) -> None:
        runner = _FakeRunner({"a": 0.1})
        base: dict[str, Any] = {"name": "exp", "model": {"name": "resnet18"}}
        run_model_comparison(runner, base, ["a"], "r2")
        assert base["model"]["name"] == "resnet18"
