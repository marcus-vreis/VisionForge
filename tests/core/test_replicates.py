from __future__ import annotations

import math
from typing import Any

import pytest

from visionforge.core.replicates import (
    ReplicateTrial,
    aggregate_replicates,
    run_replicates,
)
from visionforge.core.task_runner import RunResult


class _FakeConfig:
    """Stand-in config type whose model_validate just echoes the dict."""

    @classmethod
    def model_validate(cls, d: dict[str, Any]) -> dict[str, Any]:
        return d


class _FakeRunner:
    """Runner whose score is a deterministic function of the seed."""

    config_type = _FakeConfig

    def __init__(self, fail_seed: int | None = None) -> None:
        self._fail_seed = fail_seed
        self.seen: list[dict[str, Any]] = []

    def run(self, cfg: dict[str, Any]) -> RunResult:
        self.seen.append(cfg)
        seed = cfg["training"]["seed"]
        if self._fail_seed is not None and seed == self._fail_seed:
            return RunResult(status="failed", error="boom")
        return RunResult(
            metrics={"score": seed / 100.0},
            status="success",
            training_time_s=0.1,
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return result.metrics

    def primary_metric(self) -> str:
        return "score"


def _base() -> dict[str, Any]:
    return {
        "name": "rep",
        "model": {"name": "resnet18"},
        "training": {"learning_rate": 0.01, "seed": 42},
    }


class TestRunReplicates:
    def test_runs_one_trial_per_seed_in_seed_order(self) -> None:
        runner = _FakeRunner()
        trials = run_replicates(runner, _base(), [42, 43, 44], metric="score")
        assert [t.seed for t in trials] == [42, 43, 44]
        assert all(t.status == "success" for t in trials)
        assert [c["training"]["seed"] for c in runner.seen] == [42, 43, 44]

    def test_each_replicate_gets_a_distinct_name(self) -> None:
        runner = _FakeRunner()
        run_replicates(runner, _base(), [1, 2], metric="score")
        assert [c["name"] for c in runner.seen] == ["rep_s1", "rep_s2"]

    def test_base_config_is_not_mutated(self) -> None:
        base = _base()
        run_replicates(_FakeRunner(), base, [7, 8], metric="score")
        assert base["name"] == "rep"
        assert base["training"]["seed"] == 42

    def test_failed_seed_is_isolated_and_set_continues(self) -> None:
        trials = run_replicates(
            _FakeRunner(fail_seed=43), _base(), [42, 43, 44], metric="score"
        )
        assert [t.status for t in trials] == ["success", "failed", "success"]
        assert trials[1].error == "boom"
        assert trials[1].metrics == {}

    def test_trials_keep_seed_order_not_metric_ranking(self) -> None:
        # Higher seed → higher score; order must still follow the seed list.
        trials = run_replicates(_FakeRunner(), _base(), [90, 10, 50], metric="score")
        assert [t.seed for t in trials] == [90, 10, 50]


class TestAggregateReplicates:
    def _trial(self, seed: int, score: float | None) -> ReplicateTrial:
        if score is None:
            return ReplicateTrial(seed=seed, status="failed", error="x")
        return ReplicateTrial(seed=seed, status="success", metrics={"score": score})

    def test_mean_std_min_max(self) -> None:
        trials = [self._trial(i, v) for i, v in enumerate([0.8, 0.9, 1.0])]
        agg = aggregate_replicates(trials)["score"]
        assert agg["n"] == 3
        assert agg["mean"] == pytest.approx(0.9)
        assert agg["std"] == pytest.approx(0.1)
        assert agg["min"] == 0.8
        assert agg["max"] == 1.0

    def test_ci95_uses_student_t(self) -> None:
        trials = [self._trial(i, v) for i, v in enumerate([0.8, 0.9, 1.0])]
        agg = aggregate_replicates(trials)["score"]
        # t(0.975, dof=2) = 4.302653; half-width = 4.302653 * 0.1 / sqrt(3)
        half = 4.302653 * 0.1 / math.sqrt(3)
        assert agg["ci95_low"] == pytest.approx(0.9 - half, rel=1e-5)
        assert agg["ci95_high"] == pytest.approx(0.9 + half, rel=1e-5)

    def test_single_replicate_has_no_dispersion(self) -> None:
        agg = aggregate_replicates([self._trial(0, 0.7)])["score"]
        assert agg["n"] == 1
        assert agg["mean"] == 0.7
        assert agg["std"] is None
        assert agg["ci95_low"] is None
        assert agg["ci95_high"] is None

    def test_failed_trials_are_excluded(self) -> None:
        trials = [self._trial(0, 0.8), self._trial(1, None), self._trial(2, 1.0)]
        agg = aggregate_replicates(trials)["score"]
        assert agg["n"] == 2
        assert agg["mean"] == pytest.approx(0.9)

    def test_all_failed_yields_empty_aggregate(self) -> None:
        assert aggregate_replicates([self._trial(0, None)]) == {}

    def test_aggregates_every_metric_key(self) -> None:
        trials = [
            ReplicateTrial(seed=0, status="success", metrics={"a": 1.0, "b": 2.0}),
            ReplicateTrial(seed=1, status="success", metrics={"a": 3.0}),
        ]
        agg = aggregate_replicates(trials)
        assert agg["a"]["n"] == 2
        assert agg["a"]["mean"] == pytest.approx(2.0)
        assert agg["b"]["n"] == 1  # only present in one trial
