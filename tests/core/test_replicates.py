from __future__ import annotations

import math
import statistics
from typing import Any

import pytest

from visionforge.core.replicates import (
    ReplicateTrial,
    aggregate_replicates,
    metric_domain,
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


class _EpochRunner(_FakeRunner):
    """Runner that streams two epochs and its own "end", like a real trainer."""

    _progress_callback: Any = None

    def run(self, cfg: dict[str, Any]) -> RunResult:
        emit = self._progress_callback
        if emit is not None:
            emit({"event": "start", "total_epochs": 2})
            emit({"event": "epoch_end", "epoch": 1, "total_epochs": 2})
            emit({"event": "epoch_end", "epoch": 2, "total_epochs": 2})
            emit({"event": "end", "total_epochs": 2})
        return super().run(cfg)


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

    def test_emits_trial_start_and_end_per_seed(self) -> None:
        events: list[dict[str, Any]] = []
        run_replicates(
            _FakeRunner(fail_seed=8),
            _base(),
            [7, 8],
            metric="score",
            progress_callback=events.append,
        )
        assert [e["event"] for e in events] == [
            "trial_start",
            "trial_end",
            "trial_start",
            "trial_end",
            "end",
        ]
        assert events[0]["overrides"] == {"training.seed": 7}
        assert events[0]["total_trials"] == 2
        # trial_end carries the outcome, including for the failed seed.
        assert events[1]["status"] == "success"
        assert events[3]["status"] == "failed"

    def test_streams_each_seeds_epochs_with_its_index(self) -> None:
        """The training sheet must leave "preparing": every seed forwards its
        epochs, tagged with the seed's index, and trial_end says how many ran
        (ADR-118). The trainer's own "end" must not close the stream early."""
        events: list[dict[str, Any]] = []
        runner = _EpochRunner()
        run_replicates(
            runner,
            _base(),
            [7, 8],
            metric="score",
            progress_callback=events.append,
        )
        assert [e["event"] for e in events] == [
            "trial_start",
            "epoch_end",
            "epoch_end",
            "trial_end",
            "trial_start",
            "epoch_end",
            "epoch_end",
            "trial_end",
            "end",
        ]
        epochs = [e for e in events if e["event"] == "epoch_end"]
        assert [e["trial_index"] for e in epochs] == [0, 0, 1, 1]
        assert all(e["total_trials"] == 2 for e in epochs)
        ends = [e for e in events if e["event"] == "trial_end"]
        assert [e["total_epochs"] for e in ends] == [2, 2]
        # The runner is detached once the set is over.
        assert runner._progress_callback is None


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

    @pytest.mark.parametrize(
        ("metric", "values"),
        [
            ("mae", [0.1, 0.2, 0.9]),
            ("mse", [0.01, 0.5, 0.02]),
            ("val_loss", [0.3, 0.2, 2.0]),
        ],
    )
    def test_error_metric_interval_never_goes_below_zero(
        self, metric: str, values: list[float]
    ) -> None:
        trials = [
            ReplicateTrial(seed=i, status="success", metrics={metric: v})
            for i, v in enumerate(values)
        ]
        agg = aggregate_replicates(trials)[metric]
        assert agg["ci95_low"] == 0.0
        assert agg["ci95_clipped"] is True
        # The side that stayed inside the domain is the untouched t interval.
        half = 4.302653 * statistics.stdev(values) / math.sqrt(3)
        assert agg["ci95_high"] == pytest.approx(statistics.fmean(values) + half)

    def test_unit_interval_metric_interval_stays_inside_zero_one(self) -> None:
        # "score" is not a known bounded name: its t interval is left alone.
        trials = [self._trial(i, v) for i, v in enumerate([0.8, 0.9, 1.0])]
        assert aggregate_replicates(trials)["score"]["ci95_clipped"] is False
        acc = [
            ReplicateTrial(seed=i, status="success", metrics={"accuracy": v})
            for i, v in enumerate([0.8, 0.9, 1.0])
        ]
        agg = aggregate_replicates(acc)["accuracy"]
        assert agg["ci95_high"] == 1.0
        assert agg["ci95_low"] == pytest.approx(0.9 - 4.302653 * 0.1 / math.sqrt(3))
        assert agg["ci95_clipped"] is True

    def test_r2_interval_stops_at_one_but_may_go_negative(self) -> None:
        trials = [
            ReplicateTrial(seed=i, status="success", metrics={"r2": v})
            for i, v in enumerate([0.5, 0.9, 0.99])
        ]
        agg = aggregate_replicates(trials)["r2"]
        assert agg["ci95_high"] == 1.0
        assert agg["ci95_clipped"] is True
        wide = [
            ReplicateTrial(seed=i, status="success", metrics={"r2": v})
            for i, v in enumerate([-1.0, 0.2, 0.9])
        ]
        low = aggregate_replicates(wide)["r2"]["ci95_low"]
        assert low is not None
        assert low < -1.0

    def test_interval_inside_the_domain_is_not_flagged(self) -> None:
        trials = [
            ReplicateTrial(seed=i, status="success", metrics={"accuracy": v})
            for i, v in enumerate([0.50, 0.51, 0.52])
        ]
        agg = aggregate_replicates(trials)["accuracy"]
        assert agg["ci95_clipped"] is False
        low, high = agg["ci95_low"], agg["ci95_high"]
        assert low is not None
        assert high is not None
        assert 0.0 < low < high < 1.0

    def test_single_value_is_not_flagged(self) -> None:
        agg = aggregate_replicates([self._trial(0, 0.7)])["score"]
        assert agg["ci95_clipped"] is False

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


class TestMetricDomain:
    """Which bounds a confidence interval is clipped to (review 2026-10-17)."""

    @pytest.mark.parametrize(
        "name", ["mae", "val_mse", "test_rmse", "train_loss", "mean_absolute_error"]
    )
    def test_absolute_errors_and_losses_cannot_go_negative(self, name: str) -> None:
        assert metric_domain(name) == (0.0, None)

    @pytest.mark.parametrize("name", ["mean_error", "bias_error", "signed_error"])
    def test_a_signed_error_is_not_floored_at_zero(self, name: str) -> None:
        assert metric_domain(name) == (None, None)

    @pytest.mark.parametrize("name", ["r2", "val_r2", "r_squared", "val_r_squared"])
    def test_r_squared_is_capped_at_one_only(self, name: str) -> None:
        assert metric_domain(name) == (None, 1.0)

    def test_scores_in_the_unit_interval(self) -> None:
        assert metric_domain("val_accuracy") == (0.0, 1.0)
