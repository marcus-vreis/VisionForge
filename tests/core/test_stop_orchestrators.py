"""The orchestrators that run several trainings in one job honour a stop (ADR-111).

The token travels on the runner: the runner hands it to the training in flight,
and the orchestrator reads it between units. These fakes press "stop" while a
given unit is running, which is the moment a researcher actually presses it,
and say whether that unit was cut -- a trainer knows when its loop broke on the
token, while a unit whose stop landed in its last epoch, or a custom task that
owns its loop, runs to its end and is not cut.
"""

from __future__ import annotations

from typing import Any

import pytest

from visionforge.core.cancellation import STOPPED, STOPPED_NOTE, CancellationToken
from visionforge.core.comparison import run_model_comparison
from visionforge.core.replicated_comparison import run_replicated_comparison
from visionforge.core.replicates import aggregate_replicates, run_replicates
from visionforge.core.sweep import run_sweep
from visionforge.core.task_runner import (
    RunResult,
    give_cancel_token,
    runner_cancel_token,
)


class _EchoConfig:
    @classmethod
    def model_validate(cls, d: dict[str, Any]) -> dict[str, Any]:
        return d


class _StoppingRunner:
    """Each run succeeds at once; the stop is pressed during run ``stop_during``.

    ``cut`` is what the unit reports about that run: True when its loop broke
    on the token, False when it ran to its end anyway.
    """

    config_type = _EchoConfig
    _cancel_token: CancellationToken | None = None

    def __init__(self, stop_during: int | None, cut: bool = True) -> None:
        self.stop_during = stop_during
        self.cut = cut
        self.calls = 0
        self.tokens_seen: list[CancellationToken | None] = []

    def run(self, cfg: dict[str, Any]) -> RunResult:
        index = self.calls
        self.calls += 1
        self.tokens_seen.append(self._cancel_token)
        stopped = False
        if index == self.stop_during and self._cancel_token is not None:
            self._cancel_token.cancel()
            stopped = self.cut
        return RunResult(
            metrics={"score": 0.5 + 0.1 * index},
            status="success",
            training_time_s=0.0,
            stopped=stopped,
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return dict(result.metrics)

    def primary_metric(self) -> str:
        return "score"


def _runner(
    stop_during: int | None, cut: bool = True
) -> tuple[_StoppingRunner, CancellationToken]:
    runner = _StoppingRunner(stop_during, cut)
    token = CancellationToken()
    give_cancel_token(runner, token)
    return runner, token


def _base() -> dict[str, Any]:
    return {
        "name": "x",
        "model": {"name": "a"},
        "training": {"learning_rate": 0.01, "seed": 1},
    }


class TestTheRunnerCarriesTheToken:
    def test_a_runner_with_a_slot_gets_the_token(self) -> None:
        runner, token = _runner(None)
        assert runner_cancel_token(runner) is token

    def test_a_runner_without_a_slot_is_left_alone(self) -> None:
        class _Bare:
            pass

        bare = _Bare()
        give_cancel_token(bare, CancellationToken())

        assert not hasattr(bare, "_cancel_token")
        assert runner_cancel_token(bare) is None


class TestComparison:
    def test_stops_between_models_and_ranks_only_finished_ones(self) -> None:
        runner, token = _runner(stop_during=1)

        trials = run_model_comparison(runner, _base(), ["a", "b", "c"], "score")

        assert runner.calls == 2  # "c" never started
        assert all(seen is token for seen in runner.tokens_seen)
        by_arch = {t.model_arch: t for t in trials}
        assert set(by_arch) == {"a", "b"}
        assert by_arch["a"].status == "success"
        assert by_arch["b"].status == STOPPED
        assert by_arch["b"].error == STOPPED_NOTE
        # The cut model keeps its metrics but never outranks a finished one.
        assert trials[0].model_arch == "a"


class TestSweep:
    @pytest.mark.parametrize("mode", ["grid", "random"])
    def test_stops_between_trials(self, mode: str) -> None:
        runner, _ = _runner(stop_during=1)
        events: list[dict[str, Any]] = []
        space: dict[str, Any] = (
            {"training.learning_rate": [0.01, 0.02, 0.03, 0.04]}
            if mode == "grid"
            else {
                "training.learning_rate": {"type": "uniform", "low": 0.0, "high": 1.0}
            }
        )

        trials = run_sweep(
            runner,
            _base(),
            space,
            mode=mode,
            metric="score",
            n_trials=4,
            progress_callback=events.append,
        )

        assert runner.calls == 2
        assert sorted(t.status for t in trials) == ["stopped", "success"]
        # The live monitor sees the cut trial close as stopped, once.
        ends = [e for e in events if e["event"] == "trial_end"]
        assert [e["status"] for e in ends] == ["success", STOPPED]

    def test_optuna_samples_no_trial_after_a_stop(self) -> None:
        runner, _ = _runner(stop_during=1)

        trials = run_sweep(
            runner,
            _base(),
            {"training.learning_rate": {"type": "uniform", "low": 0.0, "high": 1.0}},
            mode="optuna",
            metric="score",
            n_trials=5,
        )

        assert runner.calls == 2
        assert sorted(t.status for t in trials) == ["stopped", "success"]


class TestReplicates:
    def test_the_aggregate_counts_only_finished_replicates(self) -> None:
        runner, _ = _runner(stop_during=2)

        trials = run_replicates(runner, _base(), [1, 2, 3, 4, 5], "score")

        assert runner.calls == 3
        assert [t.status for t in trials] == ["success", "success", STOPPED]
        aggregate = aggregate_replicates(trials)["score"]
        assert aggregate["n"] == 2
        assert aggregate["mean"] == pytest.approx((0.5 + 0.6) / 2)


class TestReplicatedComparison:
    def test_stops_inside_a_variant_and_starts_no_other(self) -> None:
        # A runs calls 0-2, B calls 3-5; the stop lands on B's second seed.
        runner, _ = _runner(stop_during=4)

        report = run_replicated_comparison(
            runner,
            _base(),
            {"A": {}, "B": {"training.learning_rate": 0.1}, "C": {}},
            [1, 2, 3],
            "score",
        )

        assert runner.calls == 5
        assert list(report["variants"]) == ["A", "B"]
        statuses = [t["status"] for t in report["variants"]["B"]["trials"]]
        assert statuses == ["success", STOPPED]
        assert report["variants"]["B"]["aggregates"]["score"]["n"] == 1
        # One finished seed is not enough to pair B with A: it is skipped, not
        # tested on a sample that does not exist.
        assert report["skipped_variants"] == ["B"]
        assert report["comparisons"] == []
        # B's one seed (0.8) beats A's mean (0.6) only because it is one seed:
        # a variant that cannot be compared is not ranked either.
        assert report["best_by_mean"] == "A"
        assert report["ranked_by_mean"] == ["A"]
        # C never started; it is named rather than silently missing.
        assert report["not_run"] == ["C"]

    def test_a_complete_comparison_has_nothing_not_run(self) -> None:
        runner, _ = _runner(stop_during=None)

        report = run_replicated_comparison(
            runner, _base(), {"A": {}, "B": {}}, [1, 2], "score"
        )

        assert report["not_run"] == []


class TestAUnitThatRanToItsEndIsNotRelabelled:
    """The stop landed during the unit, but the unit finished: it counts."""

    def test_replicates(self) -> None:
        # A custom task that owns its loop always runs its replicate to the end.
        runner, _ = _runner(stop_during=0, cut=False)

        trials = run_replicates(runner, _base(), [1, 2, 3], "score")

        assert runner.calls == 1  # no further seed after the stop
        assert [t.status for t in trials] == ["success"]
        assert aggregate_replicates(trials)["score"]["n"] == 1

    def test_comparison(self) -> None:
        runner, _ = _runner(stop_during=1, cut=False)

        trials = run_model_comparison(runner, _base(), ["a", "b", "c"], "score")

        assert runner.calls == 2
        assert [t.status for t in trials] == ["success", "success"]

    @pytest.mark.parametrize("mode", ["grid", "optuna"])
    def test_sweep(self, mode: str) -> None:
        runner, _ = _runner(stop_during=0, cut=False)
        space: dict[str, Any] = (
            {"training.learning_rate": [0.01, 0.02, 0.03]}
            if mode == "grid"
            else {
                "training.learning_rate": {"type": "uniform", "low": 0.0, "high": 1.0}
            }
        )

        trials = run_sweep(
            runner, _base(), space, mode=mode, metric="score", n_trials=3
        )

        assert runner.calls == 1
        assert [t.status for t in trials] == ["success"]


class _CountingToken(CancellationToken):
    """Counts how often the orchestrator looks at the stop signal."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    @property
    def cancelled(self) -> bool:
        self.reads += 1
        return super().cancelled


class TestTheTokenIsReadOncePerUnit:
    """Two reads per unit let a stop land between them: one decision, one read."""

    def _counted(self) -> tuple[_StoppingRunner, _CountingToken]:
        runner = _StoppingRunner(stop_during=None)
        token = _CountingToken()
        give_cancel_token(runner, token)
        return runner, token

    def test_replicates(self) -> None:
        runner, token = self._counted()

        run_replicates(runner, _base(), [1, 2, 3], "score")

        assert token.reads == 3

    def test_comparison(self) -> None:
        runner, token = self._counted()

        run_model_comparison(runner, _base(), ["a", "b"], "score")

        assert token.reads == 2

    def test_grid_sweep(self) -> None:
        runner, token = self._counted()

        run_sweep(
            runner,
            _base(),
            {"training.learning_rate": [0.01, 0.02]},
            mode="grid",
            metric="score",
        )

        assert token.reads == 2


class TestSampleStd:
    def test_one_value_has_no_spread_to_report(self) -> None:
        from visionforge.core.replicates import sample_std

        assert sample_std([0.7]) is None
        assert sample_std([]) is None

    def test_uses_n_minus_one_like_the_replicate_statistics(self) -> None:
        import statistics

        from visionforge.core.replicates import sample_std

        assert sample_std([0.5, 0.7, 0.9]) == pytest.approx(
            statistics.stdev([0.5, 0.7, 0.9])
        )


class TestWithoutAStop:
    def test_every_unit_runs(self) -> None:
        runner, _ = _runner(stop_during=None)

        trials = run_model_comparison(runner, _base(), ["a", "b", "c"], "score")

        assert runner.calls == 3
        assert all(t.status == "success" for t in trials)
