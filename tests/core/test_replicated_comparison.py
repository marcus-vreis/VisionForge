"""Replicated comparison must be honest about what it did and did not test:
the same seeds for every variant, failures kept visible, and no significance
claim where the data cannot support one.
"""

from __future__ import annotations

from typing import Any

import pytest

from visionforge.core.replicated_comparison import (
    VariantResult,
    build_report,
    run_replicated_comparison,
    validate_variants,
)
from visionforge.core.replicates import ReplicateTrial
from visionforge.core.task_runner import RunResult


class _FakeConfig:
    @classmethod
    def model_validate(cls, d: dict[str, Any]) -> dict[str, Any]:
        return d


class _FakeRunner:
    """Score depends on the learning rate, plus a small per-seed wobble.

    The wobble is shared across variants for a given seed, which is exactly
    the structure pairing is meant to cancel out.
    """

    config_type = _FakeConfig

    def __init__(self, fail_label: str | None = None) -> None:
        self._fail_label = fail_label
        self.seen: list[dict[str, Any]] = []

    def run(self, cfg: dict[str, Any]) -> RunResult:
        self.seen.append(cfg)
        if self._fail_label and self._fail_label in str(cfg.get("name", "")):
            return RunResult(metrics={}, status="failed", error="boom")
        lr = cfg["training"]["learning_rate"]
        seed = cfg["training"]["seed"]
        score = 0.90 if lr == 0.001 else 0.80
        return RunResult(
            metrics={"accuracy": score + (seed % 3) * 0.005},
            status="success",
            training_time_s=0.1,
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return result.metrics

    def primary_metric(self) -> str:
        return "accuracy"


def _base() -> dict[str, Any]:
    return {
        "name": "exp",
        "model": {"name": "resnet18"},
        "training": {"learning_rate": 0.001, "seed": 42},
    }


class TestValidation:
    def test_rejects_a_single_variant(self) -> None:
        with pytest.raises(ValueError, match="at least 2 variants"):
            validate_variants(_base(), {"only": {}})

    def test_rejects_an_unknown_override_path(self) -> None:
        with pytest.raises(ValueError, match="training.nope"):
            validate_variants(_base(), {"a": {}, "b": {"training.nope": 1}})

    def test_names_the_offending_variant(self) -> None:
        with pytest.raises(ValueError, match="Variant 'b'"):
            validate_variants(_base(), {"a": {}, "b": {"bogus.path": 1}})

    def test_accepts_a_baseline_with_no_overrides(self) -> None:
        validate_variants(
            _base(), {"baseline": {}, "alt": {"training.learning_rate": 0.01}}
        )

    def test_rejects_fewer_than_two_seeds(self) -> None:
        with pytest.raises(ValueError, match="at least 2 seeds"):
            run_replicated_comparison(
                _FakeRunner(), _base(), {"a": {}, "b": {}}, [1], "accuracy"
            )


class TestRunReplicatedComparison:
    def _variants(self) -> dict[str, dict[str, Any]]:
        return {"baseline": {}, "lr_alto": {"training.learning_rate": 0.01}}

    def test_every_variant_sees_the_same_seeds(self) -> None:
        runner = _FakeRunner()
        seeds = [1, 2, 3]
        run_replicated_comparison(runner, _base(), self._variants(), seeds, "accuracy")
        # 2 variants x 3 seeds — and each variant ran exactly this seed list,
        # which is what makes the later test paired.
        assert len(runner.seen) == 6
        baseline_seeds = [
            c["training"]["seed"] for c in runner.seen if "baseline" in c["name"]
        ]
        alt_seeds = [
            c["training"]["seed"] for c in runner.seen if "lr_alto" in c["name"]
        ]
        assert baseline_seeds == seeds
        assert alt_seeds == seeds

    def test_overrides_are_applied_per_variant(self) -> None:
        runner = _FakeRunner()
        run_replicated_comparison(runner, _base(), self._variants(), [1, 2], "accuracy")
        rates = {
            c["name"].split("_", 1)[1].rsplit("_s", 1)[0]: c["training"][
                "learning_rate"
            ]
            for c in runner.seen
        }
        assert rates["baseline"] == 0.001
        assert rates["lr_alto"] == 0.01

    def test_base_config_is_not_mutated(self) -> None:
        base = _base()
        run_replicated_comparison(
            _FakeRunner(), base, self._variants(), [1, 2], "accuracy"
        )
        assert base["name"] == "exp"
        assert base["training"]["learning_rate"] == 0.001

    def test_report_ranks_and_tests_the_variants(self) -> None:
        report = run_replicated_comparison(
            _FakeRunner(),
            _base(),
            self._variants(),
            [1, 2, 3, 4, 5, 6, 7, 8],
            "accuracy",
        )
        assert report["kind"] == "replicated_comparison"
        assert report["best_by_mean"] == "baseline"  # lr=0.001 scores higher
        assert report["ranked_by_mean"] == ["baseline", "lr_alto"]
        assert len(report["comparisons"]) == 1
        comparison = report["comparisons"][0]
        assert comparison["n_pairs"] == 8
        assert comparison["mean_difference"] == pytest.approx(0.10)
        assert comparison["significant"] is True

    def test_five_seeds_cannot_reach_significance_and_the_report_says_why(
        self,
    ) -> None:
        """Wilcoxon's p floor at n=5 is 0.0625 > 0.05. A huge, perfectly
        consistent gap still comes back 'not significant' — reporting that as
        'no effect' would be flatly wrong, so the report must explain it."""
        report = run_replicated_comparison(
            _FakeRunner(), _base(), self._variants(), [1, 2, 3, 4, 5], "accuracy"
        )
        comparison = report["comparisons"][0]
        assert comparison["mean_difference"] == pytest.approx(0.10)  # a real gap
        assert comparison["significant"] is False
        assert comparison["underpowered"] is True
        assert comparison["min_achievable_p"] == pytest.approx(0.0625)
        assert report["underpowered"] is True
        assert "6+" in report["power_note"]

    def test_enough_seeds_clears_the_power_warning(self) -> None:
        report = run_replicated_comparison(
            _FakeRunner(), _base(), self._variants(), list(range(1, 9)), "accuracy"
        )
        assert report["underpowered"] is False
        assert report["power_note"] == ""

    def test_streams_one_trial_pair_per_variant(self) -> None:
        events: list[dict[str, Any]] = []
        run_replicated_comparison(
            _FakeRunner(),
            _base(),
            self._variants(),
            [1, 2],
            "accuracy",
            progress_callback=events.append,
        )
        # The monitor tracks variants, not every seed — inner replicate events
        # would make a 2-variant run look like a 4-trial one.
        assert [e["event"] for e in events] == [
            "trial_start",
            "trial_end",
            "trial_start",
            "trial_end",
        ]
        assert events[0]["overrides"]["variant"] == "baseline"

    def test_a_failed_variant_stays_visible_but_untested(self) -> None:
        report = run_replicated_comparison(
            _FakeRunner(fail_label="lr_alto"),
            _base(),
            self._variants(),
            [1, 2, 3],
            "accuracy",
        )
        # Present in the report…
        assert "lr_alto" in report["variants"]
        assert report["variants"]["lr_alto"]["successful"] == 0
        # …but excluded from the matrix, and the omission is stated.
        assert report["comparisons"] == []
        assert report["skipped_variants"] == ["lr_alto"]


class TestBuildReport:
    def _results(self) -> list[VariantResult]:
        return [
            VariantResult(
                "a",
                {},
                [
                    ReplicateTrial(1, "success", {"accuracy": 0.90}, 0.1),
                    ReplicateTrial(2, "success", {"accuracy": 0.91}, 0.1),
                    ReplicateTrial(3, "success", {"accuracy": 0.89}, 0.1),
                ],
            ),
            VariantResult(
                "b",
                {"training.learning_rate": 0.01},
                [
                    ReplicateTrial(1, "success", {"accuracy": 0.80}, 0.1),
                    ReplicateTrial(2, "success", {"accuracy": 0.81}, 0.1),
                    ReplicateTrial(3, "success", {"accuracy": 0.79}, 0.1),
                ],
            ),
        ]

    def test_keeps_the_per_seed_vectors_for_auditing(self) -> None:
        report = build_report(self._results(), [1, 2, 3], "accuracy")
        assert report["variants"]["a"]["per_seed"] == {1: 0.90, 2: 0.91, 3: 0.89}

    def test_aggregates_carry_both_intervals(self) -> None:
        report = build_report(self._results(), [1, 2, 3], "accuracy")
        agg = report["variants"]["a"]["aggregates"]["accuracy"]
        assert agg["n"] == 3
        assert agg["ci95_low"] is not None
        assert agg["boot95_low"] is not None

    def test_best_by_mean_makes_no_significance_claim(self) -> None:
        # Two variants that differ only by noise: a winner by mean exists,
        # but nothing may be flagged significant.
        results = [
            VariantResult(
                "x",
                {},
                [
                    ReplicateTrial(s, "success", {"m": 0.5 + s * 0.001}, 0.1)
                    for s in (1, 2, 3)
                ],
            ),
            VariantResult(
                "y",
                {},
                [
                    ReplicateTrial(s, "success", {"m": 0.5 + s * 0.001}, 0.1)
                    for s in (1, 2, 3)
                ],
            ),
        ]
        report = build_report(results, [1, 2, 3], "m")
        assert report["best_by_mean"] in {"x", "y"}
        assert report["significant_pairs"] == 0

    def test_three_variants_produce_three_comparisons(self) -> None:
        results = self._results()
        results.append(
            VariantResult(
                "c",
                {},
                [
                    ReplicateTrial(s, "success", {"accuracy": 0.85}, 0.1)
                    for s in (1, 2, 3)
                ],
            )
        )
        report = build_report(results, [1, 2, 3], "accuracy")
        assert len(report["comparisons"]) == 3


class TestRankingDirection:
    """Found by running a real comparison: MAE 4.02 was crowned "best" over
    MAE 0.99 because ranking always sorted descending."""

    def _results(self, metric: str) -> list[VariantResult]:
        return [
            VariantResult(
                "poucos",
                {},
                [ReplicateTrial(s, "success", {metric: 1.0}, 0.1) for s in (1, 2, 3)],
            ),
            VariantResult(
                "muitos",
                {},
                [ReplicateTrial(s, "success", {metric: 4.0}, 0.1) for s in (1, 2, 3)],
            ),
        ]

    def test_lower_is_better_metric_ranks_the_smallest_first(self) -> None:
        report = build_report(self._results("mae"), [1, 2, 3], "mae")
        assert report["metric_direction"] == "lower"
        assert report["best_by_mean"] == "poucos"
        assert report["ranked_by_mean"] == ["poucos", "muitos"]

    def test_higher_is_better_metric_ranks_the_largest_first(self) -> None:
        report = build_report(self._results("accuracy"), [1, 2, 3], "accuracy")
        assert report["metric_direction"] == "higher"
        assert report["best_by_mean"] == "muitos"

    def test_explicit_direction_overrides_the_name_heuristic(self) -> None:
        # A custom task may declare a direction the name does not imply.
        report = build_report(
            self._results("score"), [1, 2, 3], "score", direction="lower"
        )
        assert report["best_by_mean"] == "poucos"
