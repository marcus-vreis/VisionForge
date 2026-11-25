"""Replicated comparison must be honest about what it did and did not test:
the same seeds for every variant, failures kept visible, and no significance
claim where the data cannot support one.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from visionforge.core.replicated_comparison import (
    VariantResult,
    build_report,
    min_untied_pairs,
    run_replicated_comparison,
    validate_variants,
)
from visionforge.core.replicates import ReplicateTrial
from visionforge.core.task_runner import RunResult
from visionforge.utils.messages import using_lang


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
        return dict(result.metrics)

    def primary_metric(self) -> str:
        return "accuracy"


class _EpochRunner(_FakeRunner):
    """Streams one epoch and its own "end" per run, like a real trainer."""

    _progress_callback: Any = None

    def run(self, cfg: dict[str, Any]) -> RunResult:
        emit = self._progress_callback
        if emit is not None:
            emit({"event": "start", "total_epochs": 1})
            emit({"event": "epoch_end", "epoch": 1, "total_epochs": 1})
            emit({"event": "end", "total_epochs": 1})
        return super().run(cfg)


def _base() -> dict[str, Any]:
    return {
        "name": "exp",
        "model": {"name": "resnet18"},
        "training": {"learning_rate": 0.001, "seed": 42},
    }


class TestValidation:
    def test_rejects_a_single_variant(self) -> None:
        base = _base()
        with pytest.raises(ValueError, match="at least 2 variants"):
            validate_variants(base, {"only": {}})

    def test_rejects_an_unknown_override_path(self) -> None:
        base = _base()
        with pytest.raises(ValueError, match="training.nope"):
            validate_variants(base, {"a": {}, "b": {"training.nope": 1}})

    def test_names_the_offending_variant(self) -> None:
        base = _base()
        with pytest.raises(ValueError, match="Variant 'b'"):
            validate_variants(base, {"a": {}, "b": {"bogus.path": 1}})

    def test_accepts_a_baseline_with_no_overrides(self) -> None:
        validate_variants(
            _base(), {"baseline": {}, "alt": {"training.learning_rate": 0.01}}
        )

    def test_rejects_fewer_than_two_seeds(self) -> None:
        runner = _FakeRunner()
        base = _base()
        with pytest.raises(ValueError, match="at least 2 seeds"):
            run_replicated_comparison(runner, base, {"a": {}, "b": {}}, [1], "accuracy")


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
        with using_lang("en"):
            report = run_replicated_comparison(
                _FakeRunner(), _base(), self._variants(), [1, 2, 3, 4, 5], "accuracy"
            )
        comparison = report["comparisons"][0]
        assert comparison["mean_difference"] == pytest.approx(0.10)  # a real gap
        assert comparison["significant"] is False
        assert comparison["underpowered"] is True
        assert comparison["min_achievable_p"] == pytest.approx(0.0625)
        assert report["underpowered"] is True
        assert "at least 6 paired seeds that differ" in report["power_note"]

    def test_enough_seeds_clears_the_power_warning(self) -> None:
        report = run_replicated_comparison(
            _FakeRunner(), _base(), self._variants(), list(range(1, 9)), "accuracy"
        )
        assert report["underpowered"] is False
        assert report["power_note"] == ""

    def test_streams_one_unit_per_variant_and_seed(self) -> None:
        """The unit is (variant, seed): each announced, its epochs forwarded,
        closed, one terminal "end" for the whole job (ADR-118)."""
        events: list[dict[str, Any]] = []
        run_replicated_comparison(
            _EpochRunner(),
            _base(),
            self._variants(),
            [1, 2],
            "accuracy",
            progress_callback=events.append,
        )
        kinds = [e["event"] for e in events]
        assert kinds.count("trial_start") == 4
        assert kinds.count("trial_end") == 4
        assert kinds.count("epoch_end") == 4
        assert kinds.count("end") == 1
        assert kinds[-1] == "end"
        starts = [e for e in events if e["event"] == "trial_start"]
        assert [e["trial_index"] for e in starts] == [0, 1, 2, 3]
        assert all(e["total_trials"] == 4 for e in starts)
        assert starts[0]["overrides"] == {"variant": "baseline", "training.seed": 1}
        assert starts[3]["overrides"] == {
            "variant": "lr_alto",
            "training.learning_rate": 0.01,
            "training.seed": 2,
        }
        ends = [e for e in events if e["event"] == "trial_end"]
        assert [e["total_epochs"] for e in ends] == [1, 1, 1, 1]
        epochs = [e for e in events if e["event"] == "epoch_end"]
        assert [e["trial_index"] for e in epochs] == [0, 1, 2, 3]

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


def _variant(label: str, per_seed: dict[int, float]) -> VariantResult:
    return VariantResult(
        label,
        {},
        [
            ReplicateTrial(s, "success", {"accuracy": v}, 0.1)
            for s, v in per_seed.items()
        ],
    )


class TestRankingUsesTheSameSeeds:
    """A ranking by mean is a comparison too: over the same seeds, or not at all.

    Found by stopping a comparison mid-variant (ADR-111): the cut variant's one
    finished seed out-scored a full variant's mean and was crowned best while
    the report listed it as too small to compare.
    """

    def test_a_variant_too_small_to_compare_is_not_ranked(self) -> None:
        report = build_report(
            [
                _variant("full", {1: 0.6, 2: 0.6, 3: 0.6}),
                _variant("one_seed", {1: 0.8}),
            ],
            [1, 2, 3],
            "accuracy",
        )

        assert report["skipped_variants"] == ["one_seed"]
        assert report["ranked_by_mean"] == ["full"]
        assert report["best_by_mean"] == "full"

    def test_means_are_taken_on_the_seeds_every_ranked_variant_finished(
        self,
    ) -> None:
        # Over its own three seeds "a" leads (0.633 vs 0.6); over the two seeds
        # both finished, "b" does (0.6 vs 0.5). Only the second is a comparison.
        report = build_report(
            [
                _variant("a", {1: 0.5, 2: 0.5, 3: 0.9}),
                _variant("b", {1: 0.6, 2: 0.6}),
            ],
            [1, 2, 3],
            "accuracy",
        )

        assert report["ranking_seeds"] == [1, 2]
        assert report["ranked_by_mean"] == ["b", "a"]
        assert report["best_by_mean"] == "b"

    def test_a_complete_run_ranks_exactly_as_before(self) -> None:
        report = build_report(
            [
                _variant("a", {1: 0.7, 2: 0.8, 3: 0.9}),
                _variant("b", {1: 0.6, 2: 0.6, 3: 0.6}),
            ],
            [1, 2, 3],
            "accuracy",
        )

        assert report["ranking_seeds"] == [1, 2, 3]
        assert report["ranked_by_mean"] == ["a", "b"]

    def test_the_power_note_quotes_the_seeds_actually_paired(self) -> None:
        # Six seeds were asked for; only three finished in both variants.
        with using_lang("en"):
            report = build_report(
                [
                    _variant("a", {1: 0.9, 2: 0.91, 3: 0.92}),
                    _variant("b", {1: 0.5, 2: 0.51, 3: 0.52}),
                ],
                [1, 2, 3, 4, 5, 6],
                "accuracy",
            )

        assert report["underpowered"] is True
        assert "used 3 of the 3 paired seeds (0 tied" in report["power_note"]


class TestPowerNote:
    """The note quotes the pairs the rank test kept and the alpha it ran at."""

    def test_tied_pairs_are_named_and_not_counted(self) -> None:
        # Eight seeds, four tied: a four-pair test, floor 2/2**4 = 0.125. The
        # old note said "with 8 paired seeds" and asked for "6+".
        a = {s: 0.90 + 0.01 * s for s in range(8)}
        b = {s: (a[s] if s < 4 else 0.50) for s in range(8)}

        with using_lang("en"):
            report = build_report(
                [_variant("a", a), _variant("b", b)], list(range(8)), "accuracy"
            )

        comparison = report["comparisons"][0]
        assert comparison["test"] == "wilcoxon"
        assert comparison["n_pairs"] == 8
        assert comparison["n_tested"] == 4
        note = report["power_note"]
        assert "used 4 of the 8 paired seeds (4 tied" in note
        assert "below 0.1250" in note
        assert "at least 6 paired seeds that differ" in note

    def test_the_minimum_follows_alpha(self) -> None:
        # Seven untied pairs at alpha=0.01: floor 2/2**7 = 0.0156 > 0.01, and
        # the minimum is ceil(1 - log2(0.01)) = ceil(7.64) = 8, not "6+".
        a = {s: 0.90 + 0.001 * s for s in range(7)}
        b = dict.fromkeys(a, 0.50)

        with using_lang("en"):
            report = build_report(
                [_variant("a", a), _variant("b", b)],
                list(range(7)),
                "accuracy",
                alpha=0.01,
            )

        note = report["power_note"]
        assert "used 7 of the 7 paired seeds (0 tied" in note
        assert "below 0.0156" in note
        assert "alpha=0.01" in note
        assert "at least 8 paired seeds that differ" in note

    def test_the_note_is_in_the_bound_language(self) -> None:
        with using_lang("pt"):
            report = build_report(
                [
                    _variant("a", {1: 0.9, 2: 0.91, 3: 0.92}),
                    _variant("b", {1: 0.5, 2: 0.51, 3: 0.52}),
                ],
                [1, 2, 3],
                "accuracy",
            )

        assert "usou 3 das 3 seeds pareadas" in report["power_note"]


class TestDuplicateSeeds:
    """``[42, 42, 7]`` used to train four runs, report n=3 in the aggregate and
    n_pairs=2 in the test of the same variant."""

    def test_a_repeated_seed_is_refused_before_any_training(self) -> None:
        runner = _FakeRunner()
        variants: dict[str, dict[str, Any]] = {"a": {}, "b": {}}
        base, seeds = _base(), [42, 42, 7]

        with using_lang("en"), pytest.raises(ValueError, match="must be unique"):
            run_replicated_comparison(runner, base, variants, seeds, "accuracy")

        assert runner.seen == []

    def test_two_copies_of_one_seed_are_not_two_seeds(self) -> None:
        variants: dict[str, dict[str, Any]] = {"a": {}, "b": {}}
        runner, base = _FakeRunner(), _base()

        with pytest.raises(ValueError, match="distintas"):
            run_replicated_comparison(runner, base, variants, [3, 3], "accuracy")


class TestNonFiniteMetric:
    """One NaN seed used to crash the report after every run had trained, and
    on its own it led the ranking (``sorted`` keeps NaN where it finds it)."""

    def _report(self) -> dict[str, Any]:
        a = {0: math.nan, **dict.fromkeys(range(1, 8), 0.2)}
        b = {s: 0.10 + 0.001 * s for s in range(8)}
        c = {s: 0.50 + 0.002 * s for s in range(8)}

        def variant(label: str, values: dict[int, float]) -> VariantResult:
            return VariantResult(
                label,
                {},
                [
                    ReplicateTrial(s, "success", {"rmse": v}, 0.1)
                    for s, v in values.items()
                ],
            )

        return build_report(
            [variant("A", a), variant("B", b), variant("C", c)], list(range(8)), "rmse"
        )

    def test_the_seed_is_left_out_and_recorded(self) -> None:
        report = self._report()

        assert report["non_finite_seeds"] == {"A": [0]}
        assert report["variants"]["A"]["per_seed"] == dict.fromkeys(range(1, 8), 0.2)
        assert report["variants"]["A"]["aggregates"]["rmse"]["n"] == 7
        assert report["variants"]["A"]["aggregates"]["rmse"]["n_non_finite"] == 1

    def test_every_pair_is_tested_on_finite_seeds(self) -> None:
        report = self._report()

        pairs = {(c["label_a"], c["label_b"]): c for c in report["comparisons"]}
        assert set(pairs) == {("A", "B"), ("A", "C"), ("B", "C")}
        assert pairs["A", "B"]["n_pairs"] == 7
        assert pairs["B", "C"]["n_pairs"] == 8
        assert pairs["B", "C"]["significant"] is True
        assert all(math.isfinite(c["p_value"]) for c in report["comparisons"])

    def test_the_ranking_ignores_the_nan(self) -> None:
        report = self._report()

        # Means over seeds 1..7: B 0.104, A 0.200, C 0.508; rmse is lower-better.
        assert report["ranking_seeds"] == list(range(1, 8))
        assert report["ranked_by_mean"] == ["B", "A", "C"]
        assert report["best_by_mean"] == "B"

    def test_a_clean_report_records_nothing(self) -> None:
        report = build_report(
            [
                _variant("a", {1: 0.9, 2: 0.91, 3: 0.92}),
                _variant("b", {1: 0.5, 2: 0.51, 3: 0.52}),
            ],
            [1, 2, 3],
            "accuracy",
        )

        assert report["non_finite_seeds"] == {}


class TestAlphaRange:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0, math.nan])
    def test_run_rejects_it_before_training_anything(self, alpha: float) -> None:
        runner = _FakeRunner()
        base = _base()
        variants: dict[str, dict[str, object]] = {"a": {}, "b": {}}
        with pytest.raises(ValueError, match="alpha"):
            run_replicated_comparison(
                runner, base, variants, [1, 2, 3], "accuracy", alpha=alpha
            )
        assert runner.seen == []

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0, math.nan])
    def test_build_report_and_min_untied_pairs_reject_it(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            build_report([], [1, 2], "accuracy", alpha=alpha)
        with pytest.raises(ValueError, match="alpha"):
            min_untied_pairs(alpha)
