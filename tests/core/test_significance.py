"""Statistics must be right, not merely green: these assert against values a
reader can verify by hand or against scipy's own reference implementation,
and pin the honesty guards (paired-only, Holm correction, refusals).
"""

from __future__ import annotations

import math

import pytest
from scipy import stats

from visionforge.core.significance import (
    PairedComparison,
    bootstrap_ci,
    cohens_dz,
    comparison_matrix,
    holm_correct,
    paired_comparison,
)


class TestBootstrapCI:
    def test_interval_brackets_the_mean(self) -> None:
        values = [0.80, 0.82, 0.79, 0.85, 0.81]
        ci = bootstrap_ci(values, seed=7)
        assert ci is not None
        assert ci.ci_low < ci.mean < ci.ci_high
        assert ci.mean == pytest.approx(0.8140)
        assert ci.n == 5

    def test_is_deterministic_for_a_given_seed(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert bootstrap_ci(values, seed=42) == bootstrap_ci(values, seed=42)

    def test_it_really_resamples(self) -> None:
        # At the default 10k resamples the quantiles have converged, so two
        # seeds agree — that stability is the point. Drop the count and the
        # sampling noise (hence the actual resampling) becomes visible.
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert bootstrap_ci(values, n_resamples=30, seed=1) != bootstrap_ci(
            values, n_resamples=30, seed=2
        )

    def test_default_resample_count_is_seed_stable(self) -> None:
        # A reported interval must not wobble with an arbitrary seed choice.
        values = [0.80, 0.82, 0.79, 0.85, 0.81]
        a = bootstrap_ci(values, seed=1)
        b = bootstrap_ci(values, seed=99)
        assert a is not None and b is not None
        assert a.ci_low == pytest.approx(b.ci_low, abs=0.005)
        assert a.ci_high == pytest.approx(b.ci_high, abs=0.005)

    def test_zero_variance_collapses_the_interval(self) -> None:
        ci = bootstrap_ci([0.5] * 6, seed=0)
        assert ci is not None
        assert ci.ci_low == ci.ci_high == 0.5

    def test_single_value_has_no_interval(self) -> None:
        assert bootstrap_ci([0.9]) is None

    def test_wider_confidence_widens_the_interval(self) -> None:
        values = [0.1, 0.4, 0.35, 0.8, 0.55]
        narrow = bootstrap_ci(values, confidence=0.80, seed=1)
        wide = bootstrap_ci(values, confidence=0.99, seed=1)
        assert narrow is not None and wide is not None
        assert wide.ci_low < narrow.ci_low
        assert wide.ci_high > narrow.ci_high


class TestCohensDz:
    def test_matches_the_hand_computation(self) -> None:
        # differences: mean 2.0, sample sd 1.0 -> d_z = 2.0
        assert cohens_dz([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_identical_pairs_report_zero_not_infinity(self) -> None:
        assert cohens_dz([0.5, 0.5, 0.5]) == 0.0

    def test_sign_follows_the_direction_of_the_difference(self) -> None:
        assert cohens_dz([-1.0, -2.0, -3.0]) == pytest.approx(-2.0)


class TestPairedComparison:
    def _groups(self) -> tuple[dict[int, float], dict[int, float]]:
        a = {1: 0.90, 2: 0.91, 3: 0.89, 4: 0.92, 5: 0.90}
        b = {1: 0.85, 2: 0.87, 3: 0.84, 4: 0.88, 5: 0.86}
        return a, b

    def test_uses_only_shared_seeds(self) -> None:
        a = {1: 0.9, 2: 0.9, 3: 0.9, 99: 5.0}
        b = {1: 0.8, 2: 0.8, 3: 0.8, 77: -5.0}
        result = paired_comparison("accuracy", a, b)
        # The unshared seeds (99, 77) would swamp the means if they leaked in.
        assert result.n_pairs == 3
        assert result.mean_a == pytest.approx(0.9)
        assert result.mean_b == pytest.approx(0.8)

    def test_p_value_matches_scipy_for_the_t_path(self) -> None:
        a = dict(enumerate([0.90, 0.91, 0.89, 0.92, 0.90, 0.91, 0.90, 0.92]))
        b = dict(enumerate([0.85, 0.87, 0.84, 0.88, 0.86, 0.85, 0.86, 0.87]))
        result = paired_comparison("accuracy", a, b, test="paired_t")
        expected = stats.ttest_rel(list(a.values()), list(b.values()))
        assert result.p_value == pytest.approx(float(expected.pvalue))
        assert result.test == "paired_t"

    def test_small_samples_default_to_the_rank_test(self) -> None:
        a, b = self._groups()  # 5 pairs — below the normality-check threshold
        result = paired_comparison("accuracy", a, b)
        assert result.test == "wilcoxon"
        assert "too few" in result.test_reason

    def test_reports_direction_and_effect(self) -> None:
        a, b = self._groups()
        result = paired_comparison("accuracy", a, b, label_a="novo", label_b="baseline")
        assert result.mean_difference > 0  # a is better
        assert result.effect_size > 0.8
        assert result.effect_label == "large"
        assert result.label_a == "novo" and result.label_b == "baseline"

    def test_difference_ci_excludes_zero_for_a_consistent_win(self) -> None:
        a, b = self._groups()
        result = paired_comparison("accuracy", a, b)
        assert result.ci_low > 0

    def test_identical_results_cannot_be_significant(self) -> None:
        same = {1: 0.9, 2: 0.8, 3: 0.85}
        result = paired_comparison("accuracy", same, dict(same))
        assert result.p_value == 1.0
        assert result.mean_difference == 0.0
        assert result.effect_size == 0.0

    def test_refuses_when_seeds_do_not_line_up(self) -> None:
        with pytest.raises(ValueError, match="share 1 seed"):
            paired_comparison("accuracy", {1: 0.9, 2: 0.8}, {1: 0.7, 5: 0.6})

    def test_refuses_disjoint_seeds(self) -> None:
        with pytest.raises(ValueError, match="same seeds"):
            paired_comparison("accuracy", {1: 0.9}, {2: 0.8})


class TestHolmCorrection:
    def _fake(self, p: float, label: str) -> PairedComparison:
        return paired_comparison(
            "m",
            {1: 1.0, 2: 1.0, 3: 1.0},
            {1: 1.0, 2: 1.0, 3: 1.0},
            label_a=label,
            label_b="ref",
        )

    def test_stops_at_the_first_non_rejection(self) -> None:
        # Build comparisons with controlled p-values by hand.
        from dataclasses import replace

        base = self._fake(1.0, "x")
        comparisons = [
            replace(base, p_value=0.001, label_a="a"),  # 0.05/3 = 0.0167 -> reject
            replace(base, p_value=0.02, label_a="b"),  # 0.05/2 = 0.025  -> reject
            replace(base, p_value=0.30, label_a="c"),  # 0.05/1 = 0.05   -> keep
        ]
        flagged = holm_correct(comparisons)
        assert [c.significant for c in flagged] == [True, True, False]

    def test_holm_is_stricter_than_raw_alpha(self) -> None:
        from dataclasses import replace

        base = self._fake(1.0, "x")
        # p=0.03 would pass alone, but not as the smallest of three tests.
        comparisons = [replace(base, p_value=0.03) for _ in range(3)]
        flagged = holm_correct(comparisons)
        assert [c.significant for c in flagged] == [False, False, False]

    def test_preserves_input_order(self) -> None:
        from dataclasses import replace

        base = self._fake(1.0, "x")
        comparisons = [
            replace(base, p_value=0.4, label_a="first"),
            replace(base, p_value=0.001, label_a="second"),
        ]
        flagged = holm_correct(comparisons)
        assert [c.label_a for c in flagged] == ["first", "second"]
        assert [c.significant for c in flagged] == [False, True]

    def test_empty_family_is_not_an_error(self) -> None:
        assert holm_correct([]) == []


class TestComparisonMatrix:
    def test_every_pair_once(self) -> None:
        seeds = [1, 2, 3, 4]
        groups = {
            "a": {s: 0.90 + s * 0.001 for s in seeds},
            "b": {s: 0.85 + s * 0.001 for s in seeds},
            "c": {s: 0.80 + s * 0.001 for s in seeds},
        }
        matrix = comparison_matrix("accuracy", groups)
        pairs = {(c.label_a, c.label_b) for c in matrix}
        assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}

    def test_skips_groups_that_share_no_seeds(self) -> None:
        groups = {
            "a": {1: 0.9, 2: 0.9, 3: 0.9},
            "b": {1: 0.8, 2: 0.8, 3: 0.8},
            "orphan": {90: 0.5, 91: 0.5, 92: 0.5},
        }
        matrix = comparison_matrix("accuracy", groups)
        # Only the pair that actually shares seeds is reported.
        assert [(c.label_a, c.label_b) for c in matrix] == [("a", "b")]

    def test_consistent_winner_is_flagged_significant(self) -> None:
        seeds = list(range(1, 11))
        groups = {
            "better": {s: 0.90 + 0.001 * s for s in seeds},
            "worse": {s: 0.80 + 0.001 * s for s in seeds},
        }
        matrix = comparison_matrix("accuracy", groups)
        assert len(matrix) == 1
        assert matrix[0].significant is True
        assert matrix[0].mean_difference == pytest.approx(0.10)

    def test_noise_is_not_flagged(self) -> None:
        # Same distribution, values shuffled between the groups.
        values = [0.80, 0.83, 0.79, 0.85, 0.81, 0.84, 0.78, 0.86]
        groups = {
            "a": dict(enumerate(values)),
            "b": dict(enumerate(reversed(values))),
        }
        matrix = comparison_matrix("accuracy", groups)
        assert matrix[0].significant is False
        assert not math.isnan(matrix[0].p_value)
