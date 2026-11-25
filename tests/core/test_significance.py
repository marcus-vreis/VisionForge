"""Statistics must be right, not merely green: these assert against values a
reader can verify by hand or against scipy's own reference implementation,
and pin the honesty guards (paired-only, Holm correction, refusals).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from visionforge.core.significance import (
    PairedComparison,
    bootstrap_ci,
    cohens_dz,
    comparison_matrix,
    hodges_lehmann_ci,
    holm_correct,
    infer_direction,
    min_achievable_p,
    paired_comparison,
    paired_t_ci,
    signed_rank_cdf,
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
        assert a is not None
        assert b is not None
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
        assert narrow is not None
        assert wide is not None
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
        assert result.label_a == "novo"
        assert result.label_b == "baseline"

    def test_difference_ci_excludes_zero_for_a_consistent_win(self) -> None:
        # Eight pairs: with five the Wilcoxon cannot reach p < 0.05, so its
        # dual interval cannot exclude zero either (ADR-187).
        a = {i: 0.90 + 0.01 * (i % 3) for i in range(8)}
        b = {i: 0.85 + 0.007 * (i % 4) for i in range(8)}
        result = paired_comparison("accuracy", a, b)
        assert result.ci_low is not None
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


class TestInferDirection:
    """Ranking a lower-is-better metric as if higher were better crowns the
    worst variant — and it looks authoritative next to a p-value."""

    def test_error_metrics_are_lower_is_better(self) -> None:
        for name in ("mae", "test_mae", "rmse", "mse", "val_loss", "train_loss"):
            assert infer_direction(name) == "lower", name

    def test_score_metrics_are_higher_is_better(self) -> None:
        for name in (
            "accuracy",
            "f1",
            "auc_roc",
            "r2",
            "miou",
            "dice",
            "map50",
            "auroc",
        ):
            assert infer_direction(name) == "higher", name

    def test_unknown_names_default_to_higher(self) -> None:
        # Most VisionForge ranking metrics are scores; an unknown custom
        # metric is more likely a score than an error.
        assert infer_direction("my_custom_score") == "higher"


class TestMinAchievableP:
    def test_wilcoxon_floor_matches_the_closed_form(self) -> None:
        # 2 / 2**n for a two-sided test with no ties.
        assert min_achievable_p("wilcoxon", 5) == pytest.approx(0.0625)
        assert min_achievable_p("wilcoxon", 6) == pytest.approx(0.03125)
        assert min_achievable_p("wilcoxon", 8) == pytest.approx(0.0078125)

    def test_five_seeds_cannot_reach_the_usual_alpha(self) -> None:
        assert min_achievable_p("wilcoxon", 5) > 0.05
        assert min_achievable_p("wilcoxon", 6) < 0.05

    def test_the_t_test_has_no_floor(self) -> None:
        assert min_achievable_p("paired_t", 3) == 0.0

    def test_underpowered_flag_follows_the_floor(self) -> None:
        few = paired_comparison(
            "accuracy",
            dict.fromkeys(range(5), 0.9),
            dict.fromkeys(range(5), 0.8),
        )
        assert few.underpowered is True
        many = paired_comparison(
            "accuracy",
            {s: 0.9 + s * 0.001 for s in range(8)},
            {s: 0.8 + s * 0.001 for s in range(8)},
        )
        assert many.underpowered is False


class TestFloorWithTiedPairs:
    """Wilcoxon drops zero differences, so the floor counts only the rest.

    Six seeds with three ties is a three-pair test: its smallest two-sided p is
    0.25, and calling it adequately powered because six seeds would reach
    0.03125 tells the researcher the opposite of the truth.
    """

    def test_the_floor_uses_the_non_zero_differences(self) -> None:
        a = {s: 0.80 + 0.01 * s for s in range(6)}
        b = dict(a)
        b[0] -= 0.02
        b[1] -= 0.03
        b[2] -= 0.01

        result = paired_comparison("accuracy", a, b, test="wilcoxon")

        assert result.n_pairs == 6
        assert result.min_achievable_p == pytest.approx(0.25)
        assert result.underpowered is True
        assert result.p_value >= result.min_achievable_p

    def test_without_ties_the_floor_is_unchanged(self) -> None:
        a = {s: 0.9 + s * 0.001 for s in range(6)}
        b = {s: 0.8 + s * 0.002 for s in range(6)}

        result = paired_comparison("accuracy", a, b, test="wilcoxon")

        assert result.min_achievable_p == pytest.approx(0.03125)

    def test_the_t_test_keeps_no_floor_with_ties(self) -> None:
        a = {s: 0.80 + 0.01 * s for s in range(6)}
        b = dict(a)
        b[0] -= 0.02

        result = paired_comparison("accuracy", a, b, test="paired_t")

        assert result.min_achievable_p == 0.0


# Ten seeds whose differences pass Shapiro-Wilk (auto picks the paired t) with
# p = 0.069: the percentile bootstrap of the old code gave [0.0017, 0.0263],
# excluding zero beside a non-significant p (STAT-3, ADR-187).
_BORDERLINE = [
    0.0287,
    0.0078,
    0.0046,
    0.0101,
    0.0135,
    -0.0133,
    0.0267,
    0.0538,
    0.0263,
    -0.019,
]

# Hollander & Wolfe's Hamilton depression scores (also R's ?wilcox.test
# example): first and second visit of nine patients.
_HAMILTON_X = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]
_HAMILTON_Y = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29]


def _pairs(differences: list[float]) -> tuple[dict[int, float], dict[int, float]]:
    base = 0.8
    return (
        {i: base + d for i, d in enumerate(differences)},
        dict.fromkeys(range(len(differences)), base),
    )


def _walsh(differences: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(differences.size)
    return np.sort((differences[rows] + differences[cols]) / 2.0)


def _excludes_zero(result: PairedComparison) -> bool:
    low, high = result.ci_low, result.ci_high
    assert low is not None
    assert high is not None
    return low > 0 or high < 0


class TestDifferenceIntervalMatchesTheTest:
    """The interval of the difference inverts the test beside it (ADR-187)."""

    def test_borderline_t_interval_contains_zero(self) -> None:
        a, b = _pairs(_BORDERLINE)
        result = paired_comparison("accuracy", a, b)

        assert result.test == "paired_t"
        assert 0.05 < result.p_value < 0.10
        assert result.ci_method == "t"
        assert not _excludes_zero(result)

    def test_t_interval_matches_scipy(self) -> None:
        a, b = _pairs(_BORDERLINE)
        expected = stats.ttest_rel(list(a.values()), list(b.values()))
        interval = expected.confidence_interval(0.95)

        result = paired_comparison("accuracy", a, b, test="paired_t")

        assert result.ci_low == pytest.approx(float(interval.low))
        assert result.ci_high == pytest.approx(float(interval.high))
        assert result.ci_estimate == pytest.approx(result.mean_difference)
        assert result.ci_confidence == 0.95

    def test_t_interval_formula(self) -> None:
        d = [0.2, -0.1, 0.4, 0.3]
        half = float(stats.t.ppf(0.975, 3)) * float(np.std(d, ddof=1)) / 2.0

        low, high = paired_t_ci(d)

        assert low == pytest.approx(0.2 - half)
        assert high == pytest.approx(0.2 + half)

    def test_t_interval_needs_two_differences(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            paired_t_ci([0.1])

    @pytest.mark.parametrize("sample", range(30))
    def test_t_interval_is_dual_to_the_t_test(self, sample: int) -> None:
        rng = np.random.default_rng(1000 + sample)
        n = int(rng.integers(6, 16))
        d = rng.normal(rng.normal(0.0, 0.6), 1.0, n).tolist()
        a, b = _pairs(d)

        result = paired_comparison("accuracy", a, b, test="paired_t")

        assert _excludes_zero(result) == (result.p_value < 0.05)

    @pytest.mark.parametrize("sample", range(30))
    def test_hodges_lehmann_is_dual_to_the_exact_wilcoxon(self, sample: int) -> None:
        # Continuous draws: no ties and no zero difference, so scipy's p is the
        # exact one and the duality is exact.
        rng = np.random.default_rng(2000 + sample)
        n = int(rng.integers(6, 16))
        d = rng.normal(rng.normal(0.0, 0.6), 1.0, n).tolist()
        a, b = _pairs(d)

        result = paired_comparison("accuracy", a, b, test="wilcoxon")

        assert result.ci_method == "hodges_lehmann"
        assert _excludes_zero(result) == (result.p_value < 0.05)

    def test_with_ties_the_duality_holds_up_to_a_documented_tolerance(self) -> None:
        # Ties and zero differences send scipy's p to the normal approximation
        # while the interval keeps the exact critical value: ADR-187 accepts
        # disagreement in at most 5% of such cases (1.7% measured at 1 decimal).
        rng = np.random.default_rng(7)
        disagreements = 0
        cases = 0
        for _ in range(300):
            n = int(rng.integers(6, 16))
            d = np.round(rng.normal(rng.normal(0.0, 0.5), 1.0, n), 1)
            if not np.any(d):
                continue
            a, b = _pairs(d.tolist())
            result = paired_comparison("accuracy", a, b, test="wilcoxon")
            cases += 1
            disagreements += _excludes_zero(result) != (result.p_value < 0.05)

        assert cases > 250
        assert disagreements <= 0.05 * cases


class TestHodgesLehmann:
    """Interval of the pseudo-median from Walsh averages, exact critical value."""

    def test_textbook_example(self) -> None:
        d = [x - y for x, y in zip(_HAMILTON_X, _HAMILTON_Y, strict=True)]
        walsh = _walsh(np.asarray(d))

        hl = hodges_lehmann_ci(d)

        # Hollander & Wolfe: estimate 0.46; for n = 9 the exact lower critical
        # count is P(T+ <= 5) = 10/512, so the interval is [W(6), W(40)] with
        # coverage 1 - 20/512.
        assert hl.estimate == pytest.approx(0.46)
        assert hl.ci_low == pytest.approx(walsh[5])
        assert hl.ci_high == pytest.approx(walsh[39])
        assert hl.confidence == pytest.approx(1 - 20 / 512)

    @pytest.mark.parametrize("sample", range(10))
    def test_matches_brute_force_inversion_of_the_exact_test(self, sample: int) -> None:
        # A shift is in the interval iff the exact two-sided Wilcoxon of
        # d - shift does not reject at 0.05. Probe between consecutive Walsh
        # averages (and beyond both ends) where the test is well defined.
        rng = np.random.default_rng(3000 + sample)
        d = rng.normal(0.3, 1.0, int(rng.integers(6, 13)))
        walsh = np.unique(_walsh(d))
        probes = np.concatenate(
            [[walsh[0] - 1.0], (walsh[:-1] + walsh[1:]) / 2.0, [walsh[-1] + 1.0]]
        )
        kept = [
            float(t)
            for t in probes
            if stats.wilcoxon(d - t, method="exact").pvalue > 0.05
        ]

        hl = hodges_lehmann_ci(d.tolist())

        assert hl.ci_low is not None
        assert hl.ci_high is not None
        inside = [float(t) for t in probes if hl.ci_low < t < hl.ci_high]
        assert inside == kept

    def test_signed_rank_distribution_matches_enumeration(self) -> None:
        n = 8
        totals = [
            sum(rank for rank, bit in enumerate(bits, start=1) if bit)
            for bits in itertools.product((0, 1), repeat=n)
        ]
        expected = np.cumsum(np.bincount(totals, minlength=37)) / 2**n

        np.testing.assert_allclose(signed_rank_cdf(n), expected)

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_no_finite_interval_below_six_pairs(self, n: int) -> None:
        # 2 / 2**n > 0.05: the most extreme sign pattern is not rare enough.
        hl = hodges_lehmann_ci([0.1 * (i + 1) for i in range(n)])

        assert hl.ci_low is None
        assert hl.ci_high is None

    def test_six_pairs_reach_the_extremes(self) -> None:
        d = [0.3, -0.1, 0.5, 0.2, 0.9, 0.4]

        hl = hodges_lehmann_ci(d)

        assert hl.ci_low == pytest.approx(-0.1)
        assert hl.ci_high == pytest.approx(0.9)
        assert hl.confidence == pytest.approx(1 - 2 / 64)

    def test_all_zero_differences_give_a_point_at_zero(self) -> None:
        hl = hodges_lehmann_ci([0.0] * 7)

        assert hl.ci_low == 0.0
        assert hl.ci_high == 0.0
        assert hl.estimate == 0.0

    def test_needs_a_difference(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            hodges_lehmann_ci([])

    def test_small_comparison_reports_no_interval_and_says_why(self) -> None:
        a, b = _pairs([0.02, 0.03, 0.01, 0.04, 0.02])

        result = paired_comparison("accuracy", a, b)

        assert result.test == "wilcoxon"
        assert result.underpowered
        assert result.ci_low is None
        assert result.ci_high is None
        assert result.ci_method == "hodges_lehmann"

    def test_identical_configs_get_a_zero_interval(self) -> None:
        same = {i: 0.8 + 0.01 * i for i in range(8)}

        result = paired_comparison("accuracy", same, dict(same))

        assert result.p_value == 1.0
        assert result.ci_low == 0.0
        assert result.ci_high == 0.0


class TestIntervalFollowsAlpha:
    """The interval is dual to the test at the comparison's own alpha (ADR-187).

    A 95% interval beside alpha=0.01 excluded zero while p=0.035 was not
    significant; beside alpha=0.10 it would contain zero next to a rejection.
    """

    def test_paired_t_interval_uses_one_minus_alpha(self) -> None:
        a = {0: 2.0, 1: 3.0, 2: 4.0}
        b = dict.fromkeys(a, 0.0)

        result = paired_comparison("accuracy", a, b, test="paired_t", alpha=0.01)

        # mean 3, sd 1, n 3: t(0.995, 2) = 9.9248, half-width 9.9248 / sqrt(3).
        half = float(stats.t.ppf(0.995, 2)) / math.sqrt(3)
        assert half == pytest.approx(5.730111, rel=1e-6)
        assert result.p_value == pytest.approx(0.035099, rel=1e-4)
        assert result.ci_low == pytest.approx(3.0 - half)
        assert result.ci_high == pytest.approx(3.0 + half)
        assert result.ci_confidence == pytest.approx(0.99)
        # p > alpha and the interval holds zero: the two agree.
        assert result.ci_low is not None
        assert result.ci_low < 0.0

    def test_a_looser_alpha_narrows_the_interval(self) -> None:
        a = {0: 2.0, 1: 3.0, 2: 4.0}
        b = dict.fromkeys(a, 0.0)

        result = paired_comparison("accuracy", a, b, test="paired_t", alpha=0.10)

        half = float(stats.t.ppf(0.95, 2)) / math.sqrt(3)
        assert result.ci_low == pytest.approx(3.0 - half)
        assert result.ci_high == pytest.approx(3.0 + half)
        assert result.ci_confidence == pytest.approx(0.90)

    def test_underpowered_wilcoxon_reports_no_interval_at_its_alpha(self) -> None:
        # Six pairs: floor 2/2**6 = 0.03125 > 0.01, and no 99% Hodges-Lehmann
        # interval exists either, for the same reason.
        a = {i: 1.1 + 0.1 * i for i in range(6)}
        b = dict.fromkeys(a, 1.0)

        result = paired_comparison("accuracy", a, b, alpha=0.01)

        assert result.test == "wilcoxon"
        assert result.min_achievable_p == pytest.approx(0.03125)
        assert result.underpowered is True
        assert result.ci_low is None
        assert result.ci_high is None

    def test_hodges_lehmann_reaches_the_requested_coverage(self) -> None:
        differences = [0.01 * (i + 1) for i in range(10)]
        a, b = _pairs(differences)

        result = paired_comparison("accuracy", a, b, test="wilcoxon", alpha=0.01)

        expected = hodges_lehmann_ci(differences, confidence=0.99)
        assert expected.ci_low is not None
        assert result.ci_low == pytest.approx(expected.ci_low)
        assert result.ci_high == pytest.approx(expected.ci_high)
        assert result.ci_confidence == pytest.approx(expected.confidence)
        assert expected.confidence >= 0.99

    def test_the_matrix_passes_alpha_to_the_interval(self) -> None:
        differences = [0.01 * (i + 1) for i in range(10)]
        a, b = _pairs(differences)

        (result,) = comparison_matrix("accuracy", {"a": a, "b": b}, alpha=0.01)

        # Ten evenly spaced differences pass Shapiro-Wilk: a paired t.
        assert result.ci_method == "t"
        assert result.ci_confidence == pytest.approx(0.99)
        low, high = paired_t_ci(differences, confidence=0.99)
        assert result.ci_low == pytest.approx(low)
        assert result.ci_high == pytest.approx(high)


class TestNonFiniteValues:
    """A NaN or inf is not a measurement: it stays out of the test."""

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_seed_is_left_out_of_the_pair(self, bad: float) -> None:
        a = {s: 0.2 + 0.001 * s for s in range(8)}
        b = {s: 0.1 + 0.003 * s for s in range(8)}
        a[0] = bad

        result = paired_comparison("rmse", a, b)
        clean = paired_comparison(
            "rmse",
            {s: v for s, v in a.items() if s != 0},
            {s: v for s, v in b.items() if s != 0},
        )

        assert result.n_pairs == 7
        assert result.p_value == clean.p_value
        assert result.mean_a == pytest.approx(clean.mean_a)
        assert math.isfinite(result.effect_size)

    def test_a_nan_p_value_does_not_stop_holm(self) -> None:
        from dataclasses import replace

        base = paired_comparison(
            "m", {1: 1.0, 2: 1.0, 3: 1.0}, {1: 1.0, 2: 1.0, 3: 1.0}
        )
        comparisons = [
            replace(base, p_value=math.nan, label_a="broken"),
            replace(base, p_value=0.001, label_a="clear"),  # 0.05/2 -> reject
        ]

        flagged = holm_correct(comparisons)

        assert [c.label_a for c in flagged] == ["broken", "clear"]
        assert [c.significant for c in flagged] == [False, True]

    def test_a_nan_effect_size_is_not_large(self) -> None:
        from visionforge.core.significance import _effect_label

        assert _effect_label(math.nan) == "undefined"


class TestAlphaRange:
    """alpha is a probability strictly between 0 and 1; anything else is a typo."""

    BAD = [0.0, 1.0, -0.05, 1.5, math.nan, math.inf]

    @pytest.mark.parametrize("alpha", BAD)
    def test_paired_comparison_rejects_it(self, alpha: float) -> None:
        a, b = _pairs([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="alpha"):
            paired_comparison("m", a, b, alpha=alpha)

    @pytest.mark.parametrize("alpha", BAD)
    def test_holm_correct_rejects_it(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            holm_correct([], alpha=alpha)

    @pytest.mark.parametrize("alpha", BAD)
    def test_comparison_matrix_rejects_it(self, alpha: float) -> None:
        groups = {"a": {1: 0.1, 2: 0.2, 3: 0.3}, "b": {1: 0.2, 2: 0.3, 3: 0.5}}
        with pytest.raises(ValueError, match="alpha"):
            comparison_matrix("m", groups, alpha=alpha)

    def test_the_message_is_in_both_languages(self) -> None:
        from visionforge.utils.messages import using_lang

        a, b = _pairs([0.1, 0.2, 0.3, 0.4])
        with using_lang("pt"), pytest.raises(ValueError, match="entre 0 e 1"):
            paired_comparison("m", a, b, alpha=2.0)
        with using_lang("en"), pytest.raises(ValueError, match="between 0 and 1"):
            paired_comparison("m", a, b, alpha=2.0)
