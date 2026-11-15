"""Significance testing and bootstrap intervals for replicated results (ADR-061).

ADR-056 gave every task ``metric = mean ± 95% CI`` over multiple seeds. That
answers "how uncertain is this number?" but not the question a paper actually
asks: **is A better than B, or is the gap seed noise?** This module answers it
with a paired analysis over the seeds two configs share.

Pairing is the point. Two configs trained under the same seed share the data
split, the initialization and the augmentation stream, so their difference
isolates the change under study; an unpaired test would drown that signal in
between-seed variance. The helpers here therefore refuse to compare runs whose
seeds do not line up, rather than silently producing a weaker (or wrong) test.

Everything is a pure function over lists of floats — no task, no torch — so it
is testable against textbook values and reusable by any task's report.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from visionforge.core.metric_vocabulary import is_lower_is_better

TestKind = Literal["paired_t", "wilcoxon"]
# How the interval of the paired difference was built: the dual of the test that
# produced the p-value next to it (ADR-187), so the two cannot disagree.
CiMethod = Literal["t", "hodges_lehmann"]

# Confidence of the paired-difference interval: dual to a two-sided test at
# alpha = 0.05, the comparison's significance level.
_CI_CONFIDENCE = 0.95

# Below this many pairs a normality check is meaningless, so the rank-based
# test is the honest default.
_MIN_PAIRS_FOR_NORMALITY = 8


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap interval for the mean of one metric."""

    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    n: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for the run report."""
        return asdict(self)


@dataclass(frozen=True)
class PairedComparison:
    """Outcome of comparing two configs over the seeds they share."""

    label_a: str
    label_b: str
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_difference: float  # a - b
    test: TestKind
    test_reason: str
    statistic: float
    p_value: float
    effect_size: float  # Cohen's d_z (paired)
    effect_label: str
    # 95% interval of the paired difference, built by inverting the test that
    # gave ``p_value`` (``ci_method``). None at an end the data cannot bound:
    # below 6 pairs no Hodges-Lehmann interval reaches 95% (ADR-187).
    ci_low: float | None
    ci_high: float | None
    # Smallest p the chosen test could possibly return with this many pairs.
    min_achievable_p: float = 0.0
    # True when that floor is above alpha: no result, however consistent, can
    # reach significance — "not significant" then means "not enough seeds".
    underpowered: bool = False
    significant: bool = False  # set by holm_correct over a family of tests
    # "t": Student-t interval of the mean difference (dual of the paired t).
    # "hodges_lehmann": Walsh-average interval with the exact signed-rank
    # critical value (dual of the Wilcoxon); it brackets the pseudo-median.
    ci_method: CiMethod = "t"
    # The point the interval brackets: the mean difference for "t", the
    # Hodges-Lehmann estimate (median of the Walsh averages) otherwise.
    ci_estimate: float | None = None
    # Coverage the interval actually has: the exact signed-rank distribution is
    # discrete, so the Hodges-Lehmann one is at least 95%, rarely exactly.
    ci_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for the run report."""
        return asdict(self)


def bootstrap_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapCI | None:
    """Percentile bootstrap CI for the mean; ``None`` for fewer than 2 values.

    Complements the Student-t interval of ADR-056: t assumes the sampling
    distribution of the mean is normal, which a handful of seeds cannot
    support. Reporting both is honest about how much an interval depends on
    that assumption.

    **Neither is trustworthy below ~5 seeds.** With n=2 the t interval is
    absurdly wide (routinely crossing zero for a strictly positive metric)
    while the percentile bootstrap is far too narrow — it can only resample
    the two values it has, so it describes the sample, not the population.
    Callers should surface that caveat rather than let a tight interval imply
    precision that does not exist.
    """
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    means = rng.choice(sample, size=(n_resamples, sample.size), replace=True).mean(
        axis=1
    )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return BootstrapCI(
        mean=float(sample.mean()),
        ci_low=float(low),
        ci_high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
        n=int(sample.size),
    )


def paired_t_ci(
    differences: list[float], *, confidence: float = _CI_CONFIDENCE
) -> tuple[float, float]:
    """Student-t interval of the mean paired difference.

    ``mean +- t(1 - (1 - confidence)/2, n - 1) * sd / sqrt(n)``: the set of
    shifts a two-sided paired t test at ``1 - confidence`` does not reject, so
    it excludes zero exactly when that test's p is below ``1 - confidence``.

    Raises:
        ValueError: with fewer than two differences (no spread to scale by).
    """
    n = len(differences)
    if n < 2:
        raise ValueError(f"a t interval needs at least 2 differences, got {n}")
    mean = statistics.fmean(differences)
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    half_width = critical * statistics.stdev(differences) / math.sqrt(n)
    return mean - half_width, mean + half_width


def signed_rank_cdf(n: int) -> np.ndarray:
    """``P(T+ <= t)`` for ``t = 0 .. n(n+1)/2`` under the null, without ties.

    Under the null each rank 1..n enters ``T+`` with probability 1/2,
    independently, so the distribution is the convolution of n two-point laws;
    built here rank by rank (exact up to float rounding).
    """
    total = n * (n + 1) // 2
    pmf = np.zeros(total + 1)
    pmf[0] = 1.0
    for rank in range(1, n + 1):
        shifted = np.zeros_like(pmf)
        shifted[rank:] = pmf[:-rank]
        pmf = 0.5 * (pmf + shifted)
    return np.cumsum(pmf)


@dataclass(frozen=True)
class HodgesLehmannCI:
    """Hodges-Lehmann estimate and its distribution-free interval."""

    estimate: float
    ci_low: float | None  # None when no interval reaches the confidence
    ci_high: float | None
    confidence: float  # coverage actually attained (>= the requested one)


def hodges_lehmann_ci(
    differences: list[float], *, confidence: float = _CI_CONFIDENCE
) -> HodgesLehmannCI:
    """Interval of the pseudo-median from the Walsh averages (Hollander & Wolfe).

    With the ``M = n(n+1)/2`` Walsh averages ``(d_i + d_j)/2, i <= j`` sorted,
    the interval is ``[W_(k), W_(M+1-k)]`` where ``k`` is the largest integer
    with ``P(T+ <= k-1) <= (1 - confidence)/2`` under the exact null
    distribution of the signed-rank statistic. A shift is inside it exactly
    when the exact two-sided Wilcoxon test of ``d - shift`` does not reject, so
    it excludes zero exactly when that test's p is below ``1 - confidence``.

    Exact without ties or zero differences. With them scipy's p switches to
    the normal approximation while this interval keeps the exact critical
    value, so at the boundary the two can disagree (rarely; ADR-187).

    Below 6 differences even the most extreme ranks have probability above
    2.5% (``2/2**n > 0.05``): no finite interval reaches 95%, and both ends
    come back ``None`` (the test is underpowered for the same reason).

    Raises:
        ValueError: when ``differences`` is empty.
    """
    n = len(differences)
    if n == 0:
        raise ValueError("a Hodges-Lehmann interval needs at least 1 difference")
    values = np.asarray(differences, dtype=float)
    rows, cols = np.triu_indices(n)
    walsh = np.sort((values[rows] + values[cols]) / 2.0)
    estimate = float(np.median(walsh))
    cdf = signed_rank_cdf(n)
    tail = (1.0 - confidence) / 2.0
    # Number of t with P(T+ <= t) <= tail, i.e. the largest admissible k.
    k = int(np.searchsorted(cdf, tail, side="right"))
    if k == 0:
        return HodgesLehmannCI(estimate, None, None, 1.0)
    return HodgesLehmannCI(
        estimate=estimate,
        ci_low=float(walsh[k - 1]),
        ci_high=float(walsh[walsh.size - k]),
        confidence=float(1.0 - 2.0 * cdf[k - 1]),
    )


def _difference_interval(
    kind: TestKind, differences: list[float]
) -> tuple[CiMethod, float | None, float | None, float, float]:
    """The interval dual to ``kind``: method, low, high, estimate, coverage."""
    if kind == "paired_t":
        low, high = paired_t_ci(differences)
        return "t", low, high, statistics.fmean(differences), _CI_CONFIDENCE
    hl = hodges_lehmann_ci(differences)
    return "hodges_lehmann", hl.ci_low, hl.ci_high, hl.estimate, hl.confidence


def cohens_dz(differences: list[float]) -> float:
    """Paired effect size: mean difference in units of its own SD.

    Returns 0.0 when every pair moved identically (zero spread) — the effect
    is then perfectly consistent but unmeasurable on this scale, and inf would
    poison downstream formatting.
    """
    if len(differences) < 2:
        return 0.0
    sd = statistics.stdev(differences)
    if sd == 0:
        return 0.0
    return statistics.fmean(differences) / sd


def infer_direction(metric: str) -> Literal["higher", "lower"]:
    """Guess whether a metric is better when higher or lower.

    A heuristic, and named as one: task-declared directions
    (``@register_task(metrics=...)``) should win where they exist. The
    vocabulary lives in ``core/metric_vocabulary.py`` (shared with the GUI):
    accuracy/f1/auc/r2/miou/dice/map/auroc are higher-better; losses and errors
    (mae/mse/mape/nll/wer/fid/ece...) are lower-better. Ranking an error metric
    the wrong way crowns the worst variant, next to an authoritative p-value.
    """
    return "lower" if is_lower_is_better(metric) else "higher"


def min_achievable_p(test: TestKind, n_pairs: int) -> float:
    """The smallest two-sided p this test can return with ``n_pairs`` pairs.

    Wilcoxon's signed-rank statistic is discrete: with every difference
    pointing the same way it still only reaches ``2 / 2**n``. At n=5 that is
    0.0625 — **above 0.05, so five seeds can never be significant no matter
    how large and consistent the gap is**. Surfacing this floor is the
    difference between "we found no effect" and "we could not have found one".

    The t test is continuous, so it has no such floor (0.0 is returned).
    """
    if test == "wilcoxon":
        if n_pairs < 1:
            return 1.0
        return min(1.0, 2.0 ** (1 - n_pairs))
    return 0.0


def _effect_label(dz: float) -> str:
    """Cohen's conventional bands, stated as guidance rather than truth."""
    magnitude = abs(dz)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def _choose_test(differences: list[float]) -> tuple[TestKind, str]:
    """Pick the paired test and say why, so the choice is auditable."""
    n = len(differences)
    if n < _MIN_PAIRS_FOR_NORMALITY:
        return (
            "wilcoxon",
            f"{n} pairs: too few to check normality, rank-based test is safer",
        )
    if len(set(differences)) == 1:
        return "wilcoxon", "all differences identical: normality test undefined"
    normality_p = float(stats.shapiro(differences).pvalue)
    if normality_p > 0.05:
        return (
            "paired_t",
            f"differences pass Shapiro-Wilk (p={normality_p:.3f})",
        )
    return (
        "wilcoxon",
        f"differences fail Shapiro-Wilk (p={normality_p:.3f}), using ranks",
    )


def paired_comparison(
    metric: str,
    a: dict[int, float],
    b: dict[int, float],
    *,
    label_a: str = "A",
    label_b: str = "B",
    test: TestKind | Literal["auto"] = "auto",
    alpha: float = 0.05,
) -> PairedComparison:
    """Compare two configs over their shared seeds.

    ``a`` and ``b`` map ``seed -> metric value``. Only seeds present in both
    are used, because pairing is what makes the test sensitive.

    Raises:
        ValueError: when fewer than two seeds are shared — a paired test needs
            at least two differences to have any dispersion to reason about.
    """
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        raise ValueError(
            f"'{label_a}' and '{label_b}' share {len(shared)} seed(s); a paired "
            f"test needs at least 2. Run both configs over the same seeds."
        )

    values_a = [a[s] for s in shared]
    values_b = [b[s] for s in shared]
    differences = [x - y for x, y in zip(values_a, values_b, strict=True)]

    kind, reason = _choose_test(differences) if test == "auto" else (test, "explicit")

    if all(d == 0 for d in differences):
        # Both configs produced identical numbers on every shared seed: no test
        # can reject, and scipy raises on an all-zero Wilcoxon input.
        statistic, p_value = 0.0, 1.0
        reason = "identical on every shared seed"
    elif kind == "paired_t":
        result = stats.ttest_rel(values_a, values_b)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    else:
        result = stats.wilcoxon(values_a, values_b)
        statistic, p_value = float(result.statistic), float(result.pvalue)

    dz = cohens_dz(differences)
    # The interval inverts the same test as the p-value (ADR-187): a percentile
    # bootstrap here once excluded zero beside p > 0.05 with ten seeds.
    ci_method, ci_low, ci_high, ci_estimate, ci_confidence = _difference_interval(
        kind, differences
    )
    # scipy's Wilcoxon drops zero differences (zero_method="wilcox"), so a tied
    # pair adds nothing to the test and the floor is set by the others.
    tested = (
        len(shared) if kind == "paired_t" else sum(1 for d in differences if d != 0)
    )
    floor = min_achievable_p(kind, tested)
    return PairedComparison(
        label_a=label_a,
        label_b=label_b,
        metric=metric,
        n_pairs=len(shared),
        mean_a=statistics.fmean(values_a),
        mean_b=statistics.fmean(values_b),
        mean_difference=statistics.fmean(differences),
        test=kind,
        test_reason=reason,
        statistic=statistic,
        p_value=p_value,
        effect_size=dz,
        effect_label=_effect_label(dz),
        ci_low=ci_low,
        ci_high=ci_high,
        min_achievable_p=floor,
        underpowered=floor > alpha,
        ci_method=ci_method,
        ci_estimate=ci_estimate,
        ci_confidence=ci_confidence,
    )


def holm_correct(
    comparisons: list[PairedComparison], *, alpha: float = 0.05
) -> list[PairedComparison]:
    """Flag significance with Holm-Bonferroni control of the family-wise error.

    Comparing K configs means K(K-1)/2 tests, and at alpha=0.05 roughly one in
    twenty comes up "significant" by chance alone. Holm is uniformly more
    powerful than plain Bonferroni and needs no independence assumption, so it
    is the right default for a comparison matrix. Returned in the input order
    with ``significant`` set.
    """
    if not comparisons:
        return []
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i].p_value)
    total = len(comparisons)
    flags = [False] * total
    for rank, index in enumerate(order):
        threshold = alpha / (total - rank)
        if comparisons[index].p_value <= threshold:
            flags[index] = True
        else:
            break  # Holm stops at the first non-rejection
    return [
        PairedComparison(**{**asdict(c), "significant": flag})
        for c, flag in zip(comparisons, flags, strict=True)
    ]


def comparison_matrix(
    metric: str,
    groups: dict[str, dict[int, float]],
    *,
    alpha: float = 0.05,
) -> list[PairedComparison]:
    """Every pairwise paired comparison between groups, Holm-corrected.

    Groups whose seeds do not overlap enough are skipped rather than compared
    with a weaker test — silence is better than a misleading p-value.
    """
    labels = list(groups)
    comparisons: list[PairedComparison] = []
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1 :]:
            try:
                comparisons.append(
                    paired_comparison(
                        metric,
                        groups[label_a],
                        groups[label_b],
                        label_a=label_a,
                        label_b=label_b,
                        alpha=alpha,
                    )
                )
            except ValueError:
                continue
    return holm_correct(comparisons, alpha=alpha)


__all__ = [
    "BootstrapCI",
    "CiMethod",
    "HodgesLehmannCI",
    "PairedComparison",
    "TestKind",
    "bootstrap_ci",
    "cohens_dz",
    "comparison_matrix",
    "hodges_lehmann_ci",
    "holm_correct",
    "paired_comparison",
    "paired_t_ci",
    "signed_rank_cdf",
]
