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

TestKind = Literal["paired_t", "wilcoxon"]

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
    ci_low: float  # bootstrap CI of the paired difference
    ci_high: float
    # Smallest p the chosen test could possibly return with this many pairs.
    min_achievable_p: float = 0.0
    # True when that floor is above alpha: no result, however consistent, can
    # reach significance — "not significant" then means "not enough seeds".
    underpowered: bool = False
    significant: bool = False  # set by holm_correct over a family of tests

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


# Substrings that mark a metric as lower-is-better. Ranking without this gets
# the winner exactly backwards for error metrics — a mistake that reads as
# authoritative because it comes with a p-value attached.
_LOWER_IS_BETTER = ("loss", "mae", "mse", "rmse", "error", "err", "distance")


def infer_direction(metric: str) -> Literal["higher", "lower"]:
    """Guess whether a metric is better when higher or lower.

    A heuristic, and named as one: task-declared directions
    (``@register_task(metrics=...)``) should win where they exist. It covers
    the built-in vocabulary — accuracy/f1/auc/r2/miou/dice/map/auroc are
    higher-better; mae/mse/rmse/loss are lower-better.
    """
    lowered = metric.lower()
    return "lower" if any(token in lowered for token in _LOWER_IS_BETTER) else "higher"


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
    seed: int = 0,
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
    ci = bootstrap_ci(differences, seed=seed)
    floor = min_achievable_p(kind, len(shared))
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
        ci_low=ci.ci_low if ci else math.nan,
        ci_high=ci.ci_high if ci else math.nan,
        min_achievable_p=floor,
        underpowered=floor > alpha,
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
    seed: int = 0,
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
                        seed=seed,
                    )
                )
            except ValueError:
                continue
    return holm_correct(comparisons, alpha=alpha)


__all__ = [
    "BootstrapCI",
    "PairedComparison",
    "TestKind",
    "bootstrap_ci",
    "cohens_dz",
    "comparison_matrix",
    "holm_correct",
    "paired_comparison",
]
