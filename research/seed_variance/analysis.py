"""Statistics for the seed-variance study, over the reports run_study.py saves.

Pure functions over plain numbers, so every one can be checked by hand in the
tests. The report format is VisionForge's replicated-comparison report
(``core/replicated_comparison.py::build_report``); ``per_seed`` there maps
seed -> metric, and JSON turns those seeds into strings.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from statistics import NormalDist
from typing import Any


def per_seed(report: dict[str, Any], variant: str) -> dict[int, float]:
    """``seed -> metric`` for one variant of a saved report."""
    raw = report["variants"][variant]["per_seed"]
    return {int(seed): float(value) for seed, value in raw.items()}


def paired_differences(
    report: dict[str, Any], variant: str, baseline: str = "baseline"
) -> list[float]:
    """``variant - baseline`` per seed, over the seeds both finished, in seed order."""
    a = per_seed(report, variant)
    b = per_seed(report, baseline)
    return [a[s] - b[s] for s in sorted(a.keys() & b.keys())]


def reversal_rate(
    diffs: list[float], k: int, *, draws: int = 2000, seed: int = 0
) -> float:
    """How often the mean of ``k`` seeds points the other way from the mean of all.

    This is the bet a one-seed (k=1) or three-seed ablation makes without
    saying so: had other seeds been drawn, would the variant still look better,
    or worse? Every subset is enumerated when there are at most ``draws`` of
    them; otherwise ``draws`` subsets are sampled with a fixed seed. A subset
    whose mean difference is exactly zero counts as a reversal — it would not
    have shown the effect either.

    Raises:
        ValueError: ``k`` outside ``1..len(diffs)``, or an overall mean of zero,
            which leaves no direction to reverse.
    """
    if not 1 <= k <= len(diffs):
        raise ValueError(f"k must be between 1 and {len(diffs)}, got {k}")
    overall = statistics.fmean(diffs)
    if overall == 0:
        raise ValueError("The overall mean difference is zero; nothing to reverse.")
    sign = math.copysign(1.0, overall)

    n_subsets = math.comb(len(diffs), k)
    if n_subsets <= draws:
        subsets: Any = itertools.combinations(diffs, k)
        total = n_subsets
    else:
        rng = random.Random(seed)
        subsets = (rng.sample(diffs, k) for _ in range(draws))
        total = draws
    flips = sum(1 for subset in subsets if statistics.fmean(subset) * sign <= 0)
    return flips / total


def seeds_for_power(
    diffs: list[float], *, alpha: float = 0.05, power: float = 0.8
) -> int | None:
    """Seeds a paired comparison needs to detect this effect.

    Normal approximation: n = ((z_{1-alpha/2} + z_power) * sd_d / |mean_d|)^2,
    rounded up, never below 2. It is a planning figure, not a test — the
    study's significance calls come from VisionForge's paired tests with Holm.
    Returns None when the observed mean difference is zero.
    """
    mean = statistics.fmean(diffs)
    if mean == 0:
        return None
    sd = statistics.stdev(diffs)
    z = NormalDist().inv_cdf
    n = ((z(1 - alpha / 2) + z(power)) * sd / abs(mean)) ** 2
    return max(2, math.ceil(n))


def noise_to_effect(baseline_values: list[float], diffs: list[float]) -> float:
    """Seed SD of the baseline over the absolute mean effect.

    Above 1, one run's luck is larger than the change being measured.
    Infinite when the mean effect is zero.
    """
    effect = abs(statistics.fmean(diffs))
    sd = statistics.stdev(baseline_values)
    return math.inf if effect == 0 else sd / effect
