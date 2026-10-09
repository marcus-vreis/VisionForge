"""Replicated comparison: N seeds per variant, then a paired test (ADR-061).

The honest way to claim "A beats B". A sweep ranks one run per configuration,
so its winner partly reflects seed luck (ADR-045 says so in its own table
note); replicates quantify one configuration's spread but compare nothing.
This module closes the loop: every variant is trained on the **same seed
list**, and the resulting per-seed vectors go into the paired analysis of
``core.significance`` — same seed, same split and initialization, so the
difference isolates the change under study.

Task-agnostic: it drives the uniform ``TaskRunner`` handle, so classification,
regression, segmentation, detection, anomaly and researcher-defined tasks all
get it from the same code.
"""

from __future__ import annotations

import copy
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from loguru import logger

from visionforge.core.cancellation import is_cancelled
from visionforge.core.replicates import (
    ReplicateTrial,
    aggregate_replicates,
    run_replicates,
)
from visionforge.core.significance import comparison_matrix, infer_direction
from visionforge.core.task_runner import (
    TaskRunner,
    runner_cancel_token,
    runner_metric_direction,
)


@dataclass
class VariantResult:
    """One variant's replicate set."""

    label: str
    overrides: dict[str, Any]
    trials: list[ReplicateTrial] = field(default_factory=list)

    @property
    def successful(self) -> list[ReplicateTrial]:
        """Trials that produced metrics."""
        return [t for t in self.trials if t.status == "success"]

    def per_seed(self, metric: str) -> dict[int, float]:
        """``seed -> metric`` for successful trials that reported it."""
        return {
            t.seed: float(t.metrics[metric])
            for t in self.successful
            if metric in t.metrics
        }


def _set_nested(target: dict[str, Any], dot_key: str, value: Any) -> None:
    """Set a dot-path on a config dict; raise when the path does not exist."""
    keys = dot_key.split(".")
    node = target
    for key in keys[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            raise ValueError(f"Unknown override path '{dot_key}' (failed at '{key}')")
        node = nxt
    if keys[-1] not in node:
        raise ValueError(f"Unknown override path '{dot_key}' (failed at '{keys[-1]}')")
    node[keys[-1]] = value


def validate_variants(
    base_config_dict: dict[str, Any], variants: dict[str, dict[str, Any]]
) -> None:
    """Reject unknown override paths before any GPU time is spent.

    Raises:
        ValueError: for an empty variant set, a single variant (nothing to
            compare against), or an override path the config does not have.
    """
    if len(variants) < 2:
        raise ValueError(
            f"A comparison needs at least 2 variants, got {len(variants)}. "
            "Add the baseline as its own variant with no overrides."
        )
    for label, overrides in variants.items():
        for path, value in overrides.items():
            try:
                _set_nested(copy.deepcopy(base_config_dict), path, value)
            except ValueError as exc:
                raise ValueError(f"Variant '{label}': {exc}") from exc


def run_replicated_comparison(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    seeds: list[int],
    metric: str,
    *,
    alpha: float = 0.05,
    direction: Literal["higher", "lower"] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train every variant over ``seeds`` and return the comparison report.

    Each variant's name is suffixed onto the run name so its checkpoints and
    ``run.json`` land in their own directory. A variant that fails entirely is
    kept in the report (with its error) but takes no part in the tests —
    silently dropping it would make the matrix look complete when it is not.

    A stop carried by the runner (ADR-111) ends the variant in flight at its
    current replicate and starts no further variant; the report then covers
    the variants and seeds that actually ran, paired on the seeds they share.

    Raises:
        ValueError: for invalid variants (see :func:`validate_variants`) or
            fewer than two seeds, which leaves no dispersion to test.
    """
    validate_variants(base_config_dict, variants)
    if len(seeds) < 2:
        raise ValueError(
            f"A paired comparison needs at least 2 seeds, got {len(seeds)}."
        )

    base_name = str(base_config_dict.get("name", "comparison"))
    results: list[VariantResult] = []
    total = len(variants)

    for index, (label, overrides) in enumerate(variants.items()):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "trial_start",
                    "trial_index": index,
                    "total_trials": total,
                    "overrides": {"variant": label, **overrides},
                }
            )
        variant_config = copy.deepcopy(base_config_dict)
        variant_config["name"] = f"{base_name}_{label}"
        for path, value in overrides.items():
            _set_nested(variant_config, path, value)

        logger.info(
            "Replicated comparison: variant '{}' ({}/{}) over {} seeds",
            label,
            index + 1,
            total,
            len(seeds),
        )
        # No inner progress_callback: one variant's seeds would otherwise look
        # like the whole run's trials to the live monitor.
        trials = run_replicates(runner, variant_config, seeds, metric)
        results.append(VariantResult(label=label, overrides=overrides, trials=trials))

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "trial_end",
                    "trial_index": index,
                    "total_trials": total,
                    "status": "success"
                    if any(t.status == "success" for t in trials)
                    else "failed",
                }
            )
        if is_cancelled(runner_cancel_token(runner)):
            logger.info(
                "Replicated comparison stopped after {} of {} variants.",
                index + 1,
                total,
            )
            break

    ran = {r.label for r in results}
    # The task's declared direction, where it has one, over the name heuristic.
    direction = direction or runner_metric_direction(runner, metric)
    return build_report(
        results,
        seeds,
        metric,
        alpha=alpha,
        direction=direction,
        not_run=[label for label in variants if label not in ran],
    )


def build_report(
    results: list[VariantResult],
    seeds: list[int],
    metric: str,
    *,
    alpha: float = 0.05,
    direction: Literal["higher", "lower"] | None = None,
    not_run: list[str] | None = None,
) -> dict[str, Any]:
    """Shape variant results + the paired matrix into a GUI/JSON report.

    ``direction`` says whether the metric is better high or low; when omitted
    it is inferred from the name. Getting this wrong crowns the *worst*
    variant — and the mistake reads as authoritative because it arrives with
    a p-value beside it. ``not_run`` names requested variants that never
    started, so a stopped comparison does not simply lose them (ADR-111).
    """
    groups = {r.label: r.per_seed(metric) for r in results}
    comparable = {label: values for label, values in groups.items() if len(values) >= 2}
    comparisons = comparison_matrix(metric, comparable, alpha=alpha)

    variants: dict[str, Any] = {}
    for result in results:
        variants[result.label] = {
            "overrides": result.overrides,
            "aggregates": aggregate_replicates(result.trials),
            "trials": [asdict(t) for t in result.trials],
            "successful": len(result.successful),
            "per_seed": result.per_seed(metric),
        }

    # "Best" is reported by mean only, deliberately without a claim of
    # significance — that is what the comparison matrix is for.
    resolved_direction = direction or infer_direction(metric)
    ranking_seeds, ranked = _rank_on_shared_seeds(comparable, resolved_direction)

    return {
        "kind": "replicated_comparison",
        "metric": metric,
        "metric_direction": resolved_direction,
        "seeds": list(seeds),
        "alpha": alpha,
        "variants": variants,
        "comparisons": [c.to_dict() for c in comparisons],
        "best_by_mean": ranked[0] if ranked else None,
        "ranked_by_mean": ranked,
        "ranking_seeds": ranking_seeds,
        "significant_pairs": sum(1 for c in comparisons if c.significant),
        "skipped_variants": [label for label in groups if label not in comparable],
        # Loudest possible flag: when every comparison is underpowered, a "not
        # significant" verdict says nothing about the effect — only that there
        # were too few seeds for the test to ever reject.
        "underpowered": bool(comparisons) and all(c.underpowered for c in comparisons),
        "power_note": _power_note(comparisons, alpha),
        "not_run": list(not_run or []),
    }


def _rank_on_shared_seeds(
    comparable: dict[str, dict[int, float]], direction: Literal["higher", "lower"]
) -> tuple[list[int], list[str]]:
    """Order the comparable variants by their mean over the seeds all of them finished.

    A ranking is a comparison too. Means over different seed sets (a variant
    stopped after one seed, or one whose seed failed) compare seed luck as much
    as the variants, so only variants that the paired tests could take are
    ranked, and only on their shared seeds. With every seed finished this is
    the plain mean of each variant.
    """
    if not comparable:
        return [], []
    shared = sorted(set.intersection(*(set(v) for v in comparable.values())))
    if len(shared) < 2:
        return [], []
    means = {
        label: statistics.fmean(values[seed] for seed in shared)
        for label, values in comparable.items()
    }
    ranked = sorted(means, key=means.__getitem__, reverse=direction == "higher")
    return shared, ranked


def _power_note(comparisons: list[Any], alpha: float) -> str:
    """Explain, in the report, when the seed count caps what can be concluded."""
    blocked = [c for c in comparisons if c.underpowered]
    if not blocked:
        return ""
    # The seeds actually paired, not the ones requested: after a stop or a
    # failed seed the test ran on fewer, and that is what capped it.
    worst = max(blocked, key=lambda c: c.min_achievable_p)
    return (
        f"With {worst.n_pairs} paired seeds the rank test cannot return a "
        f"p-value below {worst.min_achievable_p:.4f}, which is above "
        f"alpha={alpha}: no difference, however large and consistent, can be "
        f"flagged significant. Increase the number of seeds (6+ for "
        f"alpha=0.05) before concluding anything from a non-significant result."
    )


__all__ = [
    "VariantResult",
    "build_report",
    "run_replicated_comparison",
    "validate_variants",
]
