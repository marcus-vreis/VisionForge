"""Task-agnostic multi-seed replicates over the `TaskRunner` handle (ADR-056).

A single training run is one sample from a noisy distribution — seed-to-seed
variance in deep learning routinely exceeds the gap between two architectures,
so a conclusion drawn from one run is statistically meaningless. This module
trains the *same* config N times under different seeds and aggregates each
metric into mean / std / min / max and a 95% confidence interval, giving every
task a defensible "metric = mean ± CI" instead of a single point estimate.

It knows nothing about a task's internals — only the uniform handle — so the
same function serves classification, regression, segmentation, detection and
anomaly. Per-replicate config is built by overriding `training.seed` (shared by
every task config) and suffixing `name` so each replicate keeps its own run dir.
"""

from __future__ import annotations

import copy
import gc
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from visionforge.core.task_runner import RunResult, TaskRunner

try:  # torch is the heavy hardware extra; the cache flush is best-effort.
    import torch
except ImportError:  # pragma: no cover - torch is always present in practice
    torch = None  # type: ignore[assignment]


@dataclass
class ReplicateTrial:
    """One seed's result inside a multi-seed replicate set."""

    seed: int
    status: str = "failed"
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_s: float | None = None
    error: str = ""


def _t_critical_95(dof: int) -> float:
    """Two-sided 97.5% Student-t critical value for ``dof`` degrees of freedom.

    Uses scipy (a scikit-learn transitive dependency) when available; falls back
    to the normal approximation (1.96) so the aggregation never fails.
    """
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, dof))
    except Exception:  # noqa: BLE001 - CI must not depend on scipy internals
        return 1.96


def aggregate_replicates(
    trials: list[ReplicateTrial],
) -> dict[str, dict[str, float | int | None]]:
    """Aggregate successful trials into per-metric distribution statistics.

    For every metric key observed across successful trials, returns ``n``,
    ``mean``, ``std`` (sample, n-1), ``min``, ``max`` and a Student-t 95%
    confidence interval (``ci95_low``/``ci95_high``). With a single value the
    dispersion fields are ``None`` — one sample has no spread to report.
    """
    successful = [t for t in trials if t.status == "success"]
    keys: list[str] = list(dict.fromkeys(k for t in successful for k in t.metrics))

    aggregates: dict[str, dict[str, float | int | None]] = {}
    for key in keys:
        values = [t.metrics[key] for t in successful if key in t.metrics]
        n = len(values)
        mean = statistics.fmean(values)
        if n >= 2:
            std = statistics.stdev(values)
            half_width = _t_critical_95(n - 1) * std / math.sqrt(n)
            ci_low: float | None = mean - half_width
            ci_high: float | None = mean + half_width
        else:
            std = None  # type: ignore[assignment]
            ci_low = ci_high = None
        aggregates[key] = {
            "n": n,
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }
    return aggregates


def run_replicates(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    seeds: list[int],
    metric: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[ReplicateTrial]:
    """Train ``base_config_dict`` once per seed and return the trials in seed order.

    Each replicate overrides ``training.seed`` and suffixes ``name`` with
    ``_s{seed}`` so its checkpoints and run.json land in a distinct run dir.
    Replicates are never ranked — they are samples of one distribution, not
    competitors. A failed replicate is recorded and the set continues; GPU
    memory is released between replicates. ``progress_callback`` receives
    ``trial_start``/``trial_end`` events so the GUI overlay tracks real
    progress across the set.
    """
    trials: list[ReplicateTrial] = []
    base_name = str(base_config_dict.get("name", "replicates"))

    for index, seed in enumerate(seeds):
        trial = ReplicateTrial(seed=seed)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "trial_start",
                    "trial_index": index,
                    "total_trials": len(seeds),
                    "overrides": {"training.seed": seed},
                }
            )
        try:
            trial_dict = copy.deepcopy(base_config_dict)
            trial_dict["name"] = f"{base_name}_s{seed}"
            trial_dict["training"] = {
                **trial_dict.get("training", {}),
                "seed": seed,
            }
            cfg = runner.config_type.model_validate(trial_dict)
            result: RunResult = runner.run(cfg)
            trial.status = result.status
            trial.metrics = runner.metrics(result)
            trial.training_time_s = result.training_time_s
            trial.error = result.error
            if result.status == "success":
                logger.info(
                    "Replicate seed={} ok — {}={}",
                    seed,
                    metric,
                    trial.metrics.get(metric),
                )
            else:
                logger.warning("Replicate seed={} failed — {}", seed, result.error)
        except Exception as exc:  # noqa: BLE001
            trial.error = str(exc)
            logger.warning("Replicate seed={} failed — {}", seed, exc)
        finally:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "trial_end",
                    "trial_index": index,
                    "total_trials": len(seeds),
                    "status": trial.status,
                }
            )
        trials.append(trial)

    return trials


__all__ = ["ReplicateTrial", "aggregate_replicates", "run_replicates"]
