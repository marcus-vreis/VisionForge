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
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from visionforge.core.cancellation import STOPPED, STOPPED_NOTE, is_cancelled
from visionforge.core.significance import bootstrap_ci, infer_direction
from visionforge.core.task_runner import (
    RunResult,
    TaskRunner,
    UnitProgress,
    discard_cut_unit_resume,
    runner_cancel_token,
)

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
    # The directory this seed trained into, "" when it never got one (a failed
    # seed). History needs it to tie the run to the group it belongs to
    # (ADR-113); the folder's name is the run id the History list uses.
    run_dir: str = ""


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


# Metrics whose value lives in [0, 1] by definition, matched on the name's words
# ("val_accuracy", "map50", "f1_macro"). A name outside this list is left
# unbounded: clipping on a guess would hide a real interval.
_UNIT_INTERVAL_WORD = re.compile(
    r"^(accuracy|acc|precision|recall|f1|iou|miou|dice|auc|auroc|map|ap"
    r"|specificity|sensitivity)\d*$"
)


def metric_domain(metric: str) -> tuple[float | None, float | None]:
    """The ``(low, high)`` bounds a metric's values can take, ``None`` for open.

    Absolute errors and losses cannot be negative; the [0, 1] scores cannot
    exceed 1 (nor drop under 0); R² is bounded above by 1 only. Anything else is
    unbounded -- a signed error (``mean_error``, ``bias_error``) is "lower is
    better" yet can be negative, so the floor is kept for names that are
    absolute by construction, not for every "lower" metric.
    """
    words = re.split(r"[^a-z0-9]+", metric.lower())
    if infer_direction(metric) == "lower" and _is_absolute_error(words):
        return 0.0, None
    if any(_UNIT_INTERVAL_WORD.match(w) for w in words):
        return 0.0, 1.0
    if "r2" in words or ("r" in words and "squared" in words):
        return None, 1.0
    return None, None


_ABSOLUTE_ERROR_WORDS = frozenset(
    {"mae", "mse", "rmse", "msle", "rmsle", "mape", "smape", "loss"}
)


def _is_absolute_error(words: list[str]) -> bool:
    """A name whose values are non-negative by construction (MAE, MSE, a loss)."""
    if _ABSOLUTE_ERROR_WORDS.intersection(words):
        return True
    return "error" in words and ("absolute" in words or "squared" in words)


def _clip_to_domain(
    low: float, high: float, domain: tuple[float | None, float | None]
) -> tuple[float, float, bool]:
    """Clip an interval to the domain; the flag says whether an end moved."""
    floor, ceiling = domain
    new_low = low if floor is None else max(low, floor)
    new_high = high if ceiling is None else min(high, ceiling)
    return new_low, new_high, (new_low != low or new_high != high)


def sample_std(values: list[float]) -> float | None:
    """Sample standard deviation (n-1), or None when one value has no spread.

    The convention of the replicate and paired statistics, used for K-fold too
    so the same "± std" means the same thing everywhere. A single value
    reports None rather than a 0.0 that reads as "no variation" -- a stopped
    K-fold often has exactly one finished fold (ADR-111).
    """
    return statistics.stdev(values) if len(values) >= 2 else None


def aggregate_replicates(
    trials: list[ReplicateTrial],
) -> dict[str, dict[str, float | int | bool | None]]:
    """Aggregate successful trials into per-metric distribution statistics.

    For every metric key observed across successful trials, returns ``n``,
    ``mean``, ``std`` (sample, n-1, named by ``std_ddof``), ``min``, ``max``
    and a Student-t 95% confidence interval (``ci95_low``/``ci95_high``). With
    a single value the dispersion fields are ``None`` — one sample has no
    spread to report.

    The t interval is symmetric around the mean, so with few seeds it spills
    out of what the metric can be (an MAE of ``[-0.70, 3.76]``, an accuracy
    above 1). It is **clipped to the metric's domain** (``metric_domain``:
    errors >= 0, [0, 1] scores inside [0, 1], R² <= 1) and ``ci95_clipped``
    says it was, so the report never shows an impossible value and never hides
    that the interval was cut. Clipping, rather than swapping in the bootstrap,
    keeps ``ci95`` meaning one thing (the t interval) while the bootstrap
    stays beside it, untouched, for the reader who wants a distribution-free
    one (ADR-061).

    A percentile **bootstrap** interval (``boot95_low``/``boot95_high``,
    ADR-061) is reported alongside the t interval: t assumes the sampling
    distribution of the mean is normal, which a handful of seeds cannot
    establish. When the two disagree, that disagreement is itself the finding
    — the t interval is leaning on an assumption the data does not support.
    """
    successful = [t for t in trials if t.status == "success"]
    keys: list[str] = list(dict.fromkeys(k for t in successful for k in t.metrics))

    aggregates: dict[str, dict[str, float | int | bool | None]] = {}
    for key in keys:
        values = [t.metrics[key] for t in successful if key in t.metrics]
        n = len(values)
        mean = statistics.fmean(values)
        boot_low: float | None = None
        boot_high: float | None = None
        clipped = False
        if n >= 2:
            std = statistics.stdev(values)
            half_width = _t_critical_95(n - 1) * std / math.sqrt(n)
            ci_low: float | None
            ci_high: float | None
            ci_low, ci_high, clipped = _clip_to_domain(
                mean - half_width, mean + half_width, metric_domain(key)
            )
            boot = bootstrap_ci(values)
            if boot is not None:
                boot_low, boot_high = boot.ci_low, boot.ci_high
        else:
            std = None  # type: ignore[assignment]
            ci_low = ci_high = None
        aggregates[key] = {
            "n": n,
            "mean": mean,
            "std": std,
            # Says which std this is, as the K-fold's aggregate does: a reader
            # of the number should not have to know the code to trust it.
            "std_ddof": 1,
            "min": min(values),
            "max": max(values),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            # True when the t interval was cut to the metric's domain.
            "ci95_clipped": clipped,
            "boot95_low": boot_low,
            "boot95_high": boot_high,
        }
    return aggregates


def run_replicates(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    seeds: list[int],
    metric: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    progress: UnitProgress | None = None,
    unit_offset: int = 0,
    unit_overrides: dict[str, Any] | None = None,
) -> list[ReplicateTrial]:
    """Train ``base_config_dict`` once per seed and return the trials in seed order.

    Each replicate overrides ``training.seed`` and suffixes ``name`` with
    ``_s{seed}`` so its checkpoints and run.json land in a distinct run dir.
    Replicates are never ranked — they are samples of one distribution, not
    competitors. A failed replicate is recorded and the set continues; GPU
    memory is released between replicates. ``progress_callback`` receives the
    live stream: each seed as one unit, announced, its epochs, and closed
    (``UnitProgress``, ADR-118), then a single ``end``.

    A caller that runs several sets as one job (the replicated comparison)
    passes its own ``progress`` instead, with ``unit_offset`` (the index of this
    set's first seed in the job) and ``unit_overrides`` (what else the unit
    varies, e.g. the variant); the job then owns the terminal ``end``.

    A stop carried by the runner (ADR-111) cuts the replicate in flight and
    starts no further seed. A replicate the runner reports as cut is recorded
    as ``stopped``, outside the aggregate, whose ``n`` then counts only the
    replicates that ran to the end.
    """
    token = runner_cancel_token(runner)
    trials: list[ReplicateTrial] = []
    base_name = str(base_config_dict.get("name", "replicates"))
    owns_stream = progress is None
    stream = progress or UnitProgress(progress_callback, len(seeds))

    for index, seed in enumerate(seeds):
        trial = ReplicateTrial(seed=seed)
        unit = unit_offset + index
        stream.start(runner, unit, {**(unit_overrides or {}), "training.seed": seed})
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
            trial.run_dir = str(result.run_dir) if result.run_dir else ""
            if result.stopped and result.status == "success":
                trial.status = STOPPED
                trial.error = STOPPED_NOTE
                discard_cut_unit_resume(result)
            elif result.status == "success":
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
            stream.finish(runner, unit, trial.status)
            _release_replicate_memory()
        trials.append(trial)
        if is_cancelled(token):
            logger.info(
                "Replicates stopped after {} of {} seeds.", index + 1, len(seeds)
            )
            break
    if owns_stream:
        stream.end(len(trials))

    return trials


def _release_replicate_memory() -> None:
    """Release unused CPU and available CUDA memory between seeds."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "ReplicateTrial",
    "aggregate_replicates",
    "metric_domain",
    "run_replicates",
    "sample_std",
]
