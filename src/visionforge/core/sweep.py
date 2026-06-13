"""Task-agnostic hyperparameter sweep over the `TaskRunner` handle (ADR-045).

Grid (cartesian product of value lists) or random (sampled from per-param
distributions) search that drives any task's runner and ranks trials by the
runner's declared metric — no knowledge of task internals. Overrides are applied
by dot-path on a validated base config dict, mirroring the classification
grid/random search space format so the two stay consistent.
"""

from __future__ import annotations

import copy
import gc
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from visionforge.core.task_runner import RunResult, TaskRunner

try:  # torch is the heavy hardware extra; the cache flush is best-effort.
    import torch
except ImportError:  # pragma: no cover - torch is always present in practice
    torch = None  # type: ignore[assignment]


@dataclass
class SweepTrial:
    """One hyperparameter combination's result inside a sweep."""

    trial_index: int
    overrides: dict[str, Any]
    status: str = "failed"
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_s: float | None = None
    error: str = ""


def _set_nested(d: dict[str, Any], dot_key: str, value: Any) -> None:
    """Set a value in a nested dict by dot-path; raise if the path is unknown."""
    keys = dot_key.split(".")
    node = d
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            raise ValueError(f"Unknown sweep path '{dot_key}' (failed at '{k}')")
        node = nxt
    if keys[-1] not in node:
        raise ValueError(f"Unknown sweep path '{dot_key}' (failed at '{keys[-1]}')")
    node[keys[-1]] = value


def validate_sweep_space(base_config_dict: dict[str, Any], paths: list[str]) -> None:
    """Raise ValueError if any sweep dot-path does not exist in the base config."""
    for path in paths:
        _set_nested(copy.deepcopy(base_config_dict), path, None)


def _grid_points(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of ``{path: [values]}`` into a list of override dicts."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [
        dict(zip(keys, combo, strict=True))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def _sample_one(spec: dict[str, Any], rng: random.Random) -> Any:
    """Sample one value from a param spec (uniform / log_uniform / choice)."""
    kind = spec.get("type")
    if kind == "uniform":
        return rng.uniform(spec["low"], spec["high"])
    if kind == "log_uniform":
        return math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
    if kind == "choice":
        return rng.choice(spec["options"])
    raise ValueError(
        f"Unknown sweep param type {kind!r} (use uniform/log_uniform/choice)"
    )


def _random_points(
    space: dict[str, Any], n_trials: int, seed: int
) -> list[dict[str, Any]]:
    """Sample ``n_trials`` override dicts from the random search space."""
    rng = random.Random(seed)
    return [
        {name: _sample_one(spec, rng) for name, spec in space.items()}
        for _ in range(n_trials)
    ]


def run_sweep(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    search_space: dict[str, Any],
    *,
    mode: str,
    metric: str,
    n_trials: int = 10,
    seed: int = 0,
) -> list[SweepTrial]:
    """Run a grid or random hyperparameter sweep and return trials ranked by ``metric``.

    ``base_config_dict`` must be a fully-populated (validated then dumped) task
    config so every dot-path resolves. Successful trials come first, sorted by
    ``metric`` descending; failures keep their order at the end. GPU memory is
    released between trials.

    Raises:
        ValueError: if ``mode`` is not 'grid' or 'random'.
    """
    if mode == "grid":
        points = _grid_points(search_space)
    elif mode == "random":
        points = _random_points(search_space, n_trials, seed)
    else:
        raise ValueError(f"Unknown sweep mode {mode!r} (use 'grid' or 'random')")

    trials: list[SweepTrial] = []
    for index, overrides in enumerate(points):
        trial = SweepTrial(trial_index=index, overrides=overrides)
        try:
            cfg_dict = copy.deepcopy(base_config_dict)
            for path, value in overrides.items():
                _set_nested(cfg_dict, path, value)
            cfg = runner.config_type.model_validate(cfg_dict)
            result: RunResult = runner.run(cfg)
            trial.status = result.status
            trial.metrics = runner.metrics(result)
            trial.training_time_s = result.training_time_s
            trial.error = result.error
            if result.status == "success":
                logger.info(
                    "Sweep trial {}/{} ok — {}={}",
                    index + 1,
                    len(points),
                    metric,
                    trial.metrics.get(metric),
                )
            else:
                logger.warning("Sweep trial {} failed — {}", index, result.error)
        except Exception as exc:  # noqa: BLE001
            trial.error = str(exc)
            logger.warning("Sweep trial {} failed — {}", index, exc)
        finally:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        trials.append(trial)

    successful = [t for t in trials if t.status == "success"]
    failed = [t for t in trials if t.status != "success"]
    successful.sort(key=lambda t: t.metrics.get(metric) or 0.0, reverse=True)
    return successful + failed


__all__ = ["SweepTrial", "run_sweep", "validate_sweep_space"]
