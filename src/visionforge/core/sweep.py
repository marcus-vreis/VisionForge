"""Task-agnostic hyperparameter sweep over the `TaskRunner` handle (ADR-045/052).

Grid (cartesian product of value lists), random (sampled from per-param
distributions) or Optuna (TPE-guided adaptive search) over any task's runner,
ranking trials by the runner's declared metric — no knowledge of task internals.
Overrides are applied by dot-path on a validated base config dict, mirroring the
classification grid/random search space format so the modes stay consistent.

Optuna reuses the random search-space grammar (uniform/log_uniform/choice) but
suggests each trial adaptively from the history of previous results; it is an
optional dependency (the `optuna` extra), imported lazily.
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


def _execute_trial(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    overrides: dict[str, Any],
    index: int,
    metric: str,
    total: int,
) -> SweepTrial:
    """Apply ``overrides`` to the base config, run one trial, and record the result.

    Never raises — a failed trial is captured in the returned ``SweepTrial`` so the
    sweep continues. GPU memory is released afterwards.
    """
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
                total,
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
    return trial


def _suggest_one(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    """Ask an Optuna trial to suggest one value per the search-space param spec."""
    kind = spec.get("type")
    if kind == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if kind == "log_uniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if kind == "choice":
        return trial.suggest_categorical(name, spec["options"])
    raise ValueError(
        f"Unknown sweep param type {kind!r} (use uniform/log_uniform/choice)"
    )


def _optuna_trials(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    search_space: dict[str, Any],
    metric: str,
    n_trials: int,
    seed: int,
) -> list[SweepTrial]:
    """Run an Optuna TPE study over the search space, collecting every trial.

    Direction is ``maximize`` (the ranking metrics — accuracy/r2/mAP/mIoU/AUROC —
    are higher-is-better, matching the descending sort the other modes use). Failed
    trials are recorded and pruned so the study keeps going.

    Raises:
        ImportError: if the optional ``optuna`` dependency is not installed.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - only without the optuna extra
        raise ImportError(
            "optuna is not installed. Add the optional extra: "
            'pip install -e ".[optuna]"  (or: uv sync --extra optuna).'
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[SweepTrial] = []

    def objective(trial: Any) -> float:
        overrides = {
            name: _suggest_one(trial, name, spec) for name, spec in search_space.items()
        }
        st = _execute_trial(
            runner, base_config_dict, overrides, trial.number, metric, n_trials
        )
        trials.append(st)
        value = st.metrics.get(metric)
        if st.status != "success" or value is None:
            raise optuna.TrialPruned()
        return value

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return trials


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
    """Run a grid, random or Optuna sweep and return trials ranked by ``metric``.

    ``base_config_dict`` must be a fully-populated (validated then dumped) task
    config so every dot-path resolves. Successful trials come first, sorted by
    ``metric`` descending; failures keep their order at the end. GPU memory is
    released between trials.

    Raises:
        ValueError: if ``mode`` is not 'grid', 'random' or 'optuna'.
        ImportError: for mode 'optuna' when the optuna extra is not installed.
    """
    if mode == "grid":
        points = _grid_points(search_space)
        trials = [
            _execute_trial(runner, base_config_dict, ov, i, metric, len(points))
            for i, ov in enumerate(points)
        ]
    elif mode == "random":
        points = _random_points(search_space, n_trials, seed)
        trials = [
            _execute_trial(runner, base_config_dict, ov, i, metric, len(points))
            for i, ov in enumerate(points)
        ]
    elif mode == "optuna":
        trials = _optuna_trials(
            runner, base_config_dict, search_space, metric, n_trials, seed
        )
    else:
        raise ValueError(
            f"Unknown sweep mode {mode!r} (use 'grid', 'random' or 'optuna')"
        )

    successful = [t for t in trials if t.status == "success"]
    failed = [t for t in trials if t.status != "success"]
    successful.sort(key=lambda t: t.metrics.get(metric) or 0.0, reverse=True)
    return successful + failed


__all__ = ["SweepTrial", "run_sweep", "validate_sweep_space"]
