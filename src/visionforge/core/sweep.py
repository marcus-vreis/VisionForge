"""Task-agnostic hyperparameter sweep over the `TaskRunner` handle (ADR-045/052).

Grid (cartesian product of value lists), random (sampled from per-param
distributions) or Optuna (TPE-guided adaptive search) over any task's runner,
ranking trials by the runner's declared metric — no knowledge of task internals.
Trials are ranked, and Optuna is steered, by the metric measured on the
validation split only; the test numbers ride along for the report (ADR-131).
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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from visionforge.core.cancellation import STOPPED, STOPPED_NOTE, is_cancelled
from visionforge.core.task_runner import (
    SELECTION_SPLIT,
    RunResult,
    TaskRunner,
    UnitProgress,
    discard_cut_unit_resume,
    rank_by_metric,
    require_validation_selection,
    runner_cancel_token,
    runner_metric_direction,
)
from visionforge.utils.messages import tr

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
    # What the trial is ranked by: the runner's validation numbers (ADR-131).
    metrics: dict[str, float] = field(default_factory=dict)
    # The run's final evaluation (test split when the task has one), reported
    # beside the ranking and never used to choose a trial.
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    selection_split: str = SELECTION_SPLIT
    training_time_s: float | None = None
    error: str = ""
    # The trial's own run folder, so History can fold it under the sweep (ADR-124).
    run_dir: str | None = None


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


def _selection_problem(selection: dict[str, float], metric: str) -> str:
    """Why ``metric`` cannot rank this trial, or "" when its value is usable."""
    value = selection.get(metric)
    if value is None:
        return tr(
            "sweep.missing_validation_value",
            split=SELECTION_SPLIT,
            metric=metric,
            metrics=sorted(selection) or tr("sweep.no_metrics"),
        )
    if not math.isfinite(value):
        return tr(
            "sweep.nonfinite_validation_value",
            split=SELECTION_SPLIT,
            metric=metric,
            value=value,
        )
    return ""


def _execute_trial(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    overrides: dict[str, Any],
    index: int,
    metric: str,
    total: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SweepTrial:
    """Apply ``overrides`` to the base config, run one trial, and record the result.

    Never raises — a failed trial is captured in the returned ``SweepTrial`` so the
    sweep continues. GPU memory is released afterwards. ``progress_callback``
    receives ``trial_start``, the trial's epochs and ``trial_end`` (``UnitProgress``,
    the vocabulary the GUI's training overlay tracks), so progress is real.
    """
    trial = SweepTrial(trial_index=index, overrides=overrides)
    # The trial's own epochs reach the stream too, not only its start and end.
    progress = UnitProgress(progress_callback, total)
    progress.start(runner, index, overrides)
    try:
        cfg_dict = copy.deepcopy(base_config_dict)
        for path, value in overrides.items():
            _set_nested(cfg_dict, path, value)
        cfg = runner.config_type.model_validate(cfg_dict)
        result: RunResult = runner.run(cfg)
        trial.status = result.status
        trial.metrics = dict(result.selection_metrics)
        trial.evaluation_metrics = runner.metrics(result)
        trial.training_time_s = result.training_time_s
        trial.error = result.error
        trial.run_dir = str(result.run_dir) if result.run_dir else None
        if result.stopped and result.status == "success":
            # Cut short by the stop: kept in the list, never ranked (ADR-111).
            trial.status = STOPPED
            trial.error = STOPPED_NOTE
            discard_cut_unit_resume(result)
        elif result.status == "success" and (
            problem := _selection_problem(trial.metrics, metric)
        ):
            trial.status = "failed"
            trial.error = problem
            logger.warning("Sweep trial {} not ranked — {}", index, problem)
        elif result.status == "success":
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
        progress.finish(runner, index, trial.status)
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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    direction: Literal["higher", "lower"],
) -> list[SweepTrial]:
    """Run an Optuna TPE study over the search space, collecting every trial.

    The study maximizes a higher-is-better metric and minimizes a lower-is-better
    one, the same direction the trials are ranked in. Failed trials are recorded
    and pruned so the study keeps going.

    Raises:
        ImportError: if the optional ``optuna`` dependency is not installed.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - only without the optuna extra
        raise ImportError(
            "optuna ships with VisionForge and is missing — the install "
            'looks damaged: pip install --force-reinstall "visionforge-studio".'
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[SweepTrial] = []
    token = runner_cancel_token(runner)

    def objective(trial: Any) -> float:
        overrides = {
            name: _suggest_one(trial, name, spec) for name, spec in search_space.items()
        }
        st = _execute_trial(
            runner,
            base_config_dict,
            overrides,
            trial.number,
            metric,
            n_trials,
            progress_callback=progress_callback,
        )
        trials.append(st)
        if is_cancelled(token):
            # Optuna's own way out of `optimize`: no further trial is sampled.
            study.stop()
        value = st.metrics.get(metric)
        if st.status != "success" or value is None:
            raise optuna.TrialPruned()
        return value

    study = optuna.create_study(
        direction="maximize" if direction == "higher" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    direction: Literal["higher", "lower"] | None = None,
) -> list[SweepTrial]:
    """Run a grid, random or Optuna sweep and return trials ranked by ``metric``.

    ``base_config_dict`` must be a fully-populated (validated then dumped) task
    config so every dot-path resolves. Successful trials come first, sorted by
    ``metric`` on the validation split (best first in the metric's direction);
    a trial without a finite validation value fails, test numbers are never a
    stand-in. Failures keep their order at the end. GPU memory is
    released between trials. A stop carried by the runner (ADR-111) cuts the
    trial in flight and starts no further trial; a trial the runner reports as
    cut is recorded as ``stopped``, outside the ranking.

    Raises:
        ValueError: if ``mode`` is not 'grid', 'random' or 'optuna', or the
            runner's task has no validation objective to select on.
        ImportError: for mode 'optuna' when the optuna extra is not installed.
    """
    require_validation_selection(runner)
    resolved = direction or runner_metric_direction(runner, metric)
    if mode in ("grid", "random"):
        points = (
            _grid_points(search_space)
            if mode == "grid"
            else _random_points(search_space, n_trials, seed)
        )
        token = runner_cancel_token(runner)
        trials: list[SweepTrial] = []
        for i, ov in enumerate(points):
            trials.append(
                _execute_trial(
                    runner,
                    base_config_dict,
                    ov,
                    i,
                    metric,
                    len(points),
                    progress_callback=progress_callback,
                )
            )
            # The trial in flight already stopped at its epoch boundary; the
            # rest would only burn GPU time nobody asked for (ADR-111).
            if is_cancelled(token):
                logger.info("Sweep stopped after {} of {} trials.", i + 1, len(points))
                break
    elif mode == "optuna":
        trials = _optuna_trials(
            runner,
            base_config_dict,
            search_space,
            metric,
            n_trials,
            seed,
            progress_callback=progress_callback,
            direction=resolved,
        )
    else:
        raise ValueError(
            f"Unknown sweep mode {mode!r} (use 'grid', 'random' or 'optuna')"
        )

    successful = [t for t in trials if t.status == "success"]
    failed = [t for t in trials if t.status != "success"]
    ranked = rank_by_metric(successful, lambda t: t.metrics.get(metric), resolved)
    return ranked + failed


def planned_trials(search_space: dict[str, Any], *, mode: str, n_trials: int) -> int:
    """How many trials a sweep sets out to run: the grid's size, or ``n_trials``."""
    return len(_grid_points(search_space)) if mode == "grid" else n_trials


__all__ = ["SweepTrial", "planned_trials", "run_sweep", "validate_sweep_space"]
