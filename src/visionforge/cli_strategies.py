"""``visionforge run`` with a strategy envelope (ADR-184).

A YAML whose top level is ``strategy:`` + ``config:`` runs the same multi-unit
jobs the GUI queues: replicates, a replicated (paired) comparison, a sweep, or a
task's K-fold. ``strategy`` carries the options the matching HTTP request takes
(``seeds``, ``variants``, ``search_space``, ``n_folds``...), validated by that
same request model, and ``config`` is the task's own config, unchanged.

This module only parses, picks the task's runner and calls the orchestrator in
``core``; every statistic comes from the orchestrator, and the report, its
``.tex`` table and the History group are written by the ``finish_*`` functions
the GUI routes call too (``core.run_groups``). Nothing here may compute a
number of its own: two code paths for the paper's mean ± CI would drift.

The orchestrators and runners import torch, so they are imported inside the
functions that use them.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from visionforge.utils.messages import tr

STRATEGY_KINDS = ("replicates", "replicated_comparison", "sweep", "cv")

# Where the History group lands when the config names no output.models_dir —
# the GUI's own default, so both list the same runs.
DEFAULT_MODELS_DIR = Path("outputs/models")

_CLASSIFICATION_TASKS = ("binary", "multiclass")
_RUNNER_TASKS = (
    *_CLASSIFICATION_TASKS,
    "detection",
    "regression",
    "segmentation",
    "anomaly",
)
_CV_TASKS = ("regression", "segmentation")


def read_envelope(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(strategy, config)`` when the YAML is a strategy envelope, else None.

    A YAML without a top-level ``strategy`` is a plain task config and goes the
    way ``visionforge run`` always took it.

    Raises:
        ValueError: when ``strategy`` is present but the envelope is malformed
            (no ``kind`` among :data:`STRATEGY_KINDS`, no ``config`` mapping, or
            other top-level keys beside the two).
    """
    with config_path.open(encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, dict) or "strategy" not in raw:
        return None
    strategy = raw["strategy"]
    config = raw.get("config")
    extra = sorted(set(raw) - {"strategy", "config"})
    if not isinstance(strategy, dict) or not isinstance(config, dict) or extra:
        raise ValueError(tr("cli.strategy_shape"))
    kind = strategy.get("kind")
    if kind not in STRATEGY_KINDS:
        raise ValueError(
            tr("cli.strategy_kind", kind=kind, kinds=", ".join(STRATEGY_KINDS))
        )
    return strategy, config


def _task_of(config: dict[str, Any]) -> str:
    return str(config.get("task", "multiclass"))


def runner_for(task: str) -> Any:
    """The ``TaskRunner`` the GUI uses for ``task``.

    Raises:
        ValueError: for a task without a built-in runner.
    """
    if task in _CLASSIFICATION_TASKS:
        from visionforge.blocks.classification_runner import ClassificationRunner

        return ClassificationRunner()
    if task == "detection":
        from visionforge.blocks.detection_runner import DetectionRunner

        return DetectionRunner()
    if task == "regression":
        from visionforge.blocks.regression_runner import RegressionRunner

        return RegressionRunner()
    if task == "segmentation":
        from visionforge.blocks.segmentation_runner import SegmentationRunner

        return SegmentationRunner()
    if task == "anomaly":
        from visionforge.blocks.anomaly_runner import AnomalyRunner

        return AnomalyRunner()
    raise ValueError(tr("cli.strategy_task", task=task, tasks=", ".join(_RUNNER_TASKS)))


def _run_id(name: str, infix: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{name}_{infix}{timestamp}"


def _complete(runner: Any, config: dict[str, Any]) -> dict[str, Any]:
    """The task config validated by the runner's own class, defaults filled in."""
    complete: dict[str, Any] = runner.config_type.model_validate(config).model_dump(
        mode="json"
    )
    return complete


def _run_replicates(
    options: dict[str, Any], config: dict[str, Any], models_dir: Path
) -> dict[str, Any]:
    from visionforge.core.replicates import run_replicates
    from visionforge.core.run_groups import finish_replicates, resolve_seeds
    from visionforge.gui.api.schemas import ReplicatesRequest

    req = ReplicatesRequest.model_validate({**options, "config": config})
    runner = runner_for(_task_of(config))
    complete = _complete(runner, req.config)
    try:
        seeds = resolve_seeds(complete, req.seeds, req.n_replicates)
    except ValueError as exc:
        raise ValueError(tr("replicates.seeds_unique")) from exc
    metric = req.metric or runner.primary_metric()
    trials = run_replicates(runner, complete, seeds, metric)
    return finish_replicates(
        trials,
        seeds=seeds,
        metric=metric,
        config=complete,
        run_id=_run_id(str(config.get("name", "replicates"))),
        token=None,
        models_dir=models_dir,
    )


def _run_replicated_comparison(
    options: dict[str, Any], config: dict[str, Any], models_dir: Path
) -> dict[str, Any]:
    from visionforge.core.replicated_comparison import (
        run_replicated_comparison,
        validate_variants,
    )
    from visionforge.core.run_groups import finish_replicated_comparison, resolve_seeds
    from visionforge.gui.api.schemas import ReplicatedComparisonRequest

    req = ReplicatedComparisonRequest.model_validate({**options, "config": config})
    runner = runner_for(_task_of(config))
    complete = _complete(runner, req.config)
    try:
        seeds = resolve_seeds(complete, req.seeds, req.n_replicates)
    except ValueError as exc:
        raise ValueError(tr("compare.seeds_unique")) from exc
    # A bad override path fails before len(variants) * len(seeds) trainings.
    validate_variants(complete, req.variants)
    metric = req.metric or runner.primary_metric()
    report = run_replicated_comparison(
        runner, complete, req.variants, seeds, metric, alpha=req.alpha
    )
    return finish_replicated_comparison(
        report,
        n_planned=len(req.variants) * len(seeds),
        config=complete,
        run_id=_run_id(str(config.get("name", "comparison"))),
        token=None,
        models_dir=models_dir,
    )


def _run_sweep(
    options: dict[str, Any], config: dict[str, Any], models_dir: Path
) -> dict[str, Any]:
    from visionforge.core.run_groups import finish_sweep
    from visionforge.core.sweep import run_sweep, validate_sweep_space
    from visionforge.core.task_runner import require_validation_selection
    from visionforge.gui.api.schemas import SweepRequest

    task = _task_of(config)
    if task in _CLASSIFICATION_TASKS:
        # The GUI sweeps classification through these blocks too.
        raise ValueError(tr("cli.sweep_classification"))
    req = SweepRequest.model_validate({**options, "config": config})
    runner = runner_for(task)
    # Ranking on test is never an option (ADR-131): refused before training.
    require_validation_selection(runner)
    complete = _complete(runner, req.config)
    validate_sweep_space(complete, list(req.search_space))
    metric = req.metric or runner.primary_metric()
    trials = run_sweep(
        runner,
        complete,
        req.search_space,
        mode=req.mode,
        metric=metric,
        n_trials=req.n_trials,
        seed=req.seed,
    )
    return finish_sweep(
        trials,
        runner=runner,
        config=complete,
        search_space=req.search_space,
        mode=req.mode,
        n_trials=req.n_trials,
        metric=metric,
        run_id=_run_id(str(config.get("name", "sweep"))),
        token=None,
        models_dir=models_dir,
    )


def _cv_function(task: str) -> tuple[type[Any], Callable[..., Any]]:
    """The config class and K-fold runner of a task the GUI offers K-fold for."""
    if task == "regression":
        from visionforge.blocks.regression_cv import run_regression_cross_validation
        from visionforge.utils.regression_config import RegressionConfig

        return RegressionConfig, run_regression_cross_validation
    if task == "segmentation":
        from visionforge.blocks.segmentation_cv import (
            run_segmentation_cross_validation,
        )
        from visionforge.utils.segmentation_config import SegmentationConfig

        return SegmentationConfig, run_segmentation_cross_validation
    raise ValueError(tr("cli.cv_unavailable", task=task, tasks=", ".join(_CV_TASKS)))


def _run_cv(
    options: dict[str, Any], config: dict[str, Any], models_dir: Path
) -> dict[str, Any]:
    from visionforge.core.run_groups import finish_kfold
    from visionforge.gui.api.schemas import TaskCvRequest

    config_type, cv_fn = _cv_function(_task_of(config))
    req = TaskCvRequest.model_validate({**options, "config": config})
    validated = config_type.model_validate(req.config)
    cv = cv_fn(validated, n_folds=req.n_folds, shuffle=req.shuffle, seed=req.fold_seed)
    return finish_kfold(
        cv,
        n_folds=req.n_folds,
        config=req.config,
        validated_config=validated.model_dump(mode="json"),
        run_id=_run_id(str(validated.name), "cv_"),
        token=None,
        models_dir=models_dir,
    )


_DISPATCH: dict[
    str, Callable[[dict[str, Any], dict[str, Any], Path], dict[str, Any]]
] = {
    "replicates": _run_replicates,
    "replicated_comparison": _run_replicated_comparison,
    "sweep": _run_sweep,
    "cv": _run_cv,
}


def run_strategy(
    strategy: dict[str, Any],
    config: dict[str, Any],
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> dict[str, Any]:
    """Run ``config`` under ``strategy`` and return the report the GUI would show.

    The report carries ``report_dir``, where its JSON, ranking CSV and ``.tex``
    table were written; the History group is written under the config's
    ``output.models_dir`` (``models_dir`` when it names none).

    Raises:
        ValueError: for an invalid strategy, task or config, before training.
        RuntimeError: when every unit failed, or none reported the metric.
    """
    options = {k: v for k, v in strategy.items() if k != "kind"}
    return _DISPATCH[str(strategy["kind"])](options, config, models_dir)


__all__ = [
    "DEFAULT_MODELS_DIR",
    "STRATEGY_KINDS",
    "read_envelope",
    "run_strategy",
    "runner_for",
]
