"""Group jobs: comparison, sweep, K-fold, replicates and paired comparison (#204).

Each ``_start_*`` validates a request, names the job and hands it to the queue
through ``submission._submit_job``; each ``_execute_*`` runs it in a worker
thread, writes its report and History group (``core.run_groups``) and records
how it ended in its own job's state (ADR-185). Nothing here holds state; the
routes that call these live in ``strategy_routes.py``.

A group is written under its config's ``output.models_dir``, or under the
models folder of the profile that submitted it when the config names none
(ADR-114). A test that replaces an orchestrator (``run_model_comparison``,
``run_sweep``, ``run_replicates``, ``run_replicated_comparison``), a K-fold
front or an executor patches this module: each is looked up here when a job
starts or runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from loguru import logger
from pydantic import ValidationError

from visionforge.core.cancellation import STOPPED, CancellationToken, job_was_stopped
from visionforge.core.comparison import ComparisonTrial, run_model_comparison
from visionforge.core.replicated_comparison import (
    run_replicated_comparison,
    validate_variants,
)
from visionforge.core.replicates import run_replicates
from visionforge.core.run_groups import (
    finish_kfold,
    finish_replicated_comparison,
    finish_replicates,
    finish_sweep,
    record_sweep_group,
    refuse_output_paths,
    resolve_seeds,
    sweep_report,
    write_group_summary,
)
from visionforge.core.sweep import run_sweep, validate_sweep_space
from visionforge.core.task_runner import (
    TaskRunner,
    give_cancel_token,
    require_validation_selection,
    runner_metric_direction,
)
from visionforge.gui.api import submission
from visionforge.gui.api.execution import RUN_NO_MESSAGE_KEY, _job_state
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.schemas import (
    ComparisonRequest,
    ReplicatedComparisonRequest,
    ReplicatesRequest,
    RunResponse,
    SweepRequest,
    TaskCvRequest,
)
from visionforge.gui.api.submission import _task_label_from_config
from visionforge.gui.api.training_routes import _new_run_id
from visionforge.utils.messages import tr
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

if TYPE_CHECKING:
    from visionforge.blocks.regression_cv import CrossValidationReport

# A comparison's strategy, report name and default job name.
_COMPARISON = "comparison"
_SUCCESS = "success"

# How a running job reports progress, and what a job's body returns: its report
# and what ran ("3 trials"), for the log.
_Progress = Callable[[dict[str, Any]], None]
_Body = Callable[[_Progress, CancellationToken], Awaitable[tuple[dict[str, Any], str]]]

# Moved to core.run_groups so `visionforge run` shares them (ADR-184); the
# old names stay for the model comparison and the tests that use them.
_sweep_report = sweep_report
_write_advanced_summary = write_group_summary


# The trainers import torch, so each K-fold front imports its task's module only
# when a run starts (GUI-11, #202). They stay module attributes so a test can
# replace them here.


def run_regression_cross_validation(
    config: RegressionConfig, **kwargs: Any
) -> CrossValidationReport:
    """K-fold CV for regression, importing its trainer only when a run starts."""
    from visionforge.blocks.regression_cv import run_regression_cross_validation as run

    return run(config, **kwargs)


def run_segmentation_cross_validation(
    config: SegmentationConfig, **kwargs: Any
) -> CrossValidationReport:
    """K-fold CV for segmentation, importing its trainer only when a run starts."""
    from visionforge.blocks.segmentation_cv import (
        run_segmentation_cross_validation as run,
    )

    return run(config, **kwargs)


# ── submitting ─────────────────────────────────────────────────────────────


def _validated(config_type: type[Any], raw: dict[str, Any]) -> Any:
    """``raw`` validated against the task's config, or a 422."""
    try:
        return config_type.model_validate(raw)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _complete_config(
    config_type: type[Any], raw: dict[str, Any], profile: Profile
) -> dict[str, Any]:
    """``raw`` with every default filled in, its output inside the profile."""
    return profile.scope_dict(_validated(config_type, raw).model_dump(mode="json"))


def _job_name(config: dict[str, Any], default: str) -> str:
    """The name the request's config gives the job, else ``default``."""
    return str(config.get("name", default))


def _seeds_or_422(
    complete: dict[str, Any],
    seeds: list[int] | None,
    n_replicates: int,
    duplicate_key: str,
) -> list[int]:
    """The explicit seeds, else ``n_replicates`` from ``training.seed``.

    Raises:
        HTTPException: 422 for duplicate seeds, worded by ``duplicate_key``.
    """
    try:
        return resolve_seeds(complete, seeds, n_replicates)
    except ValueError as exc:
        # 422 is in the `responses` of every route that calls this helper.
        raise HTTPException(422, tr(duplicate_key)) from exc  # NOSONAR


def _refuse_output_paths(paths: Iterable[str], where: str) -> None:
    """422 when a dot-path would move a trial's output folders (ADR-114).

    Raises:
        HTTPException: 422 naming the first offending path.
    """
    try:
        refuse_output_paths(paths, where)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _submit_group(
    run_id: str,
    name: str,
    config_type: type[Any],
    strategy: str,
    start: Callable[[], Awaitable[None]],
    profile: Profile,
) -> RunResponse:
    """Queue a group job under the task label of ``config_type``."""
    return submission._submit_job(
        run_id,
        name,
        _task_label_from_config(config_type),
        strategy,
        start,
        profile=profile,
    )


def _start_comparison(
    req: ComparisonRequest,
    config_type: type[Any],
    runner: TaskRunner,
    profile: Profile,
) -> RunResponse:
    """Validate the base config and queue a model-comparison run.

    Raises:
        RequestValidationError: 422 if the base task config is invalid.
    """
    # Every trial and the report inherit the profile's output folders from the
    # base config (ADR-114).
    req = req.model_copy(update={"config": profile.scope_dict(req.config)})
    # Fail fast on a bad base config; each trial re-validates with its override.
    _validated(config_type, req.config)

    metric = req.metric or runner.primary_metric()
    name = _job_name(req.config, _COMPARISON)
    run_id = _new_run_id(name)
    return _submit_group(
        run_id,
        name,
        config_type,
        _COMPARISON,
        lambda: _execute_comparison(
            runner,
            req.config,
            req.model_names,
            metric,
            run_id,
            models_dir=profile.models_dir,
        ),
        profile,
    )


def _start_sweep(
    req: SweepRequest,
    config_type: type[Any],
    runner: TaskRunner,
    profile: Profile,
) -> RunResponse:
    """Validate the base config + search space and queue a sweep.

    Raises:
        HTTPException: 422 if a sweep path points under ``output``, or the task
            has no validation split to rank trials on (ADR-131).
        RequestValidationError: 422 if the base config or a sweep path is invalid.
    """
    _refuse_output_paths(req.search_space, "profile.where_sweep")
    # Validate the base config (fills defaults) and confirm every sweep path
    # resolves against it before spending GPU time.
    try:
        # Refused before the job is queued: ranking on test is never an option.
        require_validation_selection(runner)
        complete = _complete_config(config_type, req.config, profile)
        validate_sweep_space(complete, list(req.search_space))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    metric = req.metric or runner.primary_metric()
    name = _job_name(req.config, "sweep")
    run_id = _new_run_id(name)
    return _submit_group(
        run_id,
        name,
        config_type,
        f"sweep:{req.mode}" if getattr(req, "mode", None) else "sweep",
        lambda: _execute_sweep(
            runner, complete, req, metric, run_id, models_dir=profile.models_dir
        ),
        profile,
    )


def _start_task_cv(
    req: TaskCvRequest,
    config_type: type[Any],
    cv_fn: Callable[..., Any],
    task_label: str,
    profile: Profile,
) -> RunResponse:
    """Validate the config and queue a K-fold run for one task.

    ``cv_fn`` is the task's CV runner (regression/segmentation) — same keyword
    signature, same report shape.

    Raises:
        RequestValidationError: 422 if the base task config is invalid.
    """
    # Both the config the folds train from and the dict the report is written
    # from live under the profile (ADR-114).
    req = req.model_copy(update={"config": profile.scope_dict(req.config)})
    config = _validated(config_type, req.config)

    run_id = _new_run_id(f"{config.name}_cv")
    return _submit_group(
        run_id,
        str(config.name),
        config_type,
        "cv",
        lambda: _execute_task_cv(
            config, req, run_id, cv_fn, task_label, models_dir=profile.models_dir
        ),
        profile,
    )


def _start_replicated_comparison(
    req: ReplicatedComparisonRequest,
    config_type: type[Any],
    runner: TaskRunner,
    profile: Profile,
) -> RunResponse:
    """Validate the base config + variants and queue the paired comparison.

    Raises:
        HTTPException: 422 for an invalid variant set (unknown override path,
            an override under ``output``, duplicate seeds).
        RequestValidationError: 422 if the base task config is invalid.
    """
    for overrides in req.variants.values():
        _refuse_output_paths(overrides, "profile.where_comparison")
    complete = _complete_config(config_type, req.config, profile)
    seeds = _seeds_or_422(complete, req.seeds, req.n_replicates, "compare.seeds_unique")

    # Fail on a bad override path before spending len(variants)*len(seeds)
    # trainings on it.
    try:
        validate_variants(complete, req.variants)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    metric = req.metric or runner.primary_metric()
    name = _job_name(req.config, _COMPARISON)
    run_id = _new_run_id(name)
    return _submit_group(
        run_id,
        name,
        config_type,
        "replicated-comparison",
        lambda: _execute_replicated_comparison(
            runner, complete, req, seeds, metric, run_id, models_dir=profile.models_dir
        ),
        profile,
    )


def _start_replicates(
    req: ReplicatesRequest,
    config_type: type[Any],
    runner: TaskRunner,
    profile: Profile,
) -> RunResponse:
    """Validate the base config and queue a multi-seed replicate set.

    Explicit ``req.seeds`` win (duplicates rejected); otherwise ``n_replicates``
    consecutive seeds are derived from the validated config's ``training.seed``.

    Raises:
        HTTPException: 422 for duplicate seeds.
        RequestValidationError: 422 if the base task config is invalid.
    """
    # Validate the base config (fills defaults) so the seed derivation below can
    # trust `training.seed` to exist.
    complete = _complete_config(config_type, req.config, profile)
    seeds = _seeds_or_422(
        complete, req.seeds, req.n_replicates, "replicates.seeds_unique"
    )

    metric = req.metric or runner.primary_metric()
    name = _job_name(req.config, "replicates")
    run_id = _new_run_id(name)
    return _submit_group(
        run_id,
        name,
        config_type,
        "replicates",
        lambda: _execute_replicates(
            runner, complete, seeds, metric, run_id, models_dir=profile.models_dir
        ),
        profile,
    )


# ── running ────────────────────────────────────────────────────────────────


def _ended_run(
    run_id: str, report: dict[str, Any] | None, error: str | None
) -> dict[str, Any]:
    """The run dict of a finished group job: its report, or why it failed."""
    return {
        "run_id": run_id,
        "status": "failed" if error is not None else "completed",
        "error": error,
        "report": report,
        "run_dir": None,
    }


async def _run_group_job(
    run_id: str, what: str, runner: TaskRunner | None, body: _Body
) -> None:
    """Run job ``run_id``'s ``body`` and record how it ended (ADR-185).

    ``body`` gets the job's progress callback and stop token; ``what`` names the
    job in the log. A ``runner`` carries the stop into the unit in flight, and
    the orchestrator reads it between units (ADR-111).
    """
    state = _job_state(run_id)
    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    token = state.token
    if runner is not None:
        give_cancel_token(runner, token)

    try:
        report, ran = await body(_put_event, token)
        state.run = _ended_run(run_id, report, None)
        logger.success("GUI: {} {} completed ({}).", what, run_id, ran)
    except Exception as e:
        logger.exception("GUI: {} {} failed", what, run_id)
        error = f"{type(e).__name__}: {e or tr(RUN_NO_MESSAGE_KEY)}"
        state.run = _ended_run(run_id, None, error)
    finally:
        if queue is not None:
            await queue.put(None)


async def _execute_comparison(
    runner: TaskRunner,
    config_dict: dict[str, Any],
    model_names: list[str],
    metric: str,
    run_id: str,
    *,
    models_dir: Path,
) -> None:
    """Run a model comparison in a background thread and store the ranked report."""

    async def body(
        progress: _Progress, token: CancellationToken
    ) -> tuple[dict[str, Any], str]:
        # Each architecture streams as one unit, so the sheet shows which model
        # is training and its epochs instead of "preparing" until the end.
        trials = await asyncio.to_thread(
            run_model_comparison,
            runner,
            config_dict,
            model_names,
            metric,
            progress_callback=progress,
        )
        stopped = job_was_stopped([t.status for t in trials], len(model_names), token)
        if not any(t.status == _SUCCESS for t in trials) and not stopped:
            raise RuntimeError("All architectures failed — no ranking available.")
        report = _comparison_report(
            trials, metric, runner_metric_direction(runner, metric)
        )
        report["stopped"] = stopped
        report["report_dir"] = _write_advanced_summary(config_dict, _COMPARISON, report)
        _record_comparison_group(
            run_id, runner, config_dict, report, model_names, models_dir
        )
        return report, f"{len(trials)} archs"

    await _run_group_job(run_id, "Comparison", runner, body)


def _record_comparison_group(
    run_id: str,
    runner: TaskRunner,
    config_dict: dict[str, Any],
    report: dict[str, Any],
    model_names: list[str],
    models_dir: Path,
) -> None:
    """Give a finished model comparison its History entry (ADR-124)."""
    # The comparison's rows are ranked; their index is the order they ran in.
    order = {name: i for i, name in enumerate(model_names)}
    rows = [
        {
            **row,
            "trial_index": order.get(row["model_arch"]),
            "overrides": {"model.name": row["model_arch"]},
        }
        for row in report["trials"]
    ]
    top = next((r for r in rows if r["status"] == _SUCCESS), None)
    record_sweep_group(
        run_id,
        runner,
        config_dict,
        report,
        mode="models",
        planned=len(model_names),
        trials=rows,
        best_index=top["trial_index"] if top else None,
        report_name=_COMPARISON,
        models_dir=models_dir,
    )


def _comparison_report(
    trials: list[ComparisonTrial], metric: str, direction: Literal["higher", "lower"]
) -> dict[str, Any]:
    """Shape ranked comparison trials into a GUI-friendly report dict."""
    rows = [asdict(t) for t in trials]
    statuses = [t.status for t in trials]
    successful = [row for row, s in zip(rows, statuses, strict=True) if s == _SUCCESS]
    return {
        "metric": metric,
        # Which way the trials were ranked; "top" means lowest for "lower".
        "metric_direction": direction,
        "trials": rows,
        "top_3": successful[:3],
        "total_ran": len(rows),
        "failed_count": statuses.count("failed"),
        "stopped_count": statuses.count(STOPPED),
    }


async def _execute_sweep(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    req: SweepRequest,
    metric: str,
    run_id: str,
    *,
    models_dir: Path,
) -> None:
    """Run a hyperparameter sweep in a background thread and store the ranked report."""

    async def body(
        progress: _Progress, token: CancellationToken
    ) -> tuple[dict[str, Any], str]:
        trials = await asyncio.to_thread(
            run_sweep,
            runner,
            base_config_dict,
            req.search_space,
            mode=req.mode,
            metric=metric,
            n_trials=req.n_trials,
            seed=req.seed,
            progress_callback=progress,
        )
        report = finish_sweep(
            trials,
            runner=runner,
            config=base_config_dict,
            search_space=req.search_space,
            mode=req.mode,
            n_trials=req.n_trials,
            metric=metric,
            run_id=run_id,
            token=token,
            models_dir=models_dir,
        )
        return report, f"{len(trials)} trials"

    await _run_group_job(run_id, "Sweep", runner, body)


async def _execute_task_cv(
    config: Any,
    req: TaskCvRequest,
    run_id: str,
    cv_fn: Callable[..., Any],
    task_label: str,
    *,
    models_dir: Path,
) -> None:
    """Run a task's K-fold in a worker thread and store the fold report."""

    async def body(
        progress: _Progress, token: CancellationToken
    ) -> tuple[dict[str, Any], str]:
        cv = await asyncio.to_thread(
            cv_fn,
            config,
            n_folds=req.n_folds,
            shuffle=req.shuffle,
            seed=req.fold_seed,
            progress_callback=progress,
            cancel_token=token,
        )
        # The validated config names the task (config.task), which the
        # request's raw dict may leave to the default.
        report = finish_kfold(
            cv,
            n_folds=req.n_folds,
            config=req.config,
            validated_config=config.model_dump(mode="json"),
            run_id=run_id,
            token=token,
            models_dir=models_dir,
        )
        return report, f"{cv.n_folds} folds"

    # The fold runner takes the token itself: there is no runner to hand it to.
    await _run_group_job(run_id, f"{task_label} CV", None, body)


async def _execute_replicated_comparison(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    req: ReplicatedComparisonRequest,
    seeds: list[int],
    metric: str,
    run_id: str,
    *,
    models_dir: Path,
) -> None:
    """Run every variant over the shared seeds and store the paired report."""

    async def body(
        progress: _Progress, token: CancellationToken
    ) -> tuple[dict[str, Any], str]:
        raw = await asyncio.to_thread(
            run_replicated_comparison,
            runner,
            base_config_dict,
            req.variants,
            seeds,
            metric,
            alpha=req.alpha,
            progress_callback=progress,
        )
        report = finish_replicated_comparison(
            raw,
            n_planned=len(req.variants) * len(seeds),
            config=base_config_dict,
            run_id=run_id,
            token=token,
            models_dir=models_dir,
        )
        return report, f"{len(req.variants)} variants x {len(seeds)} seeds"

    await _run_group_job(run_id, "Replicated comparison", runner, body)


async def _execute_replicates(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    seeds: list[int],
    metric: str,
    run_id: str,
    *,
    models_dir: Path,
) -> None:
    """Run the replicate set in a background thread and store the aggregate report."""

    async def body(
        progress: _Progress, token: CancellationToken
    ) -> tuple[dict[str, Any], str]:
        trials = await asyncio.to_thread(
            run_replicates,
            runner,
            base_config_dict,
            seeds,
            metric,
            progress_callback=progress,
        )
        report = finish_replicates(
            trials,
            seeds=seeds,
            metric=metric,
            config=base_config_dict,
            run_id=run_id,
            token=token,
            models_dir=models_dir,
        )
        return report, f"{len(trials)} seeds"

    await _run_group_job(run_id, "Replicates", runner, body)
