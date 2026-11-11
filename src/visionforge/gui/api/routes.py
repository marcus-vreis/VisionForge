"""API endpoints for VisionForge GUI."""

from __future__ import annotations

import asyncio
import csv
import importlib
import json
import os
import sys
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
from pydantic import ValidationError

from visionforge import __version__ as _VF_VERSION
from visionforge.core.cancellation import (
    STOPPED,
    is_cancelled,
    job_was_stopped,
)
from visionforge.core.comparison import ComparisonTrial, run_model_comparison
from visionforge.core.detection_data import resolve_yolo_split
from visionforge.core.experiment_bundle import import_experiment_bundle
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
    record_run_group,
    record_sweep_group,
    refuse_output_paths,
    resolve_seeds,
    sweep_report,
    write_group_summary,
)
from visionforge.core.run_trackers import TrackerFactory
from visionforge.core.sweep import run_sweep, validate_sweep_space
from visionforge.core.task_runner import (
    TaskRunner,
    give_cancel_token,
    require_validation_selection,
    runner_metric_direction,
)
from visionforge.gui.api import execution
from visionforge.gui.api.coco_routes import router as coco_router
from visionforge.gui.api.code_tour_routes import router as code_tour_router
from visionforge.gui.api.colab_connection import PINNED_VERSION as _COLAB_PIN
from visionforge.gui.api.colab_connection import ColabCli, LoginFlows
from visionforge.gui.api.colab_drive import (
    DriveConsent,
    count_files,
    my_drive_roots,
    remote_path,
)
from visionforge.gui.api.colab_jobs import ColabJobs
from visionforge.gui.api.colab_routes import build_colab_router
from visionforge.gui.api.datasets_routes import (
    _IMAGE_EXTS,
    SCAN_BASE_MISSING_KEY,
    _image_stems,
    _remember_dataset_root,
)
from visionforge.gui.api.datasets_routes import router as datasets_router
from visionforge.gui.api.dvc_routes import build_dvc_router
from visionforge.gui.api.evaluation_routes import (
    _reject_group_action,
    build_evaluation_router,
)
from visionforge.gui.api.evidence_routes import EvidenceExports, build_evidence_router
from visionforge.gui.api.execution import (
    _LANES,
    RUN_NO_MESSAGE_KEY,
    _job_state,
    _run_state,
    _stop_point,
)
from visionforge.gui.api.execution_routes import router as execution_router
from visionforge.gui.api.explorer_routes import router as explorer_router
from visionforge.gui.api.folder_opener import (
    FolderOpenError,
    is_loopback_host,
    is_loopback_origin,
    open_folder,
)
from visionforge.gui.api.github_routes import build_github_router
from visionforge.gui.api.gpu_health import cuda_context_broken
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    CONFLICT_TEXT,
    FORBIDDEN_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
    TAG_ANOMALY,
    TAG_COLAB,
    TAG_CUSTOM_TASKS,
    TAG_DATASETS,
    TAG_DETECTION,
    TAG_EXPERIMENTS,
    TAG_REGRESSION,
    TAG_RUNS,
    TAG_SEGMENTATION,
    TAG_SYSTEM,
    UNPROCESSABLE_TEXT,
    VALIDATION_ERROR_CONTENT,
)
from visionforge.gui.api.hub_routes import build_hub_router
from visionforge.gui.api.incorporate_routes import router as incorporate_router
from visionforge.gui.api.merge_routes import router as merge_router
from visionforge.gui.api.model_card import _stopped_extent as _stopped_extent
from visionforge.gui.api.model_card import render_run_markdown as _render_run_markdown
from visionforge.gui.api.profiles import (
    PROFILE_HEADER,
    InvalidProfileError,
    Profile,
    UnknownProfileError,
    default_profile,
    resolve_profile,
)
from visionforge.gui.api.profiles_routes import build_profiles_router
from visionforge.gui.api.resplit_routes import router as resplit_router
from visionforge.gui.api.run_history import (
    RUN_BAD_JSON_KEY,
    RUN_FILENAME,
    RUN_NOT_FOUND_KEY,
)
from visionforge.gui.api.run_history import load_runs as _load_runs
from visionforge.gui.api.run_history import resume_status as _resume_status
from visionforge.gui.api.run_history import run_task as _run_task
from visionforge.gui.api.run_queue import (
    LOCAL_LANE,
    QueuedJob,
    StopPoint,
    colab_lane,
)
from visionforge.gui.api.runs_routes import (
    build_runs_router,
    read_mlflow_status,
    read_wandb_status,
)
from visionforge.gui.api.sample_dataset import (
    SampleDatasetExistsError,
    SampleTaskError,
    create_sample_dataset,
)
from visionforge.gui.api.schemas import (
    ColabDriveContinueRequest,
    ColabDriveSource,
    ColabRunRequest,
    ComparisonRequest,
    CustomTaskActionResponse,
    MlflowStatus,
    PreprocessPreviewRequest,
    PreprocessPreviewResponse,
    PreprocessPreviewStep,
    ReplicatedComparisonRequest,
    ReplicatesRequest,
    RunResponse,
    SampleDatasetRequest,
    SampleDatasetResponse,
    SweepRequest,
    TaskCvRequest,
    TaskDescriptor,
    TaskListResponse,
    WandbStatus,
)
from visionforge.gui.api.system_routes import build_system_router
from visionforge.gui.api.tracker_exports import exported, profile_trackers
from visionforge.gui.api.wandb_routes import build_wandb_router
from visionforge.gui.api.zenodo_routes import build_zenodo_router
from visionforge.tasks.base import TaskSpec
from visionforge.tasks.registry import get_task, registered_tasks
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.cuda import check_cuda
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.messages import (
    LANG_HEADER,
    current_lang,
    parse_lang,
    resolve_lang,
    set_lang,
    tr,
    using_lang,
)
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig
from visionforge.utils.workers import suggested_workers as worker_budget

if TYPE_CHECKING:
    from visionforge.blocks.regression_cv import CrossValidationReport
    from visionforge.tasks.engine import GenericTaskEngine as _GenericTaskEngine

# The blocks, runners and trainers import torch, torchvision, ultralytics and
# matplotlib. Each endpoint imports the one it runs, so starting the GUI loads
# none of them (GUI-11, #202). The three names below stay module attributes so a
# test can still replace them here.

# Resolved on first use by _generic_task_engine().
GenericTaskEngine: type[_GenericTaskEngine] | None = None


def _generic_task_engine() -> type[_GenericTaskEngine]:
    """The custom-task engine class: a test's replacement, or the real one."""
    if GenericTaskEngine is not None:
        return GenericTaskEngine
    from visionforge.tasks.engine import GenericTaskEngine as engine

    return engine


# What every trainer imports first: torch, torchvision and the plotter (with
# matplotlib, scipy and sklearn), ~3 s in a fresh process. Imported in the
# event loop, it froze every open SSE stream for that long, once per process
# (#258). Detection also loads Ultralytics through its trainer, which switches
# Ultralytics' events off on import (ADR-180).
_TRAINING_STACK: tuple[str, ...] = ("torch", "torchvision", "visionforge.core.plotter")
_DETECTION_STACK: tuple[str, ...] = ("visionforge.core.detection_trainer",)
_TASK_STACK: dict[str, tuple[str, ...]] = {"detection": _DETECTION_STACK}


def _missing_training_stack(*extra: str) -> tuple[str, ...]:
    """The modules of the training stack (plus ``extra``) not imported yet."""
    return tuple(name for name in (*_TRAINING_STACK, *extra) if name not in sys.modules)


def _import_modules(names: tuple[str, ...]) -> None:
    """Import ``names`` in order; stop quietly at the first that fails.

    The run that needs it imports it again and reports the error as its own.
    """
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - the run reports it
            logger.warning("GUI: could not preload {}: {}", name, exc)
            return


async def load_training_stack(*extra: str) -> None:
    """Import the training stack in a worker thread, so the event loop never waits.

    Call it before importing a block, runner or trainer from a handler: once the
    stack is in ``sys.modules`` that import takes milliseconds.
    """
    missing = _missing_training_stack(*extra)
    if missing:
        await asyncio.to_thread(_import_modules, missing)


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


_GPU_UNAVAILABLE_TEXT = "GPU context unavailable"


async def request_lang(
    x_vf_lang: Annotated[str | None, Header(alias=LANG_HEADER)] = None,
) -> None:
    """Bind the page's language for the rest of the request (ADR-116).

    Async on purpose: a sync dependency runs in a worker thread, and a context
    variable set there never reaches the endpoint. It always sets, so a request
    without the header is Portuguese and never inherits another one's language.
    """
    set_lang(resolve_lang(x_vf_lang))


# Router-level, so every endpoint and every dependency below it (the profile
# lookup among them) writes its messages in the language of the page.
router = APIRouter(prefix="/api", dependencies=[Depends(request_lang)])


# Default location where Trainer writes run directories. Since ADR-114 this is
# the models folder of the *default profile*; a named profile reads and writes
# under _PROFILES_DIR / <slug> instead. Both are read when a request arrives, so
# they stay relative to the folder the GUI was started from.
_MODELS_DIR = Path("outputs/models")
_PROFILES_DIR = Path("outputs/profiles")
# Where the guide's synthetic sample dataset is written (ADR-115). Datasets are
# shared across profiles, so this is not under a profile folder.
_SAMPLE_DATASETS_DIR = Path("datasets")


def current_profile(
    x_vf_profile: Annotated[str | None, Header(alias=PROFILE_HEADER)] = None,
) -> Profile:
    """The per-request resolver of the folders a request works in (ADR-114).

    Reads ``X-VF-Profile``; an absent header is the default profile, which is
    the pre-profile layout (``_MODELS_DIR``). Every History route and every
    submission goes through this instead of reading ``_MODELS_DIR`` itself.

    Raises:
        HTTPException: 400 for a header that is not a safe slug (``../x``), 404
            for a safe slug with no profile folder.
    """
    try:
        return resolve_profile(x_vf_profile, _PROFILES_DIR, _MODELS_DIR)
    except InvalidProfileError as exc:
        raise HTTPException(400, str(exc)) from exc
    except UnknownProfileError as exc:
        raise HTTPException(404, str(exc)) from exc


ProfileDep = Annotated[Profile, Depends(current_profile)]


_EVIDENCE_EXPORTS = EvidenceExports()
# The Colab CLI and the logins waiting for a code live as long as the server (ADR-163).
_COLAB_CLI = ColabCli()
_COLAB_LOGINS = LoginFlows()
_COLAB_JOBS = ColabJobs()
# The Colab jobs waiting for (or able to wait for) the Drive consent, by run_id.
_DRIVE_CONSENTS: dict[str, DriveConsent] = {}
# How long the Colab job thread waits for the loop to take one event.
_COLAB_EVENT_TIMEOUT_S = 10.0


def _task_label_from_config(config_type: type[Any]) -> str:
    """Queue display label for a shared starter, derived from its config class.

    The shared orchestrators take a config type rather than a task name — and
    for a researcher-defined task they never see its real key. Every task's
    config is named ``<Task>Config``, so the class name is the honest source
    here. This is a label for the queue panel, never a dispatch key.
    """
    if config_type is ExperimentConfig:
        return "classification"
    return config_type.__name__.removesuffix("Config").lower() or "experiment"


def _submit_job(
    run_id: str,
    label: str,
    task: str,
    strategy: str,
    start: Callable[[], Awaitable[None]],
    stop_at: StopPoint | None | Literal["auto"] = "auto",
    profile: Profile | None = None,
    resume_dir: Path | None = None,
    lane: str = LOCAL_LANE,
) -> RunResponse:
    """Queue a training job and report whether it started or is waiting.

    ``stop_at`` defaults to what the strategy implies; the starters whose stop
    depends on the config (PatchCore, a custom task that owns its loop) pass it.
    ``profile`` is only the queue's label for whose folders the job writes into;
    the config handed to ``start`` was already scoped to it by the caller.
    """
    if cuda_context_broken():
        # Every job would fail within a second, CPU ones included; say so now.
        raise HTTPException(503, tr("gpu.context_broken"))
    owner = profile or default_profile(_MODELS_DIR)
    lang = current_lang()

    async def start_in_lang() -> None:
        if lane == LOCAL_LANE:
            # The executor imports its block on the loop; load what that pulls
            # in first, off the loop (#258). A Colab job trains elsewhere.
            await load_training_stack(*_TASK_STACK.get(task, ()))
        # The queue starts the job from its own task, long after this request's
        # language is gone; the messages the trainer writes (health warnings, the
        # paging-file hint) are for the person who pressed the button.
        with using_lang(lang):
            await start()

    status = _LANES.submit(
        QueuedJob(
            run_id=run_id,
            label=label,
            task=task,
            strategy=strategy,
            start=start_in_lang,
            stop_at=_stop_point(strategy) if stop_at == "auto" else stop_at,
            profile=owner.slug,
            resume_dir=resume_dir,
            profile_name=owner.name,
        ),
        lane,
    )
    return RunResponse(run_id=run_id, status=status)  # type: ignore[arg-type]


# Strong references to background tasks to prevent premature GC (asyncio doc).
_background_tasks: set[asyncio.Task[None]] = set()


@router.get("/schema", tags=[TAG_EXPERIMENTS])
async def get_schema() -> dict[str, Any]:
    """Return the JSON Schema for ExperimentConfig."""
    return ExperimentConfig.model_json_schema()


# Read probes per request so runtime overrides remain effective (ADR-132).
# Three loader pools reserve memory for train, validation and test (ADR-098).
router.include_router(
    build_system_router(lambda: check_cuda(), lambda: worker_budget(loader_pools=3))
)


# Profile endpoints live in profiles_routes.py (ADR-132). The paths stay owned
# here so tests that repoint routes._MODELS_DIR / _PROFILES_DIR still reach them.
router.include_router(
    build_hub_router(
        lambda header: current_profile(header),
        lambda run_id, profile: _find_run_dir(run_id, profile),
    )
)
router.include_router(build_profiles_router(lambda: (_PROFILES_DIR, _MODELS_DIR)))
router.include_router(
    build_wandb_router(
        lambda header: current_profile(header),
        lambda run_id, profile: _find_run_dir(run_id, profile),
        lambda run_dir: _retry_record(
            run_dir, read_wandb_status(run_dir), "wandb.retry_unavailable"
        ),
        lambda: _PROFILES_DIR.parent.resolve(),
        _EVIDENCE_EXPORTS,
    )
)
# A run's record, without weights, committed to a private GitHub repository (ADR-175).
router.include_router(
    build_github_router(
        lambda header: current_profile(header),
        lambda run_id, profile: _find_run_dir(run_id, profile),
    )
)
# Testing, Grad-CAM, batch prediction and ONNX export of a saved run (#204).
router.include_router(
    build_evaluation_router(
        lambda run_id, profile: _find_run_dir(run_id, profile),
        lambda header: current_profile(header),
    )
)
# A run versioned with DVC in a local repository of experiments (ADR-176).
router.include_router(
    build_dvc_router(
        lambda header: current_profile(header),
        lambda run_id, profile: _find_run_dir(run_id, profile),
        lambda request: _client_host(request),
    )
)
# Unpublished Zenodo drafts of a run's bundle (ADR-174).
router.include_router(
    build_zenodo_router(
        lambda header: current_profile(header),
        lambda run_id, profile: _find_run_dir(run_id, profile),
    )
)
# The real training code behind the "Código" tab (ADR-173).
router.include_router(code_tour_router)
# Dataset, picker and credential endpoints live in datasets_routes.py (ADR-132).
router.include_router(datasets_router)
# The read-only dataset explorer (ADR-165).
router.include_router(explorer_router)
# Adding new images to a dataset as a new dataset (ADR-166).
router.include_router(incorporate_router)
# Merging two datasets into a new one (ADR-167).
router.include_router(merge_router)
# Re-splitting a dataset into a new one (ADR-168).
router.include_router(resplit_router)
router.include_router(coco_router)
# History list, detail and delete live in runs_routes.py (ADR-132); the lookups
# stay here, the job state in execution.py (ADR-185); both read per request.
router.include_router(
    build_runs_router(
        lambda run_id, profile: _find_run_dir(run_id, profile),
        lambda header: current_profile(header),
        _run_state,
        lambda request: _client_host(request),
    )
)
router.include_router(
    build_colab_router(
        lambda header: current_profile(header), _COLAB_CLI, _COLAB_LOGINS, _COLAB_JOBS
    )
)
router.include_router(
    build_evidence_router(
        lambda run_id, profile: _find_run_dir(run_id, profile),
        lambda header: current_profile(header),
        lambda run_id, run_dir, profile: _LANES.is_writing(
            run_id, run_dir, profile.slug
        ),
        _EVIDENCE_EXPORTS,
    )
)


def _client_host(request: Request) -> str | None:
    """The address the request came from, None when the transport has none."""
    return request.client.host if request.client else None


# Why the button cannot install, as the catalog words it (ADR-120).
_UPDATE_BLOCKED = {
    "source": "update.blocked_source",
    "no_installer": "update.blocked_no_installer",
    "remote": "update.blocked_remote",
    "busy": "update.blocked_busy",
}


@router.get("/update", tags=[TAG_SYSTEM])
async def get_update(request: Request) -> dict[str, Any]:
    """Whether a newer release exists, and whether this page may install it (ADR-120).

    The PyPI lookup blocks for up to a few seconds when the network is slow, so it
    runs off the event loop; its answer is cached for hours.
    """
    from visionforge.gui.api.updates import update_status

    return await asyncio.to_thread(
        update_status,
        client_is_local=is_loopback_host(_client_host(request)),
        busy=_LANES.is_busy(),
    )


@router.post(
    "/update/apply",
    responses={
        403: {"description": FORBIDDEN_TEXT},
        409: {"description": CONFLICT_TEXT},
        500: {"description": SERVER_ERROR_TEXT},
    },
    tags=[TAG_SYSTEM],
)
async def apply_update(request: Request) -> dict[str, str]:
    """Install the announced release and restart the GUI (ADR-120).

    Refused for a page of another site and for a client on another machine, the
    same two checks as "open folder": reinstalling the server is the server
    owner's decision. Refused while a job runs or waits, which the restart would
    kill. The updater runs in a process of its own; this server exits right after
    answering, so the updater can replace its files and reuse its port.
    """
    from visionforge.gui.api.updates import spawn_updater, update_status
    from visionforge.gui.server import request_shutdown

    origin = request.headers.get("origin")
    if origin is not None and not is_loopback_origin(origin):
        raise HTTPException(403, tr("update.foreign_origin"))
    local = is_loopback_host(_client_host(request))
    busy = _LANES.is_busy()
    status = await asyncio.to_thread(update_status, client_is_local=local, busy=busy)
    if not status["available"]:
        raise HTTPException(409, tr("update.none"))
    if not status["can_apply"]:
        raise HTTPException(
            403 if status["blocked"] == "remote" else 409,
            tr(_UPDATE_BLOCKED[status["blocked"]]),
        )
    host, port = request.scope.get("server") or ("127.0.0.1", 8000)
    try:
        spawn_updater(version=status["latest"], host=str(host), port=int(port))
    except (OSError, RuntimeError) as exc:
        raise HTTPException(500, tr("update.spawn_failed", error=exc)) from exc
    request_shutdown()
    return {"status": "restarting", "target": status["latest"]}


@router.post(
    "/runs/{run_id}/reveal",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        403: {"description": FORBIDDEN_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        500: {"description": SERVER_ERROR_TEXT},
    },
    tags=[TAG_RUNS],
)
async def reveal_run_folder(
    run_id: str, request: Request, profile: ProfileDep
) -> dict[str, str]:
    """Open a run's folder in the file manager of the machine running the server.

    The folder is found by ``_find_run_dir`` alone: the request carries no path,
    so it cannot point the file manager anywhere else. And the window opens on
    the *server's* desktop, which is the clicker's only when both are the same
    machine, so a client that is not loopback is refused (403) -- the page hides
    the button for it too (``can_reveal`` on the run detail).

    A page on another site can POST here from the user's own browser, which makes
    the client loopback too (CSRF). A browser names that page in ``Origin``, so a
    request whose ``Origin`` is not this machine is refused. No ``Origin`` is a
    same-origin navigation or a client that is not a browser: allowed.
    """
    origin = request.headers.get("origin")
    if origin is not None and not is_loopback_origin(origin):
        raise HTTPException(403, tr("reveal.foreign_origin"))
    if not is_loopback_host(_client_host(request)):
        raise HTTPException(403, tr("reveal.remote_client"))
    run_dir = _find_run_dir(run_id, profile)
    if run_dir is None:
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))
    try:
        open_folder(run_dir)
    except FolderOpenError as exc:
        raise HTTPException(500, tr("reveal.failed", error=exc)) from exc
    return {"run_id": run_dir.name, "run_dir": str(run_dir.resolve())}


_MLFLOW_UNAVAILABLE = "mlflow.retry_unavailable"


def _retry_record(
    run_dir: Path,
    status: MlflowStatus | WandbStatus | None,
    unavailable: str = _MLFLOW_UNAVAILABLE,
) -> dict[str, Any]:
    """The run.json of a finished, exported run, or 409/500.

    ``status`` is the exporter's sidecar: a retry needs one that is not still
    running.
    """
    if status is None or status.status == "RUNNING":
        raise HTTPException(409, tr(unavailable))
    try:
        data = json.loads((run_dir / RUN_FILENAME).read_bytes())
    except (OSError, ValueError) as exc:
        raise HTTPException(500, tr(RUN_BAD_JSON_KEY, error=exc)) from exc
    # Any task: the sidecar above proves the run was exported (ADR-197).
    if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
        raise HTTPException(409, tr(unavailable))
    return data


@router.post(
    "/runs/{run_id}/mlflow/retry",
    responses={
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
        500: {"description": SERVER_ERROR_TEXT},
    },
    tags=[TAG_RUNS],
)
def retry_mlflow(run_id: str, profile: ProfileDep) -> MlflowStatus:
    """Reconcile a completed local run with its profile's MLflow store."""
    from visionforge.core.mlflow_tracking import MlflowTracker

    run_dir = _find_run_dir(run_id, profile)
    if (
        run_dir is None
        or not run_dir.resolve().is_relative_to(profile.models_dir.resolve())
        or (run_dir / RUN_FILENAME).is_symlink()
    ):
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))
    if not profile.settings.mlflow.enabled or not _EVIDENCE_EXPORTS.reserve(run_dir):
        raise HTTPException(409, tr(_MLFLOW_UNAVAILABLE))
    try:
        data = _retry_record(run_dir, read_mlflow_status(run_dir))
        tracker = MlflowTracker(_PROFILES_DIR.parent / "mlflow", profile.slug, run_dir)
        tracker.retry(data)
        result = read_mlflow_status(run_dir)
        if result is None:
            raise HTTPException(500, tr(_MLFLOW_UNAVAILABLE))
        return result
    finally:
        _EVIDENCE_EXPORTS.release(run_dir)


@router.post(
    "/runs/{run_id}/resume",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": "Run not found"},
        409: {"description": "Run is busy or cannot continue"},
        500: {"description": "Run metadata is unreadable"},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_RUNS],
)
async def resume_run(run_id: str, profile: ProfileDep) -> RunResponse:
    """Continue a stopped run in its own directory (ADR-092/093).

    The config comes from the run's own ``run.json`` rather than from the
    browser: continuing with different hyperparameters would produce one
    directory whose history describes two different experiments. The run is
    looked up in the caller's profile only, and the config is scoped to that
    profile again so its reports and graphics follow the run (ADR-114).
    """
    run_dir = _find_run_dir(run_id, profile)
    if run_dir is None:
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))
    if _EVIDENCE_EXPORTS.is_exporting(run_dir):
        raise HTTPException(409, tr("evidence.export_busy"))
    _reject_group_action(run_dir)
    try:
        data: dict[str, Any] = json.loads(
            (run_dir / RUN_FILENAME).read_text(encoding="utf-8")
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, tr(RUN_BAD_JSON_KEY, error=exc)) from exc

    resumable, _ = _resume_status(run_dir, data)
    if not resumable:
        raise HTTPException(409, tr("run.nothing_to_continue", run_id=run_id))

    # Built here rather than at module level: the executors are defined further
    # down this file and do not exist yet while it is being imported.
    task = _run_task(data)
    executors: dict[str, tuple[Any, Any]] = {
        "classification": (ExperimentConfig, execution._execute_experiment),
        "detection": (DetectionConfig, _execute_detection),
        "regression": (RegressionConfig, _execute_regression),
        "segmentation": (SegmentationConfig, _execute_segmentation),
        "anomaly": (AnomalyConfig, _execute_anomaly),
    }
    if task not in executors:
        raise HTTPException(400, tr("run.cannot_resume_task", task=task))
    config_cls, executor = executors[task]

    try:
        config = config_cls.model_validate(data["config"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            400, tr("run.stored_config_invalid", run_id=run_id, error=exc)
        ) from exc

    config = profile.scope_model(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_id = f"{config.name}_{timestamp}"
    return _submit_job(
        job_id,
        tr("run.resuming_label", name=config.name),
        task,
        "simple",
        lambda: executor(config, job_id, resume_dir=run_dir),
        profile=profile,
        resume_dir=run_dir,
    )


@router.get(
    "/runs/{run_id}/export_md",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        500: {"description": SERVER_ERROR_TEXT},
    },
    tags=[TAG_RUNS],
)
async def export_run_markdown(
    run_id: str,
    profile: ProfileDep,
    x_vf_lang: Annotated[str | None, Header(alias=LANG_HEADER)] = None,
) -> Response:
    """Render a model card (markdown) for the run, ready for paper inclusion.

    The card is an artifact: it follows the language the page asked for, and is
    English when the request names none (a script calling the endpoint).
    """
    run_dir = _find_run_dir(run_id, profile)
    if run_dir is None:
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))
    run_json_path = run_dir / RUN_FILENAME
    try:
        data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, tr(RUN_BAD_JSON_KEY, error=exc)) from exc

    with using_lang(parse_lang(x_vf_lang) or "en"):
        md = _render_run_markdown(run_dir, data)
    filename = f"{data.get('experiment', run_dir.name)}_{run_dir.name}.md"
    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/dataset/preview_preprocess",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
    },
    tags=[TAG_DATASETS],
)
async def preview_preprocess(
    req: PreprocessPreviewRequest,
) -> PreprocessPreviewResponse:
    """Generate a strip of PNG previews showing the preprocessing pipeline.

    Picks one image from the requested split (first one alphabetically inside
    ``class_name`` or the first class found) and runs every step. Each step's
    output is saved as a PNG under ``outputs/preview_cache/<run-id>/`` and
    served via the existing /api/artifacts/{path} endpoint.

    Raises:
        HTTPException: 422 when a step names a filter that does not exist —
            that is bad input, not a server fault, and the message lists the
            filters that do (ADR-078).
    """
    try:
        return await asyncio.to_thread(_render_preprocess_preview, req)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post(
    "/sample-dataset",
    status_code=201,
    response_model=SampleDatasetResponse,
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        409: {"description": CONFLICT_TEXT},
        500: {"description": SERVER_ERROR_TEXT},
    },
    tags=[TAG_DATASETS],
)
async def sample_dataset(
    req: SampleDatasetRequest,
) -> SampleDatasetResponse | JSONResponse:
    """Write the synthetic sample dataset of a task for the guide (ADR-115, ADR-117).

    Local and offline: the self-test builder draws tiny images with a
    deterministic seed. Every built-in task has one, in its own layout.

    Raises:
        HTTPException: 400 for an unknown task, 409 when
            ``datasets/exemplo-<tarefa>/`` already holds files (the body carries
            ``path`` so the guide can use that folder), 500 when the OS refuses
            to write.
    """
    try:
        built = await asyncio.to_thread(
            create_sample_dataset, _SAMPLE_DATASETS_DIR, req.task
        )
    except SampleTaskError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SampleDatasetExistsError as exc:
        # `path` rides next to `detail` instead of inside it, so a client that
        # only knows string details still shows a readable message.
        return JSONResponse(
            status_code=409, content={"detail": str(exc), "path": str(exc.path)}
        )
    except OSError as exc:
        logger.warning("GUI: could not write the sample dataset: {}", exc)
        raise HTTPException(500, tr("sample.write_failed", error=exc)) from exc
    _remember_dataset_root(built.path)
    logger.info("GUI: wrote the {} sample dataset at {}", req.task, built.path)
    return SampleDatasetResponse(
        task=built.task,
        path=str(built.path),
        classes=built.classes,
        counts=built.counts,
        targets=built.targets,
    )


def _tracker_factories(
    config: ExperimentConfig, profile: Profile
) -> tuple[TrackerFactory, ...]:
    """The exporters this profile turned on, for a classification training only."""
    if config.block != "classification" or config.classification.mode != "train":
        return ()
    return profile_trackers(profile, _PROFILES_DIR.parent.resolve())


@router.post(
    "/experiment/run",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_EXPERIMENTS],
)
async def run_experiment(config: ExperimentConfig, profile: ProfileDep) -> RunResponse:
    """Start a training experiment, or queue it if one is already running."""
    config = profile.scope_model(config)
    trackers = _tracker_factories(config, profile)
    await asyncio.to_thread(_check_classification_classes, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "classification",
        config.block,
        lambda: (
            execution._execute_experiment(config, run_id, trackers=trackers)
            if trackers
            else execution._execute_experiment(config, run_id)
        ),
        profile=profile,
    )


@router.post(
    "/colab/experiment/run",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        409: {
            "description": "The Colab CLI is not installed or the profile is not logged in"
        },
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_COLAB],
)
async def run_colab_experiment(
    req: ColabRunRequest, profile: ProfileDep
) -> RunResponse:
    """Train a classification experiment on a Colab VM (ADR-163, phase 2).

    It goes through the same queue and the same /experiment/{status,events,result}
    endpoints as a local run. The dataset comes from ``req.source`` on the VM; the
    local ``data.base_dir`` is not read.
    """
    if _COLAB_CLI.installed_version() != _COLAB_PIN:
        raise HTTPException(409, tr("colab.not_installed"))
    if not _COLAB_CLI.logged_in(profile.slug):
        raise HTTPException(409, tr("colab.not_logged_in"))
    config = _validate_colab_config(req.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_colab_{timestamp}"
    payload = {
        "version": _VF_VERSION,
        # The validated form, so the VM gets the defaults this version fills in;
        # the runner points data and outputs at the VM.
        "config": config.model_dump(mode="json"),
        "source": req.source.model_dump(mode="json"),
        "drive_export": req.drive_export,
    }
    if isinstance(req.source, ColabDriveSource):
        payload["source"] = await asyncio.to_thread(
            _drive_source, req.config, req.source.my_drive_root
        )
    return _submit_job(
        run_id,
        f"{config.name} · Colab {req.accelerator}",
        "classification",
        "colab",
        lambda: _execute_colab(
            run_id, profile.slug, req.accelerator, payload, profile.models_dir
        ),
        stop_at="vm",
        profile=profile,
        lane=await asyncio.to_thread(_colab_lane_of, profile.slug),
    )


@router.post(
    "/colab/drive/continue",
    responses={404: {"description": "No job is waiting for the Drive consent"}},
    tags=[TAG_COLAB],
)
async def continue_drive_consent(req: ColabDriveContinueRequest) -> dict[str, bool]:
    """Answer the Drive consent prompt of a Colab job (ADR-164).

    The GUI calls it when the person comes back from the consent page, or
    presses "I already allowed". The server then sends the Enter the CLI waits
    for; nobody touches a terminal.
    """
    consent = _DRIVE_CONSENTS.get(req.run_id)
    if consent is None or not consent.grant():
        raise HTTPException(404, tr("colab.drive_not_waiting"))
    return {"sent": True}


def _drive_source(config: dict[str, Any], my_drive_root: str | None) -> dict[str, Any]:
    """The form's dataset folder as a path on the VM's mounted Drive (ADR-164).

    The local file count goes along: the VM compares it after copying, which
    catches a folder Drive for desktop has not finished uploading.
    """
    data = config.get("data")
    base_dir = data.get("base_dir") if isinstance(data, dict) else None
    local = Path(str(base_dir or ""))
    remote = remote_path(local, my_drive_roots(my_drive_root)) if base_dir else None
    if remote is None:
        raise HTTPException(400, tr("colab.drive_not_in_drive", path=str(local)))
    files = count_files(local)
    if files == 0:
        raise HTTPException(400, tr("colab.drive_empty", path=str(local)))
    return {"kind": "drive", "path": remote.as_posix(), "files": files}


def _colab_lane_of(slug: str) -> str:
    """The queue of the Google account the profile is logged into (ADR-163, phase 4).

    The session quota and the compute units are the account's, so two profiles
    on the same account share one queue. A login whose e-mail ``colab whoami``
    did not report gets a queue of its own rather than one shared by guess.
    """
    account = _COLAB_CLI.account(slug)
    return colab_lane(account.lower() if account else f"profile:{slug}")


@router.get("/detection/schema", tags=[TAG_DETECTION])
async def get_detection_schema() -> dict[str, Any]:
    """Return the JSON Schema for DetectionConfig (drives the detection form)."""
    return DetectionConfig.model_json_schema()


@router.post(
    "/detection/run",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_DETECTION],
)
async def run_detection(config: DetectionConfig, profile: ProfileDep) -> RunResponse:
    """Start an object-detection training run in the background.

    Detection shares the run queue and the /experiment/{status,events,result}
    endpoints (one run at a time, one GPU) but dispatches its own standalone
    DetectionBlock (ADR-033) rather than the ExperimentConfig path.
    """
    config = profile.scope_model(config)
    trackers = profile_trackers(profile, _PROFILES_DIR.parent.resolve())
    await asyncio.to_thread(_check_detection_dataset, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "detection",
        "simple",
        lambda: exported(_execute_detection(config, run_id), run_id, trackers),
        profile=profile,
    )


@router.get("/regression/schema", tags=[TAG_REGRESSION])
async def get_regression_schema() -> dict[str, Any]:
    """Return the JSON Schema for RegressionConfig (drives the regression form)."""
    return RegressionConfig.model_json_schema()


@router.post(
    "/regression/run",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_REGRESSION],
)
async def run_regression(config: RegressionConfig, profile: ProfileDep) -> RunResponse:
    """Start an image-regression training run in the background.

    Regression shares the run queue and the /experiment/{status,events,result}
    endpoints (one run at a time, one GPU) but dispatches its own standalone
    RegressionBlock (ADR-036) rather than the ExperimentConfig path.
    """
    config = profile.scope_model(config)
    trackers = profile_trackers(profile, _PROFILES_DIR.parent.resolve())
    await asyncio.to_thread(_check_regression_dataset, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "regression",
        "simple",
        lambda: exported(_execute_regression(config, run_id), run_id, trackers),
        profile=profile,
    )


@router.get("/segmentation/schema", tags=[TAG_SEGMENTATION])
async def get_segmentation_schema() -> dict[str, Any]:
    """Return the JSON Schema for SegmentationConfig (drives the segmentation form)."""
    return SegmentationConfig.model_json_schema()


@router.post(
    "/segmentation/run",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_SEGMENTATION],
)
async def run_segmentation(
    config: SegmentationConfig, profile: ProfileDep
) -> RunResponse:
    """Start a semantic-segmentation training run in the background.

    Segmentation shares the run queue and the /experiment/{status,events,result}
    endpoints (one run at a time, one GPU) but dispatches its own standalone
    SegmentationBlock (ADR-037) rather than the ExperimentConfig path.
    """
    config = profile.scope_model(config)
    trackers = profile_trackers(profile, _PROFILES_DIR.parent.resolve())
    await asyncio.to_thread(_check_segmentation_dataset, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "segmentation",
        "simple",
        lambda: exported(_execute_segmentation(config, run_id), run_id, trackers),
        profile=profile,
    )


@router.get("/anomaly/schema", tags=[TAG_ANOMALY])
async def get_anomaly_schema() -> dict[str, Any]:
    """Return the JSON Schema for AnomalyConfig (drives the anomaly form)."""
    return AnomalyConfig.model_json_schema()


@router.post(
    "/anomaly/run",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_ANOMALY],
)
async def run_anomaly(config: AnomalyConfig, profile: ProfileDep) -> RunResponse:
    """Start an anomaly-detection training run in the background.

    Anomaly shares the run queue and the /experiment/{status,events,result}
    endpoints (one run at a time, one GPU) but dispatches its own standalone
    AnomalyBlock (ADR-038) rather than the ExperimentConfig path.
    """
    config = profile.scope_model(config)
    trackers = profile_trackers(profile, _PROFILES_DIR.parent.resolve())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "anomaly",
        "simple",
        lambda: exported(_execute_anomaly(config, run_id), run_id, trackers),
        # PatchCore has no epochs; it stops between its phases (ADR-111).
        stop_at="phase" if config.model.name == "patchcore" else "epoch",
        profile=profile,
    )


@router.post(
    "/regression/compare",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_REGRESSION],
)
async def compare_regression(
    req: ComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Train N regression backbones on the same dataset and rank them (ADR-044)."""
    await load_training_stack()
    from visionforge.blocks.regression_runner import RegressionRunner

    return _start_comparison(req, RegressionConfig, RegressionRunner(), profile)


@router.post(
    "/segmentation/compare",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_SEGMENTATION],
)
async def compare_segmentation(
    req: ComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Train N segmentation backbones on the same dataset and rank them (ADR-044)."""
    await load_training_stack()
    from visionforge.blocks.segmentation_runner import SegmentationRunner

    return _start_comparison(req, SegmentationConfig, SegmentationRunner(), profile)


@router.post(
    "/regression/sweep",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_REGRESSION],
)
async def sweep_regression(req: SweepRequest, profile: ProfileDep) -> RunResponse:
    """Run a grid/random hyperparameter sweep for regression (ADR-045)."""
    await load_training_stack()
    from visionforge.blocks.regression_runner import RegressionRunner

    return _start_sweep(req, RegressionConfig, RegressionRunner(), profile)


@router.post(
    "/segmentation/sweep",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_SEGMENTATION],
)
async def sweep_segmentation(req: SweepRequest, profile: ProfileDep) -> RunResponse:
    """Run a grid/random hyperparameter sweep for segmentation (ADR-045)."""
    await load_training_stack()
    from visionforge.blocks.segmentation_runner import SegmentationRunner

    return _start_sweep(req, SegmentationConfig, SegmentationRunner(), profile)


@router.post(
    "/detection/compare",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_DETECTION],
)
async def compare_detection(req: ComparisonRequest, profile: ProfileDep) -> RunResponse:
    """Train N detectors on the same dataset and rank them (ADR-044)."""
    await load_training_stack(*_DETECTION_STACK)
    from visionforge.blocks.detection_runner import DetectionRunner

    return _start_comparison(req, DetectionConfig, DetectionRunner(), profile)


@router.post(
    "/detection/sweep",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_DETECTION],
)
async def sweep_detection(req: SweepRequest, profile: ProfileDep) -> RunResponse:
    """Run a grid/random hyperparameter sweep for detection (ADR-045)."""
    await load_training_stack(*_DETECTION_STACK)
    from visionforge.blocks.detection_runner import DetectionRunner

    return _start_sweep(req, DetectionConfig, DetectionRunner(), profile)


@router.post(
    "/anomaly/compare",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_ANOMALY],
)
async def compare_anomaly(req: ComparisonRequest, profile: ProfileDep) -> RunResponse:
    """Train N anomaly models on the same dataset and rank them (ADR-044)."""
    await load_training_stack()
    from visionforge.blocks.anomaly_runner import AnomalyRunner

    return _start_comparison(req, AnomalyConfig, AnomalyRunner(), profile)


@router.post(
    "/anomaly/sweep",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_ANOMALY],
)
async def sweep_anomaly(req: SweepRequest, profile: ProfileDep) -> RunResponse:
    """Run a grid/random hyperparameter sweep for anomaly detection (ADR-045)."""
    await load_training_stack()
    from visionforge.blocks.anomaly_runner import AnomalyRunner

    return _start_sweep(req, AnomalyConfig, AnomalyRunner(), profile)


# Built-in task descriptors — accents mirror frontend/src/types/tasks.ts.
_BUILTIN_TASK_DESCRIPTORS = [
    TaskDescriptor(
        key="classification",
        label="Classificação",
        accent="#f16363",
        description="Categorize imagens em rótulos discretos",
        primary_metric="accuracy",
    ),
    TaskDescriptor(
        key="detection",
        label="Detecção de Objeto",
        accent="#48cf8e",
        description="Localize objetos com caixas delimitadoras",
        primary_metric="map50_95",
    ),
    TaskDescriptor(
        key="regression",
        label="Regressão",
        accent="#5b9fff",
        description="Estime valores contínuos a partir das entradas",
        primary_metric="r2",
    ),
    TaskDescriptor(
        key="segmentation",
        label="Segmentação",
        accent="#b079ff",
        description="Classifique cada pixel da imagem",
        primary_metric="miou",
    ),
    TaskDescriptor(
        key="anomaly",
        label="Anomalia",
        accent="#f5a524",
        description="Detecte defeitos treinando só com imagens normais",
        primary_metric="auroc",
    ),
]


@router.get("/tasks", tags=[TAG_EXPERIMENTS])
async def list_tasks() -> TaskListResponse:
    """Every renderable task: the five built-ins + registered custom tasks.

    Scans ``user_tasks/`` so a freshly dropped task file appears without a
    server restart (ADR-058).
    """
    from visionforge.tasks.manage import hidden_tasks
    from visionforge.tasks.registry import load_user_tasks

    load_user_tasks()
    # A hidden task stays registered — it can still be re-run from a YAML and
    # its past runs still resolve — it just does not claim a tab.
    hidden = hidden_tasks()
    custom = [
        TaskDescriptor(
            key=t.key,
            label=t.label,
            accent=t.accent,
            description=t.description,
            custom=True,
            metrics=t.metrics,
            primary_metric=t.primary_metric,
        )
        for t in registered_tasks()
        if t.key not in hidden
    ]
    return TaskListResponse(tasks=[*_BUILTIN_TASK_DESCRIPTORS, *custom])


@router.post(
    "/custom/{key}/hide",
    responses={404: {"description": NOT_FOUND_TEXT}},
    tags=[TAG_CUSTOM_TASKS],
)
async def hide_custom_task(key: str) -> CustomTaskActionResponse:
    """Stop rendering a custom task's tab, keeping its file untouched."""
    from visionforge.tasks.manage import set_hidden

    _get_custom_task_or_404(key)
    changed = set_hidden(key, True)
    return CustomTaskActionResponse(
        key=key,
        action="hidden",
        detail=tr("custom.hidden") if changed else tr("custom.already_hidden"),
    )


@router.post("/custom/{key}/unhide", tags=[TAG_CUSTOM_TASKS])
async def unhide_custom_task(key: str) -> CustomTaskActionResponse:
    """Bring a hidden custom task's tab back."""
    from visionforge.tasks.manage import set_hidden

    set_hidden(key, False)
    return CustomTaskActionResponse(
        key=key, action="unhidden", detail=tr("custom.unhidden")
    )


@router.delete(
    "/custom/{key}",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
    },
    tags=[TAG_CUSTOM_TASKS],
)
async def delete_custom_task(key: str, confirm: str = "") -> CustomTaskActionResponse:
    """Delete a custom task's source file. ``confirm`` must repeat the key.

    Typing the name is deliberately more work than clicking: this removes code
    the researcher wrote, and there is no undo.
    """
    from visionforge.tasks.manage import delete_task

    try:
        removed = delete_task(key, confirm)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return CustomTaskActionResponse(
        key=key, action="deleted", detail=tr("custom.deleted", removed=removed)
    )


def _get_custom_task_or_404(key: str) -> Any:
    try:
        return get_task(key)
    except KeyError as exc:
        # str() of a KeyError is the repr of its argument, quotes included.
        raise HTTPException(404, exc.args[0] if exc.args else str(exc)) from exc


@router.get(
    "/custom/{key}/schema",
    responses={404: {"description": NOT_FOUND_TEXT}},
    tags=[TAG_CUSTOM_TASKS],
)
async def custom_task_schema(key: str) -> dict[str, Any]:
    """JSON Schema of a custom task's Config — drives the generic form."""
    info = _get_custom_task_or_404(key)
    schema: dict[str, Any] = info.spec_cls.Config.model_json_schema()
    return _localize_task_schema(schema)


# The hints the generic form shows (SchemaForm renders each `description`) that
# VisionForge writes itself, in the shared training block every researcher task
# inherits. The researcher's own descriptions are theirs and stay as written.
_TASK_SCHEMA_HINTS = {
    ("TaskTrainingConfig", "early_stopping_patience"): "schema.early_stopping_patience",
    ("TaskTrainingConfig", "deterministic"): "schema.deterministic",
}


def _localize_task_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema with VisionForge's own field hints in the page's language (ADR-116)."""
    defs = schema.get("$defs") or {}
    for (model, field), key in _TASK_SCHEMA_HINTS.items():
        prop = (defs.get(model) or {}).get("properties", {}).get(field)
        if isinstance(prop, dict) and "description" in prop:
            prop["description"] = tr(key)
    return schema


@router.post(
    "/custom/{key}/run",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_CUSTOM_TASKS],
)
async def run_custom_task(
    key: str, config: dict[str, Any], profile: ProfileDep
) -> RunResponse:
    """Start a custom-task training run in the background (ADR-058).

    Validates against the task's own Config; reuses the shared run queue and the
    /experiment/{status,events,result} endpoints.

    Raises:
        HTTPException: 404 unknown key.
        RequestValidationError: 422 if the config is invalid for the task.
    """
    info = _get_custom_task_or_404(key)

    try:
        cfg = info.spec_cls.Config.model_validate(profile.scope_dict(config))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    trackers = profile_trackers(profile, _PROFILES_DIR.parent.resolve())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{cfg.name}_{timestamp}"

    return _submit_job(
        run_id,
        str(cfg.name),
        f"custom:{info.key}",
        "simple",
        lambda: exported(_execute_custom_task(info, cfg, run_id), run_id, trackers),
        stop_at=None if _owns_its_loop(info) else "epoch",
        profile=profile,
    )


def _owns_its_loop(info: Any) -> bool:
    """True for a Level 2 custom task, whose own ``run`` never reads a stop request."""
    return info.spec_cls is not None and info.spec_cls.run is not TaskSpec.run


async def _execute_custom_task(info: Any, cfg: Any, run_id: str) -> None:
    """Run a custom task's engine in a worker thread and store the report."""
    state = _job_state(run_id)

    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        engine = _generic_task_engine()(info, cfg)
        logger.info("GUI: Starting custom task '{}' run {}", info.key, run_id)
        result = await asyncio.to_thread(engine.run, _put_event, state.token)

        report = {
            "task": f"custom:{info.key}",
            "metrics": result.metrics,
            "best_epoch": result.best_epoch,
            "total_epochs": result.total_epochs,
            "device_used": result.device_used,
            "run_dir": str(result.run_dir),
            # True only when the stop cut the training (ADR-111).
            "stopped": result.stopped,
        }
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": str(result.run_dir),
        }
        logger.success("GUI: Custom task {} completed.", run_id)
    except Exception as e:
        logger.exception("GUI: Custom task {} failed", run_id)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e or tr('run.no_message')}",
            "report": None,
            "run_dir": None,
        }
    finally:
        if queue is not None:
            await queue.put(None)


@router.post(
    "/custom/{key}/sweep",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_CUSTOM_TASKS],
)
async def sweep_custom_task(
    key: str, req: SweepRequest, profile: ProfileDep
) -> RunResponse:
    """Run a grid/random hyperparameter sweep for a custom task (ADR-045/058).

    ``CustomTaskRunner`` gives the task the same orchestrator the built-ins
    use; the search space is validated against the task's own Config. There is
    deliberately no /custom compare endpoint: comparison overrides
    ``model.name``, which ``BaseTaskConfig`` does not guarantee — a one-axis
    sweep over whichever field the task declares covers that use case.
    """
    await load_training_stack()
    from visionforge.tasks.runner import CustomTaskRunner

    info = _get_custom_task_or_404(key)
    return _start_sweep(req, info.spec_cls.Config, CustomTaskRunner(info), profile)


@router.post(
    "/custom/{key}/replicates",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_CUSTOM_TASKS],
)
async def replicates_custom_task(
    key: str, req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one custom-task config N times under different seeds (ADR-056/058)."""
    await load_training_stack()
    from visionforge.tasks.runner import CustomTaskRunner

    info = _get_custom_task_or_404(key)
    return _start_replicates(req, info.spec_cls.Config, CustomTaskRunner(info), profile)


@router.post(
    "/regression/cv",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_REGRESSION],
)
async def cv_regression(req: TaskCvRequest, profile: ProfileDep) -> RunResponse:
    """Run K-fold cross-validation for regression (ADR-050) in the background."""
    return _start_task_cv(
        req, RegressionConfig, run_regression_cross_validation, "Regression", profile
    )


@router.post(
    "/segmentation/cv",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_SEGMENTATION],
)
async def cv_segmentation(req: TaskCvRequest, profile: ProfileDep) -> RunResponse:
    """Run K-fold cross-validation for segmentation in the background."""
    return _start_task_cv(
        req,
        SegmentationConfig,
        run_segmentation_cross_validation,
        "Segmentation",
        profile,
    )


@router.post(
    "/classification/replicates",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_EXPERIMENTS],
)
async def replicates_classification(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one classification config N times under different seeds (ADR-056)."""
    await load_training_stack()
    from visionforge.blocks.classification_runner import ClassificationRunner

    return _start_replicates(req, ExperimentConfig, ClassificationRunner(), profile)


@router.post(
    "/regression/replicates",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_REGRESSION],
)
async def replicates_regression(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one regression config N times under different seeds (ADR-056)."""
    await load_training_stack()
    from visionforge.blocks.regression_runner import RegressionRunner

    return _start_replicates(req, RegressionConfig, RegressionRunner(), profile)


@router.post(
    "/segmentation/replicates",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_SEGMENTATION],
)
async def replicates_segmentation(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one segmentation config N times under different seeds (ADR-056)."""
    await load_training_stack()
    from visionforge.blocks.segmentation_runner import SegmentationRunner

    return _start_replicates(req, SegmentationConfig, SegmentationRunner(), profile)


@router.post(
    "/detection/replicates",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_DETECTION],
)
async def replicates_detection(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one detection config N times under different seeds (ADR-056)."""
    await load_training_stack(*_DETECTION_STACK)
    from visionforge.blocks.detection_runner import DetectionRunner

    return _start_replicates(req, DetectionConfig, DetectionRunner(), profile)


@router.post(
    "/anomaly/replicates",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_ANOMALY],
)
async def replicates_anomaly(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one anomaly config N times under different seeds (ADR-056)."""
    await load_training_stack()
    from visionforge.blocks.anomaly_runner import AnomalyRunner

    return _start_replicates(req, AnomalyConfig, AnomalyRunner(), profile)


@router.post(
    "/classification/replicated-comparison",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_EXPERIMENTS],
)
async def replicated_comparison_classification(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare classification variants over shared seeds with a paired test."""
    await load_training_stack()
    from visionforge.blocks.classification_runner import ClassificationRunner

    return _start_replicated_comparison(
        req, ExperimentConfig, ClassificationRunner(), profile
    )


@router.post(
    "/regression/replicated-comparison",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_REGRESSION],
)
async def replicated_comparison_regression(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare regression variants over shared seeds with a paired test."""
    await load_training_stack()
    from visionforge.blocks.regression_runner import RegressionRunner

    return _start_replicated_comparison(
        req, RegressionConfig, RegressionRunner(), profile
    )


@router.post(
    "/segmentation/replicated-comparison",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_SEGMENTATION],
)
async def replicated_comparison_segmentation(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare segmentation variants over shared seeds with a paired test."""
    await load_training_stack()
    from visionforge.blocks.segmentation_runner import SegmentationRunner

    return _start_replicated_comparison(
        req, SegmentationConfig, SegmentationRunner(), profile
    )


@router.post(
    "/detection/replicated-comparison",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_DETECTION],
)
async def replicated_comparison_detection(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare detection variants over shared seeds with a paired test."""
    await load_training_stack(*_DETECTION_STACK)
    from visionforge.blocks.detection_runner import DetectionRunner

    return _start_replicated_comparison(
        req, DetectionConfig, DetectionRunner(), profile
    )


@router.post(
    "/anomaly/replicated-comparison",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_ANOMALY],
)
async def replicated_comparison_anomaly(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare anomaly variants over shared seeds with a paired test."""
    await load_training_stack()
    from visionforge.blocks.anomaly_runner import AnomalyRunner

    return _start_replicated_comparison(req, AnomalyConfig, AnomalyRunner(), profile)


@router.post(
    "/custom/{key}/replicated-comparison",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        503: {"description": _GPU_UNAVAILABLE_TEXT},
    },
    tags=[TAG_CUSTOM_TASKS],
)
async def replicated_comparison_custom(
    key: str, req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare a custom task's variants over shared seeds (ADR-058 + ADR-061)."""
    await load_training_stack()
    from visionforge.tasks.runner import CustomTaskRunner

    info = _get_custom_task_or_404(key)
    return _start_replicated_comparison(
        req, info.spec_cls.Config, CustomTaskRunner(info), profile
    )


# A started job's status, event stream and result, and the queue (ADR-185).
router.include_router(execution_router)


@router.get(
    "/artifacts/{file_path:path}",
    responses={
        403: {"description": "Path outside outputs"},
        404: {"description": "File not found"},
    },
    tags=[TAG_RUNS],
)
async def serve_artifact(file_path: str) -> FileResponse:
    """Serve output files (plots, models) with path traversal protection."""
    candidate = Path(os.path.abspath(file_path))
    outputs_root = Path(os.path.abspath("outputs"))
    if not candidate.is_relative_to(outputs_root):
        raise HTTPException(403, tr("file.access_denied"))

    resolved = candidate.resolve()
    outputs_dir = outputs_root.resolve()

    if not resolved.is_relative_to(outputs_dir):
        raise HTTPException(403, tr("file.access_denied"))

    if not resolved.is_file():
        raise HTTPException(404, tr("file.not_found"))

    return FileResponse(resolved)


# ── private helpers ──────────────────────────────────────────────────────────


def _find_run_dir(run_id: str, profile: Profile | None = None) -> Path | None:
    """Locate a run directory by its timestamp folder name, inside one profile.

    The search root is the profile's models folder (``_MODELS_DIR`` for the
    default one). A run id is only compared with folder names and ``run.json``
    ids, never joined into a path, and for a named profile a hit whose real path
    left the root (a link) is skipped (ADR-114). The default profile keeps
    following links, exactly as before profiles existed.
    """
    profile = profile or default_profile(_MODELS_DIR)
    models_dir = profile.models_dir.resolve()
    if not models_dir.exists():
        return None
    # Only a named profile is confined: the default one is the pre-profile
    # layout, where a link under outputs/models (old runs moved to another
    # drive) is legitimate and the History already lists it.
    confine = not profile.is_default
    # run_id == timestamp folder OR experiment_name+timestamp; check both forms.
    for run_json in models_dir.rglob(RUN_FILENAME):
        d = run_json.parent
        if confine and not d.resolve().is_relative_to(models_dir):
            continue
        if d.name == run_id:
            return d
        try:
            data = json.loads(run_json.read_text(encoding="utf-8"))
            if data.get("id") == run_id:
                return d
        except Exception:  # noqa: BLE001
            continue
    return None


_PREVIEW_CACHE_DIR = Path("outputs/preview_cache")


def _render_preprocess_preview(
    req: PreprocessPreviewRequest,
) -> PreprocessPreviewResponse:
    """Save original + per-step previews and return their artifact paths."""
    from PIL import Image

    from visionforge.core.preprocessing import (
        apply_pipeline,
        available_kinds,
    )

    _remember_dataset_root(req.base_dir)
    base = Path(req.base_dir)
    split_dir = base / req.split
    yolo_split = None if split_dir.is_dir() else resolve_yolo_split(base, req.split)
    if not split_dir.is_dir() and yolo_split is None:
        return PreprocessPreviewResponse(
            original="",
            steps=[],
            final="",
            source_image="",
            available_kinds=available_kinds(),
            message=tr("scan.split_missing", split=req.split),
        )

    # Pick the first available class + first image inside it. The images/<split>
    # layout has no class folders: its images sit loose in that one directory.
    class_dirs = (
        [yolo_split[0]]
        if yolo_split is not None
        else [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
    )
    if not class_dirs:
        return PreprocessPreviewResponse(
            original="",
            steps=[],
            final="",
            source_image="",
            available_kinds=available_kinds(),
            message=tr("scan.split_no_classes"),
        )

    target_class = None
    if req.class_name is not None:
        target_class = next((d for d in class_dirs if d.name == req.class_name), None)
    if target_class is None:
        target_class = class_dirs[0]

    img_path = next(
        (
            p
            for p in sorted(target_class.iterdir())
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ),
        None,
    )
    if img_path is None:
        return PreprocessPreviewResponse(
            original="",
            steps=[],
            final="",
            source_image="",
            available_kinds=available_kinds(),
            message=tr("scan.class_no_images", path=target_class),
        )

    # Cache dir is deterministic per source image so repeated previews don't
    # accumulate hundreds of files. Each invocation overwrites prior renders.
    cache_dir = _PREVIEW_CACHE_DIR / target_class.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass

    # Unique token per invocation: the browser caches /dataset/file by URL, so
    # reusing a filename (99_final.png) shows the PREVIOUS render after the
    # pipeline changes. Fresh names every call — old ones were unlinked above.
    token = uuid.uuid4().hex[:8]

    original_img = Image.open(img_path).convert("RGB")
    original_path = cache_dir / f"00_original_{token}.png"
    original_img.save(original_path)

    final_img, intermediates = apply_pipeline(original_img, req.steps)

    step_records: list[PreprocessPreviewStep] = []
    for i, (step_cfg, img) in enumerate(zip(req.steps, intermediates, strict=False)):
        step_path = (
            cache_dir / f"{i + 1:02d}_{step_cfg.get('kind', 'step')}_{token}.png"
        )
        img.save(step_path)
        step_records.append(
            PreprocessPreviewStep(
                kind=str(step_cfg.get("kind", "")),
                artifact=str(step_path),
                params={k: v for k, v in step_cfg.items() if k != "kind"},
            )
        )

    if step_records:
        # Save a separate "final" copy so callers don't depend on intermediates.
        final_path = cache_dir / f"99_final_{token}.png"
        final_img.save(final_path)
    else:
        final_path = original_path

    return PreprocessPreviewResponse(
        original=str(original_path),
        steps=step_records,
        final=str(final_path),
        source_image=str(img_path.resolve()),
        available_kinds=available_kinds(),
    )


# ── dataset checks at submit time ───────────────────────────────────────────
# The run queue accepts any config that validates, so an empty or mismatched
# dataset used to enter the queue and die on the first batch with a raw English
# exception. These checks list directories and read one CSV row at most (the
# /dataset/stats walks count every file); they reuse the stats probes' split
# resolution and messages so the form and the submit agree on what is usable.


def _refuse(message: str) -> NoReturn:
    raise HTTPException(422, message)


def _check_detection_dataset(config: DetectionConfig) -> None:
    """Refuse a YOLO ``base_dir`` with no split or no train images."""
    data = config.data
    if data.data_yaml is not None or data.base_dir is None:
        return  # an explicit data.yaml names its own splits; Ultralytics owns them
    base = Path(data.base_dir)
    if not base.is_dir():
        _refuse(tr(SCAN_BASE_MISSING_KEY, path=base))
    splits = {s: resolve_yolo_split(base, s) for s in ("train", "val", "test")}
    if all(resolved is None for resolved in splits.values()):
        _refuse(tr("scan.no_yolo_split"))
    train = splits["train"]
    if train is None or not _has_detection_image(train[0]):
        _refuse(tr("submit.detection_empty_train"))


def _has_detection_image(directory: Path) -> bool:
    """Any image Ultralytics would read under ``directory``, subfolders included.

    Stricter than this would refuse datasets the trainer accepts: Ultralytics
    walks subfolders and reads formats (``.tif``, ``.avif``...) the stats probe
    does not count.
    """
    from visionforge.utils.third_party_privacy import apply_opt_outs

    apply_opt_outs()  # before ultralytics' first import in a `visionforge gui`
    from ultralytics.data.utils import IMG_FORMATS

    return any(
        p.is_file() and p.suffix.lower().lstrip(".") in IMG_FORMATS
        for p in directory.rglob("*")
    )


def _check_segmentation_dataset(config: SegmentationConfig) -> None:
    """Refuse a base_dir with no split or no image-mask pair in train."""
    data = config.data
    base = Path(data.base_dir)
    split_dirs = (data.train_dir, data.val_dir, data.test_dir)
    if not any((base / d).is_dir() for d in split_dirs):
        _refuse(tr("scan.no_seg_split"))
    train = base / data.train_dir
    if not train.is_dir():
        _refuse(tr("submit.segmentation_no_train_dir", path=train))
    # The dataset's own extension lists: the trainer pairs .tif images and masks.
    from visionforge.core.segmentation_data import IMAGE_EXTS, MASK_EXTS

    images = _image_stems(train / data.images_subdir, IMAGE_EXTS)
    masks = _image_stems(train / data.masks_subdir, MASK_EXTS)
    if not images & masks:
        _refuse(
            tr(
                "submit.segmentation_no_pairs",
                images=len(images),
                masks=len(masks),
            )
        )


def _check_regression_dataset(config: RegressionConfig) -> None:
    """Refuse a missing train CSV, absent columns or a manifest with no rows."""
    data = config.data
    base = Path(data.base_dir)
    expected = [data.image_column, *data.target_columns]
    train_csv = base / data.train_csv
    if not train_csv.is_file():
        _refuse(tr("submit.regression_no_train_csv", path=train_csv))
    for name in (data.train_csv, data.val_csv, data.test_csv):
        path = base / name
        if not path.is_file():
            continue  # val/test are optional here; the trainer reports what it needs
        try:
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                has_rows = next(reader, None) is not None
        except (OSError, UnicodeDecodeError, csv.Error):
            continue  # an unreadable file fails with its own message at load time
        missing = [c for c in expected if c not in header]
        if missing:
            _refuse(
                tr(
                    "submit.regression_missing_columns",
                    csv=name,
                    missing=", ".join(missing),
                    found=", ".join(header) or "-",
                )
            )
        if path == train_csv and not has_rows:
            _refuse(tr("submit.regression_empty_manifest", csv=name))


def _check_classification_classes(config: ExperimentConfig) -> None:
    """Refuse ``num_classes`` that disagrees with the train class folders.

    ``binary`` has one output for two folders (ADR on binary heads), so it is
    checked against 2 folders, not against ``num_classes``.
    """
    train = Path(config.data.base_dir) / config.data.train_dir
    if not train.is_dir():
        return
    # ImageFolder counts every sub-directory as a class, empty or hidden included.
    names = sorted(p.name for p in train.iterdir() if p.is_dir())
    if not names:
        return
    shown = ", ".join(names[:10]) + (", ..." if len(names) > 10 else "")
    if config.task == "binary":
        if len(names) != 2:
            _refuse(tr("submit.binary_folders", found=len(names), names=shown))
        return
    if len(names) != config.model.num_classes:
        _refuse(
            tr(
                "submit.num_classes_mismatch",
                num_classes=config.model.num_classes,
                found=len(names),
                names=shown,
            )
        )


def _validate_colab_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Validate a config whose dataset is not on this machine.

    ``base_dir`` is swapped for a folder that exists only for the check; the VM
    sets the real one. Any other mistake is still a 422 here, before a runtime
    is paid for.
    """
    data = raw.get("data")
    probe = {
        **raw,
        "data": {**(data if isinstance(data, dict) else {}), "base_dir": "."},
    }
    try:
        return ExperimentConfig.model_validate(probe)
    except ValidationError as exc:
        raise RequestValidationError(
            [{**err, "loc": ("body", "config", *err["loc"])} for err in exc.errors()]
        ) from None


def _import_colab_run(
    bundle: Path, models_dir: Path, accelerator: str
) -> tuple[str | None, str | None]:
    """Import the run a Colab job brought back; (run_dir, None) or (None, error).

    The downloaded zip is removed either way: the imported run is the copy kept.
    """
    origin = {
        "kind": "colab",
        "accelerator": accelerator,
        "imported_at": datetime.now().isoformat(),
    }
    try:
        run_dir = import_experiment_bundle(bundle, models_dir, origin=origin)
    except (ValueError, OSError) as exc:
        logger.warning("GUI: Colab run in {} not imported: {}", bundle, exc)
        return None, tr("colab.import_failed", detail=f"{type(exc).__name__}: {exc}")
    finally:
        bundle.unlink(missing_ok=True)
    logger.info("GUI: Colab run imported into {}", run_dir)
    return str(run_dir), None


async def _execute_colab(
    run_id: str,
    slug: str,
    accelerator: str,
    payload: dict[str, Any],
    models_dir: Path,
) -> None:
    """Run one job on Colab off the event loop, import its run and record how it ended."""
    state = _job_state(run_id)

    loop = asyncio.get_running_loop()
    queue = state.log
    token = state.token

    def _put_event(event: dict[str, Any]) -> None:
        # The job thread waits until the event is in the log, so by the time
        # run_job returns every event is there and the `None` below cannot
        # overtake one. Callbacks merely scheduled from the thread did lose the
        # race on the Linux py3.13 CI: the stream closed with the event missing.
        if queue is None:
            return
        delivered = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        try:
            delivered.result(timeout=_COLAB_EVENT_TIMEOUT_S)
        except TimeoutError:
            # A stalled loop must not stall the VM; the job keeps going.
            delivered.cancel()
            logger.warning(
                "GUI: Colab job {} dropped an event after {}s: {}",
                run_id,
                _COLAB_EVENT_TIMEOUT_S,
                event.get("event"),
            )

    try:
        logger.info("GUI: Starting Colab job {} ({})", run_id, accelerator)
        consent = _DRIVE_CONSENTS.setdefault(run_id, DriveConsent())
        result = await asyncio.to_thread(
            _COLAB_JOBS.run_job,
            slug,
            accelerator,
            payload,
            _put_event,
            lambda: is_cancelled(token),
            consent=consent,
        )
        run_dir, error = None, result.error
        if result.bundle is not None:
            run_dir, error = await asyncio.to_thread(
                _import_colab_run, result.bundle, models_dir, accelerator
            )
        state.run = {
            "run_id": run_id,
            "status": "completed"
            if result.outcome == "completed" and error is None
            else "failed",
            "error": error,
            "report": result.report,
            "run_dir": run_dir,
        }
    except Exception as e:
        logger.exception("GUI: Colab job {} failed", run_id)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "report": None,
            "run_dir": None,
        }
    finally:
        _DRIVE_CONSENTS.pop(run_id, None)
        # The thread has returned, and each event it sent is already in the log.
        if queue is not None:
            await queue.put(None)


async def _execute_detection(
    config: DetectionConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run an Ultralytics detection training in a background thread."""
    from visionforge.blocks.detection import DetectionBlock

    state = _job_state(run_id)

    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = DetectionBlock()
        block.setup(config)
        block._progress_callback = _put_event
        block._cancel_token = state.token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting detection {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("detection", {}).get("run_dir")
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Detection {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Detection {} failed", run_id)
        cls = type(e).__name__
        msg = str(e) or tr(RUN_NO_MESSAGE_KEY)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{cls}: {msg}",
            "report": None,
            "run_dir": None,
        }

    finally:
        if queue is not None:
            await queue.put(None)


async def _execute_regression(
    config: RegressionConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run an image-regression training in a background thread."""
    from visionforge.blocks.regression import RegressionBlock

    state = _job_state(run_id)

    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = RegressionBlock()
        block.setup(config)
        block._progress_callback = _put_event
        block._cancel_token = state.token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting regression {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("train", {}).get("run_dir")
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Regression {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Regression {} failed", run_id)
        cls = type(e).__name__
        msg = str(e) or tr(RUN_NO_MESSAGE_KEY)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{cls}: {msg}",
            "report": None,
            "run_dir": None,
        }

    finally:
        if queue is not None:
            await queue.put(None)


async def _execute_segmentation(
    config: SegmentationConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run a semantic-segmentation training in a background thread."""
    from visionforge.blocks.segmentation import SegmentationBlock

    state = _job_state(run_id)

    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = SegmentationBlock()
        block.setup(config)
        block._progress_callback = _put_event
        block._cancel_token = state.token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting segmentation {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("train", {}).get("run_dir")
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Segmentation {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Segmentation {} failed", run_id)
        cls = type(e).__name__
        msg = str(e) or tr(RUN_NO_MESSAGE_KEY)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{cls}: {msg}",
            "report": None,
            "run_dir": None,
        }

    finally:
        if queue is not None:
            await queue.put(None)


async def _execute_anomaly(
    config: AnomalyConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run an anomaly-detection training in a background thread."""
    from visionforge.blocks.anomaly import AnomalyBlock

    state = _job_state(run_id)

    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = AnomalyBlock()
        block.setup(config)
        block._progress_callback = _put_event
        block._cancel_token = state.token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting anomaly {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("train", {}).get("run_dir")
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Anomaly {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Anomaly {} failed", run_id)
        cls = type(e).__name__
        msg = str(e) or tr(RUN_NO_MESSAGE_KEY)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{cls}: {msg}",
            "report": None,
            "run_dir": None,
        }

    finally:
        if queue is not None:
            await queue.put(None)


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
    try:
        config_type.model_validate(req.config)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    metric = req.metric or runner.primary_metric()
    name = str(req.config.get("name", "comparison"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{name}_{timestamp}"

    return _submit_job(
        run_id,
        name,
        _task_label_from_config(config_type),
        "comparison",
        lambda: _execute_comparison(
            runner, req.config, req.model_names, metric, run_id
        ),
        profile=profile,
    )


async def _execute_comparison(
    runner: TaskRunner,
    config_dict: dict[str, Any],
    model_names: list[str],
    metric: str,
    run_id: str,
) -> None:
    """Run a model comparison in a background thread and store the ranked report."""
    state = _job_state(run_id)
    queue = state.log
    token = state.token
    # The runner carries the stop into the architecture in flight, and the
    # comparison reads it between architectures (ADR-111).
    give_cancel_token(runner, token)
    loop = asyncio.get_running_loop()

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        # Each architecture streams as one unit, so the sheet shows which model
        # is training and its epochs instead of "preparing" until the end.
        trials = await asyncio.to_thread(
            run_model_comparison,
            runner,
            config_dict,
            model_names,
            metric,
            progress_callback=_put_event,
        )
        stopped = job_was_stopped([t.status for t in trials], len(model_names), token)
        if not any(t.status == "success" for t in trials) and not stopped:
            raise RuntimeError("All architectures failed — no ranking available.")
        report = _comparison_report(
            trials, metric, runner_metric_direction(runner, metric)
        )
        report["stopped"] = stopped
        report["report_dir"] = _write_advanced_summary(
            config_dict, "comparison", report
        )
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
        top = next((r for r in rows if r["status"] == "success"), None)
        _record_sweep_group(
            run_id,
            runner,
            config_dict,
            report,
            mode="models",
            planned=len(model_names),
            trials=rows,
            best_index=top["trial_index"] if top else None,
            report_name="comparison",
        )
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": None,
        }
        logger.success("GUI: Comparison {} completed ({} archs).", run_id, len(trials))
    except Exception as e:
        logger.exception("GUI: Comparison {} failed", run_id)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e or tr('run.no_message')}",
            "report": None,
            "run_dir": None,
        }
    finally:
        if queue is not None:
            await queue.put(None)


def _comparison_report(
    trials: list[ComparisonTrial], metric: str, direction: Literal["higher", "lower"]
) -> dict[str, Any]:
    """Shape ranked comparison trials into a GUI-friendly report dict."""
    rows = [asdict(t) for t in trials]
    successful = [t for t in rows if t["status"] == "success"]
    return {
        "metric": metric,
        # Which way the trials were ranked; "top" means lowest for "lower".
        "metric_direction": direction,
        "trials": rows,
        "top_3": successful[:3],
        "total_ran": len(rows),
        "failed_count": sum(1 for t in rows if t["status"] == "failed"),
        "stopped_count": sum(1 for t in rows if t["status"] == STOPPED),
    }


def _refuse_output_paths(paths: Iterable[str], where: str) -> None:
    """422 when a dot-path would move a trial's output folders (ADR-114).

    Raises:
        HTTPException: 422 naming the first offending path.
    """
    try:
        refuse_output_paths(paths, where)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


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
        complete = profile.scope_dict(
            config_type.model_validate(req.config).model_dump(mode="json")
        )
        validate_sweep_space(complete, list(req.search_space))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    metric = req.metric or runner.primary_metric()
    name = str(req.config.get("name", "sweep"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{name}_{timestamp}"

    return _submit_job(
        run_id,
        name,
        _task_label_from_config(config_type),
        f"sweep:{req.mode}" if getattr(req, "mode", None) else "sweep",
        lambda: _execute_sweep(runner, complete, req, metric, run_id),
        profile=profile,
    )


async def _execute_sweep(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    req: SweepRequest,
    metric: str,
    run_id: str,
) -> None:
    """Run a hyperparameter sweep in a background thread and store the ranked report."""
    state = _job_state(run_id)
    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    token = state.token
    # The runner carries the stop into the trial in flight, and the sweep reads
    # it between trials (ADR-111).
    give_cancel_token(runner, token)

    try:
        trials = await asyncio.to_thread(
            run_sweep,
            runner,
            base_config_dict,
            req.search_space,
            mode=req.mode,
            metric=metric,
            n_trials=req.n_trials,
            seed=req.seed,
            progress_callback=_put_event,
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
            models_dir=_MODELS_DIR,
        )
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": None,
        }
        logger.success("GUI: Sweep {} completed ({} trials).", run_id, len(trials))
    except Exception as e:
        logger.exception("GUI: Sweep {} failed", run_id)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e or tr('run.no_message')}",
            "report": None,
            "run_dir": None,
        }
    finally:
        if queue is not None:
            await queue.put(None)


# Moved to core.run_groups so `visionforge run` shares them (ADR-184); the
# old names stay for the model comparison and the tests that use them.
_sweep_report = sweep_report
_write_advanced_summary = write_group_summary


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
    try:
        config = config_type.model_validate(req.config)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_cv_{timestamp}"

    return _submit_job(
        run_id,
        str(config.name),
        _task_label_from_config(config_type),
        "cv",
        lambda: _execute_task_cv(config, req, run_id, cv_fn, task_label),
        profile=profile,
    )


async def _execute_task_cv(
    config: Any,
    req: TaskCvRequest,
    run_id: str,
    cv_fn: Callable[..., Any],
    task_label: str,
) -> None:
    """Run a task's K-fold in a worker thread and store the fold report."""
    state = _job_state(run_id)
    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    token = state.token

    try:
        cv = await asyncio.to_thread(
            cv_fn,
            config,
            n_folds=req.n_folds,
            shuffle=req.shuffle,
            seed=req.fold_seed,
            progress_callback=_put_event,
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
            models_dir=_MODELS_DIR,
        )
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": None,
        }
        logger.success(
            "GUI: {} CV {} completed ({} folds).", task_label, run_id, cv.n_folds
        )
    except Exception as e:
        logger.exception("GUI: {} CV {} failed", task_label, run_id)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e or tr('run.no_message')}",
            "report": None,
            "run_dir": None,
        }
    finally:
        if queue is not None:
            await queue.put(None)


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
    try:
        complete = profile.scope_dict(
            config_type.model_validate(req.config).model_dump(mode="json")
        )
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    try:
        seeds = resolve_seeds(complete, req.seeds, req.n_replicates)
    except ValueError as exc:
        # 422 is in the `responses` of every route that calls this helper.
        raise HTTPException(422, tr("compare.seeds_unique")) from exc  # NOSONAR

    # Fail on a bad override path before spending len(variants)*len(seeds)
    # trainings on it.
    try:
        validate_variants(complete, req.variants)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    metric = req.metric or runner.primary_metric()
    name = str(req.config.get("name", "comparison"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{name}_{timestamp}"

    return _submit_job(
        run_id,
        name,
        _task_label_from_config(config_type),
        "replicated-comparison",
        lambda: _execute_replicated_comparison(
            runner, complete, req, seeds, metric, run_id
        ),
        profile=profile,
    )


async def _execute_replicated_comparison(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    req: ReplicatedComparisonRequest,
    seeds: list[int],
    metric: str,
    run_id: str,
) -> None:
    """Run every variant over the shared seeds and store the paired report."""
    state = _job_state(run_id)
    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    token = state.token
    # The runner carries the stop into the replicate in flight, and the
    # comparison reads it between replicates and variants (ADR-111).
    give_cancel_token(runner, token)

    try:
        report = await asyncio.to_thread(
            run_replicated_comparison,
            runner,
            base_config_dict,
            req.variants,
            seeds,
            metric,
            alpha=req.alpha,
            progress_callback=_put_event,
        )
        report = finish_replicated_comparison(
            report,
            n_planned=len(req.variants) * len(seeds),
            config=base_config_dict,
            run_id=run_id,
            token=token,
            models_dir=_MODELS_DIR,
        )
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": None,
        }
        logger.success(
            "GUI: Replicated comparison {} completed ({} variants x {} seeds).",
            run_id,
            len(req.variants),
            len(seeds),
        )
    except Exception as e:
        logger.exception("GUI: Replicated comparison {} failed", run_id)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e or tr('run.no_message')}",
            "report": None,
            "run_dir": None,
        }
    finally:
        if queue is not None:
            await queue.put(None)


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
    try:
        complete = profile.scope_dict(
            config_type.model_validate(req.config).model_dump(mode="json")
        )
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    try:
        seeds = resolve_seeds(complete, req.seeds, req.n_replicates)
    except ValueError as exc:
        # 422 is in the `responses` of every route that calls this helper.
        raise HTTPException(422, tr("replicates.seeds_unique")) from exc  # NOSONAR

    metric = req.metric or runner.primary_metric()
    name = str(req.config.get("name", "replicates"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{name}_{timestamp}"

    return _submit_job(
        run_id,
        name,
        _task_label_from_config(config_type),
        "replicates",
        lambda: _execute_replicates(runner, complete, seeds, metric, run_id),
        profile=profile,
    )


async def _execute_replicates(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    seeds: list[int],
    metric: str,
    run_id: str,
) -> None:
    """Run the replicate set in a background thread and store the aggregate report."""
    state = _job_state(run_id)
    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    token = state.token
    # The runner carries the stop into the replicate in flight, and the set
    # reads it between replicates (ADR-111).
    give_cancel_token(runner, token)

    try:
        trials = await asyncio.to_thread(
            run_replicates,
            runner,
            base_config_dict,
            seeds,
            metric,
            progress_callback=_put_event,
        )
        report = finish_replicates(
            trials,
            seeds=seeds,
            metric=metric,
            config=base_config_dict,
            run_id=run_id,
            token=token,
            models_dir=_MODELS_DIR,
        )
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": None,
        }
        logger.success("GUI: Replicates {} completed ({} seeds).", run_id, len(trials))
    except Exception as e:
        logger.exception("GUI: Replicates {} failed", run_id)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e or tr('run.no_message')}",
            "report": None,
            "run_dir": None,
        }
    finally:
        if queue is not None:
            await queue.put(None)


def _record_sweep_group(
    run_id: str,
    runner: TaskRunner,
    config_dict: dict[str, Any],
    report: dict[str, Any],
    *,
    mode: str,
    planned: int,
    trials: list[dict[str, Any]],
    best_index: int | None,
    report_name: str,
) -> None:
    """Give a finished model comparison its History entry (ADR-124)."""
    record_sweep_group(
        run_id,
        runner,
        config_dict,
        report,
        mode=mode,
        planned=planned,
        trials=trials,
        best_index=best_index,
        report_name=report_name,
        models_dir=_MODELS_DIR,
    )


def _record_run_group(
    kind: Literal["replicates", "replicated_comparison", "kfold", "sweep"],
    run_id: str,
    config: dict[str, Any],
    report: dict[str, Any],
    report_name: str,
) -> None:
    """Give a finished group job its History entry under the GUI's models dir."""
    record_run_group(kind, run_id, config, report, report_name, models_dir=_MODELS_DIR)


__all__ = ["router", "_load_runs"]
