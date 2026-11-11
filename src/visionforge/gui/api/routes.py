"""API endpoints for VisionForge GUI."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
from pydantic import ValidationError

from visionforge.core.cancellation import (
    STOPPED,
    job_was_stopped,
)
from visionforge.core.comparison import ComparisonTrial, run_model_comparison
from visionforge.core.detection_data import resolve_yolo_split
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
from visionforge.gui.api import execution, submission, training_routes
from visionforge.gui.api.coco_routes import router as coco_router
from visionforge.gui.api.code_tour_routes import router as code_tour_router
from visionforge.gui.api.colab_routes import build_colab_router
from visionforge.gui.api.colab_run_routes import (
    _COLAB_CLI,
    _COLAB_JOBS,
    _COLAB_LOGINS,
    build_colab_run_router,
)
from visionforge.gui.api.custom_task_routes import (
    _get_custom_task_or_404,
    build_custom_task_run_router,
)
from visionforge.gui.api.custom_task_routes import router as custom_tasks_router
from visionforge.gui.api.datasets_routes import (
    _IMAGE_EXTS,
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
    _job_state,
    _run_state,
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
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    CONFLICT_TEXT,
    FORBIDDEN_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
    TAG_ANOMALY,
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
from visionforge.gui.api.profile_tokens import read_token
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
    ComparisonRequest,
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
    WandbStatus,
)
from visionforge.gui.api.submission import (
    _DETECTION_STACK,
    _GPU_UNAVAILABLE_TEXT,
    _task_label_from_config,
    load_training_stack,
)
from visionforge.gui.api.system_routes import build_system_router
from visionforge.gui.api.training_routes import (
    build_experiment_run_router,
    build_task_run_router,
)
from visionforge.gui.api.wandb_routes import build_wandb_router
from visionforge.gui.api.zenodo_routes import build_zenodo_router
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.cuda import check_cuda
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.messages import (
    LANG_HEADER,
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

# The blocks, runners and trainers import torch, torchvision, ultralytics and
# matplotlib. Each endpoint imports the one it runs, so starting the GUI loads
# none of them (GUI-11, #202). The two names below stay module attributes so a
# test can still replace them here.


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
    """The run.json of a finished classification run, or 409/500.

    ``status`` is the exporter's sidecar: a retry needs one that is not still
    running.
    """
    if status is None or status.status == "RUNNING":
        raise HTTPException(409, tr(unavailable))
    try:
        data = json.loads((run_dir / RUN_FILENAME).read_bytes())
    except (OSError, ValueError) as exc:
        raise HTTPException(500, tr(RUN_BAD_JSON_KEY, error=exc)) from exc
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("config"), dict)
        or _run_task(data) != "classification"
    ):
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
    """Reconcile a completed local classification run with its profile's MLflow store."""
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

    # Built per request rather than at module level: the executors are looked up
    # in their modules when the run resumes, so a test that replaces one reaches it.
    task = _run_task(data)
    executors: dict[str, tuple[Any, Any]] = {
        "classification": (ExperimentConfig, execution._execute_experiment),
        "detection": (DetectionConfig, training_routes._execute_detection),
        "regression": (RegressionConfig, training_routes._execute_regression),
        "segmentation": (SegmentationConfig, training_routes._execute_segmentation),
        "anomaly": (AnomalyConfig, training_routes._execute_anomaly),
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
    return submission._submit_job(
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
    from visionforge.core.mlflow_tracking import MlflowTracker
    from visionforge.core.wandb_tracking import WandbTarget, WandbTracker

    outputs = _PROFILES_DIR.parent.resolve()
    factories: list[TrackerFactory] = []
    if profile.settings.mlflow.enabled:
        factories.append(partial(MlflowTracker, outputs / "mlflow", profile.slug))
    wandb = profile.settings.wandb
    if wandb.enabled:
        target = WandbTarget(wandb.project, wandb.entity, read_token("wandb", profile))
        factories.append(
            partial(WandbTracker, outputs / "wandb", profile.slug, target=target)
        )
    return tuple(factories)


# Starting one run: classification here and on Colab, then each standalone
# task's schema and run endpoint (#204, ADR-196). Included where they used to be
# defined, so the OpenAPI document and the route order do not change.
router.include_router(
    build_experiment_run_router(
        lambda header: current_profile(header),
        lambda config, profile: _tracker_factories(config, profile),
    )
)
router.include_router(build_colab_run_router(lambda header: current_profile(header)))
router.include_router(build_task_run_router(lambda header: current_profile(header)))


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


# The task list and the researcher's own tasks (#204, ADR-196).
router.include_router(custom_tasks_router)
router.include_router(
    build_custom_task_run_router(lambda header: current_profile(header))
)


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

    return submission._submit_job(
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

    return submission._submit_job(
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

    return submission._submit_job(
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

    return submission._submit_job(
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

    return submission._submit_job(
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
