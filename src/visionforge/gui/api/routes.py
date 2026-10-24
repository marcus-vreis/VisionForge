"""API endpoints for VisionForge GUI."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from loguru import logger
from pydantic import ValidationError

from visionforge.blocks.anomaly import AnomalyBlock
from visionforge.blocks.anomaly_runner import AnomalyRunner
from visionforge.blocks.batch_prediction import BatchPredictionBlock
from visionforge.blocks.classification import ClassificationBlock
from visionforge.blocks.classification_runner import ClassificationRunner
from visionforge.blocks.cross_validation import CrossValidationBlock
from visionforge.blocks.detection import DetectionBlock
from visionforge.blocks.detection_runner import DetectionRunner
from visionforge.blocks.export_onnx import ExportONNXBlock
from visionforge.blocks.grid_search import GridSearchBlock
from visionforge.blocks.model_comparison import ModelComparisonBlock
from visionforge.blocks.random_search import RandomSearchBlock
from visionforge.blocks.regression import RegressionBlock
from visionforge.blocks.regression_cv import run_regression_cross_validation
from visionforge.blocks.regression_runner import RegressionRunner
from visionforge.blocks.segmentation import SegmentationBlock
from visionforge.blocks.segmentation_cv import run_segmentation_cross_validation
from visionforge.blocks.segmentation_runner import SegmentationRunner
from visionforge.blocks.transfer_learning import TransferLearningBlock
from visionforge.core.cancellation import (
    STOPPED,
    CancellationToken,
    job_was_stopped,
)
from visionforge.core.comparison import ComparisonTrial, run_model_comparison
from visionforge.core.data import DataModule
from visionforge.core.detection_data import resolve_yolo_split
from visionforge.core.evaluator import Evaluator
from visionforge.core.latex_export import report_to_latex
from visionforge.core.loader_lifecycle import describe_worker_spawn_failure
from visionforge.core.plotter import MetricsPlotter
from visionforge.core.replicated_comparison import (
    run_replicated_comparison,
    validate_variants,
)
from visionforge.core.replicates import (
    ReplicateTrial,
    aggregate_replicates,
    run_replicates,
)
from visionforge.core.run_groups import write_run_group
from visionforge.core.sweep import (
    SweepTrial,
    planned_trials,
    run_sweep,
    validate_sweep_space,
)
from visionforge.core.task_runner import (
    SELECTION_SPLIT,
    TaskRunner,
    give_cancel_token,
    require_validation_selection,
    runner_metric_direction,
)
from visionforge.gui.api.datasets_routes import (
    _IMAGE_EXTS,
    SCAN_BASE_MISSING_KEY,
    _image_stems,
    _remember_dataset_root,
)
from visionforge.gui.api.datasets_routes import router as datasets_router
from visionforge.gui.api.detection_export import export_detection_run
from visionforge.gui.api.detection_testing import evaluate_detection_run
from visionforge.gui.api.evidence_routes import EvidenceExports, build_evidence_router
from visionforge.gui.api.folder_opener import (
    FolderOpenError,
    is_loopback_host,
    is_loopback_origin,
    open_folder,
)
from visionforge.gui.api.gpu_health import cuda_context_broken
from visionforge.gui.api.http_responses import BAD_REQUEST as _BAD_REQUEST
from visionforge.gui.api.http_responses import CONFLICT as _CONFLICT
from visionforge.gui.api.http_responses import NOT_FOUND as _NOT_FOUND
from visionforge.gui.api.http_responses import SERVER_ERROR as _SERVER_ERROR
from visionforge.gui.api.profiles import (
    PROFILE_HEADER,
    InvalidProfileError,
    Profile,
    UnknownProfileError,
    default_profile,
    resolve_profile,
)
from visionforge.gui.api.profiles_routes import build_profiles_router
from visionforge.gui.api.run_history import (
    CUSTOM_TASK_PREFIX,
    RUN_BAD_JSON_KEY,
    RUN_FILENAME,
    RUN_NOT_FOUND_KEY,
)
from visionforge.gui.api.run_history import load_runs as _load_runs
from visionforge.gui.api.run_history import resume_status as _resume_status
from visionforge.gui.api.run_history import run_task as _run_task
from visionforge.gui.api.run_queue import (
    NotStoppableError,
    QueuedJob,
    RunQueue,
    StopPoint,
)
from visionforge.gui.api.runs_routes import build_runs_router
from visionforge.gui.api.sample_dataset import (
    SampleDatasetExistsError,
    SampleTaskError,
    create_sample_dataset,
)
from visionforge.gui.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ComparisonRequest,
    CustomTaskActionResponse,
    ExportOnnxRequest,
    ExportOnnxResponse,
    GradCamItem,
    GradCamRequest,
    GradCamResponse,
    PreprocessPreviewRequest,
    PreprocessPreviewResponse,
    PreprocessPreviewStep,
    QueuedJobInfo,
    QueueSnapshot,
    ReplicatedComparisonRequest,
    ReplicatesRequest,
    RunResponse,
    RunResult,
    RunStatus,
    RunTestRequest,
    RunTestResponse,
    SampleDatasetRequest,
    SampleDatasetResponse,
    SweepRequest,
    TaskCvRequest,
    TaskDescriptor,
    TaskListResponse,
)
from visionforge.gui.api.system_routes import build_system_router
from visionforge.gui.api.torch_batch_predict import (
    batch_predict_anomaly_run,
    batch_predict_regression_run,
)
from visionforge.gui.api.torch_onnx_export import (
    export_regression_run,
    export_segmentation_run,
)
from visionforge.models.factory import ModelFactory
from visionforge.tasks.base import TaskSpec
from visionforge.tasks.engine import GenericTaskEngine
from visionforge.tasks.registry import get_task, registered_tasks
from visionforge.tasks.runner import CustomTaskRunner
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
    tr_artifact,
    using_lang,
)
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig
from visionforge.utils.workers import suggested_workers as worker_budget

RUN_NO_MESSAGE_KEY = "run.no_message"
MARKDOWN_TABLE_SEPARATOR = "|---|---|"

_FORBIDDEN: dict[int | str, dict[str, Any]] = {403: {"description": "Access denied"}}
_UNPROCESSABLE: dict[int | str, dict[str, Any]] = {
    422: {
        "description": "Invalid request",
        # Keeps FastAPI's validation-error schema that a description-only entry would drop.
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/HTTPValidationError"}
            }
        },
    }
}
_UNAVAILABLE: dict[int | str, dict[str, Any]] = {
    503: {"description": "GPU context unavailable"}
}


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

# State of the run that is executing right now (the machine still trains one at
# a time); submissions that arrive while it is busy wait in _RUN_QUEUE.
_current_run: dict[str, Any] | None = None

# Per-run SSE queue; None when no run is active.
# The Trainer worker thread puts dicts onto this queue via asyncio.run_coroutine_threadsafe;
# the SSE generator pulls from it and serialises each item as an SSE data line.
# A None sentinel signals end-of-stream so the generator can close cleanly.
_event_queue: asyncio.Queue[dict[str, Any] | None] | None = None

# Cancellation token of the job executing right now; None when idle. Handed to
# the block so DELETE /api/queue/{id} reaches the trainer's epoch boundary
# (ADR-088) instead of only flipping a flag nobody reads.
_active_cancel_token: CancellationToken | None = None

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


def _begin_job(job: QueuedJob) -> None:
    """Prepare per-run state the moment a queued job actually starts (ADR-075).

    The fresh SSE queue is created here rather than at submission time: doing it
    in the request handler was safe only while a second submit was refused, and
    with queueing it would replace the live queue of the job still running.

    The job's cancellation token is stashed for the same reason: the executor
    coroutine is built at submission time and cannot close over a job that does
    not exist yet, so it reads the token here — the queue calls this immediately
    before it awaits ``job.start()``.
    """
    global _current_run, _event_queue, _active_cancel_token

    _event_queue = asyncio.Queue()
    _active_cancel_token = job.cancel_token
    _current_run = {
        "run_id": job.run_id,
        "status": "running",
        "error": None,
        "report": None,
        "run_dir": None,
    }


def _finish_job(job: QueuedJob) -> dict[str, Any] | None:
    """Hand the finished run's terminal state to the queue to keep by run_id."""
    if _current_run is not None and _current_run.get("run_id") == job.run_id:
        return dict(_current_run)
    return None


_RUN_QUEUE = RunQueue(on_start=_begin_job, on_finish=_finish_job)
_EVIDENCE_EXPORTS = EvidenceExports()


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


# Where each kind of job stops when asked (ADR-111), by the strategy it is
# queued under. A strategy missing here is reported as not stoppable: claiming a
# stop that the executor never checks for is the ADR-094 failure mode.
_STOP_POINTS: dict[str, StopPoint] = {
    "simple": "epoch",
    # Classification queues under `config.block`; these four all train through
    # ClassificationBlock or TransferLearningBlock in _execute_experiment.
    "classification": "epoch",
    "transfer_learning": "epoch",
    "batch_prediction": "epoch",
    "export_onnx": "epoch",
    "grid_search": "trial",
    "random_search": "trial",
    "sweep": "trial",
    "cross_validation": "fold",
    "cv": "fold",
    "model_comparison": "model",
    "comparison": "model",
    "replicates": "replicate",
    "replicated-comparison": "replicate",
}


def _stop_point(strategy: str) -> StopPoint | None:
    """Where a job queued under ``strategy`` stops; ``sweep:grid`` reads as ``sweep``."""
    return _STOP_POINTS.get(strategy.split(":", 1)[0])


def _submit_job(
    run_id: str,
    label: str,
    task: str,
    strategy: str,
    start: Callable[[], Awaitable[None]],
    stop_at: StopPoint | None | Literal["auto"] = "auto",
    profile: Profile | None = None,
    resume_dir: Path | None = None,
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
        # The queue starts the job from its own task, long after this request's
        # language is gone; the messages the trainer writes (health warnings, the
        # paging-file hint) are for the person who pressed the button.
        with using_lang(lang):
            await start()

    status = _RUN_QUEUE.submit(
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
        )
    )
    return RunResponse(run_id=run_id, status=status)  # type: ignore[arg-type]


# Strong references to background tasks to prevent premature GC (asyncio doc).
_background_tasks: set[asyncio.Task[None]] = set()


@router.get("/schema")
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
router.include_router(build_profiles_router(lambda: (_PROFILES_DIR, _MODELS_DIR)))
# Dataset, picker and credential endpoints live in datasets_routes.py (ADR-132).
router.include_router(datasets_router)
# History list, detail and delete live in runs_routes.py (ADR-132); the lookups
# and the running job's state stay owned here and are read per request.
router.include_router(
    build_runs_router(
        lambda run_id, profile: _find_run_dir(run_id, profile),
        lambda header: current_profile(header),
        lambda: _current_run,
        lambda request: _client_host(request),
    )
)
router.include_router(
    build_evidence_router(
        lambda run_id, profile: _find_run_dir(run_id, profile),
        lambda header: current_profile(header),
        lambda run_id, run_dir, profile: _RUN_QUEUE.is_writing(
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


@router.get("/update")
async def get_update(request: Request) -> dict[str, Any]:
    """Whether a newer release exists, and whether this page may install it (ADR-120).

    The PyPI lookup blocks for up to a few seconds when the network is slow, so it
    runs off the event loop; its answer is cached for hours.
    """
    from visionforge.gui.api.updates import update_status

    return await asyncio.to_thread(
        update_status,
        client_is_local=is_loopback_host(_client_host(request)),
        busy=_RUN_QUEUE.is_busy() or _RUN_QUEUE.pending_count() > 0,
    )


@router.post("/update/apply", responses={**_FORBIDDEN, **_CONFLICT, **_SERVER_ERROR})
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
    busy = _RUN_QUEUE.is_busy() or _RUN_QUEUE.pending_count() > 0
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
    responses={**_BAD_REQUEST, **_FORBIDDEN, **_NOT_FOUND, **_SERVER_ERROR},
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


@router.post(
    "/runs/{run_id}/resume",
    responses={
        **_BAD_REQUEST,
        **_UNAVAILABLE,
        404: {"description": "Run not found"},
        409: {"description": "Run is busy or cannot continue"},
        500: {"description": "Run metadata is unreadable"},
    },
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
        "classification": (ExperimentConfig, _execute_experiment),
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
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_SERVER_ERROR},
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


def _reject_group_action(run_dir: Path) -> None:
    """Refuse a checkpoint-based action on a replicate group (ADR-113).

    A group trained nothing itself: its ``run.json`` has no checkpoint, so
    continuing, testing, batch prediction, ONNX export and Grad-CAM have
    nothing to act on. They used to fail by accident (a 409 on resume, a "no
    checkpoint" error from deep inside the others); the researcher now gets the
    reason and the way out. Deleting the group (removes only its summary) and
    revealing or exporting it stay allowed, so they do not call this.

    Raises:
        HTTPException: 400 pointing at the seeds, which are ordinary runs.
    """
    try:
        data = json.loads((run_dir / RUN_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict) and isinstance(data.get("group"), dict):
        raise HTTPException(400, tr("run.group_action"))


def _reject_custom_task_action(run_dir: Path, action: str) -> None:
    """Refuse a per-run action on a researcher-defined task's run.

    These four actions rebuild the model from ``config.model`` and assume one of
    the five built-in shapes. A custom run.json has no ``config.task`` and no
    ``config.model`` — the task owns those — so the dispatchers fell through to
    the classification default and died inside ``load_state_dict`` with a raw
    ResNet key mismatch. The task's own key lives at the top level of run.json
    (ADR-013), which is what makes an honest refusal possible here.

    Raises:
        HTTPException: 400 naming the task, so the GUI can hide the action.
    """
    run_json = run_dir / RUN_FILENAME
    if not run_json.exists():
        return
    try:
        task = json.loads(run_json.read_text(encoding="utf-8")).get("task", "")
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(task, str) and task.startswith(CUSTOM_TASK_PREFIX):
        raise HTTPException(
            400,
            tr(
                "run.custom_action", action=action, task=task[len(CUSTOM_TASK_PREFIX) :]
            ),
        )


@router.post(
    "/runs/{run_id}/test", responses={**_BAD_REQUEST, **_NOT_FOUND, **_SERVER_ERROR}
)
async def test_run_on_dataset(
    run_id: str, req: RunTestRequest, profile: ProfileDep
) -> RunTestResponse:
    """Run the saved checkpoint against a new dataset and record the result.

    The test result is appended to run.json under the 'tests' array so each
    saved model accumulates its own per-dataset history.
    """
    run_dir = _find_run_dir(run_id, profile)
    if run_dir is None:
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))

    _reject_group_action(run_dir)
    _reject_custom_task_action(run_dir, "test")

    # Run the actual evaluation in a worker thread (PyTorch + disk I/O).
    try:
        result = await asyncio.to_thread(_execute_run_test, run_dir, req)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Test on run {} failed", run_id)
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return result


@router.post(
    "/runs/{run_id}/gradcam",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_SERVER_ERROR},
)
async def gradcam_run(
    run_id: str, req: GradCamRequest, profile: ProfileDep
) -> GradCamResponse:
    """Generate Grad-CAM overlays for sample images using this run's checkpoint.

    Classification-only (detection/regression/segmentation/anomaly have no
    softmax-classifier CAM path). Overlays are written to ``<run_dir>/gradcam/``.
    """
    run_dir = _find_run_dir(run_id, profile)
    if run_dir is None:
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))
    _reject_group_action(run_dir)
    _reject_custom_task_action(run_dir, "gradcam")
    try:
        return await asyncio.to_thread(_execute_run_gradcam, run_dir, req)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Grad-CAM on run {} failed", run_id)
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@router.post(
    "/runs/{run_id}/batch_predict",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_SERVER_ERROR},
)
async def batch_predict_run(
    run_id: str, req: BatchPredictRequest, profile: ProfileDep
) -> BatchPredictResponse:
    """Run inference of this run's checkpoint over a folder of images.

    Outputs a CSV with one row per image. ``output_csv`` defaults to a
    timestamped file in ``<run_dir>/predictions/`` so consecutive batch runs
    don't overwrite each other.
    """
    run_dir = _find_run_dir(run_id, profile)
    if run_dir is None:
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))

    _reject_group_action(run_dir)
    _reject_custom_task_action(run_dir, "batch_predict")
    try:
        result = await asyncio.to_thread(_execute_batch_predict, run_dir, req)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch prediction on run {} failed", run_id)
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return result


@router.post(
    "/runs/{run_id}/export_onnx",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_SERVER_ERROR},
)
async def export_run_to_onnx(
    run_id: str, req: ExportOnnxRequest, profile: ProfileDep
) -> ExportOnnxResponse:
    """Export this run's saved checkpoint to ONNX, optionally validating it.

    Reuses the run's stored ``config.data.transforms.image_size`` and
    ``config.model`` to build the dummy input and the architecture for export.
    """
    run_dir = _find_run_dir(run_id, profile)
    if run_dir is None:
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))

    _reject_group_action(run_dir)
    _reject_custom_task_action(run_dir, "export_onnx")
    try:
        result = await asyncio.to_thread(_execute_onnx_export, run_dir, req)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("ONNX export on run {} failed", run_id)
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return result


@router.post(
    "/dataset/preview_preprocess", responses={**_BAD_REQUEST, **_UNPROCESSABLE}
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
    responses={**_BAD_REQUEST, **_CONFLICT, **_SERVER_ERROR},
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


@router.post(
    "/experiment/run",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def run_experiment(config: ExperimentConfig, profile: ProfileDep) -> RunResponse:
    """Start a training experiment, or queue it if one is already running."""
    config = profile.scope_model(config)
    await asyncio.to_thread(_check_classification_classes, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "classification",
        config.block,
        lambda: _execute_experiment(config, run_id),
        profile=profile,
    )


@router.get("/detection/schema")
async def get_detection_schema() -> dict[str, Any]:
    """Return the JSON Schema for DetectionConfig (drives the detection form)."""
    return DetectionConfig.model_json_schema()


@router.post(
    "/detection/run",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def run_detection(config: DetectionConfig, profile: ProfileDep) -> RunResponse:
    """Start an object-detection training run in the background.

    Detection shares the run queue and the /experiment/{status,events,result}
    endpoints (one run at a time, one GPU) but dispatches its own standalone
    DetectionBlock (ADR-033) rather than the ExperimentConfig path.
    """
    config = profile.scope_model(config)
    await asyncio.to_thread(_check_detection_dataset, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "detection",
        "simple",
        lambda: _execute_detection(config, run_id),
        profile=profile,
    )


@router.get("/regression/schema")
async def get_regression_schema() -> dict[str, Any]:
    """Return the JSON Schema for RegressionConfig (drives the regression form)."""
    return RegressionConfig.model_json_schema()


@router.post(
    "/regression/run",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def run_regression(config: RegressionConfig, profile: ProfileDep) -> RunResponse:
    """Start an image-regression training run in the background.

    Regression shares the run queue and the /experiment/{status,events,result}
    endpoints (one run at a time, one GPU) but dispatches its own standalone
    RegressionBlock (ADR-036) rather than the ExperimentConfig path.
    """
    config = profile.scope_model(config)
    await asyncio.to_thread(_check_regression_dataset, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "regression",
        "simple",
        lambda: _execute_regression(config, run_id),
        profile=profile,
    )


@router.get("/segmentation/schema")
async def get_segmentation_schema() -> dict[str, Any]:
    """Return the JSON Schema for SegmentationConfig (drives the segmentation form)."""
    return SegmentationConfig.model_json_schema()


@router.post(
    "/segmentation/run",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
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
    await asyncio.to_thread(_check_segmentation_dataset, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "segmentation",
        "simple",
        lambda: _execute_segmentation(config, run_id),
        profile=profile,
    )


@router.get("/anomaly/schema")
async def get_anomaly_schema() -> dict[str, Any]:
    """Return the JSON Schema for AnomalyConfig (drives the anomaly form)."""
    return AnomalyConfig.model_json_schema()


@router.post("/anomaly/run", responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNAVAILABLE})
async def run_anomaly(config: AnomalyConfig, profile: ProfileDep) -> RunResponse:
    """Start an anomaly-detection training run in the background.

    Anomaly shares the run queue and the /experiment/{status,events,result}
    endpoints (one run at a time, one GPU) but dispatches its own standalone
    AnomalyBlock (ADR-038) rather than the ExperimentConfig path.
    """
    config = profile.scope_model(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    return _submit_job(
        run_id,
        config.name,
        "anomaly",
        "simple",
        lambda: _execute_anomaly(config, run_id),
        # PatchCore has no epochs; it stops between its phases (ADR-111).
        stop_at="phase" if config.model.name == "patchcore" else "epoch",
        profile=profile,
    )


@router.post(
    "/regression/compare", responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNAVAILABLE}
)
async def compare_regression(
    req: ComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Train N regression backbones on the same dataset and rank them (ADR-044)."""
    return _start_comparison(req, RegressionConfig, RegressionRunner(), profile)


@router.post(
    "/segmentation/compare", responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNAVAILABLE}
)
async def compare_segmentation(
    req: ComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Train N segmentation backbones on the same dataset and rank them (ADR-044)."""
    return _start_comparison(req, SegmentationConfig, SegmentationRunner(), profile)


@router.post(
    "/regression/sweep",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def sweep_regression(req: SweepRequest, profile: ProfileDep) -> RunResponse:
    """Run a grid/random hyperparameter sweep for regression (ADR-045)."""
    return _start_sweep(req, RegressionConfig, RegressionRunner(), profile)


@router.post(
    "/segmentation/sweep",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def sweep_segmentation(req: SweepRequest, profile: ProfileDep) -> RunResponse:
    """Run a grid/random hyperparameter sweep for segmentation (ADR-045)."""
    return _start_sweep(req, SegmentationConfig, SegmentationRunner(), profile)


@router.post(
    "/detection/compare", responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNAVAILABLE}
)
async def compare_detection(req: ComparisonRequest, profile: ProfileDep) -> RunResponse:
    """Train N detectors on the same dataset and rank them (ADR-044)."""
    return _start_comparison(req, DetectionConfig, DetectionRunner(), profile)


@router.post(
    "/detection/sweep",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def sweep_detection(req: SweepRequest, profile: ProfileDep) -> RunResponse:
    """Run a grid/random hyperparameter sweep for detection (ADR-045)."""
    return _start_sweep(req, DetectionConfig, DetectionRunner(), profile)


@router.post(
    "/anomaly/compare", responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNAVAILABLE}
)
async def compare_anomaly(req: ComparisonRequest, profile: ProfileDep) -> RunResponse:
    """Train N anomaly models on the same dataset and rank them (ADR-044)."""
    return _start_comparison(req, AnomalyConfig, AnomalyRunner(), profile)


@router.post(
    "/anomaly/sweep",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def sweep_anomaly(req: SweepRequest, profile: ProfileDep) -> RunResponse:
    """Run a grid/random hyperparameter sweep for anomaly detection (ADR-045)."""
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


@router.get("/tasks")
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


@router.post("/custom/{key}/hide", responses=_NOT_FOUND)
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


@router.post("/custom/{key}/unhide")
async def unhide_custom_task(key: str) -> CustomTaskActionResponse:
    """Bring a hidden custom task's tab back."""
    from visionforge.tasks.manage import set_hidden

    set_hidden(key, False)
    return CustomTaskActionResponse(
        key=key, action="unhidden", detail=tr("custom.unhidden")
    )


@router.delete("/custom/{key}", responses={**_BAD_REQUEST, **_NOT_FOUND})
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


@router.get("/custom/{key}/schema", responses=_NOT_FOUND)
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
    "/custom/{key}/run", responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNAVAILABLE}
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{cfg.name}_{timestamp}"

    return _submit_job(
        run_id,
        str(cfg.name),
        f"custom:{info.key}",
        "simple",
        lambda: _execute_custom_task(info, cfg, run_id),
        stop_at=None if _owns_its_loop(info) else "epoch",
        profile=profile,
    )


def _owns_its_loop(info: Any) -> bool:
    """True for a Level 2 custom task, whose own ``run`` never reads a stop request."""
    return info.spec_cls is not None and info.spec_cls.run is not TaskSpec.run


async def _execute_custom_task(info: Any, cfg: Any, run_id: str) -> None:
    """Run a custom task's engine in a worker thread and store the report."""
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        engine = GenericTaskEngine(info, cfg)
        logger.info("GUI: Starting custom task '{}' run {}", info.key, run_id)
        result = await asyncio.to_thread(engine.run, _put_event, _active_cancel_token)

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
        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": str(result.run_dir),
        }
        logger.success("GUI: Custom task {} completed.", run_id)
    except Exception as e:
        logger.exception("GUI: Custom task {} failed", run_id)
        _current_run = {
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
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
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
    info = _get_custom_task_or_404(key)
    return _start_sweep(req, info.spec_cls.Config, CustomTaskRunner(info), profile)


@router.post(
    "/custom/{key}/replicates",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicates_custom_task(
    key: str, req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one custom-task config N times under different seeds (ADR-056/058)."""
    info = _get_custom_task_or_404(key)
    return _start_replicates(req, info.spec_cls.Config, CustomTaskRunner(info), profile)


@router.post("/regression/cv", responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNAVAILABLE})
async def cv_regression(req: TaskCvRequest, profile: ProfileDep) -> RunResponse:
    """Run K-fold cross-validation for regression (ADR-050) in the background."""
    return _start_task_cv(
        req, RegressionConfig, run_regression_cross_validation, "Regression", profile
    )


@router.post(
    "/segmentation/cv", responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNAVAILABLE}
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
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicates_classification(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one classification config N times under different seeds (ADR-056)."""
    return _start_replicates(req, ExperimentConfig, ClassificationRunner(), profile)


@router.post(
    "/regression/replicates",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicates_regression(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one regression config N times under different seeds (ADR-056)."""
    return _start_replicates(req, RegressionConfig, RegressionRunner(), profile)


@router.post(
    "/segmentation/replicates",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicates_segmentation(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one segmentation config N times under different seeds (ADR-056)."""
    return _start_replicates(req, SegmentationConfig, SegmentationRunner(), profile)


@router.post(
    "/detection/replicates",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicates_detection(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one detection config N times under different seeds (ADR-056)."""
    return _start_replicates(req, DetectionConfig, DetectionRunner(), profile)


@router.post(
    "/anomaly/replicates",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicates_anomaly(
    req: ReplicatesRequest, profile: ProfileDep
) -> RunResponse:
    """Train one anomaly config N times under different seeds (ADR-056)."""
    return _start_replicates(req, AnomalyConfig, AnomalyRunner(), profile)


@router.post(
    "/classification/replicated-comparison",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicated_comparison_classification(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare classification variants over shared seeds with a paired test."""
    return _start_replicated_comparison(
        req, ExperimentConfig, ClassificationRunner(), profile
    )


@router.post(
    "/regression/replicated-comparison",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicated_comparison_regression(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare regression variants over shared seeds with a paired test."""
    return _start_replicated_comparison(
        req, RegressionConfig, RegressionRunner(), profile
    )


@router.post(
    "/segmentation/replicated-comparison",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicated_comparison_segmentation(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare segmentation variants over shared seeds with a paired test."""
    return _start_replicated_comparison(
        req, SegmentationConfig, SegmentationRunner(), profile
    )


@router.post(
    "/detection/replicated-comparison",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicated_comparison_detection(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare detection variants over shared seeds with a paired test."""
    return _start_replicated_comparison(
        req, DetectionConfig, DetectionRunner(), profile
    )


@router.post(
    "/anomaly/replicated-comparison",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicated_comparison_anomaly(
    req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare anomaly variants over shared seeds with a paired test."""
    return _start_replicated_comparison(req, AnomalyConfig, AnomalyRunner(), profile)


@router.post(
    "/custom/{key}/replicated-comparison",
    responses={**_BAD_REQUEST, **_NOT_FOUND, **_UNPROCESSABLE, **_UNAVAILABLE},
)
async def replicated_comparison_custom(
    key: str, req: ReplicatedComparisonRequest, profile: ProfileDep
) -> RunResponse:
    """Compare a custom task's variants over shared seeds (ADR-058 + ADR-061)."""
    info = _get_custom_task_or_404(key)
    return _start_replicated_comparison(
        req, info.spec_cls.Config, CustomTaskRunner(info), profile
    )


@router.get("/experiment/events")
async def experiment_events() -> StreamingResponse:
    """Stream live training progress as Server-Sent Events."""
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_generator():  # type: ignore[return]
    """Pull events off _event_queue and yield SSE-formatted lines."""
    queue = _event_queue
    if queue is None:
        yield "data: {}\n\n"
        return

    while True:
        item = await queue.get()
        if item is None:
            break
        yield f"data: {json.dumps(item)}\n\n"


@router.get("/experiment/status")
async def get_status() -> RunStatus:
    """Poll the current experiment status, plus how many are queued behind it."""
    queued = _RUN_QUEUE.pending_count()
    if _current_run is None:
        return RunStatus(status="idle", queued=queued)

    return RunStatus(
        status=_current_run["status"],
        run_id=_current_run["run_id"],
        error=_current_run.get("error"),
        queued=queued,
    )


@router.get("/queue")
async def get_queue() -> QueueSnapshot:
    """The active run and everything waiting behind it, in submission order."""
    snap = _RUN_QUEUE.snapshot()
    return QueueSnapshot(
        active=QueuedJobInfo(**snap["active"]) if snap["active"] else None,
        pending=[QueuedJobInfo(**job) for job in snap["pending"]],
    )


@router.delete("/queue/{run_id}", responses={**_NOT_FOUND, **_CONFLICT})
async def cancel_queued_run(run_id: str) -> dict[str, str]:
    """Stop a job, whether it is waiting or already training.

    A pending job is dropped. A running one is asked to stop at the boundary its
    ``stop_at`` names and keeps the best checkpoint it has earned (ADR-088,
    ADR-111), so a 200 here means the request was delivered, not that training
    has already ended.

    Raises:
        HTTPException: 404 if no job carries that id — already finished, or
            never submitted. 409 if the job is running and cannot be stopped
            (``stop_at`` is null); nothing is asked of it.
    """
    try:
        found = _RUN_QUEUE.cancel(run_id)
    except NotStoppableError:
        raise HTTPException(409, tr("queue.not_stoppable")) from None
    if not found:
        raise HTTPException(404, tr("queue.unknown_id"))
    return {"run_id": run_id, "status": "cancelled"}


@router.get(
    "/experiment/result/{run_id}",
    responses={**_NOT_FOUND, **_CONFLICT, **_SERVER_ERROR},
)
async def get_result(run_id: str) -> RunResult:
    """Fetch the result of a completed experiment.

    Reads the live state when the id is the current run, and otherwise the
    queue's per-run record: with jobs queued back to back, the next one is
    already running by the time the browser asks for the previous one's report.
    """
    state = _current_run if (_current_run or {}).get("run_id") == run_id else None
    if state is None:
        state = _RUN_QUEUE.record(run_id)
    if state is None:
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))

    if state["status"] == "running":
        raise HTTPException(409, tr("run.still_running"))

    if state["status"] == "failed":
        raise HTTPException(
            500, tr("run.failed", error=state.get("error", "unknown error"))
        )

    run_dir = state.get("run_dir")
    run_json_path = Path(run_dir) / RUN_FILENAME if run_dir else None

    report = state.get("report") or {}
    stopped = _report_stopped(report)
    if run_json_path and run_json_path.exists():
        data = json.loads(run_json_path.read_text(encoding="utf-8"))
        return RunResult(
            run_id=run_id,
            metrics=data.get("metrics", {}),
            metric_cis=data.get("metric_cis", {}),
            report=report,
            artifacts=data.get("artifacts", {}),
            stopped=stopped,
        )

    return RunResult(
        run_id=run_id,
        metrics={},
        report=report,
        artifacts={},
        stopped=stopped,
    )


def _report_stopped(report: dict[str, Any]) -> bool:
    """Whether a stop cut this run, read from wherever its report keeps it.

    Multi-unit reports carry ``stopped`` at the top; single runs carry it in
    their training section (``train``, or ``detection`` for detection).
    """
    if "stopped" in report:
        return bool(report["stopped"])
    for section in ("train", "detection"):
        part = report.get(section)
        if isinstance(part, dict) and "stopped" in part:
            return bool(part["stopped"])
    return False


@router.get(
    "/artifacts/{file_path:path}",
    responses={
        403: {"description": "Path outside outputs"},
        404: {"description": "File not found"},
    },
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


def _stopped_extent(data: dict[str, Any]) -> str:
    """How far a stopped run got, in the unit that run counts (ADR-111)."""
    metrics: dict[str, Any] = data.get("metrics") or {}
    config: dict[str, Any] = data.get("config") or {}
    cv = metrics.get("cv_aggregate")
    if isinstance(cv, dict):
        return tr_artifact(
            "card.stopped_folds",
            ok=cv.get("n_folds_ok", "?"),
            n=cv.get("n_folds", "?"),
        )
    if (config.get("model") or {}).get("name") == "patchcore":
        # PatchCore has phases, not epochs: one fit counts the bank as built.
        return tr_artifact(
            "card.stopped_bank_built"
            if metrics.get("total_epochs")
            else "card.stopped_before_bank"
        )
    planned = (config.get("training") or {}).get("epochs", "?")
    return tr_artifact(
        "card.stopped_epochs", done=metrics.get("total_epochs", "?"), planned=planned
    )


def _card_status(status: Any) -> str:
    """The run's status in the card's language; an unknown one is kept as is."""
    if status is None:
        return "—"
    if status == "completed":
        return tr_artifact("card.status_completed")
    if status == "failed":
        return tr_artifact("card.status_failed")
    if status == "running":
        return tr_artifact("card.status_running")
    if status == "stopped":
        return tr_artifact("card.status_stopped")
    return str(status)


def _card_value(value: Any) -> str:
    """A metric cell: four decimals for a float, a dash for a missing value."""
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _render_run_markdown(run_dir: Path, data: dict[str, Any]) -> str:
    """Format a run.json into a paper-ready model card."""
    lines: list[str] = []
    name = data.get("experiment", run_dir.name)
    config: dict[str, Any] = data.get("config", {})
    metrics: dict[str, Any] = data.get("metrics", {})
    history: list[dict[str, Any]] = data.get("history", [])

    lines.extend([f"# {name}", ""])
    # The History lists a run by its folder name, so the card carries the same
    # id; run.json's own "id" is "<experiment>_<folder>" and matched nothing the
    # user sees. A bare Path (no folder) falls back to it.
    lines.append(
        f"- **{tr_artifact('card.run_id')}:** `{run_dir.name or data.get('id', '')}`"
    )
    lines.append(f"- **{tr_artifact('card.timestamp')}:** {data.get('timestamp', '?')}")
    lines.append(
        f"- **{tr_artifact('card.status')}:** {_card_status(data.get('status'))}"
    )
    if data.get("stopped"):
        lines.append(
            f"- **{tr_artifact('card.stopped')}:** {tr_artifact('card.yes')} "
            f"({_stopped_extent(data)})"
        )
    if data.get("device_used"):
        lines.append(
            f"- **{tr_artifact('card.device_used')}:** `{data['device_used']}`"
        )
    lines.append(
        f"- **{tr_artifact('card.run_dir')}:** `{data.get('run_dir', run_dir)}`"
    )
    artifacts = data.get("artifacts", {})
    if artifacts.get("model"):
        lines.append(f"- **{tr_artifact('card.checkpoint')}:** `{artifacts['model']}`")
    lines.extend(
        [
            "",
            f"## {tr_artifact('card.configuration')}",
            "",
        ]
    )
    model_cfg = config.get("model") or {}
    training_cfg = config.get("training") or {}
    # Each task has its own config shape: a row whose field the task does not
    # have is left out rather than printed as "?".
    task_label = data.get("task_label")
    rows: list[tuple[str, Any]] = [
        (
            tr_artifact("card.architecture"),
            model_cfg.get("name") or task_label,
        ),
        ("num_classes", model_cfg.get("num_classes")),
        ("num_targets", model_cfg.get("num_targets")),
        (tr_artifact("card.pretrained"), model_cfg.get("pretrained")),
        (
            tr_artifact("card.task"),
            task_label or config.get("task") or data.get("task"),
        ),
        (tr_artifact("card.optimizer"), training_cfg.get("optimizer")),
        (tr_artifact("card.learning_rate"), training_cfg.get("learning_rate")),
        (tr_artifact("card.batch_size"), training_cfg.get("batch_size")),
        (tr_artifact("card.epochs_max"), training_cfg.get("epochs")),
        # "Seed" and "Scheduler" are the words a Portuguese paper uses too.
        ("Seed", training_cfg.get("seed")),
    ]
    for label, value in rows:
        if value is not None:
            lines.append(f"- **{label}:** {value}")
    if training_cfg.get("scheduler", {}).get("kind", "none") != "none":
        sched = training_cfg["scheduler"]
        lines.append(f"- **Scheduler:** {sched['kind']} ({sched})")
    if training_cfg.get("mixed_precision"):
        lines.append(
            f"- **{tr_artifact('card.mixed_precision')}:** {tr_artifact('card.enabled')}"
        )
    lines.append("")

    data_cfg = config.get("data", {})
    pp_steps = data_cfg.get("preprocessing", {}).get("steps", [])
    if pp_steps:
        lines.extend(
            [
                f"## {tr_artifact('card.preprocessing')}",
                "",
                tr_artifact("card.preprocessing_intro"),
                "",
            ]
        )
        for i, step in enumerate(pp_steps, start=1):
            kind = step.get("kind", "?")
            params = ", ".join(f"{k}={v!r}" for k, v in step.items() if k != "kind")
            lines.append(f"{i}. **{kind}**" + (f" ({params})" if params else ""))
        lines.append("")

    tx = dict(data_cfg.get("transforms") or {})
    # Anomaly and segmentation resize to data.image_size; the image_size inside
    # transforms is a classification default they never read, so the card would
    # state a resolution the model never saw.
    if tx and isinstance(data_cfg.get("image_size"), int):
        tx["image_size"] = data_cfg["image_size"]
    if tx:
        lines.extend(
            [
                f"## {tr_artifact('card.augmentation')}",
                "",
                f"| {tr_artifact('card.field')} | {tr_artifact('card.value')} |",
                MARKDOWN_TABLE_SEPARATOR,
            ]
        )
        for k, v in tx.items():
            lines.append(f"| {k} | `{v}` |")
        lines.append("")

    lines.append(f"## {tr_artifact('card.final_metrics')}")
    lines.append("")
    if metrics:
        cis: dict[str, Any] = data.get("metric_cis", {})
        has_cis = bool(cis)
        header = f"| {tr_artifact('card.metric')} | {tr_artifact('card.value')} |"
        if has_cis:
            header += f" {tr_artifact('card.ci_bootstrap')} |"
        lines.append(header)
        lines.append("|---|---|---|" if has_cis else MARKDOWN_TABLE_SEPARATOR)
        for k, v in metrics.items():
            value = _card_value(v)
            if not has_cis:
                lines.append(f"| {k} | {value} |")
                continue
            # run.json keys the CIs by bare metric name; the metrics block
            # prefixes the test-split ones with "test_".
            ci = cis.get(k[len("test_") :] if k.startswith("test_") else k)
            interval = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]" if ci else "—"
            lines.append(f"| {k} | {value} | {interval} |")
        if has_cis:
            sample = next(iter(cis.values()))
            lines.append("")
            lines.append(
                tr_artifact(
                    "card.bootstrap_note",
                    n_samples=sample["n_samples"],
                    n_resamples=sample["n_resamples"],
                )
            )
    else:
        lines.append(tr_artifact("card.none_recorded"))
    lines.append("")

    if history:
        # Columns come from the history itself rather than a fixed
        # classification shape: a segmentation epoch has val_miou and no
        # val_accuracy, a custom task streams whatever metrics it declared, and
        # a hardcoded header formatted every missing cell with ":.4f" — which
        # raised on the "?" fallback and made this endpoint 500 for every task
        # except classification.
        columns: list[str] = []
        for epoch in history:
            for key in epoch:
                if key != "epoch" and key not in columns:
                    columns.append(key)

        lines.extend(
            [
                f"## {tr_artifact('card.history', n=len(history))}",
                "",
                f"| {tr_artifact('card.epoch')} | " + " | ".join(columns) + " |",
                "|---" * (len(columns) + 1) + "|",
            ]
        )
        for epoch in history:
            cells = [str(epoch.get("epoch", "?"))]
            for key in columns:
                cell = epoch.get(key)
                if isinstance(cell, float):
                    formatted_cell = f"{cell:.4f}"
                else:
                    formatted_cell = str(cell) if cell is not None else "—"
                cells.append(formatted_cell)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    graphics = artifacts.get("graphics", [])
    if graphics:
        lines.append(f"## {tr_artifact('card.plots')}")
        lines.append("")
        for g in graphics:
            lines.append(f"- `{g}`")
        lines.append("")

    tests = data.get("tests", [])
    if tests:
        lines.append(f"## {tr_artifact('card.tests', n=len(tests))}")
        lines.append("")
        for t in tests:
            lines.extend(
                [
                    f"### {t.get('label', t.get('test_id'))}",
                    f"- **{tr_artifact('card.test_base_dir')}:** `{t.get('base_dir')}`",
                    f"- **{tr_artifact('card.test_timestamp')}:** {t.get('timestamp')}",
                ]
            )
            m = t.get("metrics", {})
            if m:
                lines.append(
                    f"| {tr_artifact('card.metric')} | {tr_artifact('card.value')} |"
                )
                lines.append(MARKDOWN_TABLE_SEPARATOR)
                for k, v in m.items():
                    lines.append(f"| {k} | {_card_value(v)} |")
            lines.append("")

    return "\n".join(lines)


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
    if not split_dir.is_dir():
        return PreprocessPreviewResponse(
            original="",
            steps=[],
            final="",
            source_image="",
            available_kinds=available_kinds(),
            message=tr("scan.split_missing", split=req.split),
        )

    # Pick the first available class + first image inside it.
    class_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
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


def _execute_batch_predict(
    run_dir: Path, req: BatchPredictRequest
) -> BatchPredictResponse:
    """Dispatch folder batch-prediction to the run's task path.

    Classification runs go through ``BatchPredictionBlock`` (softmax/sigmoid CSV);
    regression and anomaly runs reuse the shared ``core.batch_predict`` helper
    (continuous targets / anomaly score + decision → CSV, ADR-041 slice 3).
    Detection/segmentation are not yet supported and are rejected. The request
    supplies ``input_dir``, ``output_csv`` (default: timestamped under
    ``run_dir/predictions/``) and the recursive flag.
    """
    run_json_path = run_dir / RUN_FILENAME
    data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    task = str(data.get("config", {}).get("task", ""))

    input_dir = Path(req.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"input_dir does not exist or is not a directory: {input_dir}"
        )
    _remember_dataset_root(input_dir)

    if req.output_csv:
        output_csv = Path(req.output_csv)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_csv = run_dir / "predictions" / f"predictions_{ts}.csv"

    if task in ("regression", "anomaly"):
        runner = (
            batch_predict_regression_run
            if task == "regression"
            else batch_predict_anomaly_run
        )
        summary = runner(
            run_name=run_dir.name,
            data=data,
            input_dir=input_dir,
            output_csv=output_csv.resolve(),
            recursive=req.recursive,
        )
        return BatchPredictResponse(
            output_csv=summary.output_csv,
            total_processed=summary.total_processed,
            failed_count=len(summary.failed_files),
            failed_files=summary.failed_files,
        )

    if task in ("detection", "segmentation"):
        raise ValueError(tr("testrun.batch_unsupported", task=task))

    base_config_dict: dict[str, Any] = data["config"]

    checkpoint_path = data.get("artifacts", {}).get("model")
    if not checkpoint_path:
        raise FileNotFoundError(
            f"Run '{run_dir.name}' has no checkpoint recorded in artifacts.model."
        )

    batch_config_dict = {
        **base_config_dict,
        "block": "batch_prediction",
        "batch_prediction": {
            "checkpoint_path": checkpoint_path,
            "input_dir": str(input_dir.resolve()),
            "output_csv": str(output_csv.resolve()),
            "recursive": req.recursive,
            "class_names": req.class_names,
        },
    }
    config = ExperimentConfig.model_validate(batch_config_dict)

    block = BatchPredictionBlock()
    block.setup(config)
    block.run()
    report = block.report()

    failed: list[str] = list(report.get("failed_files", []))
    return BatchPredictResponse(
        output_csv=str(output_csv.resolve()),
        total_processed=int(report.get("total_processed", 0)),
        failed_count=len(failed),
        failed_files=failed,
    )


def _execute_onnx_export(run_dir: Path, req: ExportOnnxRequest) -> ExportOnnxResponse:
    """Build a one-shot ExperimentConfig for ExportONNXBlock and run it.

    The run's stored config supplies the architecture and image size; the
    request supplies the export-specific knobs. The exported file defaults
    to ``<run_dir>/best_model.onnx`` next to the checkpoint.
    """
    run_json_path = run_dir / RUN_FILENAME
    data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    # Detection exports via Ultralytics; regression/segmentation via the shared
    # core helper (their own factory builds the model); classification below.
    task = data.get("config", {}).get("task")
    if task == "detection":
        return export_detection_run(run_dir, req, data)
    if task == "regression":
        return export_regression_run(run_dir, req, data)
    if task == "segmentation":
        return export_segmentation_run(run_dir, req, data)
    base_config_dict: dict[str, Any] = data["config"]

    checkpoint_path = data.get("artifacts", {}).get("model")
    if not checkpoint_path:
        raise FileNotFoundError(
            f"Run '{run_dir.name}' has no checkpoint recorded in artifacts.model."
        )

    output_onnx = req.output_onnx or str((run_dir / "best_model.onnx").resolve())

    export_config_dict = {
        **base_config_dict,
        "block": "export_onnx",
        "export_onnx": {
            "checkpoint_path": checkpoint_path,
            "output_onnx": output_onnx,
            "opset_version": req.opset_version,
            "dynamic_axes": req.dynamic_axes,
            "validate": req.run_validate,
            "benchmark": req.benchmark,
            "benchmark_runs": req.benchmark_runs,
        },
    }
    config = ExperimentConfig.model_validate(export_config_dict)

    block = ExportONNXBlock()
    block.setup(config)
    block.run()
    report = block.report()

    file_size = int(report.get("file_size_bytes", 0))
    return ExportOnnxResponse(
        output_onnx=output_onnx,
        file_size_bytes=file_size,
        validation=report.get("validation"),
        benchmark=report.get("benchmark"),
    )


def _evaluate_standalone_run(
    run_dir: Path, req: RunTestRequest, data: dict[str, Any], task: str
) -> RunTestResponse:
    """Score a regression / segmentation / anomaly checkpoint on one folder.

    Each task's trainer already knows how to evaluate; all that was missing was
    a way to point its *test* split at the folder the researcher chose. For the
    folder-shaped tasks that is ``base_dir=<parent>`` + ``test_dir=<name>``;
    regression's manifest model makes the chosen path a ``.csv`` instead, so its
    parent becomes the base and the file becomes ``test_csv``.

    Raises:
        FileNotFoundError: the run has no checkpoint on disk.
        ValueError: the chosen path does not exist, or an anomaly run has no
            normal training split to calibrate its threshold on.
    """
    import torch

    # What the researcher just chose is validated first: a complaint about the
    # run's checkpoint is confusing when the actual problem is the path they
    # typed a second ago.
    chosen = Path(req.data_dir)
    if not chosen.exists():
        raise ValueError(tr("testrun.path_not_found", path=chosen))
    if task == "regression" and chosen.is_dir():
        raise ValueError(tr("testrun.regression_needs_csv"))

    checkpoint = (data.get("artifacts") or {}).get("model")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            tr("testrun.no_checkpoint", run=run_dir.name, model=repr(checkpoint))
        )

    config_dict = dict(data["config"])
    data_dict = dict(config_dict.get("data", {}))
    if task == "regression":
        data_dict |= {"base_dir": str(chosen.parent), "test_csv": chosen.name}
    else:
        data_dict |= {"base_dir": str(chosen.parent), "test_dir": chosen.name}
    config_dict["data"] = data_dict

    if task == "regression":
        from visionforge.core.regression_data import RegressionDataModule
        from visionforge.core.regression_trainer import RegressionTrainer
        from visionforge.models.regression_factory import RegressionModelFactory
        from visionforge.utils.regression_config import RegressionConfig

        reg_config = RegressionConfig.model_validate(config_dict)
        reg_model = RegressionModelFactory.create(reg_config.model)
        reg_model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        # The chosen manifest is the only split: a held-out set rarely has
        # train.csv / val.csv beside it.
        reg_loader = RegressionDataModule(
            reg_config, evaluation_only=True
        ).test_loader()
        if reg_loader is None:
            raise ValueError(tr("testrun.no_rows", path=chosen))
        mse, rmse, mae, r2 = RegressionTrainer(reg_config).evaluate(
            reg_model, reg_loader
        )
        metrics: dict[str, Any] = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    elif task == "segmentation":
        from visionforge.core.segmentation_data import SegmentationDataModule
        from visionforge.core.segmentation_trainer import SegmentationTrainer
        from visionforge.models.segmentation_factory import SegmentationModelFactory
        from visionforge.utils.segmentation_config import SegmentationConfig

        seg_config = SegmentationConfig.model_validate(config_dict)
        # Without this the pairing step walks into the missing folder and the
        # operating system's own message (in its own language) reaches the screen.
        test_split = seg_config.data.base_dir / seg_config.data.test_dir
        masks_dir = test_split / seg_config.data.masks_subdir
        if (test_split / seg_config.data.images_subdir).is_dir() and (
            not masks_dir.is_dir()
        ):
            raise ValueError(
                tr(
                    "testrun.masks_missing",
                    split=test_split,
                    images=seg_config.data.images_subdir,
                    masks=seg_config.data.masks_subdir,
                    masks_path=masks_dir,
                )
            )
        seg_model = SegmentationModelFactory.create(seg_config.model)
        seg_model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        seg_loader = SegmentationDataModule(
            seg_config, evaluation_only=True
        ).test_loader()
        if seg_loader is None:
            raise ValueError(tr("testrun.no_pairs", path=chosen))
        miou, dice, pixel_acc = SegmentationTrainer(seg_config).evaluate(
            seg_model, seg_loader
        )
        metrics = {"miou": miou, "dice": dice, "pixel_acc": pixel_acc}

    else:  # anomaly
        from visionforge.core.anomaly_data import AnomalyDataModule
        from visionforge.core.anomaly_trainer import AnomalyTrainer
        from visionforge.models.anomaly_factory import AnomalyModelFactory
        from visionforge.utils.anomaly_config import AnomalyConfig

        anom_config = AnomalyConfig.model_validate(config_dict)
        # Checked before the checkpoint is loaded: the data module's own message
        # ("no normal training images") does not tell someone who just picked a
        # test folder why a training folder is being asked for.
        train_normal = (
            anom_config.data.base_dir
            / anom_config.data.train_dir
            / anom_config.data.normal_dir
        )
        if not train_normal.is_dir():
            raise ValueError(
                tr(
                    "testrun.anomaly_train_missing",
                    path=train_normal,
                    train_dir=anom_config.data.train_dir,
                    normal_dir=anom_config.data.normal_dir,
                )
            )
        anom_model = AnomalyModelFactory.create(anom_config.model)
        anom_model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        modules = AnomalyDataModule(anom_config)
        # The threshold comes from the *normal training* distribution, so that
        # split is still needed even though only the chosen folder is being
        # scored (ADR-076) — read without augmentation, which is the
        # distribution the model is applied to (ADR-098).
        auroc, threshold, image_f1 = AnomalyTrainer(anom_config).evaluate(
            anom_model, modules.calibration_loader(), modules.test_loader()
        )
        metrics = {"auroc": auroc, "threshold": threshold, "image_f1": image_f1}

    timestamp = datetime.now()
    test_id = f"test_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    label = req.label or chosen.name or "test"
    resolved = str(chosen.resolve())
    record: dict[str, Any] = {
        "test_id": test_id,
        "label": label,
        "base_dir": resolved,
        "timestamp": timestamp.isoformat(),
        "metrics": metrics,
        "artifacts": {},
    }
    data.setdefault("tests", []).append(record)
    (run_dir / RUN_FILENAME).write_text(json.dumps(data, indent=2), encoding="utf-8")

    return RunTestResponse(
        test_id=test_id,
        run_id=run_dir.name,
        label=label,
        base_dir=resolved,
        timestamp=timestamp,
        metrics=metrics,
        artifacts={},
    )


def _execute_run_test(run_dir: Path, req: RunTestRequest) -> RunTestResponse:
    """Reload a saved checkpoint and evaluate it on a new dataset path.

    Updates run.json so each model accumulates its own test history.
    """
    run_json_path = run_dir / RUN_FILENAME
    data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    task = _run_task(data)

    # Each family evaluates through its own trainer (ADR-080). Detection had a
    # native path already; the other three used to fall into the classification
    # branch below and surface a raw ExperimentConfig validation error.
    if task == "detection":
        return evaluate_detection_run(run_dir, req, data)
    if task in ("regression", "segmentation", "anomaly"):
        return _evaluate_standalone_run(run_dir, req, data, task)

    base_config_dict: dict[str, Any] = data["config"]

    # The same guard as the other families: a run stopped before its first
    # epoch records no checkpoint (ADR-111), and evaluating "None" would fail
    # with an error about a path instead of saying what is missing.
    checkpoint = (data.get("artifacts") or {}).get("model")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            tr("testrun.no_checkpoint", run=run_dir.name, model=repr(checkpoint))
        )

    # Point the *test* split at the chosen folder: base_dir is its parent and the
    # split name is the folder itself, so one folder is all the researcher gives
    # and the existing DataModule still does the loading.
    chosen = Path(req.data_dir)
    test_config_dict = {
        **base_config_dict,
        "classification": {
            "mode": "evaluate",
            "checkpoint_path": checkpoint,
        },
        "data": {
            **base_config_dict.get("data", {}),
            "base_dir": str(chosen.parent),
            "test_dir": chosen.name,
        },
    }
    config = ExperimentConfig.model_validate(test_config_dict)

    model = ModelFactory.create(config.model)
    import torch  # local import — keeps top-level lightweight on cold start

    state_dict = torch.load(
        str(config.classification.checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)  # type: ignore[arg-type]

    data_module = DataModule(config)
    eval_result = Evaluator(config).evaluate(model, data_module.test_loader())

    timestamp = datetime.now()
    test_id = f"test_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    tests_dir = run_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Render the full evaluation-time plot set for the test dataset so a
    # per-model test history mirrors what the original training produced.
    artifacts: dict[str, Any] = {}
    cm_path = tests_dir / f"{test_id}_confusion_matrix.png"
    MetricsPlotter.confusion_matrix_plot(
        eval_result.confusion_matrix, data_module.class_names, cm_path
    )
    artifacts["confusion_matrix"] = str(cm_path)

    cm_norm_path = tests_dir / f"{test_id}_confusion_matrix_normalized.png"
    MetricsPlotter.confusion_matrix_normalized(
        eval_result.confusion_matrix, data_module.class_names, cm_norm_path
    )
    artifacts["confusion_matrix_normalized"] = str(cm_norm_path)

    roc_path = tests_dir / f"{test_id}_roc_curve.png"
    if MetricsPlotter.roc_curve_plot(
        eval_result.y_true,
        eval_result.y_proba_full,
        data_module.class_names,
        roc_path,
    ):
        artifacts["roc_curve"] = str(roc_path)

    pr_path = tests_dir / f"{test_id}_precision_recall_curve.png"
    if MetricsPlotter.precision_recall_curve_plot(
        eval_result.y_true,
        eval_result.y_proba_full,
        data_module.class_names,
        pr_path,
    ):
        artifacts["precision_recall_curve"] = str(pr_path)

    metrics = {
        "accuracy": eval_result.accuracy,
        "f1": eval_result.f1,
        "precision": eval_result.precision,
        "recall": eval_result.recall,
        "auc_roc": eval_result.auc_roc,
    }

    label = req.label or Path(req.data_dir).name or "test"
    base_dir_str = str(Path(req.data_dir).resolve())
    record: dict[str, Any] = {
        "test_id": test_id,
        "label": label,
        "base_dir": base_dir_str,
        "timestamp": timestamp.isoformat(),
        "metrics": metrics,
        "artifacts": artifacts,
    }
    data.setdefault("tests", []).append(record)
    run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return RunTestResponse(
        test_id=test_id,
        run_id=run_dir.name,
        label=label,
        base_dir=base_dir_str,
        timestamp=timestamp,
        metrics=metrics,
        artifacts=artifacts,
    )


_GRADCAM_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _gradcam_class_names(data: dict[str, Any]) -> list[str]:
    """Class names for a classification run, in the model's own index order.

    Recovered from the training folder rather than run.json, which never stored
    them: ``ImageFolder`` assigns indices by sorting the class sub-directories,
    so the sorted folder names *are* the index→name mapping the checkpoint was
    trained with. Returns an empty list when the folder is gone or the task is
    not classification — the caller then reports indices, as before.
    """
    config = data.get("config", {})
    if config.get("task") not in ("binary", "multiclass"):
        return []
    # Runs written from 0.7.1 on record the mapping, which is the only form that
    # survives the dataset being renamed or moved.
    recorded = data.get("class_names")
    if isinstance(recorded, list) and recorded:
        return [str(c) for c in recorded]
    data_config = config.get("data", {})
    base_dir = data_config.get("base_dir")
    if not base_dir:
        return []
    train_dir = Path(base_dir) / str(data_config.get("train_dir", "train"))
    if not train_dir.is_dir():
        return []
    names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    # A binary run trained with a single output unit maps 1 -> the second class;
    # anything else that does not line up is safer left unnamed.
    expected = config.get("model", {}).get("num_classes")
    if expected and len(names) != expected and not (expected == 1 and len(names) == 2):
        return []
    return names


def _execute_run_gradcam(run_dir: Path, req: GradCamRequest) -> GradCamResponse:
    """Rebuild this run's model and write Grad-CAM overlays for sample images.

    Dispatches by task (classification / regression / segmentation) via
    ``build_gradcam``; detection and anomaly are unsupported. Model + transform are
    read leniently from run.json so a moved dataset path does not block it; the
    checkpoint supplies the weights (architecture rebuilt with ``pretrained=False``,
    no download).
    """
    import torch
    from PIL import Image

    from visionforge.core.gradcam import GradCAM, overlay_cam, resolve_target_layer
    from visionforge.gui.api.torch_gradcam import build_gradcam

    data: dict[str, Any] = json.loads(
        (run_dir / RUN_FILENAME).read_text(encoding="utf-8")
    )
    checkpoint_path = data.get("artifacts", {}).get("model")
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Run '{run_dir.name}' has no usable checkpoint at artifacts.model."
        )

    setup = build_gradcam(
        data, target_index=req.target_class, checkpoint=checkpoint_path
    )

    input_dir = Path(req.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"input_dir is not a directory: {input_dir}")
    walker = input_dir.rglob("*") if req.recursive else input_dir.iterdir()
    images = sorted(
        p for p in walker if p.is_file() and p.suffix.lower() in _GRADCAM_IMAGE_EXTS
    )[: req.num_samples]
    if not images:
        raise ValueError(f"No image files found under {input_dir}.")

    target_layer = resolve_target_layer(setup.model)
    cam = GradCAM(setup.model, target_layer)
    out_dir = run_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = _gradcam_class_names(data)

    items: list[GradCamItem] = []
    try:
        for img_path in images:
            with Image.open(img_path) as im:
                tensor = setup.transform(im.convert("RGB")).unsqueeze(0)
            heat = cam(tensor, target_fn=setup.target_fn)
            with torch.no_grad():
                output = setup.model(tensor)
            predicted_class, prediction = setup.describe(output)
            overlay = overlay_cam(tensor[0], heat, alpha=req.alpha)
            overlay_path = out_dir / f"{img_path.stem}_gradcam.png"
            overlay.save(overlay_path)

            predicted_label = (
                class_names[predicted_class]
                if class_names
                and predicted_class is not None
                and 0 <= predicted_class < len(class_names)
                else None
            )
            # Only a folder named after one of this run's classes counts as
            # ground truth. A flat folder of unlabeled images has none, and
            # inventing one would be worse than showing nothing.
            parent = img_path.parent.name
            true_class = parent if parent in class_names else None
            items.append(
                GradCamItem(
                    source=str(img_path),
                    overlay=str(overlay_path),
                    predicted_class=predicted_class,
                    prediction=prediction,
                    predicted_label=predicted_label,
                    true_class=true_class,
                    correct=(
                        predicted_label == true_class
                        if true_class is not None and predicted_label is not None
                        else None
                    ),
                )
            )
    finally:
        cam.remove()

    return GradCamResponse(
        run_id=run_dir.name,
        count=len(items),
        target_layer=type(target_layer).__name__,
        items=items,
    )


async def _execute_experiment(
    config: ExperimentConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run the experiment in a background thread."""
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue  # capture at task-start to survive queue replacement

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        # Dispatch to the right block based on config.block. The classification
        # path stays default; cross_validation reuses the same /experiment/run
        # endpoint so the frontend doesn't need a separate route. Adding more
        # blocks here is a one-liner — keep them limited to ones whose GUI
        # surface is actually wired up.
        block: (
            ClassificationBlock
            | CrossValidationBlock
            | TransferLearningBlock
            | ModelComparisonBlock
            | GridSearchBlock
            | RandomSearchBlock
        )
        if config.block == "cross_validation":
            block = CrossValidationBlock()
        elif config.block == "transfer_learning":
            block = TransferLearningBlock()
        elif config.block == "model_comparison":
            block = ModelComparisonBlock()
        elif config.block == "grid_search":
            block = GridSearchBlock()
        elif config.block == "random_search":
            block = RandomSearchBlock()
        else:
            block = ClassificationBlock()
        block.setup(config)
        # Blocks that stream live epoch progress over SSE get the event pump
        # wired here. Grid/random search and K-fold CV forward each inner
        # trial/fold's epochs (annotated with trial context) plus
        # trial_start/trial_end banners.
        if isinstance(
            block,
            ClassificationBlock
            | GridSearchBlock
            | RandomSearchBlock
            | CrossValidationBlock
            | TransferLearningBlock
            | ModelComparisonBlock,
        ):
            block._progress_callback = _put_event
        block._cancel_token = _active_cancel_token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting experiment {} (block={})", run_id, config.block)
        await asyncio.to_thread(block.run)

        report = block.report()

        # Classification and TransferLearning produce a TrainResult with a
        # model_path so the run_dir is the checkpoint's parent folder. CV
        # writes per-fold run.jsons in subdirs and has no single run_dir.
        run_dir: str | None = None
        if isinstance(block, ClassificationBlock | TransferLearningBlock):
            if block._train_result and block._train_result.model_path:
                run_dir = str(block._train_result.model_path.parent)

        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Experiment {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Experiment {} failed", run_id)
        cls = type(e).__name__
        msg = str(e) or tr(RUN_NO_MESSAGE_KEY)
        # A worker-spawn failure surfaces as a torch DLL error, which sends the
        # reader after the wrong thing entirely.
        spawn_hint = describe_worker_spawn_failure(e)
        _current_run = {
            "run_id": run_id,
            "status": "failed",
            "error": spawn_hint or f"{cls}: {msg}",
            "report": None,
            "run_dir": None,
        }

    finally:
        if queue is not None:
            await queue.put(None)


async def _execute_detection(
    config: DetectionConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run an Ultralytics detection training in a background thread."""
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = DetectionBlock()
        block.setup(config)
        block._progress_callback = _put_event
        block._cancel_token = _active_cancel_token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting detection {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("detection", {}).get("run_dir")
        _current_run = {
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
        _current_run = {
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
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = RegressionBlock()
        block.setup(config)
        block._progress_callback = _put_event
        block._cancel_token = _active_cancel_token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting regression {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("train", {}).get("run_dir")
        _current_run = {
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
        _current_run = {
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
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = SegmentationBlock()
        block.setup(config)
        block._progress_callback = _put_event
        block._cancel_token = _active_cancel_token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting segmentation {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("train", {}).get("run_dir")
        _current_run = {
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
        _current_run = {
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
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = AnomalyBlock()
        block.setup(config)
        block._progress_callback = _put_event
        block._cancel_token = _active_cancel_token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting anomaly {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("train", {}).get("run_dir")
        _current_run = {
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
        _current_run = {
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
    global _current_run, _event_queue
    queue = _event_queue
    token = _active_cancel_token
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
        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": None,
        }
        logger.success("GUI: Comparison {} completed ({} archs).", run_id, len(trials))
    except Exception as e:
        logger.exception("GUI: Comparison {} failed", run_id)
        _current_run = {
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
    """Reject a dot-path that would move a trial's output folders.

    ``Profile.scope_dict`` points ``output`` at the profile before a sweep or a
    comparison starts, but both apply their dot-paths to that dict once per
    trial. A path under ``output`` would therefore undo the scoping (ADR-114),
    and no experiment legitimately varies where its files go.

    ``where`` is the catalog key of the phrase that ends the sentence ("in a
    sweep"), because that phrase differs between languages.

    Raises:
        HTTPException: 422 naming the first offending path.
    """
    for path in paths:
        if str(path).split(".", 1)[0] == "output":
            raise HTTPException(
                422, tr("profile.output_locked", where=tr(where), path=path)
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
    global _current_run, _event_queue
    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    token = _active_cancel_token
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
        planned = planned_trials(req.search_space, mode=req.mode, n_trials=req.n_trials)
        stopped = job_was_stopped([t.status for t in trials], planned, token)
        finished = any(t.status == "success" for t in trials)
        if not finished and not stopped:
            raise RuntimeError("All sweep trials failed — no ranking available.")
        if finished:
            _require_reported_metric(metric, [t.metrics for t in trials], "sweep")
        report = _sweep_report(
            trials, req.mode, metric, runner_metric_direction(runner, metric)
        )
        report["stopped"] = stopped
        # total_trials counts the trials that ran; the table and the GUI need
        # to say of how many (ADR-111).
        report["planned_trials"] = planned
        report["report_dir"] = _write_advanced_summary(
            base_config_dict, "sweep", report
        )
        best = report["best_trial"]
        _record_sweep_group(
            run_id,
            runner,
            base_config_dict,
            report,
            mode=req.mode,
            planned=planned,
            trials=report["trials"],
            best_index=best["trial_index"] if best else None,
            report_name="sweep",
        )
        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": None,
        }
        logger.success("GUI: Sweep {} completed ({} trials).", run_id, len(trials))
    except Exception as e:
        logger.exception("GUI: Sweep {} failed", run_id)
        _current_run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e or tr('run.no_message')}",
            "report": None,
            "run_dir": None,
        }
    finally:
        if queue is not None:
            await queue.put(None)


def _require_reported_metric(
    metric: str, per_trial_metrics: list[dict[str, float]], kind: str
) -> None:
    """Fail loudly when no trial reported the metric the run is ranked/aggregated by.

    Without this the failure is silent and misleading: a sweep ranks every
    trial as 0.0 and calls an arbitrary one "best", and a replicate set returns
    ``headline: None`` that renders as an empty card. The metric is chosen
    before any training, so the earliest honest place to catch a typo (or a
    name the runner never projects, e.g. ``test_accuracy`` vs ``accuracy``) is
    here, once results exist.

    Raises:
        RuntimeError: naming the metric and what the trials actually reported.
    """
    if any(metric in m for m in per_trial_metrics):
        return
    available = sorted({key for m in per_trial_metrics for key in m})
    raise RuntimeError(
        f"No {kind} reported the metric '{metric}' — "
        f"available: {available or 'none'}. Pick one of those."
    )


def _sweep_report(
    trials: list[SweepTrial],
    mode: str,
    metric: str,
    direction: Literal["higher", "lower"],
) -> dict[str, Any]:
    """Shape ranked sweep trials into a GUI-friendly report dict."""
    rows = [asdict(t) for t in trials]
    successful = [t for t in rows if t["status"] == "success"]
    return {
        "mode": mode,
        "metric": metric,
        # Which way the trials were ranked; best_trial is the lowest for "lower".
        "metric_direction": direction,
        # Each trial's `metrics` are measured on this split and rank it; its
        # `evaluation_metrics` are the final (test) numbers, not used (ADR-131).
        "selection_split": SELECTION_SPLIT,
        "trials": rows,
        "best_trial": successful[0] if successful else None,
        "total_trials": len(rows),
        "successful_trials": len(successful),
        "failed_count": sum(1 for t in rows if t["status"] == "failed"),
        "stopped_count": sum(1 for t in rows if t["status"] == STOPPED),
    }


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
    global _current_run, _event_queue
    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    token = _active_cancel_token

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
        stopped = job_was_stopped([f.status for f in cv.folds], req.n_folds, token)
        if not any(f.status == "success" for f in cv.folds) and not stopped:
            raise RuntimeError("All folds failed — no aggregate available.")
        report: dict[str, Any] = {
            "n_folds": cv.n_folds,
            "metric": cv.metric,
            "fold_results": [asdict(f) for f in cv.folds],
            "aggregate": cv.aggregate,
            "successful_folds": sum(1 for f in cv.folds if f.status == "success"),
            "stopped": stopped,
        }
        report["report_dir"] = _write_advanced_summary(req.config, "cv", report)
        # One History entry for the job, its folds under it (ADR-123).
        # The validated config: it names the task (config.task), which the
        # request's raw dict may leave to the default.
        _record_run_group("kfold", run_id, config.model_dump(mode="json"), report, "cv")
        _current_run = {
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
        _current_run = {
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

    if req.seeds is not None:
        if len(set(req.seeds)) != len(req.seeds):
            raise HTTPException(422, tr("compare.seeds_unique"))
        seeds = list(req.seeds)
    else:
        base_seed = int(complete.get("training", {}).get("seed", 42))
        seeds = [base_seed + i for i in range(req.n_replicates)]

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
    global _current_run, _event_queue
    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    token = _active_cancel_token
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
        # A stopped comparison reports the variants that did run, even with no
        # pair left to test: the researcher asked for the stop (ADR-111).
        stopped = job_was_stopped(
            [
                t["status"]
                for v in report.get("variants", {}).values()
                for t in v.get("trials", [])
            ],
            len(req.variants) * len(seeds),
            token,
        )
        report["stopped"] = stopped
        if not report["comparisons"] and not stopped:
            raise RuntimeError(
                "No variant produced enough successful replicates to compare "
                f"(skipped: {report['skipped_variants']})."
            )
        report["report_dir"] = _write_advanced_summary(
            base_config_dict, "comparison", report
        )
        _record_run_group(
            "replicated_comparison", run_id, base_config_dict, report, "comparison"
        )
        _current_run = {
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
        _current_run = {
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

    if req.seeds is not None:
        if len(set(req.seeds)) != len(req.seeds):
            raise HTTPException(422, tr("replicates.seeds_unique"))
        seeds = list(req.seeds)
    else:
        base_seed = int(complete.get("training", {}).get("seed", 42))
        seeds = [base_seed + i for i in range(req.n_replicates)]

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
    global _current_run, _event_queue
    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    token = _active_cancel_token
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
        stopped = job_was_stopped([t.status for t in trials], len(seeds), token)
        finished = any(t.status == "success" for t in trials)
        if not finished and not stopped:
            raise RuntimeError("All replicates failed — no aggregate available.")
        if finished:
            _require_reported_metric(metric, [t.metrics for t in trials], "replicate")
        report = _replicates_report(trials, seeds, metric)
        report["stopped"] = stopped
        report["report_dir"] = _write_advanced_summary(
            base_config_dict, "replicates", report
        )
        _record_run_group("replicates", run_id, base_config_dict, report, "replicates")
        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": None,
        }
        logger.success("GUI: Replicates {} completed ({} seeds).", run_id, len(trials))
    except Exception as e:
        logger.exception("GUI: Replicates {} failed", run_id)
        _current_run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(e).__name__}: {e or tr('run.no_message')}",
            "report": None,
            "run_dir": None,
        }
    finally:
        if queue is not None:
            await queue.put(None)


def _record_run_group(
    kind: Literal["replicates", "replicated_comparison", "kfold", "sweep"],
    run_id: str,
    config: dict[str, Any],
    report: dict[str, Any],
    report_name: str,
) -> None:
    """Give a finished replicate job or K-fold its own entry in History (ADR-113, ADR-123).

    The job's trainings are ordinary runs and its numbers sit in the report
    just written; this writes the ``run.json`` that ties them together, copied
    from that report. It is the last step of a job whose training and report
    are already done, so a failure here is logged rather than turned into a
    failed job: the researcher keeps a report and loses only the grouped entry.
    """
    models_dir = Path((config.get("output") or {}).get("models_dir") or _MODELS_DIR)
    try:
        write_run_group(
            kind=kind,
            group_id=run_id,
            name=str(config.get("name", report_name)),
            config=config,
            report=report,
            report_json=Path(report["report_dir"]) / f"{report_name}_summary.json",
            models_dir=models_dir,
        )
    # the report is on disk; do not fail the job
    except Exception:  # noqa: BLE001
        logger.exception("GUI: could not write the History entry of group {}", run_id)


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
    """Give a finished sweep or model comparison its History entry (ADR-124)."""
    # The validated config names the task (config.task); the raw dict may not.
    try:
        config = runner.config_type.model_validate(config_dict).model_dump(mode="json")
    # the entry still stands on the raw config
    except Exception:  # noqa: BLE001
        config = config_dict
    group_report = {
        "metric": report.get("metric"),
        "metric_direction": report.get("metric_direction"),
        "mode": mode,
        "n_planned": planned,
        "trials": trials,
        "best_index": best_index,
        "stopped": report.get("stopped", False),
        "report_dir": report.get("report_dir"),
    }
    _record_run_group("sweep", run_id, config, group_report, report_name)


def _replicates_report(
    trials: list[ReplicateTrial], seeds: list[int], metric: str
) -> dict[str, Any]:
    """Shape replicate trials + per-metric aggregates into a GUI-friendly report."""
    rows = [asdict(t) for t in trials]
    aggregates = aggregate_replicates(trials)
    return {
        "metric": metric,
        "seeds": seeds,
        "trials": rows,
        "aggregates": aggregates,
        "headline": aggregates.get(metric),
        "total_replicates": len(rows),
        "successful_replicates": sum(1 for r in rows if r["status"] == "success"),
    }


def _flatten_trial(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten a comparison/sweep trial row (nested metrics/overrides) for CSV."""
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if key in ("metrics", "overrides") and isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _write_advanced_summary(
    config_dict: dict[str, Any], kind: str, report: dict[str, Any]
) -> str:
    """Persist a comparison/sweep report (JSON + ranking CSV + LaTeX) to outputs/reports.

    Mirrors the classification ModelComparisonBlock artifact layout so a
    standalone-task comparison/sweep leaves the same durable record on disk
    instead of vanishing when the result view closes. The ``.tex`` table
    (ADR-061) is the paper-ready form: retyping numbers into a manuscript is
    where transcription errors enter, and nothing downstream catches them.
    """
    name = str(config_dict.get("name", kind))
    reports_dir = config_dict.get("output", {}).get("reports_dir") or "outputs/reports"
    out_dir = Path(reports_dir) / name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"{kind}_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    rows = report.get("trials") or report.get("fold_results") or []
    # Like the classification comparison's ranking.csv: only units that ran to
    # the end, so a cut unit's half-trained numbers never sit among finished
    # ones (ADR-111). The JSON summary above keeps every unit with its status.
    finished = [r for r in rows if r.get("status", "success") == "success"]
    flat = [_flatten_trial(r) for r in finished]
    if kind in ("comparison", "sweep"):
        # Already sorted by the metric, best first.
        flat = [{"rank": i, **row} for i, row in enumerate(flat, start=1)]
    if flat:
        fieldnames = list(dict.fromkeys(k for row in flat for k in row))
        with (out_dir / f"{kind}_ranking.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat)

    latex = report_to_latex(report, experiment=name)
    if latex:
        (out_dir / f"{kind}_table.tex").write_text(latex, encoding="utf-8")

    return str(out_dir)


__all__ = ["router", "_load_runs", "_sse_generator"]
