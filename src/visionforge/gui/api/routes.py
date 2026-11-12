"""API endpoints for VisionForge GUI."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response

from visionforge.core.run_trackers import TrackerFactory
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
from visionforge.gui.api.comet_routes import build_comet_router
from visionforge.gui.api.custom_task_routes import build_custom_task_run_router
from visionforge.gui.api.custom_task_routes import router as custom_tasks_router
from visionforge.gui.api.dataset_tools_routes import router as dataset_tools_router
from visionforge.gui.api.datasets_routes import router as datasets_router
from visionforge.gui.api.dvc_routes import build_dvc_router
from visionforge.gui.api.evaluation_routes import (
    _reject_group_action,
    build_evaluation_router,
)
from visionforge.gui.api.evidence_routes import EvidenceExports, build_evidence_router
from visionforge.gui.api.execution import _LANES, _run_state
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
    TAG_EXPERIMENTS,
    TAG_RUNS,
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
from visionforge.gui.api.runs_routes import (
    build_runs_router,
    read_comet_status,
    read_mlflow_status,
    read_wandb_status,
)
from visionforge.gui.api.schemas import MlflowStatus, RunResponse
from visionforge.gui.api.strategy_routes import build_group_router, build_search_router
from visionforge.gui.api.submission import _GPU_UNAVAILABLE_TEXT
from visionforge.gui.api.system_routes import build_system_router
from visionforge.gui.api.tracker_exports import profile_trackers, retry_record
from visionforge.gui.api.training_routes import (
    build_experiment_run_router,
    build_task_run_router,
)
from visionforge.gui.api.update_routes import build_update_router
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
        lambda run_dir: retry_record(
            run_dir, read_wandb_status(run_dir), "wandb.retry_unavailable"
        ),
        lambda: _PROFILES_DIR.parent.resolve(),
        _EVIDENCE_EXPORTS,
    )
)
# Comet, on the same contract as W&B (ADR-200).
router.include_router(
    build_comet_router(
        lambda header: current_profile(header),
        lambda run_id, profile: _find_run_dir(run_id, profile),
        lambda run_dir: retry_record(
            run_dir, read_comet_status(run_dir), "comet.retry_unavailable"
        ),
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


# The update check and the update itself (ADR-120).
router.include_router(build_update_router(lambda request: _client_host(request)))


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
        data = retry_record(run_dir, read_mlflow_status(run_dir), _MLFLOW_UNAVAILABLE)
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


# The preprocessing preview and the guide's sample dataset (#204).
router.include_router(dataset_tools_router)


def _tracker_factories(
    config: ExperimentConfig, profile: Profile
) -> tuple[TrackerFactory, ...]:
    """The exporters this profile turned on, for a classification training only."""
    if config.block != "classification" or config.classification.mode != "train":
        return ()
    return _profile_trackers(profile)


def _profile_trackers(profile: Profile) -> tuple[TrackerFactory, ...]:
    """The exporters this profile turned on (ADR-170, ADR-172, ADR-197)."""
    return profile_trackers(profile, _PROFILES_DIR.parent.resolve())


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
router.include_router(
    build_task_run_router(lambda header: current_profile(header), _profile_trackers)
)


# Compare and sweep of the built-in tasks, the task list and the researcher's
# own tasks, then K-fold, replicates and the paired comparison (#204, ADR-196).
# Included where they used to be defined, so the route order does not change.
router.include_router(build_search_router(lambda header: current_profile(header)))
router.include_router(custom_tasks_router)
router.include_router(
    build_custom_task_run_router(
        lambda header: current_profile(header), _profile_trackers
    )
)
router.include_router(build_group_router(lambda header: current_profile(header)))


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


__all__ = ["router", "_load_runs"]
