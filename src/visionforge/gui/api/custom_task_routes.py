"""The task list and the researcher's own tasks (#204, ADR-058, ADR-196).

``GET /api/tasks`` lists the five built-ins and every registered custom task;
``/api/custom/{key}/...`` hides, restores and deletes a custom task, serves its
form schema and starts a run of it. The run goes to the queue through
``submission._submit_job`` and runs in ``_execute_custom_task``, which writes
only its own job's state (ADR-185). A custom task's sweep, replicates and
paired comparison are with the other strategies in ``strategy_routes.py``.

``routes.py`` includes ``router`` and the run router where these routes used to
be defined, so the OpenAPI document and the route order do not change. A test
that replaces ``GenericTaskEngine``, ``_execute_custom_task`` or
``_get_custom_task_or_404`` patches this module.

No ``from __future__ import annotations``: ``ProfileDep`` is built inside the
builder, and FastAPI resolves string annotations against module globals only.
The engine is imported when a run starts, so importing this module stays light
(GUI-11, #202).
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from loguru import logger
from pydantic import ValidationError

from visionforge.gui.api import submission
from visionforge.gui.api.execution import _job_state
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    NOT_FOUND_TEXT,
    TAG_CUSTOM_TASKS,
    TAG_EXPERIMENTS,
)
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.schemas import (
    CustomTaskActionResponse,
    RunResponse,
    TaskDescriptor,
    TaskListResponse,
)
from visionforge.gui.api.submission import (
    _GPU_UNAVAILABLE_TEXT,
    ProfileResolver,
    profile_dependency,
)
from visionforge.tasks.base import TaskSpec
from visionforge.tasks.registry import get_task, registered_tasks
from visionforge.utils.messages import tr

if TYPE_CHECKING:
    from visionforge.tasks.engine import GenericTaskEngine as _GenericTaskEngine

# The engine imports torch; a run imports it when it starts, so starting the GUI
# loads none of it (GUI-11, #202). This stays a module attribute so a test can
# replace it here; resolved on first use by _generic_task_engine().
GenericTaskEngine: "type[_GenericTaskEngine] | None" = None


def _generic_task_engine() -> "type[_GenericTaskEngine]":
    """The custom-task engine class: a test's replacement, or the real one."""
    if GenericTaskEngine is not None:
        return GenericTaskEngine
    from visionforge.tasks.engine import GenericTaskEngine as engine

    return engine


router = APIRouter()


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


def build_custom_task_run_router(resolve_profile: ProfileResolver) -> APIRouter:
    """Mount ``POST /custom/{key}/run``; ``routes.py`` owns the profile lookup."""
    run_router = APIRouter()
    ProfileDep = Annotated[Profile, Depends(profile_dependency(resolve_profile))]

    @run_router.post(
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
        cfg = _validated_task_config(info, profile.scope_dict(config))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{cfg.name}_{timestamp}"

        return submission._submit_job(
            run_id,
            str(cfg.name),
            f"custom:{info.key}",
            "simple",
            lambda: _execute_custom_task(info, cfg, run_id),
            stop_at=None if _owns_its_loop(info) else "epoch",
            profile=profile,
        )

    return run_router


def _validated_task_config(info: Any, raw: dict[str, Any]) -> Any:
    """``raw`` validated against the task's own Config, or a 422."""
    try:
        return info.spec_cls.Config.model_validate(raw)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


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
