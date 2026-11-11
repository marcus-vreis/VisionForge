"""Handing a training job to the run queue (#204, ADR-196).

Every endpoint that starts a run goes through ``_submit_job``: it refuses at
once when the GPU context is broken, binds the page's language to the job, and
loads the training stack off the event loop before a local job starts. The
queue itself and every started job's state belong to ``execution.py``
(ADR-185); nothing here holds state.

A test that replaces ``_submit_job``, ``cuda_context_broken`` or a piece of the
training stack (``_TRAINING_STACK``, ``_DETECTION_STACK``, ``_TASK_STACK``,
``_import_modules``) patches this module: every caller looks them up here.
``profile_dependency`` builds the ``X-VF-Profile`` parameter of a split-out
router from the resolver ``routes.py`` owns.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Header, HTTPException
from loguru import logger

from visionforge.gui.api.execution import _LANES, _stop_point
from visionforge.gui.api.gpu_health import cuda_context_broken
from visionforge.gui.api.profiles import DEFAULT_SLUG, PROFILE_HEADER, Profile
from visionforge.gui.api.run_queue import LOCAL_LANE, QueuedJob, StopPoint
from visionforge.gui.api.schemas import RunResponse
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.messages import current_lang, tr, using_lang

# The 503 every submission documents: a broken CUDA context refuses every job.
_GPU_UNAVAILABLE_TEXT = "GPU context unavailable"

# How a split-out router resolves ``X-VF-Profile``: ``routes.current_profile``,
# which reads the profile folders when the request arrives (ADR-114).
ProfileResolver = Callable[[str | None], Profile]


def profile_dependency(resolve_profile: ProfileResolver) -> Callable[..., Profile]:
    """The dependency behind the ``profile`` parameter of a split-out route.

    It reads ``X-VF-Profile`` as ``routes.current_profile`` does and hands it to
    ``resolve_profile``, called per request.
    """

    def profile_dep(
        x_vf_profile: Annotated[str | None, Header(alias=PROFILE_HEADER)] = None,
    ) -> Profile:
        return resolve_profile(x_vf_profile)

    return profile_dep


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
    the config handed to ``start`` was already scoped to it by the caller. With
    none, the job is the default profile's, labelled by its slug (ADR-114).
    """
    if cuda_context_broken():
        # Every job would fail within a second, CPU ones included; say so now.
        raise HTTPException(503, tr("gpu.context_broken"))
    slug, name = (
        (profile.slug, profile.name) if profile else (DEFAULT_SLUG, DEFAULT_SLUG)
    )
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
            profile=slug,
            resume_dir=resume_dir,
            profile_name=name,
        ),
        lane,
    )
    return RunResponse(run_id=run_id, status=status)  # type: ignore[arg-type]
