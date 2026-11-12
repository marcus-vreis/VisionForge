"""Training a classification run on a Colab VM (#204, ADR-163, ADR-164, ADR-196).

``POST /api/colab/experiment/run`` queues the job on the lane of the Google
account the profile is logged into; ``POST /api/colab/drive/continue`` answers
the Drive consent prompt of a job waiting for it. ``_execute_colab`` runs the job
off the event loop, imports the run it brings back and writes only its own job's
state (ADR-185).

This module owns the Colab objects that live as long as the server: the CLI,
the logins waiting for a code, the jobs and the Drive consents. ``routes.py``
passes the first three to ``build_colab_router`` and includes this router where
these routes used to be defined, so the OpenAPI document and the route order do
not change. A test that replaces one of them, ``_execute_colab``,
``_drive_source`` or ``_colab_lane_of`` patches this module.

No ``from __future__ import annotations``: ``ProfileDep`` is built inside the
builder, and FastAPI resolves string annotations against module globals only.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from loguru import logger
from pydantic import ValidationError

from visionforge import __version__ as _VF_VERSION
from visionforge.core.cancellation import is_cancelled
from visionforge.core.experiment_bundle import import_experiment_bundle
from visionforge.gui.api import submission
from visionforge.gui.api.colab_connection import PINNED_VERSION as _COLAB_PIN
from visionforge.gui.api.colab_connection import ColabCli, LoginFlows
from visionforge.gui.api.colab_drive import (
    DriveConsent,
    count_files,
    my_drive_roots,
    remote_path,
)
from visionforge.gui.api.colab_jobs import ColabJobs
from visionforge.gui.api.execution import _job_state
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    NOT_FOUND_TEXT,
    TAG_COLAB,
    UNPROCESSABLE_TEXT,
    VALIDATION_ERROR_CONTENT,
)
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.run_queue import colab_lane
from visionforge.gui.api.schemas import (
    ColabDriveContinueRequest,
    ColabDriveSource,
    ColabRunRequest,
    RunResponse,
)
from visionforge.gui.api.submission import (
    _GPU_UNAVAILABLE_TEXT,
    ProfileResolver,
    profile_dependency,
)
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.messages import tr

# The Colab CLI and the logins waiting for a code live as long as the server (ADR-163).
_COLAB_CLI = ColabCli()
_COLAB_LOGINS = LoginFlows()
_COLAB_JOBS = ColabJobs()
# The Colab jobs waiting for (or able to wait for) the Drive consent, by run_id.
_DRIVE_CONSENTS: dict[str, DriveConsent] = {}
# How long the Colab job thread waits for the loop to take one event.
_COLAB_EVENT_TIMEOUT_S = 10.0


def build_colab_run_router(resolve_profile: ProfileResolver) -> APIRouter:
    """Mount the Colab run and the Drive consent answer; ``routes.py`` owns the lookup."""
    router = APIRouter()
    ProfileDep = Annotated[Profile, Depends(profile_dependency(resolve_profile))]

    @router.post(
        "/colab/experiment/run",
        responses={
            400: {"description": BAD_REQUEST_TEXT},
            404: {"description": NOT_FOUND_TEXT},
            409: {
                "description": "The Colab CLI is not installed or the profile is not logged in"
            },
            422: {
                "description": UNPROCESSABLE_TEXT,
                "content": VALIDATION_ERROR_CONTENT,
            },
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
        return submission._submit_job(
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

    return router


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
