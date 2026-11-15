"""The experiment trackers a profile turned on, and the export of a task run.

Classification feeds its trackers epoch by epoch (ADR-170, ADR-172, ADR-200).
The other tasks have no tracker hooks: their finished ``run.json`` is sent once
the job ends (ADR-197). The History's retries read the record back here too.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from fastapi import HTTPException

from visionforge.core.run_trackers import TrackerFactory, export_run_record
from visionforge.gui.api import execution
from visionforge.gui.api.profile_tokens import read_token
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.run_history import RUN_BAD_JSON_KEY, RUN_FILENAME
from visionforge.utils.messages import tr


class ExportSidecar(Protocol):
    """What a retry reads from an exporter's sidecar."""

    @property
    def status(self) -> str:
        """``RUNNING`` while the export is open, else its terminal status."""


def profile_trackers(profile: Profile, outputs: Path) -> tuple[TrackerFactory, ...]:
    """The exporters this profile turned on, writing their stores under ``outputs``."""
    from visionforge.core.comet_tracking import CometTarget, CometTracker
    from visionforge.core.mlflow_tracking import MlflowTracker
    from visionforge.core.wandb_tracking import WandbTarget, WandbTracker

    factories: list[TrackerFactory] = []
    if profile.settings.mlflow.enabled:
        factories.append(partial(MlflowTracker, outputs / "mlflow", profile.slug))
    wandb = profile.settings.wandb
    if wandb.enabled:
        target = WandbTarget(wandb.project, wandb.entity, read_token("wandb", profile))
        factories.append(
            partial(WandbTracker, outputs / "wandb", profile.slug, target=target)
        )
    comet = profile.settings.comet
    if comet.enabled:
        token = read_token("comet", profile)
        destination = CometTarget(comet.project, comet.workspace, token)
        factories.append(partial(CometTracker, profile.slug, target=destination))
    return tuple(factories)


def retry_record(
    run_dir: Path, status: ExportSidecar | None, unavailable: str
) -> dict[str, Any]:
    """The run.json of a finished, exported run, or 409/500.

    ``status`` is the exporter's sidecar: a retry needs one that is not still
    running; ``unavailable`` is the message key of the 409.
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


async def exported(
    job: Awaitable[None], run_id: str, trackers: tuple[TrackerFactory, ...]
) -> None:
    """Run a task's executor, then send its ``run.json`` when the run completed.

    The executor has already ended the live events; a run that raised has no
    folder to send.
    """
    await job
    run = execution._run_state(run_id) or {}
    run_dir = run.get("run_dir")
    if trackers and run.get("status") == "completed" and run_dir:
        await asyncio.to_thread(export_run_record, trackers, Path(run_dir))
