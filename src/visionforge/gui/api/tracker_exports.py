"""The experiment trackers a profile turned on, and the export of a task run.

Classification feeds its trackers epoch by epoch (ADR-170, ADR-172). The other
tasks have no tracker hooks: their finished ``run.json`` is sent once the job
ends (ADR-197).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from functools import partial
from pathlib import Path

from visionforge.core.run_trackers import TrackerFactory, export_run_record
from visionforge.gui.api import execution
from visionforge.gui.api.profile_tokens import read_token
from visionforge.gui.api.profiles import Profile


def profile_trackers(profile: Profile, outputs: Path) -> tuple[TrackerFactory, ...]:
    """The exporters this profile turned on, writing their stores under ``outputs``."""
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
    return tuple(factories)


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
