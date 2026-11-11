"""Weights & Biases token of a profile and the History's retry (ADR-172)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from visionforge.core.wandb_tracking import WandbTarget, WandbTracker
from visionforge.gui.api.evidence_routes import EvidenceExports
from visionforge.gui.api.http_responses import (
    CONFLICT_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
    TAG_INTEGRATIONS,
)
from visionforge.gui.api.profile_tokens import (
    ProfileHeader,
    add_token_routes,
    quiet_validation_route,
    read_token,
)
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.run_history import RUN_FILENAME, RUN_NOT_FOUND_KEY
from visionforge.gui.api.runs_routes import read_wandb_status
from visionforge.gui.api.schemas import WandbStatus
from visionforge.utils.messages import tr

_UNAVAILABLE = "wandb.retry_unavailable"


def build_wandb_router(
    resolve: Callable[[str | None], Profile],
    find: Callable[[str, Profile], Path | None],
    record: Callable[[Path], dict[str, Any]],
    outputs: Callable[[], Path],
    exports: EvidenceExports,
) -> APIRouter:
    """Mount the W&B token routes and the retry of one finished run.

    ``record`` returns the run.json a retry may send (or raises the HTTP
    error), and ``outputs`` the folder that holds the W&B store.
    """
    router = APIRouter(
        prefix="/wandb",
        tags=[TAG_INTEGRATIONS],
        route_class=quiet_validation_route("wandb.request_invalid"),
    )
    add_token_routes(router, "wandb", resolve)

    @router.post(
        "/runs/{run_id}/retry",
        responses={
            404: {"description": NOT_FOUND_TEXT},
            409: {"description": CONFLICT_TEXT},
            500: {"description": SERVER_ERROR_TEXT},
        },
    )
    def retry(run_id: str, header: ProfileHeader = None) -> WandbStatus:
        """Send a finished run to the same W&B run again, with this profile's token."""
        profile = resolve(header)
        run_dir = find(run_id, profile)
        if (
            run_dir is None
            or not run_dir.resolve().is_relative_to(profile.models_dir.resolve())
            or (run_dir / RUN_FILENAME).is_symlink()
        ):
            raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))
        settings = profile.settings.wandb
        if not settings.enabled or not exports.reserve(run_dir):
            raise HTTPException(409, tr(_UNAVAILABLE))
        try:
            data = record(run_dir)
            token = read_token("wandb", profile)
            if token is None:
                raise HTTPException(409, tr("wandb.token_missing"))
            target = WandbTarget(settings.project, settings.entity, token)
            store = outputs() / "wandb"
            WandbTracker(store, profile.slug, run_dir, target).retry(data)
            result = read_wandb_status(run_dir)
            if result is None:
                raise HTTPException(500, tr(_UNAVAILABLE))
            return result
        finally:
            exports.release(run_dir)

    return router
