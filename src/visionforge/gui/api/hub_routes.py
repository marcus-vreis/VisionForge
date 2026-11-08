"""Private Hugging Face Hub operations for the selected History profile."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from visionforge.core.hub_export import (
    PreviewMismatchError,
    PrivateRepositoryRequiredError,
    export_run,
    is_valid_repo_id,
    preview_run,
)
from visionforge.gui.api.http_responses import (
    CONFLICT_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
)
from visionforge.gui.api.profile_tokens import (
    ProfileHeader,
    add_token_routes,
    quiet_validation_route,
    read_token,
)
from visionforge.gui.api.profiles import Profile
from visionforge.utils.messages import tr

_ARTIFACT_MISSING = "hub.artifact_missing"
_PACKAGE_INVALID = "hub.package_invalid"


class HubExport(BaseModel):
    """Destination and content confirmed by the researcher."""

    run_id: str
    repo_id: str
    digest: str


def build_hub_router(
    resolve: Callable[[str | None], Profile],
    find: Callable[[str, Profile], Path | None],
) -> APIRouter:
    """Mount profile-scoped credentials and explicit private upload routes."""
    router = APIRouter(
        prefix="/hub", route_class=quiet_validation_route("hub.request_invalid")
    )

    def locate(run_id: str, profile: Profile) -> Path:
        directory = find(run_id, profile)
        if directory is None:
            raise HTTPException(404, tr("hub.run_not_found"))
        return directory

    add_token_routes(router, "hub", resolve)

    @router.get(
        "/preview",
        responses={
            404: {"description": NOT_FOUND_TEXT},
        },
    )
    def preview(run_id: str, header: ProfileHeader = None) -> dict[str, Any]:
        """Inspect the selected package without contacting the Hub."""
        directory = locate(run_id, resolve(header))
        try:
            return preview_run(directory)
        except FileNotFoundError as exc:
            raise HTTPException(404, tr(_ARTIFACT_MISSING)) from exc
        except ValueError as exc:
            raise HTTPException(422, tr(_PACKAGE_INVALID, reason=str(exc))) from exc

    @router.post(
        "/export",
        responses={
            404: {"description": NOT_FOUND_TEXT},
            409: {"description": CONFLICT_TEXT},
            502: {"description": SERVER_ERROR_TEXT},
        },
    )
    def send(body: HubExport, header: ProfileHeader = None) -> dict[str, Any]:
        """Send the confirmed package using only the selected profile's token."""
        if not is_valid_repo_id(body.repo_id):
            raise HTTPException(422, tr("hub.repo_invalid"))
        profile = resolve(header)
        directory = locate(body.run_id, profile)
        token = read_token("hub", profile)
        if token is None:
            raise HTTPException(409, tr("hub.token_missing"))
        try:
            return export_run(directory, body.repo_id, token, body.digest)
        except FileNotFoundError as exc:
            raise HTTPException(404, tr(_ARTIFACT_MISSING)) from exc
        except PreviewMismatchError as exc:
            raise HTTPException(409, tr("hub.preview_changed")) from exc
        except PrivateRepositoryRequiredError as exc:
            raise HTTPException(409, tr("hub.private_required")) from exc
        except ValueError as exc:
            raise HTTPException(422, tr(_PACKAGE_INVALID, reason=str(exc))) from exc
        except RuntimeError as exc:
            raise HTTPException(502, tr("hub.export_failed")) from exc

    return router
