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
    TAG_INTEGRATIONS,
    UNPROCESSABLE_TEXT,
    VALIDATION_ERROR_CONTENT,
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


def _locate(
    find: Callable[[str, Profile], Path | None], run_id: str, profile: Profile
) -> Path:
    directory = find(run_id, profile)
    if directory is None:
        raise HTTPException(404, tr("hub.run_not_found"))
    return directory


def _preview(directory: Path) -> dict[str, Any]:
    try:
        return preview_run(directory)
    except FileNotFoundError as exc:
        raise HTTPException(404, tr(_ARTIFACT_MISSING)) from exc
    except ValueError as exc:
        raise HTTPException(422, tr(_PACKAGE_INVALID, reason=str(exc))) from exc


def _profile_token(profile: Profile) -> str:
    token = read_token("hub", profile)
    if token is None:
        raise HTTPException(409, tr("hub.token_missing"))
    return token


def _export(directory: Path, body: HubExport, token: str) -> dict[str, Any]:
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


def build_hub_router(
    resolve: Callable[[str | None], Profile],
    find: Callable[[str, Profile], Path | None],
) -> APIRouter:
    """Mount profile-scoped credentials and explicit private upload routes."""
    router = APIRouter(
        prefix="/hub",
        tags=[TAG_INTEGRATIONS],
        route_class=quiet_validation_route("hub.request_invalid"),
    )

    add_token_routes(router, "hub", resolve)

    @router.get(
        "/preview",
        responses={
            404: {"description": NOT_FOUND_TEXT},
            422: {
                "description": UNPROCESSABLE_TEXT,
                "content": VALIDATION_ERROR_CONTENT,
            },
        },
    )
    def preview(run_id: str, header: ProfileHeader = None) -> dict[str, Any]:
        """Inspect the selected package without contacting the Hub."""
        return _preview(_locate(find, run_id, resolve(header)))

    @router.post(
        "/export",
        responses={
            404: {"description": NOT_FOUND_TEXT},
            409: {"description": CONFLICT_TEXT},
            422: {
                "description": UNPROCESSABLE_TEXT,
                "content": VALIDATION_ERROR_CONTENT,
            },
            502: {"description": SERVER_ERROR_TEXT},
        },
    )
    def send(body: HubExport, header: ProfileHeader = None) -> dict[str, Any]:
        """Send the confirmed package using only the selected profile's token."""
        if not is_valid_repo_id(body.repo_id):
            raise HTTPException(422, tr("hub.repo_invalid"))
        profile = resolve(header)
        directory = _locate(find, body.run_id, profile)
        return _export(directory, body, _profile_token(profile))

    return router
