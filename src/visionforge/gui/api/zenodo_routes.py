"""Zenodo drafts for the selected History profile (ADR-174).

``/api/zenodo/token`` holds the profile's zenodo.org token and
``/api/zenodo/sandbox/token`` its sandbox.zenodo.org one: the two sites have
separate accounts, so a token from one never reaches the other.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from visionforge.core.hub_export import PreviewMismatchError
from visionforge.core.zenodo_export import ZenodoExportError, export_run, preview_run
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
from visionforge.utils.messages import tr

_ARTIFACT_MISSING = "zenodo.artifact_missing"
_PACKAGE_INVALID = "zenodo.package_invalid"
_REQUEST_INVALID = "zenodo.request_invalid"
# Zenodo's answer to a token it does not know (401) or one without deposit:write (403).
_REJECTED = {401, 403}


class ZenodoExport(BaseModel):
    """The previewed bundle the researcher confirmed, and which Zenodo gets it."""

    run_id: str
    digest: str
    sandbox: bool = False


def build_zenodo_router(
    resolve: Callable[[str | None], Profile],
    find: Callable[[str, Profile], Path | None],
) -> APIRouter:
    """Mount the profile tokens, the preview and the draft upload."""
    router = APIRouter(
        prefix="/zenodo",
        tags=[TAG_INTEGRATIONS],
        route_class=quiet_validation_route(_REQUEST_INVALID),
    )
    sandbox = APIRouter(
        prefix="/sandbox", route_class=quiet_validation_route(_REQUEST_INVALID)
    )
    add_token_routes(router, "zenodo", resolve)
    add_token_routes(sandbox, "zenodo-sandbox", resolve)
    router.include_router(sandbox)

    def locate(run_id: str, profile: Profile) -> Path:
        directory = find(run_id, profile)
        if directory is None:
            raise HTTPException(404, tr("zenodo.run_not_found"))
        return directory

    @router.get("/preview", responses={404: {"description": NOT_FOUND_TEXT}})
    def preview(run_id: str, header: ProfileHeader = None) -> dict[str, Any]:
        """List the bundle a draft would receive, without contacting Zenodo."""
        directory = locate(run_id, resolve(header))
        try:
            return preview_run(directory)
        except (FileNotFoundError, ValueError) as exc:
            raise _http_error(exc) from exc

    @router.post(
        "/export",
        responses={
            404: {"description": NOT_FOUND_TEXT},
            409: {"description": CONFLICT_TEXT},
            502: {"description": SERVER_ERROR_TEXT},
        },
    )
    def send(body: ZenodoExport, header: ProfileHeader = None) -> dict[str, Any]:
        """Create an unpublished draft with the confirmed bundle."""
        profile = resolve(header)
        directory = locate(body.run_id, profile)
        token = read_token("zenodo-sandbox" if body.sandbox else "zenodo", profile)
        if token is None:
            raise HTTPException(409, tr("zenodo.token_missing"))
        try:
            return export_run(directory, token, body.digest, sandbox=body.sandbox)
        except (FileNotFoundError, ValueError, ZenodoExportError) as exc:
            raise _http_error(exc) from exc

    return router


def _http_error(exc: Exception) -> HTTPException:
    """The answer for a preview or export failure, never Zenodo's own text."""
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, tr(_ARTIFACT_MISSING))
    # Before ValueError: a stale preview is a ValueError too.
    if isinstance(exc, PreviewMismatchError):
        return HTTPException(409, tr("zenodo.preview_changed"))
    if isinstance(exc, ValueError):
        return HTTPException(422, tr(_PACKAGE_INVALID, reason=str(exc)))
    if isinstance(exc, ZenodoExportError) and exc.status in _REJECTED:
        return HTTPException(502, tr("zenodo.token_rejected"))
    return HTTPException(502, tr("zenodo.export_failed"))
