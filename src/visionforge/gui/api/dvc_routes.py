"""A History run versioned with DVC in a local repository of experiments (ADR-176).

Every route writes to, or reads, a folder the request names on the server and
runs git and DVC there, so all of them answer only a client on this machine,
and refuse a page of another site (the same checks as "open folder").
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from visionforge.core.dvc_export import (
    DvcExportError,
    InvalidRepositoryPathError,
    RepositoryNotReadyError,
    ToolMissingError,
    Tools,
    export_run,
    find_tools,
    prepare_repository,
    preview_run,
)
from visionforge.core.hub_export import PreviewMismatchError
from visionforge.gui.api.folder_opener import is_loopback_host, is_loopback_origin
from visionforge.gui.api.http_responses import (
    CONFLICT_TEXT,
    FORBIDDEN_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
)
from visionforge.gui.api.profile_tokens import ProfileHeader, quiet_validation_route
from visionforge.gui.api.profiles import Profile
from visionforge.utils.messages import tr

_LOCAL_ONLY: dict[int | str, dict[str, Any]] = {403: {"description": FORBIDDEN_TEXT}}
_FAILURES: dict[int | str, dict[str, Any]] = {
    **_LOCAL_ONLY,
    404: {"description": NOT_FOUND_TEXT},
    409: {"description": CONFLICT_TEXT},
    500: {"description": SERVER_ERROR_TEXT},
}
_MISSING = {"git": "dvc.git_missing", "dvc": "dvc.dvc_missing"}
_FAILED = {"git": "dvc.git_failed", "dvc": "dvc.dvc_failed"}
_EXPORT_FAILURES = (FileNotFoundError, ValueError, ToolMissingError, DvcExportError)


class DvcRepository(BaseModel):
    """The folder of the repository of experiments."""

    repo: str


class DvcExport(BaseModel):
    """The previewed files and the repository the researcher confirmed."""

    run_id: str
    repo: str
    digest: str


def build_dvc_router(
    resolve: Callable[[str | None], Profile],
    find: Callable[[str, Profile], Path | None],
    client_host: Callable[[Request], str | None],
    tools: Callable[[], Tools] | None = None,
) -> APIRouter:
    """Mount the preview, the folder preparation and the export.

    ``tools`` finds git and DVC per request; by default, ``find_tools``.
    """
    current_tools = tools or (lambda: find_tools())
    router = APIRouter(
        prefix="/dvc",
        route_class=quiet_validation_route("dvc.request_invalid"),
        dependencies=[Depends(_local_only(client_host))],
    )

    def locate(run_id: str, header: str | None) -> Path:
        directory = find(run_id, resolve(header))
        if directory is None:
            raise HTTPException(404, tr("dvc.run_not_found"))
        return directory

    @router.get("/preview", responses=_FAILURES)
    def preview(run_id: str, repo: str, header: ProfileHeader = None) -> dict[str, Any]:
        """The folder, its files (git or DVC) and the repository's state."""
        directory = locate(run_id, header)
        try:
            return preview_run(directory, repo, current_tools())
        except (FileNotFoundError, ValueError) as exc:
            raise _http_error(exc) from exc

    @router.post("/prepare", responses=_FAILURES)
    def prepare(body: DvcRepository) -> dict[str, Any]:
        """Make the folder a git repository with DVC (git init, dvc init)."""
        try:
            return prepare_repository(body.repo, current_tools())
        except RepositoryNotReadyError as exc:
            raise HTTPException(409, tr("dvc.prepare_refused")) from exc
        except (ValueError, ToolMissingError, DvcExportError) as exc:
            raise _http_error(exc) from exc

    @router.post("/export", responses=_FAILURES)
    def send(body: DvcExport, header: ProfileHeader = None) -> dict[str, Any]:
        """Write the folder, dvc add the weights, commit it and push them."""
        directory = locate(body.run_id, header)
        try:
            return export_run(directory, body.repo, body.digest, current_tools())
        except _EXPORT_FAILURES as exc:
            raise _http_error(exc) from exc

    return router


def _local_only(
    client_host: Callable[[Request], str | None],
) -> Callable[[Request], None]:
    """A dependency that refuses another machine and a page of another site."""

    def check(request: Request) -> None:
        origin = request.headers.get("origin")
        if origin is not None and not is_loopback_origin(origin):
            raise HTTPException(403, tr("dvc.foreign_origin"))
        if not is_loopback_host(client_host(request)):
            raise HTTPException(403, tr("dvc.remote_client"))

    return check


def _http_error(exc: Exception) -> HTTPException:
    """The answer for a failure; git's and DVC's own output stays in the log."""
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, tr("dvc.artifact_missing"))
    if isinstance(exc, ToolMissingError):
        return HTTPException(409, tr(_MISSING[exc.tool]))
    if isinstance(exc, DvcExportError):
        return HTTPException(500, tr(_FAILED[exc.tool], step=exc.step))
    # These are ValueErrors: check them before the generic one.
    if isinstance(exc, InvalidRepositoryPathError):
        return HTTPException(422, tr("dvc.repo_invalid"))
    if isinstance(exc, RepositoryNotReadyError):
        return HTTPException(409, tr("dvc.repo_not_ready"))
    if isinstance(exc, PreviewMismatchError):
        return HTTPException(409, tr("dvc.preview_changed"))
    return HTTPException(422, tr("dvc.package_invalid", reason=str(exc)))
