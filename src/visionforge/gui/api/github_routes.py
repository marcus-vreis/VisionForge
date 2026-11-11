"""A History run committed, without weights, to a private GitHub repository (ADR-175)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from visionforge.core.github_export import (
    GitHubExportError,
    InvalidRepositoryError,
    export_run,
    is_valid_github_repo,
    preview_run,
)
from visionforge.core.hub_export import (
    PreviewMismatchError,
    PrivateRepositoryRequiredError,
)
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

# GitHub's status -> what the page says. 404 also means "no access": GitHub
# hides a private repository from a token that cannot read it.
_GITHUB_STATUS = {
    401: "github.token_rejected",
    403: "github.token_forbidden",
    404: "github.repo_not_found",
}


class GitHubExport(BaseModel):
    """The previewed files and the repository the researcher confirmed."""

    run_id: str
    repo: str
    digest: str


def build_github_router(
    resolve: Callable[[str | None], Profile],
    find: Callable[[str, Profile], Path | None],
) -> APIRouter:
    """Mount the profile token, the preview and the commit."""
    router = APIRouter(
        prefix="/github",
        tags=[TAG_INTEGRATIONS],
        route_class=quiet_validation_route("github.request_invalid"),
    )
    add_token_routes(router, "github", resolve)

    def locate(run_id: str, profile: Profile) -> Path:
        directory = find(run_id, profile)
        if directory is None:
            raise HTTPException(404, tr("github.run_not_found"))
        return directory

    @router.get("/preview", responses={404: {"description": NOT_FOUND_TEXT}})
    def preview(run_id: str, header: ProfileHeader = None) -> dict[str, Any]:
        """List the files and the README a commit would hold, offline."""
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
    def send(body: GitHubExport, header: ProfileHeader = None) -> dict[str, Any]:
        """Commit the confirmed files to the private repository."""
        if not is_valid_github_repo(body.repo):
            raise _http_error(InvalidRepositoryError(body.repo))
        profile = resolve(header)
        directory = locate(body.run_id, profile)
        token = read_token("github", profile)
        if token is None:
            raise HTTPException(409, tr("github.token_missing"))
        try:
            return export_run(directory, body.repo, token, body.digest)
        except (FileNotFoundError, ValueError, GitHubExportError) as exc:
            raise _http_error(exc) from exc

    return router


def _http_error(exc: Exception) -> HTTPException:
    """The answer for a preview or export failure, never GitHub's own text."""
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, tr("github.artifact_missing"))
    # These are ValueErrors: check them before the generic one.
    if isinstance(exc, InvalidRepositoryError):
        return HTTPException(422, tr("github.repo_invalid"))
    if isinstance(exc, PreviewMismatchError):
        return HTTPException(409, tr("github.preview_changed"))
    if isinstance(exc, PrivateRepositoryRequiredError):
        return HTTPException(409, tr("github.private_required"))
    if isinstance(exc, ValueError):
        return HTTPException(422, tr("github.package_invalid", reason=str(exc)))
    status = exc.status if isinstance(exc, GitHubExportError) else None
    key = _GITHUB_STATUS.get(status or 0, "github.export_failed")
    return HTTPException(502, tr(key))
