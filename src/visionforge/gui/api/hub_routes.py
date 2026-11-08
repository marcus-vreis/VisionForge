"""Private Hugging Face Hub operations for the selected History profile."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, SecretStr

from visionforge.core.hub_export import (
    PreviewMismatchError,
    PrivateRepositoryRequiredError,
    export_run,
    is_valid_repo_id,
    preview_run,
)
from visionforge.gui.api.folder_opener import is_loopback_host
from visionforge.gui.api.http_responses import (
    CONFLICT_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
)
from visionforge.gui.api.profiles import PROFILE_HEADER, Profile
from visionforge.utils.credentials import config_dir
from visionforge.utils.messages import tr

ProfileHeader = Annotated[str | None, Header(alias=PROFILE_HEADER)]

_ARTIFACT_MISSING = "hub.artifact_missing"
_PACKAGE_INVALID = "hub.package_invalid"


class _HubRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError:
                # FastAPI's default 422 includes invalid input verbatim.
                raise HTTPException(422, tr("hub.request_invalid")) from None

        return handle


class HubToken(BaseModel):
    """Credential supplied only when the profile owner saves it."""

    token: SecretStr


class HubExport(BaseModel):
    """Destination and content confirmed by the researcher."""

    run_id: str
    repo_id: str
    digest: str


def _secret(profile: Profile) -> Path:
    identity = f"{profile.slug}\0{os.path.normcase(str(profile.models_dir.absolute()))}"
    key = hashlib.sha256(identity.encode()).hexdigest()
    return config_dir() / "hub-tokens" / key


def build_hub_router(
    resolve: Callable[[str | None], Profile],
    find: Callable[[str, Profile], Path | None],
) -> APIRouter:
    """Mount profile-scoped credentials and explicit private upload routes."""
    router = APIRouter(prefix="/hub", route_class=_HubRoute)

    def locate(run_id: str, profile: Profile) -> Path:
        directory = find(run_id, profile)
        if directory is None:
            raise HTTPException(404, tr("hub.run_not_found"))
        return directory

    def token_state(request: Request, header: str | None) -> dict[str, bool]:
        return {
            "configured": _secret(resolve(header)).is_file(),
            "shared": not is_loopback_host(
                getattr(request.app.state, "bind_host", None)
            ),
        }

    @router.get("/token")
    def status(request: Request, header: ProfileHeader = None) -> dict[str, bool]:
        """Report credential presence and whether other clients may use it."""
        return token_state(request, header)

    @router.put("/token")
    def save(
        body: HubToken, request: Request, header: ProfileHeader = None
    ) -> dict[str, bool]:
        """Replace the profile credential atomically with private file permissions."""
        target = _secret(resolve(header))
        value = body.token.get_secret_value().strip()
        if not value or len(value) > 4096 or any(c.isspace() for c in value):
            raise HTTPException(422, tr("hub.token_invalid"))
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            path = Path(tmp.name)
        try:
            path.write_text(value, encoding="utf-8")
            path.replace(target)
        finally:
            path.unlink(missing_ok=True)
        return token_state(request, header)

    @router.delete("/token")
    def forget(request: Request, header: ProfileHeader = None) -> dict[str, bool]:
        """Forget only the current profile's upload token."""
        _secret(resolve(header)).unlink(missing_ok=True)
        return token_state(request, header)

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
        try:
            token = _secret(profile).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise HTTPException(409, tr("hub.token_missing")) from exc
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
