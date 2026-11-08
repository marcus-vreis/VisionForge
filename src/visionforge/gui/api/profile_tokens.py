"""One private token per profile and service, saved where only the server reads it.

The Hugging Face Hub (ADR-171) and Weights & Biases (ADR-172) keep their tokens
the same way: a file named by a hash of the profile, never sent back to the
page. The API only says whether one exists, and whether the server listens
beyond this machine, where whoever picks the profile can use it.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, SecretStr

from visionforge.gui.api.folder_opener import is_loopback_host
from visionforge.gui.api.profiles import PROFILE_HEADER, Profile
from visionforge.utils.credentials import config_dir
from visionforge.utils.messages import tr

ProfileHeader = Annotated[str | None, Header(alias=PROFILE_HEADER)]
Service = Literal["hub", "wandb"]
_MAX_TOKEN = 4096
_INVALID: dict[Service, str] = {
    "hub": "hub.token_invalid",
    "wandb": "wandb.token_invalid",
}


class ProfileToken(BaseModel):
    """Credential supplied only when the profile owner saves it."""

    token: SecretStr


def quiet_validation_route(message_key: str) -> type[APIRoute]:
    """A route class whose 422 says ``tr(message_key)`` instead of the input.

    FastAPI's default 422 echoes the request body, and here the body can hold a
    token.
    """

    class QuietRoute(APIRoute):
        def get_route_handler(
            self,
        ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handler = super().get_route_handler()

            async def handle(request: Request) -> Response:
                try:
                    return await handler(request)
                except RequestValidationError:
                    raise HTTPException(422, tr(message_key)) from None

            return handle

    return QuietRoute


def token_file(service: Service, profile: Profile) -> Path:
    """Where ``service``'s token for ``profile`` lives (the file may not exist)."""
    models = os.path.normcase(str(profile.models_dir.absolute()))
    key = hashlib.sha256(f"{profile.slug}\0{models}".encode()).hexdigest()
    return config_dir() / f"{service}-tokens" / key


def read_token(service: Service, profile: Profile) -> str | None:
    """The saved token, or ``None`` when the profile has none."""
    try:
        return token_file(service, profile).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def add_token_routes(
    router: APIRouter,
    service: Service,
    resolve: Callable[[str | None], Profile],
) -> None:
    """Mount ``GET``/``PUT``/``DELETE /token`` on ``router`` for ``service``."""

    def token_state(request: Request, header: str | None) -> dict[str, bool]:
        return {
            "configured": token_file(service, resolve(header)).is_file(),
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
        body: ProfileToken, request: Request, header: ProfileHeader = None
    ) -> dict[str, bool]:
        """Replace the profile credential atomically with private file permissions."""
        target = token_file(service, resolve(header))
        value = body.token.get_secret_value().strip()
        if not value or len(value) > _MAX_TOKEN or any(c.isspace() for c in value):
            raise HTTPException(422, tr(_INVALID[service]))
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            path = Path(tmp.name)
        try:
            # The file name is a SHA-256 hex digest under a fixed folder, and the
            # temporary file sits beside it: no request text reaches the path.
            path.write_text(value, encoding="utf-8")  # NOSONAR pythonsecurity:S2083
            path.replace(target)
        finally:
            path.unlink(missing_ok=True)
        return token_state(request, header)

    @router.delete("/token")
    def forget(request: Request, header: ProfileHeader = None) -> dict[str, bool]:
        """Forget only the current profile's token for this service."""
        token_file(service, resolve(header)).unlink(missing_ok=True)
        return token_state(request, header)
