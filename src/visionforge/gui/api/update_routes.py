"""The update button's endpoints: ``GET /api/update`` and ``/api/update/apply`` (#204).

What they answer and do is ``updates.py``'s (ADR-120): whether a newer release
exists, whether this page may install it, and the updater that installs it in a
process of its own.

``routes.py`` includes the router where these routes used to be defined, so the
OpenAPI document and the route order do not change, and passes in how it reads
a request's client address. ``updates.py`` and the server are imported per
request, as before.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from visionforge.gui.api.execution import _LANES
from visionforge.gui.api.folder_opener import is_loopback_host, is_loopback_origin
from visionforge.gui.api.http_responses import (
    CONFLICT_TEXT,
    FORBIDDEN_TEXT,
    SERVER_ERROR_TEXT,
    TAG_SYSTEM,
)
from visionforge.utils.messages import tr

# The address a request came from, None when the transport has none.
ClientHost = Callable[[Request], "str | None"]

# Why the button cannot install, as the catalog words it (ADR-120).
_UPDATE_BLOCKED = {
    "source": "update.blocked_source",
    "no_installer": "update.blocked_no_installer",
    "remote": "update.blocked_remote",
    "busy": "update.blocked_busy",
}


def build_update_router(client_host: ClientHost) -> APIRouter:
    """Mount the update check and the update; ``routes.py`` reads the client."""
    router = APIRouter()

    @router.get("/update", tags=[TAG_SYSTEM])
    async def get_update(request: Request) -> dict[str, Any]:
        """Whether a newer release exists, and whether this page may install it (ADR-120).

        The PyPI lookup blocks for up to a few seconds when the network is slow, so it
        runs off the event loop; its answer is cached for hours.
        """
        from visionforge.gui.api.updates import update_status

        return await asyncio.to_thread(
            update_status,
            client_is_local=is_loopback_host(client_host(request)),
            busy=_LANES.is_busy(),
        )

    @router.post(
        "/update/apply",
        responses={
            403: {"description": FORBIDDEN_TEXT},
            409: {"description": CONFLICT_TEXT},
            500: {"description": SERVER_ERROR_TEXT},
        },
        tags=[TAG_SYSTEM],
    )
    async def apply_update(request: Request) -> dict[str, str]:
        """Install the announced release and restart the GUI (ADR-120).

        Refused for a page of another site and for a client on another machine, the
        same two checks as "open folder": reinstalling the server is the server
        owner's decision. Refused while a job runs or waits, which the restart would
        kill. The updater runs in a process of its own; this server exits right after
        answering, so the updater can replace its files and reuse its port.
        """
        from visionforge.gui.api.updates import spawn_updater, update_status
        from visionforge.gui.server import request_shutdown

        origin = request.headers.get("origin")
        if origin is not None and not is_loopback_origin(origin):
            raise HTTPException(403, tr("update.foreign_origin"))
        local = is_loopback_host(client_host(request))
        busy = _LANES.is_busy()
        status = await asyncio.to_thread(
            update_status, client_is_local=local, busy=busy
        )
        if not status["available"]:
            raise HTTPException(409, tr("update.none"))
        if not status["can_apply"]:
            raise HTTPException(
                403 if status["blocked"] == "remote" else 409,
                tr(_UPDATE_BLOCKED[status["blocked"]]),
            )
        host, port = request.scope.get("server") or ("127.0.0.1", 8000)
        try:
            spawn_updater(version=status["latest"], host=str(host), port=int(port))
        except (OSError, RuntimeError) as exc:
            raise HTTPException(500, tr("update.spawn_failed", error=exc)) from exc
        request_shutdown()
        return {"status": "restarting", "target": status["latest"]}

    return router
