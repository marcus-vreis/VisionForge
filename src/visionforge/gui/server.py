"""FastAPI server for VisionForge GUI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from visionforge import __version__
from visionforge.gui.api.routes import router as api_router

STATIC_DIR = Path(__file__).parent / "static"


def _read_spa_bundle() -> str:
    """Name of the SPA bundle `index.html` referenced when this process started.

    Static files are read from disk per request, so a rebuild reaches the browser
    immediately. Python modules are not: they are loaded once, at import. A
    server left running across a rebuild therefore serves *new* JavaScript from
    *old* Python — the SPA asks for fields the stale routes never send, and the
    feature silently does nothing. Capturing the bundle name here, at import,
    gives the SPA something to compare itself against and name the cause.
    """
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        return ""
    match = re.search(
        r"assets/(index-[A-Za-z0-9_-]+\.js)", index.read_text(encoding="utf-8")
    )
    return match.group(1) if match else ""


_BOOT_SPA_BUNDLE = _read_spa_bundle()

app = FastAPI(title="VisionForge", version=__version__)


@app.on_event("startup")
async def _sweep_filtered_datasets() -> None:
    """Remove filtered dataset copies a killed process left behind (ADR-084, ADR-130).

    The context manager cleans up after itself, including on exception, but
    nothing in a process that was killed outright ever runs again. A copy whose
    operating-system lease is still held belongs to another live process (a
    second GUI on the same workspace) and is left alone.
    """
    from visionforge.core.materialized_dataset import sweep_orphans

    sweep_orphans(Path("outputs/models/_filtered"))
    # A named profile keeps its filtered copies under its own models folder (ADR-114).
    for cache in sorted(Path("outputs/profiles").glob("*/models/_filtered")):
        sweep_orphans(cache)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Version and the SPA build this process booted against.

    `spa_bundle` is what lets the running page detect that the server predates
    it and needs restarting, rather than leaving the researcher to conclude the
    feature is broken.
    """
    return {"version": __version__, "spa_bundle": _BOOT_SPA_BUNDLE}


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],  # NOSONAR python:S5332 - local dev frontend only
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


# Vite fingerprints every asset filename, so a bundle can be cached forever —
# a new build produces a new name. index.html is the opposite: its name never
# changes and its whole job is to point at the current bundle. Letting a browser
# cache it serves yesterday's app from today's server, which has twice been
# mistaken for a broken build.
_INDEX_HEADERS = {"Cache-Control": "no-cache, must-revalidate"}


@app.get("/{path:path}")
async def spa_fallback(path: str) -> FileResponse:
    """Serve the React SPA — static files or index.html fallback."""
    # An unknown /api/... is a client error, not a deep link: answering 200 with
    # index.html made typos and stale clients fail as JSON parse errors.
    if path == "api" or path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    root = STATIC_DIR.resolve()
    # Treat Windows separators consistently even when the server runs on Linux.
    try:
        file_path = (root / path.replace("\\", "/")).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail="Not Found") from exc
    if not file_path.is_relative_to(root):
        raise HTTPException(status_code=404, detail="Not Found")
    if file_path.is_file() and file_path.name != "index.html":
        return FileResponse(file_path)  # NOSONAR python:S2083
    return FileResponse(  # NOSONAR python:S2083
        root / "index.html", headers=_INDEX_HEADERS
    )


# The running server, when this process started it through ``start_server``, so
# an update can stop it cleanly (ADR-120). None under ``uvicorn ...:app`` or tests.
_SERVER: Any = None


def start_server(
    *, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True
) -> None:
    """Start the VisionForge GUI server.

    ``open_browser`` is False when the updater restarts the server: the page the
    researcher had open reloads itself, and a second tab would only confuse.
    """
    import webbrowser

    import uvicorn

    global _SERVER
    logger.info(
        "Starting VisionForge GUI at http://{}:{}", host, port
    )  # NOSONAR python:S5332 - local server URL
    if open_browser:
        webbrowser.open(
            f"http://{host}:{port}"
        )  # NOSONAR python:S5332 - local server URL
    _SERVER = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )
    _SERVER.run()


def request_shutdown(grace_seconds: float = 1.0, hard_after: float = 15.0) -> None:
    """Stop this server soon, after the current response has gone out.

    A server started by ``start_server`` is asked to exit the way Ctrl+C does;
    any other (``uvicorn ...:app``) gets SIGINT. If it is still alive
    ``hard_after`` seconds later, the process exits outright: the updater is
    waiting for the port, and an update that never starts is worse than an abrupt
    stop with nothing running (the route refuses while a job is active).
    """
    import os
    import signal
    import threading

    def _stop() -> None:
        if _SERVER is not None:
            _SERVER.should_exit = True
        else:
            signal.raise_signal(signal.SIGINT)

    threading.Timer(grace_seconds, _stop).start()
    hard = threading.Timer(hard_after, lambda: os._exit(0))
    hard.daemon = True
    hard.start()


__all__ = ["app", "request_shutdown", "start_server"]
