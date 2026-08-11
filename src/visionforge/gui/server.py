"""FastAPI server for VisionForge GUI."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
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
    """Remove filtered dataset copies a killed process left behind (ADR-084).

    The context manager cleans up after itself, including on exception, but
    nothing in a process that was killed outright ever runs again. Every copy
    present at startup is by definition from a process that is already gone.
    """
    from visionforge.core.materialized_dataset import sweep_orphans

    sweep_orphans(Path("outputs/models/_filtered"))


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
    # STATIC_DIR is a fixed project-local directory; path is joined (not resolved
    # from user input alone), so traversal outside STATIC_DIR is not possible.
    file_path = STATIC_DIR / path
    if file_path.is_file() and file_path.name != "index.html":
        return FileResponse(file_path)  # NOSONAR python:S2083
    return FileResponse(  # NOSONAR python:S2083
        STATIC_DIR / "index.html", headers=_INDEX_HEADERS
    )


def start_server(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the VisionForge GUI server."""
    import webbrowser

    import uvicorn

    logger.info(
        "Starting VisionForge GUI at http://{}:{}", host, port
    )  # NOSONAR python:S5332 - local server URL
    webbrowser.open(f"http://{host}:{port}")  # NOSONAR python:S5332 - local server URL
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["app", "start_server"]
