"""FastAPI server for VisionForge GUI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from visionforge import __version__
from visionforge.gui.api.routes import router as api_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="VisionForge", version=__version__)

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
