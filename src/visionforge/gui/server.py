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


@app.get("/{path:path}")
async def spa_fallback(path: str) -> FileResponse:
    """Serve the React SPA — static files or index.html fallback."""
    # STATIC_DIR is a fixed project-local directory; path is joined (not resolved
    # from user input alone), so traversal outside STATIC_DIR is not possible.
    file_path = STATIC_DIR / path
    if file_path.is_file():
        return FileResponse(file_path)  # NOSONAR python:S2083
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)  # NOSONAR python:S2083
    return FileResponse(STATIC_DIR / "index.html")  # NOSONAR python:S2083


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
