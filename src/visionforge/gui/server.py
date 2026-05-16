"""FastAPI server for VisionForge GUI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from visionforge.gui.api.routes import router as api_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="VisionForge", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/{path:path}")
async def spa_fallback(path: str) -> FileResponse:
    """Serve the React SPA — static files or index.html fallback."""
    file_path = STATIC_DIR / path
    if file_path.is_file():
        return FileResponse(file_path)
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return FileResponse(STATIC_DIR / "index.html")


def start_server(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the VisionForge GUI server."""
    import webbrowser

    import uvicorn

    logger.info("Starting VisionForge GUI at http://{}:{}", host, port)
    webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["app", "start_server"]
