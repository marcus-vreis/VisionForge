"""Local image audits and contained run downloads for the browser."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.background import BackgroundTask

from visionforge.core.dataset_audit import audit_dataset
from visionforge.core.experiment_bundle import export_experiment_bundle
from visionforge.gui.api.profiles import PROFILE_HEADER, Profile
from visionforge.utils.messages import tr


class DatasetAuditRequest(BaseModel):
    """Image split paths and optional researcher-supplied grouping metadata."""

    model_config = ConfigDict(extra="forbid")
    base_dir: Path
    splits: dict[str, str] | None = None
    group_manifest: Path | None = None
    strict_groups: bool = Field(default=False)

    @field_validator("base_dir", "group_manifest", mode="before")
    @classmethod
    def nonempty_paths(cls, value: Any) -> Any:
        """Reject empty paths rather than auditing the server working directory."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("A nonempty path is required")
        return value


class EvidenceExports:
    """Reserve source directories while their archives are being created."""

    def __init__(self) -> None:
        self._paths: set[Path] = set()
        self._lock = Lock()

    def reserve(self, run_dir: Path) -> bool:
        """Acquire an exclusive archive reservation without waiting."""
        path = run_dir.resolve()
        with self._lock:
            if path in self._paths:
                return False
            self._paths.add(path)
            return True

    def release(self, run_dir: Path) -> None:
        """Release a completed or failed archive reservation."""
        with self._lock:
            self._paths.discard(run_dir.resolve())

    def is_exporting(self, run_dir: Path) -> bool:
        """Whether continuation must wait for a source archive to finish."""
        with self._lock:
            return run_dir.resolve() in self._paths


async def _finish_export(worker: asyncio.Future[Path]) -> None:
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(worker)
            break
        except asyncio.CancelledError as exc:
            if worker.cancelled():
                raise
            cancellation = cancellation or exc
        except Exception:
            if cancellation is not None:
                raise cancellation from None
            raise
    if cancellation is not None:
        raise cancellation


async def _bundle_response(run_dir: Path) -> FileResponse:
    temporary: TemporaryDirectory[str] | None = None
    try:
        temporary = TemporaryDirectory(prefix="visionforge-bundle-")
        destination = Path(temporary.name) / "experiment.zip"
        # Shield the executor Future itself: cancellation cannot close the
        # source lease or remove files while the writer still owns them.
        worker = asyncio.get_running_loop().run_in_executor(
            None, export_experiment_bundle, run_dir, destination
        )
        await _finish_export(worker)
        return FileResponse(
            destination,
            media_type="application/zip",
            filename="experiment.zip",
            background=BackgroundTask(temporary.cleanup),
        )
    except (OSError, ValueError) as exc:
        if temporary is not None:
            temporary.cleanup()
        raise HTTPException(400, tr("evidence.bundle_failed", error=exc)) from exc
    except BaseException:
        if temporary is not None:
            temporary.cleanup()
        raise


def build_evidence_router(
    resolve_run: Callable[[str, Profile], Path | None],
    resolve_profile: Callable[[str | None], Profile],
    is_writing: Callable[[str, Path, Profile], bool],
    exports: EvidenceExports,
) -> APIRouter:
    """Build endpoints with profile/run lookups owned by the parent router."""
    router = APIRouter()

    @router.post(
        "/dataset/audit", responses={400: {"description": "Dataset audit failed"}}
    )
    async def audit_images(req: DatasetAuditRequest) -> dict[str, Any]:
        """Read image splits without blocking status polling or changing data."""
        try:
            return await asyncio.to_thread(
                audit_dataset,
                req.base_dir,
                req.splits,
                group_manifest=req.group_manifest,
                strict_groups=req.strict_groups,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(400, tr("evidence.audit_failed", error=exc)) from exc

    @router.get(
        "/runs/{run_id}/bundle",
        response_class=FileResponse,
        responses={
            400: {"description": "Invalid profile or experiment bundle"},
            404: {"description": "Profile or run not found"},
            409: {"description": "Run is being written or exported"},
        },
    )
    async def download_bundle(
        run_id: str,
        x_vf_profile: Annotated[str | None, Header(alias=PROFILE_HEADER)] = None,
    ) -> FileResponse:
        """Download a contained record from the active profile and clean it up."""
        profile = resolve_profile(x_vf_profile)
        run_dir = resolve_run(run_id, profile)
        if run_dir is None:
            raise HTTPException(404, tr("run.not_found", run_id=run_id))
        if not exports.reserve(run_dir):
            raise HTTPException(409, tr("evidence.export_busy"))
        try:
            if is_writing(run_id, run_dir, profile):
                raise HTTPException(409, tr("evidence.run_busy"))
            return await _bundle_response(run_dir)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, tr("evidence.bundle_failed", error=exc)) from exc
        finally:
            exports.release(run_dir)

    return router
