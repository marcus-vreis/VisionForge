"""Server mode: the datasets in a person's own folder, listed and deleted (ADR-227).

On a server a person writes datasets only into their own folder (ADR-219),
and that folder counts against their disk quota (ADR-225). These two routes
let them see what is there and delete one, so a full folder is theirs to fix.
On a local install nobody has such a folder: the list is empty and there is
nothing to delete.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from visionforge.gui.api import execution
from visionforge.gui.api.dataset_scope import DatasetScope, current_scope
from visionforge.gui.api.http_responses import (
    CONFLICT_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
    TAG_DATASETS,
)
from visionforge.gui.disk_quota import folder_bytes
from visionforge.utils.messages import tr

router = APIRouter(tags=[TAG_DATASETS])


class OwnDataset(BaseModel):
    """One folder directly inside the person's dataset folder."""

    name: str
    path: str
    size_bytes: int


class OwnDatasetDeleted(BaseModel):
    """The dataset that is gone."""

    name: str
    status: str = "deleted"


class OwnDatasetsResponse(BaseModel):
    """The person's own datasets, by name; empty on a local install."""

    datasets: list[OwnDataset]


def _own_folder() -> tuple[DatasetScope, Path] | None:
    scope = current_scope()
    if scope is None or scope.own is None:
        return None
    return scope, scope.own


def _listed(own: Path) -> list[OwnDataset]:
    try:
        folders = sorted(
            (p for p in own.iterdir() if p.is_dir() and not p.is_symlink()),
            key=lambda p: p.name.lower(),
        )
    except OSError:
        return []
    return [
        OwnDataset(name=p.name, path=str(p), size_bytes=folder_bytes(p))
        for p in folders
    ]


@router.get("/datasets/own")
async def list_own_datasets() -> OwnDatasetsResponse:
    """The folders in the signed-in person's dataset folder, with their sizes."""
    found = _own_folder()
    if found is None:
        return OwnDatasetsResponse(datasets=[])
    return OwnDatasetsResponse(datasets=await asyncio.to_thread(_listed, found[1]))


def _has_jobs(owner: str) -> bool:
    """True while a job of ``owner`` runs or waits: it may be reading a dataset."""
    for lane in execution._LANES.snapshots():
        jobs = [lane.get("active"), *lane.get("pending", [])]
        if any(job and job.get("profile") == owner for job in jobs):
            return True
    return False


def _target(own: Path, name: str) -> Path | None:
    """``own/name`` when ``name`` is one real folder directly inside ``own``."""
    if not name or name in {".", ".."} or Path(name).name != name:
        return None
    if any(sep in name for sep in ("/", "\\", ":")):
        return None
    folder = own / name
    if folder.is_symlink() or not folder.is_dir():
        return None
    return folder if folder.resolve().parent == own.resolve() else None


@router.delete(
    "/datasets/own/{name}",
    responses={
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
        500: {"description": SERVER_ERROR_TEXT},
    },
)
async def delete_own_dataset(name: str) -> OwnDatasetDeleted:
    """Delete one dataset of the signed-in person's own folder, for good.

    Raises:
        HTTPException: 404 when there is no such folder directly inside the
            person's dataset folder (always, on a local install); 409 while a
            training of theirs runs or waits, since it may be reading it; 500
            when the OS refuses.
    """
    found = _own_folder()
    folder = _target(found[1], name) if found is not None else None
    if found is None or folder is None:
        raise HTTPException(404, tr("own_dataset.not_found", name=name[:80]))
    if _has_jobs(found[0].owner or ""):
        raise HTTPException(409, tr("own_dataset.jobs_running"))
    try:
        await asyncio.to_thread(shutil.rmtree, folder)
    except OSError as exc:
        logger.warning("GUI: could not delete the dataset {}: {}", folder, exc)
        reason = exc.strerror or type(exc).__name__
        raise HTTPException(
            500, tr("own_dataset.delete_failed", name=name[:80], reason=reason)
        ) from exc
    logger.info("GUI: deleted the dataset {}", folder)
    return OwnDatasetDeleted(name=folder.name)
