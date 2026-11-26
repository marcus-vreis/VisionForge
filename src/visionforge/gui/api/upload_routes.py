"""Server mode: a person sends a dataset of their own as a .zip, in parts (ADR-246).

On a shared server a person could bring data only by a provider download or
the sample dataset (ADR-219). These routes take a .zip from their browser:

- ``POST /api/uploads`` declares the folder name, the file and its size. The
  size is checked against ``--max-upload-gb`` and the quota, and the archive's
  file is made at that size at once, so the space counts from the start. It
  may ask ``delete_after_run``: the dataset goes after the run that uses it
  (``upload_cleanup``, ADR-249).
- ``PUT /api/uploads/{id}/parts/{n}`` writes part ``n`` (``PART_SIZE`` bytes,
  the last one shorter) at its place. Sending a part again rewrites the same
  bytes, so a broken connection is resumed by sending what is missing.
- ``GET /api/uploads/{id}`` says which parts arrived.
- ``POST /api/uploads/{id}/complete`` checks the archive (``upload_archive``),
  unpacks it into a folder beside it and moves that into the person's own
  dataset folder under the declared name; an existing folder is never
  replaced.
- ``DELETE /api/uploads/{id}`` drops it.

Everything is staged in the person's profile folder (``uploads/<id>``), so it
counts against their quota and nobody else can name it: an id of another
account is not found, the same answer as an id that never existed. An upload
untouched for ``EXPIRE_AFTER_S`` is removed the next time that person starts
one. On a local install nobody has an own folder and every route answers 404.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from visionforge.gui.api.dataset_scope import current_scope
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    CONFLICT_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
    TAG_DATASETS,
    UNPROCESSABLE_TEXT,
    VALIDATION_ERROR_CONTENT,
)
from visionforge.gui.api.own_datasets import OwnDataset
from visionforge.gui.api.upload_cleanup import set_mark
from visionforge.gui.disk_quota import folder_bytes, size_text
from visionforge.gui.server_mode import USAGE, ServerMode, mode_of
from visionforge.gui.upload_archive import ArchiveError, extract, inspect
from visionforge.utils.messages import current_lang, tr

router = APIRouter(tags=[TAG_DATASETS])

PART_SIZE = 8 * 1024 * 1024
EXPIRE_AFTER_S = 24 * 3600.0
# Uploads one person may have open at once; each holds its declared size.
MAX_OPEN = 3
# Unpacked, an archive may hold at most this many times the largest upload.
UNPACKED_FACTOR = 4
NAME_MAX = 120

TOO_LARGE_TEXT = "Upload larger than the server allows"
QUOTA_TEXT = "The person's folder has no room for it"

_STAGING = "uploads"
_RECORD = "upload.json"
_ARCHIVE = "archive.zip"
_RECEIVED = "received"
_UNPACKED = "unpacked"
_ID = re.compile(r"[0-9a-f]{32}")
# Characters no folder name may hold on any of the systems a server runs on.
_FORBIDDEN = frozenset('/\\:*?"<>|')
# The uploads being unpacked now; a part or a cancel for them waits.
_FINISHING: set[str] = set()


class UploadStart(BaseModel):
    """A .zip the person is about to send, and the folder it will become."""

    name: str = Field(
        min_length=1,
        max_length=NAME_MAX,
        description="The new folder's name inside the person's dataset folder.",
    )
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, description="The .zip's size in bytes.")
    delete_after_run: bool = Field(
        default=False,
        description="Delete the dataset once the next run that uses it ends.",
    )


class UploadStatus(BaseModel):
    """An open upload: what it will become and which parts arrived."""

    id: str
    name: str
    filename: str
    size: int
    part_size: int
    parts: int
    received: list[int]
    delete_after_run: bool = False


class UploadPart(BaseModel):
    """One part that arrived."""

    part: int
    size: int


class UploadCancelled(BaseModel):
    """The upload that was dropped."""

    id: str
    status: str = "cancelled"


@dataclass(frozen=True)
class Staged:
    """An upload's record and the folder that holds it."""

    id: str
    name: str
    filename: str
    size: int
    part_size: int
    folder: Path
    delete_after_run: bool = False

    @property
    def parts(self) -> int:
        return math.ceil(self.size / self.part_size)

    def length(self, part: int) -> int:
        """The bytes part ``part`` must carry: ``part_size``, the last one less."""
        return min(self.part_size, self.size - part * self.part_size)


def _person() -> tuple[Path, Path] | None:
    """``(own dataset folder, profile folder)`` of the signed-in person."""
    scope = current_scope()
    if scope is None or scope.own is None:
        return None
    return scope.own, scope.own.parent


def _not_found() -> HTTPException:
    return HTTPException(404, tr("upload.not_found"))


def plain_name(name: str) -> bool:
    """True when ``name`` is one folder name, the same on every system."""
    return (
        0 < len(name) <= NAME_MAX
        and name == name.strip()
        and not name.startswith(".")
        and not name.endswith(".")
        and all(ch.isprintable() and ch not in _FORBIDDEN for ch in name)
        and Path(name).name == name
    )


def _taken(own: Path, name: str) -> bool:
    folder = own / name
    return folder.exists() or folder.is_symlink()


def _read(folder: Path) -> Staged | None:
    """The record of the upload in ``folder``; None when absent or damaged."""
    try:
        raw = json.loads((folder / _RECORD).read_text(encoding="utf-8"))
        staged = Staged(
            id=str(raw["id"]),
            name=str(raw["name"]),
            filename=str(raw["filename"]),
            size=int(raw["size"]),
            part_size=int(raw["part_size"]),
            folder=folder,
            delete_after_run=raw.get("delete_after_run") is True,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if staged.id != folder.name or staged.size <= 0 or staged.part_size <= 0:
        return None
    return staged


def _idle_for(path: Path, now: float) -> float:
    try:
        return now - path.stat().st_mtime
    except OSError:
        return math.inf


def _discard(folder: Path) -> None:
    shutil.rmtree(folder, ignore_errors=True)


def _open_uploads(staging: Path) -> list[Staged]:
    """The person's live uploads; the expired and the damaged are removed."""
    if not staging.is_dir():
        return []
    now = time.time()
    live: list[Staged] = []
    for folder in staging.iterdir():
        if not folder.is_dir() or folder.is_symlink():
            continue
        staged = _read(folder)
        # A record written moments ago by a start still running is not damage.
        touched = folder if staged is None else folder / _RECORD
        if _idle_for(touched, now) > EXPIRE_AFTER_S:
            logger.info("GUI: removing the expired upload {}", folder)
            _discard(folder)
        elif staged is not None:
            live.append(staged)
    return live


def _find(upload_id: str) -> Staged:
    """The signed-in person's upload ``upload_id``, or a 404 for anyone else."""
    person = _person()
    if person is None or not _ID.fullmatch(upload_id):
        raise _not_found()
    folder = person[1] / _STAGING / upload_id
    staged = _read(folder)
    if staged is None:
        raise _not_found()
    if _idle_for(folder / _RECORD, time.time()) > EXPIRE_AFTER_S:
        _discard(folder)
        raise _not_found()
    return staged


def _received(staged: Staged) -> list[int]:
    try:
        names = [p.name for p in (staged.folder / _RECEIVED).iterdir()]
    except OSError:
        return []
    return sorted(int(n) for n in names if n.isdigit() and int(n) < staged.parts)


def _status(staged: Staged) -> UploadStatus:
    return UploadStatus(
        id=staged.id,
        name=staged.name,
        filename=staged.filename,
        size=staged.size,
        part_size=staged.part_size,
        parts=staged.parts,
        received=_received(staged),
        delete_after_run=staged.delete_after_run,
    )


def _write_record(staged: Staged) -> None:
    record = {
        "id": staged.id,
        "name": staged.name,
        "filename": staged.filename,
        "size": staged.size,
        "part_size": staged.part_size,
        "delete_after_run": staged.delete_after_run,
    }
    (staged.folder / _RECORD).write_text(json.dumps(record), encoding="utf-8")


def _create(staging: Path, req: UploadStart) -> Staged:
    """Make the upload's folder and its archive file at the declared size."""
    upload_id = secrets.token_hex(16)
    folder = staging / upload_id
    (folder / _RECEIVED).mkdir(parents=True)
    with (folder / _ARCHIVE).open("wb") as out:
        out.truncate(req.size)
    staged = Staged(
        id=upload_id,
        name=req.name,
        filename=req.filename,
        size=req.size,
        part_size=PART_SIZE,
        folder=folder,
        delete_after_run=req.delete_after_run,
    )
    _write_record(staged)
    return staged


def _write_part(staged: Staged, part: int, data: bytes) -> None:
    with (staged.folder / _ARCHIVE).open("r+b") as out:
        out.seek(part * staged.part_size)
        out.write(data)
    (staged.folder / _RECEIVED / str(part)).touch()
    # The last activity, for the expiry.
    os.utime(staged.folder / _RECORD)


def _sizes(**sizes: int) -> dict[str, str]:
    lang = current_lang()
    return {key: size_text(value, lang) for key, value in sizes.items()}


async def _check_quota(mode: ServerMode, root: Path, needed: int) -> None:
    """507 when ``needed`` more bytes do not fit under the person's ceiling."""
    if mode.quota_bytes is None:
        return
    used = await asyncio.to_thread(USAGE.of, root)
    if used + needed > mode.quota_bytes:
        raise HTTPException(
            507,
            tr(
                "upload.quota",
                **_sizes(used=used, limit=mode.quota_bytes, size=needed),
            ),
        )


@router.post(
    "/uploads",
    responses={
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
        413: {"description": TOO_LARGE_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        507: {"description": QUOTA_TEXT},
    },
)
async def start_upload(req: UploadStart, request: Request) -> UploadStatus:
    """Open an upload of a .zip that will become ``name`` in the person's folder.

    The same name, file and size as an upload already open answers that one,
    so a page that was reloaded resumes instead of starting over; it keeps the
    ``delete_after_run`` this request asks.

    Raises:
        HTTPException: 404 on a local install; 422 for a name that is not one
            plain folder name or a file that is not a .zip; 413 above
            ``--max-upload-gb``; 409 when the folder exists, another upload
            goes to it, or ``MAX_OPEN`` are open; 507 when it does not fit
            under the quota.
    """
    person = _person()
    if person is None:
        raise _not_found()
    own, root = person
    mode = mode_of(request.app)
    if not plain_name(req.name):
        raise HTTPException(422, tr("upload.bad_name"))
    if not req.filename.lower().endswith(".zip"):
        raise HTTPException(422, tr("upload.not_zip_name"))
    if req.size > mode.max_upload_bytes:
        raise HTTPException(
            413,
            tr("upload.too_big", **_sizes(size=req.size, limit=mode.max_upload_bytes)),
        )
    if _taken(own, req.name):
        raise HTTPException(409, tr("upload.exists", name=req.name))
    staging = root / _STAGING
    live = await asyncio.to_thread(_open_uploads, staging)
    for staged in live:
        if staged.name != req.name:
            continue
        if staged.filename == req.filename and staged.size == req.size:
            if staged.delete_after_run != req.delete_after_run:
                staged = dataclasses.replace(
                    staged, delete_after_run=req.delete_after_run
                )
                await asyncio.to_thread(_write_record, staged)
            return _status(staged)
        raise HTTPException(409, tr("upload.name_busy", name=req.name))
    if len(live) >= MAX_OPEN:
        raise HTTPException(409, tr("upload.too_many_open", count=MAX_OPEN))
    await _check_quota(mode, root, req.size)
    staged = await asyncio.to_thread(_create, staging, req)
    USAGE.forget(root)
    logger.info(
        "GUI: upload {} opened for {} ({} bytes)", staged.id, req.name, req.size
    )
    return _status(staged)


async def _body(request: Request, part: int, expected: int) -> bytes:
    """The request's bytes, read no further than one byte past ``expected``."""
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > expected:
            break
    if len(data) != expected:
        raise HTTPException(
            400,
            tr("upload.bad_part", part=part, expected=expected, received=len(data)),
        )
    return bytes(data)


@router.put(
    "/uploads/{upload_id}/parts/{part}",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
        507: {"description": QUOTA_TEXT},
    },
)
async def put_upload_part(upload_id: str, part: int, request: Request) -> UploadPart:
    """Write part ``part`` of the upload; sending it again rewrites the same place.

    Raises:
        HTTPException: 404 for an upload that is not the person's (or gone);
            400 for a part outside the upload or of the wrong length; 409
            while the upload is being unpacked; 507 when the person's folder
            went over the quota since the upload started.
    """
    staged = _find(upload_id)
    if not 0 <= part < staged.parts:
        raise HTTPException(400, tr("upload.no_part", part=part, last=staged.parts - 1))
    if upload_id in _FINISHING:
        raise HTTPException(409, tr("upload.busy"))
    mode = mode_of(request.app)
    if mode.quota_bytes is not None:
        # The declared size already counts (the archive is made at that size):
        # a folder over the ceiling now grew from something else meanwhile.
        root = staged.folder.parent.parent
        used = await asyncio.to_thread(USAGE.of, root)
        if used > mode.quota_bytes:
            raise HTTPException(
                507,
                tr("server.quota_full", **_sizes(used=used, limit=mode.quota_bytes)),
            )
    expected = staged.length(part)
    data = await _body(request, part, expected)
    try:
        await asyncio.to_thread(_write_part, staged, part, data)
    except FileNotFoundError as exc:
        # Cancelled while this part was on its way.
        raise _not_found() from exc
    return UploadPart(part=part, size=expected)


@router.get("/uploads/{upload_id}", responses={404: {"description": NOT_FOUND_TEXT}})
async def upload_status(upload_id: str) -> UploadStatus:
    """Which parts of the upload arrived, to resume after a failure.

    Raises:
        HTTPException: 404 for an upload that is not the person's, or gone.
    """
    return _status(_find(upload_id))


def _move(source: Path, target: Path) -> None:
    """Rename ``source`` to ``target``; never over something already there."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    os.rename(source, target)


async def _unpack(staged: Staged, own: Path, mode: ServerMode) -> Path:
    """Check and unpack the archive into ``own/<name>``; the new folder."""
    archive = staged.folder / _ARCHIVE
    unpacked = staged.folder / _UNPACKED
    max_unpacked = UNPACKED_FACTOR * mode.max_upload_bytes
    target = own / staged.name
    try:
        checked = await asyncio.to_thread(inspect, archive, max_unpacked)
        # Unpacking needs the archive's room and its contents' at once.
        USAGE.forget(own.parent)
        await _check_quota(mode, own.parent, checked.unpacked_bytes)
        await asyncio.to_thread(_discard, unpacked)
        dataset = await asyncio.to_thread(extract, archive, unpacked, max_unpacked)
        await asyncio.to_thread(_move, dataset, target)
    except ArchiveError as exc:
        # The bytes can never make a dataset: the person sends another file.
        await asyncio.to_thread(_discard, staged.folder)
        raise HTTPException(422, tr(exc.key, **exc.params)) from exc
    except FileExistsError as exc:
        await asyncio.to_thread(_discard, unpacked)
        raise HTTPException(409, tr("upload.exists", name=staged.name)) from exc
    except OSError as exc:
        await asyncio.to_thread(_discard, unpacked)
        logger.warning("GUI: could not unpack the upload {}: {}", staged.id, exc)
        reason = exc.strerror or type(exc).__name__
        raise HTTPException(500, tr("upload.failed", reason=reason)) from exc
    return target


@router.post(
    "/uploads/{upload_id}/complete",
    responses={
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
        500: {"description": SERVER_ERROR_TEXT},
        507: {"description": QUOTA_TEXT},
    },
)
async def complete_upload(upload_id: str, request: Request) -> OwnDataset:
    """Unpack the finished upload into the person's own dataset folder.

    Raises:
        HTTPException: 404 for an upload that is not the person's, or gone;
            409 while parts are missing, while it is already being unpacked,
            or when the folder exists by now; 422 when the archive is refused
            (the upload is dropped: its bytes can never make a dataset); 507
            when the unpacked dataset does not fit under the quota (the
            upload is kept, to complete again after freeing space); 500 when
            the OS refuses.
    """
    staged = _find(upload_id)
    person = _person()
    if person is None:
        raise _not_found()
    own = person[0]
    missing = staged.parts - len(_received(staged))
    if missing:
        raise HTTPException(409, tr("upload.missing_parts", count=missing))
    if upload_id in _FINISHING:
        raise HTTPException(409, tr("upload.busy"))
    if _taken(own, staged.name):
        raise HTTPException(409, tr("upload.exists", name=staged.name))
    _FINISHING.add(upload_id)
    try:
        target = await _unpack(staged, own, mode_of(request.app))
    finally:
        _FINISHING.discard(upload_id)
    await asyncio.to_thread(_discard, staged.folder)
    USAGE.forget(own.parent)
    # A mark left by an earlier dataset of the same name is cleared too.
    await asyncio.to_thread(set_mark, own, target.name, staged.delete_after_run)
    logger.info(
        "GUI: upload {} unpacked into {} (delete after run: {})",
        upload_id,
        target,
        staged.delete_after_run,
    )
    size = await asyncio.to_thread(folder_bytes, target)
    return OwnDataset(
        name=target.name,
        path=str(target),
        size_bytes=size,
        delete_after_run=staged.delete_after_run,
    )


@router.delete(
    "/uploads/{upload_id}",
    responses={
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def cancel_upload(upload_id: str) -> UploadCancelled:
    """Drop an upload and every part of it.

    Raises:
        HTTPException: 404 for an upload that is not the person's, or gone;
            409 while it is being unpacked.
    """
    staged = _find(upload_id)
    if upload_id in _FINISHING:
        raise HTTPException(409, tr("upload.busy"))
    await asyncio.to_thread(_discard, staged.folder)
    logger.info("GUI: upload {} cancelled", upload_id)
    return UploadCancelled(id=upload_id)
