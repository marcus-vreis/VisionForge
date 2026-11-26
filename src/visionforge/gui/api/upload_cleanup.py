"""Server mode: a dataset sent to be deleted after the run that uses it (ADR-249).

A person who sends a dataset (ADR-246) may ask, when the upload starts, that it
be deleted once trained on: a dataset used for one run should not hold their
quota until they remember it. The choice is a marker beside their dataset
folder, never inside it, so no loader reads it as data:
``<profile>/delete_after_run/<name>``.

The request that queues a job notes which of the person's own datasets its
config reads (``config_scope.take_reads``). When the job ends, completed,
failed or stopped, :func:`clean_after` deletes each marked one that no other
running or waiting job still names. A strategy (K-fold, search, comparison) is
one job, so its dataset goes when the whole strategy ends. The deletion is
logged and returned as entries the caller writes into the run's ``run.json``
(``dataset_cleanup``); a deletion that fails is an entry that says so, never an
error of the job.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import shutil
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from visionforge.core.experiment_bundle import RUN_FILENAME
from visionforge.core.strict_json import dumps_strict
from visionforge.gui.api.run_queue import QueuedJob
from visionforge.gui.disk_quota import folder_bytes
from visionforge.gui.server_mode import USAGE

# Beside the person's dataset folder, inside their profile.
MARKS = "delete_after_run"
# The key of run.json that lists what was deleted after the run.
RUN_KEY = "dataset_cleanup"


def _mark(own: Path, name: str) -> Path:
    return own.parent / MARKS / name


def set_mark(own: Path, name: str, delete_after_run: bool) -> None:
    """Mark ``own/name`` to be deleted after its next run, or clear the mark."""
    mark = _mark(own, name)
    if delete_after_run:
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.touch()
    else:
        mark.unlink(missing_ok=True)


def is_marked(own: Path, name: str) -> bool:
    """True when ``own/name`` goes after the next run that uses it."""
    return _mark(own, name).is_file()


def own_folders_read(reads: Iterable[Path], own: Path | None) -> tuple[Path, ...]:
    """The folders directly inside ``own`` that the resolved ``reads`` lie in."""
    if own is None:
        return ()
    found: dict[Path, None] = {}
    for path in reads:
        if path != own and path.is_relative_to(own):
            found[own / path.relative_to(own).parts[0]] = None
    return tuple(found)


def _entry(folder: Path, size: int, error: str | None) -> dict[str, Any]:
    return {
        "dataset": folder.name,
        # Relative to the profile: the person's view of their own folder.
        "path": folder.relative_to(folder.parent.parent).as_posix(),
        "size_bytes": size,
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "failed" if error else "deleted",
        "error": error,
    }


def _delete(folder: Path) -> dict[str, Any]:
    """Delete ``folder`` and its mark; the entry that says how it went.

    The folder is first renamed to a hidden name: from that moment no new run
    can name it, and a deletion cut halfway leaves no half dataset under the
    old name, only a folder the person sees and can delete.
    """
    own = folder.parent
    size = folder_bytes(folder)
    try:
        gone = own / f".{folder.name}.deleting-{secrets.token_hex(4)}"
        os.rename(folder, gone)
    except OSError as exc:
        # The dataset is intact and still marked: the next run tries again.
        logger.warning("GUI: could not delete the dataset {}: {}", folder, exc)
        return _entry(folder, size, exc.strerror or type(exc).__name__)
    _mark(own, folder.name).unlink(missing_ok=True)
    try:
        shutil.rmtree(gone)
    except OSError as exc:
        logger.warning("GUI: could not delete the dataset {}: {}", folder, exc)
        return _entry(folder, size, exc.strerror or type(exc).__name__)
    finally:
        USAGE.forget(own.parent)
    logger.info("GUI: deleted the dataset {} after its run ({} bytes)", folder, size)
    return _entry(folder, size, None)


def _run_json(job: QueuedJob, run_dir: str | None) -> Path | None:
    """The ``run.json`` the job left: its run folder's, else the newest it wrote.

    A K-fold, a search or a comparison names no single run folder; the group
    ``run.json`` it writes last is the newest under the models folder.
    """
    if job.models_dir is None:
        return None
    if run_dir:
        # The folder may be named by the job's child process (ADR-248): it is
        # believed only inside the models folder the job was given.
        named = (Path(run_dir) / RUN_FILENAME).resolve()
        if named.is_relative_to(job.models_dir.resolve()) and named.is_file():
            return named
    if job.started_at is None:
        return None
    newest: tuple[float, Path] | None = None
    for path in job.models_dir.rglob(RUN_FILENAME):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= job.started_at and (newest is None or mtime > newest[0]):
            newest = (mtime, path)
    return newest[1] if newest is not None else None


def _record(path: Path, entries: list[dict[str, Any]]) -> None:
    """Add ``entries`` to the ``dataset_cleanup`` list of the run.json ``path``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("GUI: could not read {} to record a deletion: {}", path, exc)
        return
    if not isinstance(data, dict):
        return
    before = data.get(RUN_KEY)
    data[RUN_KEY] = [*(before if isinstance(before, list) else []), *entries]
    tmp = path.with_name(path.name + ".tmp")
    try:
        # The path is a run.json found inside the job's own models folder
        # (`_run_json`); what the file held reaches the content, not the path.
        text = dumps_strict(data)
        tmp.write_text(text, encoding="utf-8")  # NOSONAR pythonsecurity:S2083
        tmp.replace(path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        logger.warning("GUI: could not record a deletion in {}: {}", path, exc)


def _in_use(folder: Path, others: Iterable[QueuedJob]) -> bool:
    return any(folder in job.datasets for job in others)


async def clean_after(
    job: QueuedJob, others: Iterable[QueuedJob], run_dir: str | None
) -> list[dict[str, Any]]:
    """Delete the marked datasets ``job`` read that no job in ``others`` names.

    Returns the entries, also written into the job's ``run.json`` when it left
    one. Nothing here raises: a failure is an entry with its reason.
    """
    waiting = list(others)
    entries: list[dict[str, Any]] = []
    for folder in job.datasets:
        if not is_marked(folder.parent, folder.name) or not folder.is_dir():
            continue
        if _in_use(folder, waiting):
            logger.info(
                "GUI: keeping the dataset {} after {}: another job uses it",
                folder,
                job.run_id,
            )
            continue
        entries.append(await asyncio.to_thread(_delete, folder))
    if entries:
        target = await asyncio.to_thread(_run_json, job, run_dir)
        if target is not None:
            await asyncio.to_thread(_record, target, entries)
    return entries


__all__ = [
    "MARKS",
    "RUN_KEY",
    "clean_after",
    "is_marked",
    "own_folders_read",
    "set_mark",
]
