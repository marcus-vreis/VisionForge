"""A started job's status, event stream and result, and the queue (ADR-185).

The state these routes read is owned by ``execution.py``; they read it through
its functions and command the lanes through their methods, so no route here
writes a job's state (ADR-132).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from visionforge.gui.api.execution import _LANES, _run_state, _sse_generator
from visionforge.gui.api.http_responses import (
    CONFLICT_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
    TAG_EXPERIMENTS,
)
from visionforge.gui.api.run_history import RUN_FILENAME, RUN_NOT_FOUND_KEY
from visionforge.gui.api.run_queue import NotStoppableError
from visionforge.gui.api.schemas import (
    QueuedJobInfo,
    QueueLane,
    QueueSnapshot,
    RunResult,
    RunStatus,
)
from visionforge.utils.messages import tr

router = APIRouter()


@router.get("/experiment/events", tags=[TAG_EXPERIMENTS])
async def experiment_events(
    run_id: str | None = None, after: Annotated[int, Query(ge=0)] = 0
) -> StreamingResponse:
    """Stream a run's progress as Server-Sent Events, from its start (ADR-159).

    ``after`` skips the events the client already has, for a stream that dropped.
    ``run_id`` names the run the client follows. Each job has its own log
    (ADR-163, phase 4), so a local run and a Colab run stream side by side; with
    no ``run_id`` the stream is the job started last. An id with no log here
    closes at once rather than serving another job's events as its own.
    """
    return StreamingResponse(
        _sse_generator(run_id, after),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/experiment/status", tags=[TAG_EXPERIMENTS])
async def get_status(run_id: str | None = None) -> RunStatus:
    """Poll a run's status, plus how many jobs wait over all queues.

    ``run_id`` names the run; without it, the job started last (ADR-163, phase 4).
    """
    queued = _LANES.pending_count()
    run = _run_state(run_id)
    if run is None and run_id is not None:
        run = _LANES.record(run_id)
    if run is None:
        return RunStatus(status="idle", queued=queued)

    return RunStatus(
        status=run["status"],
        run_id=run["run_id"],
        error=run.get("error"),
        queued=queued,
    )


@router.get("/queue", tags=[TAG_EXPERIMENTS])
async def get_queue() -> QueueSnapshot:
    """Every queue: its running job and what waits behind it, in submission order.

    ``active`` and ``pending`` stay the local queue's, as before ADR-163 phase 4;
    ``lanes`` lists the local queue first and then one per Google account.
    """
    lanes = [
        QueueLane(
            lane=snap["lane"],
            active=QueuedJobInfo(**snap["active"]) if snap["active"] else None,
            pending=[QueuedJobInfo(**job) for job in snap["pending"]],
        )
        for snap in _LANES.snapshots()
    ]
    return QueueSnapshot(active=lanes[0].active, pending=lanes[0].pending, lanes=lanes)


@router.delete(
    "/queue/{run_id}",
    responses={
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
    tags=[TAG_EXPERIMENTS],
)
async def cancel_queued_run(run_id: str) -> dict[str, str]:
    """Stop a job, whether it is waiting or already training.

    A pending job is dropped. A running one is asked to stop at the boundary its
    ``stop_at`` names and keeps the best checkpoint it has earned (ADR-088,
    ADR-111), so a 200 here means the request was delivered, not that training
    has already ended.

    Raises:
        HTTPException: 404 if no job carries that id — already finished, or
            never submitted. 409 if the job is running and cannot be stopped
            (``stop_at`` is null); nothing is asked of it.
    """
    try:
        found = _LANES.cancel(run_id)
    except NotStoppableError:
        raise HTTPException(409, tr("queue.not_stoppable")) from None
    if not found:
        raise HTTPException(404, tr("queue.unknown_id"))
    return {"run_id": run_id, "status": "cancelled"}


@router.get(
    "/experiment/result/{run_id}",
    responses={
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
        500: {"description": SERVER_ERROR_TEXT},
    },
    tags=[TAG_EXPERIMENTS],
)
async def get_result(run_id: str) -> RunResult:
    """Fetch the result of a completed experiment.

    Reads the job's live state while it is kept, and otherwise its queue's
    per-run record: with jobs queued back to back, the next one is already
    running by the time the browser asks for the previous one's report.
    """
    state = _run_state(run_id) or _LANES.record(run_id)
    if state is None:
        raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))

    if state["status"] == "running":
        raise HTTPException(409, tr("run.still_running"))

    if state["status"] == "failed":
        raise HTTPException(
            500, tr("run.failed", error=state.get("error", "unknown error"))
        )

    run_dir = state.get("run_dir")
    run_json_path = Path(run_dir) / RUN_FILENAME if run_dir else None

    report = state.get("report") or {}
    stopped = _report_stopped(report)
    if run_json_path and run_json_path.exists():
        data = json.loads(run_json_path.read_text(encoding="utf-8"))
        return RunResult(
            run_id=run_id,
            metrics=data.get("metrics", {}),
            metric_cis=data.get("metric_cis", {}),
            report=report,
            artifacts=data.get("artifacts", {}),
            stopped=stopped,
        )

    return RunResult(
        run_id=run_id,
        metrics={},
        report=report,
        artifacts={},
        stopped=stopped,
    )


def _report_stopped(report: dict[str, Any]) -> bool:
    """Whether a stop cut this run, read from wherever its report keeps it.

    Multi-unit reports carry ``stopped`` at the top; single runs carry it in
    their training section (``train``, or ``detection`` for detection).
    """
    if "stopped" in report:
        return bool(report["stopped"])
    for section in ("train", "detection"):
        part = report.get(section)
        if isinstance(part, dict) and "stopped" in part:
            return bool(part["stopped"])
    return False
