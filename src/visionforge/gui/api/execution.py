"""The run queue's lanes and the state of every started job (ADR-132, ADR-185).

This module is the one owner of what used to live in ``routes.py``: the lanes
that drain submitted jobs (ADR-075, ADR-163), the per-job state the queue
creates when a job starts (event log, cancellation token, run dict), and the
classification executor that fills that state in.

Who may touch what:

- The queue adds and drops a job's state, through ``_begin_job`` and
  ``_finish_job``; nothing else writes ``_JOB_STATES`` or ``_latest_run_id``.
- An executor writes only its own job's ``run`` and log, reached through
  ``_job_state(run_id)``.
- The routes read through ``_run_state``, ``_sse_generator`` and ``_LANES``, and
  command the lanes through their methods (submit, cancel, reset).

A test that replaces a value here (``_JOB_STATES``, ``_latest_run_id``,
``_MAX_FINISHED_STATES``, ``_execute_experiment``) patches this module: the
readers look the names up here, not in ``routes.py``. Importing it starts no
training and loads no block; the blocks are imported when a run starts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from visionforge.core.cancellation import CancellationToken
from visionforge.core.loader_lifecycle import describe_worker_spawn_failure
from visionforge.core.run_trackers import TrackerFactory
from visionforge.gui.api.run_events import RunEventLog
from visionforge.gui.api.run_queue import (
    LOCAL_LANE,
    QueuedJob,
    RunLanes,
    RunQueue,
    StopPoint,
)
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.messages import tr

if TYPE_CHECKING:
    from visionforge.blocks.classification import ClassificationBlock
    from visionforge.blocks.cross_validation import CrossValidationBlock
    from visionforge.blocks.grid_search import GridSearchBlock
    from visionforge.blocks.model_comparison import ModelComparisonBlock
    from visionforge.blocks.random_search import RandomSearchBlock
    from visionforge.blocks.transfer_learning import TransferLearningBlock

    _Block = (
        ClassificationBlock
        | CrossValidationBlock
        | TransferLearningBlock
        | ModelComparisonBlock
        | GridSearchBlock
        | RandomSearchBlock
    )

RUN_NO_MESSAGE_KEY = "run.no_message"


@dataclass
class JobState:
    """What one started job owns (ADR-163, phase 4).

    A local job and a Colab job run at the same time, so the state that used to
    be a single "current run" lives here, one per ``run_id``:

    - ``log``: the run's event log (ADR-159). The trainer thread puts events on
      it; each SSE reader walks it from where its client stands, and ``None``
      ends the stream.
    - ``token``: the job's cancellation token, handed to the block so
      ``DELETE /api/queue/{id}`` reaches that job's stop boundary (ADR-088).
    - ``run``: the status, error, report and run folder that
      ``/experiment/status`` and ``/experiment/result`` read.
    """

    run_id: str
    log: RunEventLog
    token: CancellationToken
    lane: str = LOCAL_LANE
    run: dict[str, Any] = dataclasses.field(default_factory=dict)


# Started jobs by run_id, oldest first. A finished job stays a while so its
# stream can be replayed and its result fetched; running ones are never dropped.
_JOB_STATES: dict[str, JobState] = {}
_MAX_FINISHED_STATES = 50
# The job started last: what a client that names no run_id follows.
_latest_run_id: str | None = None


def _job_state(run_id: str) -> JobState:
    """The state of a started job; the executors read their own through this."""
    return _JOB_STATES[run_id]


def _run_state(run_id: str | None = None) -> dict[str, Any] | None:
    """A started job's run dict: the one named, or the latest when none is."""
    key = run_id if run_id is not None else _latest_run_id
    state = _JOB_STATES.get(key) if key is not None else None
    return state.run if state is not None else None


def _drop_finished_states() -> None:
    finished = [k for k, s in _JOB_STATES.items() if s.run.get("status") != "running"]
    for key in finished[: max(0, len(finished) - _MAX_FINISHED_STATES)]:
        del _JOB_STATES[key]


def _begin_job(job: QueuedJob) -> None:
    """Prepare per-run state the moment a queued job actually starts (ADR-075).

    The fresh SSE queue is created here rather than at submission time: doing it
    in the request handler was safe only while a second submit was refused, and
    with queueing it would replace the live queue of the job still running.

    The job's cancellation token is stashed for the same reason: the executor
    coroutine is built at submission time and cannot close over a job that does
    not exist yet, so it reads the token here — the queue calls this immediately
    before it awaits ``job.start()``.
    """
    global _latest_run_id

    _JOB_STATES[job.run_id] = JobState(
        run_id=job.run_id,
        log=RunEventLog(job.run_id),
        token=job.cancel_token,
        lane=job.lane,
        run={
            "run_id": job.run_id,
            "status": "running",
            "error": None,
            "report": None,
            "run_dir": None,
        },
    )
    _latest_run_id = job.run_id
    _drop_finished_states()


def _finish_job(job: QueuedJob) -> dict[str, Any] | None:
    """Hand the finished run's terminal state to its lane to keep by run_id."""
    state = _JOB_STATES.get(job.run_id)
    return dict(state.run) if state is not None else None


def _new_lane() -> RunQueue:
    return RunQueue(on_start=_begin_job, on_finish=_finish_job)


# The local lane keeps its name: it is the queue ADR-075 introduced.
_RUN_QUEUE = _new_lane()
_LANES = RunLanes(_RUN_QUEUE, _new_lane)


# Where each kind of job stops when asked (ADR-111), by the strategy it is
# queued under. A strategy missing here is reported as not stoppable: claiming a
# stop that the executor never checks for is the ADR-094 failure mode.
_STOP_POINTS: dict[str, StopPoint] = {
    "simple": "epoch",
    # Classification queues under `config.block`; these four all train through
    # ClassificationBlock or TransferLearningBlock in _execute_experiment.
    "classification": "epoch",
    "transfer_learning": "epoch",
    "batch_prediction": "epoch",
    "export_onnx": "epoch",
    "grid_search": "trial",
    "random_search": "trial",
    "sweep": "trial",
    "cross_validation": "fold",
    "cv": "fold",
    "model_comparison": "model",
    "comparison": "model",
    "replicates": "replicate",
    "replicated-comparison": "replicate",
    # A Colab job (ADR-163): the stop releases the VM, mid-epoch.
    "colab": "vm",
}


def _stop_point(strategy: str) -> StopPoint | None:
    """Where a job queued under ``strategy`` stops; ``sweep:grid`` reads as ``sweep``."""
    return _STOP_POINTS.get(strategy.split(":", 1)[0])


async def _sse_generator(  # type: ignore[return]
    run_id: str | None = None, after: int = 0
):
    """Read the run's event log and yield SSE-formatted lines."""
    key = run_id if run_id is not None else _latest_run_id
    state = _JOB_STATES.get(key) if key is not None else None
    if state is None:
        yield "data: {}\n\n"
        return
    log = state.log

    async for item in log.read(after):
        yield f"data: {json.dumps(item)}\n\n"


# ── the classification executor ─────────────────────────────────────────────


def _block_types() -> dict[str, type[_Block]]:
    """The classification blocks by ``config.block``, imported when a run starts.

    ``classification`` is also what any other value of ``config.block`` runs.
    """
    from visionforge.blocks.classification import ClassificationBlock
    from visionforge.blocks.cross_validation import CrossValidationBlock
    from visionforge.blocks.grid_search import GridSearchBlock
    from visionforge.blocks.model_comparison import ModelComparisonBlock
    from visionforge.blocks.random_search import RandomSearchBlock
    from visionforge.blocks.transfer_learning import TransferLearningBlock

    types: dict[str, type[_Block]] = {
        "classification": ClassificationBlock,
        "cross_validation": CrossValidationBlock,
        "transfer_learning": TransferLearningBlock,
        "model_comparison": ModelComparisonBlock,
        "grid_search": GridSearchBlock,
        "random_search": RandomSearchBlock,
    }
    return types


def _wire_block(
    block: _Block,
    config: ExperimentConfig,
    trackers: tuple[TrackerFactory, ...],
    put_event: Callable[[dict[str, Any]], None],
    token: CancellationToken,
    resume_dir: Path | None,
) -> None:
    """Set the block up, then hand it the trackers, event pump, stop token and resume dir."""
    from visionforge.blocks.classification import ClassificationBlock
    from visionforge.blocks.cross_validation import CrossValidationBlock
    from visionforge.blocks.grid_search import GridSearchBlock
    from visionforge.blocks.model_comparison import ModelComparisonBlock
    from visionforge.blocks.random_search import RandomSearchBlock
    from visionforge.blocks.transfer_learning import TransferLearningBlock

    block.setup(config)
    if isinstance(block, ClassificationBlock):
        block.tracker_factories = trackers
    # Blocks that stream live epoch progress over SSE get the event pump
    # wired here. Grid/random search and K-fold CV forward each inner
    # trial/fold's epochs (annotated with trial context) plus
    # trial_start/trial_end banners.
    if isinstance(
        block,
        ClassificationBlock
        | GridSearchBlock
        | RandomSearchBlock
        | CrossValidationBlock
        | TransferLearningBlock
        | ModelComparisonBlock,
    ):
        block._progress_callback = put_event
    block._cancel_token = token
    block._resume_dir = resume_dir


def _run_dir_of(block: _Block) -> str | None:
    """The folder of the one model a block trained, None when it has no single one.

    Classification and TransferLearning produce a TrainResult with a
    model_path so the run_dir is the checkpoint's parent folder. CV
    writes per-fold run.jsons in subdirs and has no single run_dir.
    """
    from visionforge.blocks.classification import ClassificationBlock
    from visionforge.blocks.transfer_learning import TransferLearningBlock

    if not isinstance(block, ClassificationBlock | TransferLearningBlock):
        return None
    if block._train_result and block._train_result.model_path:
        return str(block._train_result.model_path.parent)
    return None


def _failed_run(run_id: str, error: Exception) -> dict[str, Any]:
    """The run dict of a job whose block raised ``error``."""
    cls = type(error).__name__
    msg = str(error) or tr(RUN_NO_MESSAGE_KEY)
    # A worker-spawn failure surfaces as a torch DLL error, which sends the
    # reader after the wrong thing entirely.
    spawn_hint = describe_worker_spawn_failure(error)
    return {
        "run_id": run_id,
        "status": "failed",
        "error": spawn_hint or f"{cls}: {msg}",
        "report": None,
        "run_dir": None,
    }


async def _execute_experiment(
    config: ExperimentConfig,
    run_id: str,
    resume_dir: Path | None = None,
    *,
    trackers: tuple[TrackerFactory, ...] = (),
) -> None:
    """Run the experiment in a background thread."""
    # Imported before the try, as before the split: an import error is not a
    # failed run of this config.
    block_types = _block_types()

    state = _job_state(run_id)

    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        # Dispatch to the right block based on config.block. The classification
        # path stays default; cross_validation reuses the same /experiment/run
        # endpoint so the frontend doesn't need a separate route. Adding more
        # blocks here is a one-liner — keep them limited to ones whose GUI
        # surface is actually wired up.
        block = block_types.get(config.block, block_types["classification"])()
        _wire_block(block, config, trackers, _put_event, state.token, resume_dir)

        logger.info("GUI: Starting experiment {} (block={})", run_id, config.block)
        await asyncio.to_thread(block.run)

        report = block.report()

        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": _run_dir_of(block),
        }
        logger.success("GUI: Experiment {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Experiment {} failed", run_id)
        state.run = _failed_run(run_id, e)

    finally:
        if queue is not None:
            await queue.put(None)
