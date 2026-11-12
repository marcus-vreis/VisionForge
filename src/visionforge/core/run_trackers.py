"""The contract every optional experiment tracker of a training follows.

MLflow (ADR-170) and Weights & Biases (ADR-172) are built per run folder, opened
when it exists, fed each epoch and closed on every exit path. None of the calls
may raise: a tracker that fails records it beside the run and the training goes
on. The tasks other than classification send their finished ``run.json`` the
same way, epoch by epoch (ADR-197).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from loguru import logger


class RunTracker(Protocol):
    """One external copy of a local run's parameters and epoch metrics."""

    def start(self, config: dict[str, Any]) -> None:
        """Open the external run with the run's config."""

    def epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Record one finished epoch."""

    def finish(self, status: str) -> None:
        """Close the run as ``FINISHED``, ``KILLED`` or ``FAILED``."""


# Built once the trainer knows the run folder.
TrackerFactory = Callable[[Path], RunTracker]


def export_run_record(factories: Iterable[TrackerFactory], run_dir: Path) -> None:
    """Send a finished run's ``run.json`` to each tracker, one epoch per history row.

    Never raises: a missing or unreadable record, or a tracker that cannot be
    built, is logged and the others go on.
    """
    path = run_dir / "run.json"
    try:
        if path.is_symlink():
            raise ValueError("Unsafe run record")
        record = json.loads(path.read_text("utf-8"))
        if not isinstance(record, dict):
            raise ValueError("Invalid run record")
    except (OSError, ValueError) as exc:
        logger.warning("Run not exported to trackers ({})", type(exc).__name__)
        return
    config = record.get("config")
    history = record.get("history")
    rows = [
        row
        for row in (history if isinstance(history, list) else [])
        if isinstance(row, dict)
        and isinstance(row.get("epoch"), int)
        and not isinstance(row.get("epoch"), bool)
    ]
    for build in factories:
        try:
            tracker = build(run_dir)
        except Exception as exc:  # noqa: BLE001
            # Optional export boundary.
            logger.warning("Tracker unavailable ({})", type(exc).__name__)
            continue
        tracker.start(config if isinstance(config, dict) else {})
        for row in rows:
            metrics = {key: value for key, value in row.items() if key != "epoch"}
            tracker.epoch(row["epoch"], metrics)
        tracker.finish(_terminal_status(record))


def _terminal_status(record: dict[str, Any]) -> str:
    if record.get("status") == "failed":
        return "FAILED"
    return "KILLED" if record.get("stopped") else "FINISHED"
