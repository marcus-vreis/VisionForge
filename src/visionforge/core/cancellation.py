"""A cooperative stop signal for training loops.

ADR-075 refused to cancel a running job, and the reason it gave was sound: the
trainers owned their loops and had no point at which stopping was safe, so a
"cancel" would either lie about having worked or leave a half-written run
directory behind.

The missing piece was never the mechanism — it was a safe point. Every trainer
already pauses between epochs to write a checkpoint and emit progress. That is
where a run can stop with everything on disk consistent, and it is where the
trainers read this token. The orchestrators that run several trainings in one
job (K-fold, comparison, sweeps, replicates) also read it between units, so
nothing new starts after a stop; PatchCore, which has no epochs, reads it
between its phases (ADR-111).

**Cancelling keeps what the run has earned.** A researcher usually cancels
because the curve already answered the question, not because the work is
garbage: the best checkpoint so far, its metrics and its plots stay. Discarding
them would make the button something people avoid pressing, which defeats it.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

# Status of a unit (fold, trial, model, replicate) that was training when the
# stop arrived (ADR-111). Its record keeps whatever metrics it reached, but it
# stays out of every aggregate and ranking: a fold cut at epoch 2 averaged with
# folds that ran to epoch 30 gives a mean that no configuration produced.
STOPPED = "stopped"
STOPPED_NOTE = "Parado a pedido antes de terminar; fica fora da agregação."


class CancellationToken:
    """A one-way flag: once cancelled, always cancelled.

    Thread-safe because the GUI sets it from the request thread while the
    trainer reads it from the worker thread. `threading.Event` gives that for
    free and needs no lock of our own.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Ask the run to stop at its next epoch boundary."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True once `cancel()` has been called."""
        return self._event.is_set()

    def __bool__(self) -> bool:
        """So `if token:` reads as "was it cancelled", not "does it exist"."""
        return self.cancelled


def is_cancelled(token: CancellationToken | None) -> bool:
    """Read a token that may be absent.

    Trainers take the token optionally so the CLI path, which has nobody to
    press a button, does not have to invent one.
    """
    return token is not None and token.cancelled


def job_was_stopped(
    statuses: Sequence[str], planned: int, token: CancellationToken | None
) -> bool:
    """Whether a stop cut a multi-unit job: a unit was cut, or units were left unrun.

    The run-level marker the result and run.json carry (ADR-111), so a reader
    never has to infer a stop from a message. A stop that arrived during the
    last unit, which then finished, cut nothing: the job ran in full.
    """
    return STOPPED in statuses or (is_cancelled(token) and len(statuses) < planned)


__all__ = [
    "STOPPED",
    "STOPPED_NOTE",
    "CancellationToken",
    "is_cancelled",
    "job_was_stopped",
]
