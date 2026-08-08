"""A cooperative stop signal for training loops.

ADR-075 refused to cancel a running job, and the reason it gave was sound: the
trainers owned their loops and had no point at which stopping was safe, so a
"cancel" would either lie about having worked or leave a half-written run
directory behind.

The missing piece was never the mechanism — it was a safe point. Every trainer
already pauses between epochs to write a checkpoint and emit progress. That is
where a run can stop with everything on disk consistent, and it is the only
place this token is ever read.

**Cancelling keeps what the run has earned.** A researcher usually cancels
because the curve already answered the question, not because the work is
garbage: the best checkpoint so far, its metrics and its plots stay. Discarding
them would make the button something people avoid pressing, which defeats it.
"""

from __future__ import annotations

import threading


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


__all__ = ["CancellationToken", "is_cancelled"]
