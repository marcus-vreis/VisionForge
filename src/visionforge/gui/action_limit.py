"""Server mode: how many heavy actions run at once outside the queue (ADR-226).

A training waits its turn in the queue. The actions below do their work inside
the request that asks for them: they load a model, read a whole dataset or
talk to another service. On a server several people share, nothing bounded
how many ran together. ``--max-actions`` does: past it the next one is told
the server is busy, and nothing waits in line.
"""

from __future__ import annotations

from threading import Lock

DEFAULT_MAX_ACTIONS = 2

# What runs at once, in the request, and is heavy enough to count. A Laboratory
# job is not here: it runs after its request ends and has its own bound, one
# per person and kind (ADR-220).
HEAVY_ROUTES = frozenset(
    {
        # A finished run's checkpoint is loaded and run over a folder of images.
        "POST /api/runs/{run_id}/test",
        "POST /api/runs/{run_id}/gradcam",
        "POST /api/runs/{run_id}/batch_predict",
        "POST /api/runs/{run_id}/export_onnx",
        # A whole dataset is fetched, or every image of one is hashed.
        "POST /api/dataset/download",
        "POST /api/dataset/audit",
        # A run is packed and sent to another service.
        "POST /api/hub/export",
        "POST /api/zenodo/export",
        "POST /api/github/export",
        "POST /api/wandb/runs/{run_id}/retry",
        "POST /api/comet/runs/{run_id}/retry",
    }
)


def heavy(route: str) -> bool:
    """True when ``route`` (``"METHOD /api/template"``) counts against the limit."""
    return route in HEAVY_ROUTES


class ActionSlots:
    """How many heavy actions are running now, across everybody."""

    def __init__(self) -> None:
        self._busy = 0
        self._lock = Lock()

    def take(self, limit: int) -> bool:
        """Count one more action when fewer than ``limit`` are running."""
        with self._lock:
            if self._busy >= limit:
                return False
            self._busy += 1
            return True

    def give(self) -> None:
        """The action that took a slot ended, however it ended."""
        with self._lock:
            self._busy = max(0, self._busy - 1)

    @property
    def busy(self) -> int:
        """The actions running now."""
        with self._lock:
            return self._busy
