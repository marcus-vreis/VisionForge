"""Shared helpers for tests that wait on background work (OPS-13)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

_TERMINAL_STATUSES = ("completed", "failed")


class _StatusClient(Protocol):
    """Anything with a ``get`` returning a response with ``json()`` (TestClient)."""

    def get(self, url: str) -> Any: ...


def wait_until[T](
    predicate: Callable[[], T],
    timeout: float = 30.0,
    interval: float = 0.05,
    message: str = "condition",
) -> T:
    """Poll ``predicate`` until it returns a truthy value and return that value.

    Raises ``AssertionError`` naming ``message`` if ``timeout`` seconds pass
    first, so a stuck background job fails the test explicitly instead of
    letting the caller read a stale value.
    """
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"timed out after {timeout}s waiting for {message}")
        # The one legitimate polling sleep: every test waits through this helper.
        time.sleep(min(interval, remaining))


def wait_experiment_done(
    client: _StatusClient, timeout: float = 30.0
) -> dict[str, Any]:
    """Wait for ``/api/experiment/status`` to reach completed/failed; return it."""
    last: dict[str, Any] = {}

    def _done() -> bool:
        nonlocal last
        last = client.get("/api/experiment/status").json()
        return last["status"] in _TERMINAL_STATUSES

    wait_until(_done, timeout, message="the experiment to finish")
    return last
