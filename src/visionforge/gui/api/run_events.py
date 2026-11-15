"""The events of one run, kept so that anyone can read them from the start (ADR-159).

A plain ``asyncio.Queue`` hands each event to one reader and forgets it. That
left a page reloaded mid-run with an empty log, and let a second tab steal
epochs from the first. Here the events stay for the life of the run and every
reader walks them at its own pace. A run reports per epoch, not per batch, so
the list stays small, and it is dropped when the next run starts.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any


class RunEventLog:
    """Append-only event list of one run; ``None`` closes it."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id
        self._items: list[dict[str, Any] | None] = []
        self._changed = asyncio.Event()

    # Async on purpose: worker threads hand events over with
    # run_coroutine_threadsafe(log.put(...)), so the append runs on the loop.
    async def put(self, item: dict[str, Any] | None) -> None:  # NOSONAR python:S7503
        """Append an event, or the ``None`` that ends the stream."""
        self._items.append(item)
        self._changed.set()
        self._changed = asyncio.Event()

    async def read(self, after: int = 0) -> AsyncIterator[dict[str, Any]]:
        """Yield the events past the first ``after``, then new ones until closed."""
        index = after
        while True:
            # No await between the length check and taking the event, so a put
            # cannot slip in between and be missed.
            while index < len(self._items):
                item = self._items[index]
                index += 1
                if item is None:
                    return
                yield item
            if self._items and self._items[-1] is None:
                # `after` already past the end of a closed run.
                return
            await self._changed.wait()
