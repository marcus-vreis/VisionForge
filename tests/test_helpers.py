from __future__ import annotations

import asyncio
import threading
import time

import pytest

from tests.helpers import wait_experiment_done, wait_until, wait_until_async


def test_wait_until_returns_the_truthy_value() -> None:
    calls = {"n": 0}

    def _third_call() -> str:
        calls["n"] += 1
        return "ready" if calls["n"] >= 3 else ""

    assert wait_until(_third_call, timeout=5, interval=0.001) == "ready"
    assert calls["n"] == 3


def test_wait_until_false_predicate_raises_before_timeout_plus_one_second() -> None:
    start = time.monotonic()
    with pytest.raises(AssertionError, match="the thing"):
        wait_until(lambda: False, timeout=0.2, interval=0.01, message="the thing")
    assert time.monotonic() - start < 1.2


def test_wait_until_checks_once_even_with_zero_timeout() -> None:
    assert wait_until(lambda: True, timeout=0)


class _Resp:
    def __init__(self, body: dict[str, str]) -> None:
        self._body = body

    def json(self) -> dict[str, str]:
        return self._body


class _Client:
    def __init__(self, statuses: list[str]) -> None:
        self._statuses = statuses

    def get(self, url: str) -> _Resp:
        assert url == "/api/experiment/status"
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return _Resp({"status": status})


def test_wait_experiment_done_returns_terminal_status() -> None:
    client = _Client(["running", "running", "failed"])
    assert wait_experiment_done(client, timeout=5) == {"status": "failed"}


def test_wait_experiment_done_fails_when_stuck_running() -> None:
    client = _Client(["running"])
    with pytest.raises(AssertionError, match="experiment to finish"):
        wait_experiment_done(client, timeout=0.1)


# ── on an event loop (#383) ──────────────────────────────────────────────────
# Python 3.12 on Windows reads a 15.6 ms clock, and asyncio runs a timer due
# within one clock tick at once: sleep(0.01) does not wait. A wait counted in
# turns of sleep(0.01) then gave up in under a second on the CI, while an open
# SSE stream kept the loop from ever blocking.
_COARSE_CLOCK = 0.015625


async def _on_a_coarse_clock() -> asyncio.Task[None]:
    """Mimic py3.12 on Windows, with a stream that is ready on every turn."""
    asyncio.get_running_loop()._clock_resolution = _COARSE_CLOCK  # type: ignore[attr-defined]

    async def stream() -> None:
        while True:
            await asyncio.sleep(0)

    return asyncio.create_task(stream())


def test_the_async_wait_lasts_its_timeout_in_seconds() -> None:
    async def scenario() -> float:
        stream = await _on_a_coarse_clock()
        start = time.monotonic()
        with pytest.raises(AssertionError, match="the queue to drain"):
            await wait_until_async(
                lambda: False, timeout=0.3, message="the queue to drain"
            )
        stream.cancel()
        return time.monotonic() - start

    assert 0.3 <= asyncio.run(scenario()) < 1.3


def test_the_async_wait_sees_work_a_thread_finishes_later() -> None:
    done = threading.Event()
    timer = threading.Timer(0.2, done.set)

    async def scenario() -> bool:
        stream = await _on_a_coarse_clock()
        timer.start()
        seen = await wait_until_async(done.is_set, timeout=5)
        stream.cancel()
        return seen

    try:
        assert asyncio.run(scenario()) is True
    finally:
        timer.cancel()
