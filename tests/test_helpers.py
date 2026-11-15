from __future__ import annotations

import time

import pytest

from tests.helpers import wait_experiment_done, wait_until


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
