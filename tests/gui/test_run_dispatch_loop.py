"""A dispatched run must reach ``block.run`` even when its thread starts late (#77).

``RunQueue.submit`` spawns the drain task on the loop of the request that
submitted the job. ``TestClient`` used outside ``with`` gives every request its
own loop and closes it when the response is back: closing cancels the drain
task, and a ``to_thread`` work item that no pool thread has picked up yet is
cancelled with it, so ``setup()`` ran and ``run()`` never does. Under the full
suite with coverage a fresh pool thread can be that slow; the gate below makes
the delay certain instead of likely.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionforge.utils.selftest_data import build_regression_dataset

from .conftest import release_queue


class _GatedPool(ThreadPoolExecutor):
    """One worker held on ``gate``; flags ``queued`` once real work waits behind it."""

    def __init__(self, gate: threading.Event, queued: threading.Event) -> None:
        super().__init__(max_workers=1)
        self._queued = queued
        super().submit(gate.wait)

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def, override]
        future = super().submit(fn, *args, **kwargs)
        self._queued.set()
        return future


@dataclass
class _Gated:
    captured: dict[str, object]
    gate: threading.Event
    queued: threading.Event


@pytest.fixture
def gated_regression() -> Iterator[_Gated]:
    """Fake RegressionBlock whose ``run`` waits behind a busy pool thread."""
    import visionforge.gui.api.routes as routes_mod
    from visionforge.blocks.regression import RegressionBlock

    gated = _Gated(captured={}, gate=threading.Event(), queued=threading.Event())
    pools: list[ThreadPoolExecutor] = []
    originals = (RegressionBlock.setup, RegressionBlock.run, RegressionBlock.report)

    def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
        gated.captured["setup"] = True
        # The only pool thread is busy until the gate opens, so the run submitted
        # right after this returns waits in the queue, like a slow thread start.
        pool = _GatedPool(gated.gate, gated.queued)
        pools.append(pool)
        asyncio.get_running_loop().set_default_executor(pool)
        drain = asyncio.current_task()
        assert drain is not None
        # A closing loop cancels the drain; open the gate then, or the loop's
        # executor shutdown would wait on it forever.
        drain.add_done_callback(lambda _task: gated.gate.set())

    def fake_run(self) -> None:  # type: ignore[no-untyped-def]
        gated.captured["run"] = True

    def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
        return {"train": {"run_dir": None}}

    RegressionBlock.setup = fake_setup  # type: ignore[method-assign]
    RegressionBlock.run = fake_run  # type: ignore[method-assign]
    RegressionBlock.report = fake_report  # type: ignore[method-assign]
    try:
        yield gated
    finally:
        gated.gate.set()
        for pool in pools:
            pool.shutdown(wait=True)
        (RegressionBlock.setup, RegressionBlock.run, RegressionBlock.report) = (  # type: ignore[method-assign]
            originals
        )
        release_queue(routes_mod)


def _payload(tmp_path: Path) -> dict[str, object]:
    base = build_regression_dataset(tmp_path / "ds")
    return {
        "name": "late_thread",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(base), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }


def test_run_reaches_the_block_when_its_thread_starts_late(
    gated_regression: _Gated,
    live_client: TestClient,
    tmp_path: Path,
) -> None:
    resp = live_client.post("/api/regression/run", json=_payload(tmp_path))
    assert resp.status_code == 200, resp.text
    # Open the gate only once run() is waiting in the pool, so the late start
    # is certain rather than a race the test may or may not hit.
    assert gated_regression.queued.wait(30.0)
    assert gated_regression.captured.get("setup") is True
    gated_regression.gate.set()

    deadline = time.monotonic() + 30.0
    while "run" not in gated_regression.captured and time.monotonic() < deadline:
        time.sleep(0.05)
    assert gated_regression.captured.get("run") is True
