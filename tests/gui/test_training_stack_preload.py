"""The first run imports the training stack off the event loop (#258).

Starting the GUI no longer imports torch (#202), so the first run of each process
paid for it (~3 s) inside an ``async`` handler: the event loop stopped, and with
it every open SSE stream. The import now runs in a worker thread.

The tests stand in for torch with a module whose import blocks on a gate, so
they prove where the import runs without timing anything.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import sys
import threading
import types
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from visionforge.gui.api.run_queue import colab_lane

from .conftest import release_queue

_ROUTES_FILE = Path(__file__).resolve().parents[2] / "src/visionforge/gui/api/routes.py"
_LOADER = "load_training_stack"
_GATE_TIMEOUT = 10


class _Gate:
    """What the fake module reports while it is being imported."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.thread: int | None = None


@pytest.fixture
def routes_mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    import visionforge.gui.api.routes as routes

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(routes, "_MODELS_DIR", Path("outputs/models"))
    monkeypatch.setattr(routes, "_PROFILES_DIR", Path("outputs/profiles"))
    release_queue(routes)
    yield routes
    release_queue(routes)


@pytest.fixture
def slow_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, _Gate]]:
    """A module whose import waits for ``gate.release``, standing in for torch."""
    gate = _Gate()
    suffix = uuid.uuid4().hex[:8]
    gate_name = f"_vf_gate_{suffix}"
    name = f"_vf_slow_stack_{suffix}"
    (tmp_path / f"{name}.py").write_text(
        "import sys, threading\n"
        f"_gate = sys.modules[{gate_name!r}].gate\n"
        "_gate.thread = threading.get_ident()\n"
        "_gate.entered.set()\n"
        f"_gate.release.wait(timeout={_GATE_TIMEOUT})\n"
        "_gate.finished.set()\n",
        encoding="utf-8",
    )
    holder = types.ModuleType(gate_name)
    holder.gate = gate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, gate_name, holder)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, gate
    gate.release.set()
    sys.modules.pop(name, None)


def _use_stack(monkeypatch: pytest.MonkeyPatch, routes: Any, *names: str) -> None:
    # raising=False: before #258 there was no stack to replace, and the test must
    # then fail on what the loop does, not on a missing attribute.
    monkeypatch.setattr(routes, "_TRAINING_STACK", names, raising=False)


async def _wait_until_idle(routes: Any) -> None:
    for _ in range(1000):
        if not routes._LANES.is_busy():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the queue never drained")


def test_the_first_run_imports_the_stack_while_the_loop_keeps_running(
    routes_mod: Any,
    slow_module: tuple[str, _Gate],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name, gate = slow_module
    _use_stack(monkeypatch, routes_mod, name)
    started: list[str] = []
    progress: dict[str, bool] = {}

    async def start() -> None:
        # Like every executor: import what it trains with, on the loop.
        importlib.import_module(name)
        started.append(name)

    async def scenario() -> None:
        stream: list[int] = []

        async def sse() -> None:  # an open SSE stream, one event per loop turn
            while True:
                stream.append(len(stream))
                await asyncio.sleep(0)

        ticker = asyncio.create_task(sse())
        routes_mod._submit_job("first-run", "first-run", "regression", "simple", start)
        progress["entered"] = await asyncio.to_thread(gate.entered.wait, _GATE_TIMEOUT)
        seen = len(stream)
        for _ in range(5):
            await asyncio.sleep(0)
        progress["stream_moved"] = len(stream) > seen
        progress["import_done"] = gate.finished.is_set()
        gate.release.set()
        await _wait_until_idle(routes_mod)
        ticker.cancel()

    asyncio.run(scenario())

    assert progress["entered"]
    # Before #258 the loop ran the import itself: by the time it could look again
    # the import was over, and nothing else had moved meanwhile.
    assert progress["import_done"] is False, "the import ran on the event loop"
    assert progress["stream_moved"]
    assert started == [name]
    assert gate.thread != threading.get_ident()


def test_the_loader_imports_in_a_worker_thread(
    routes_mod: Any,
    slow_module: tuple[str, _Gate],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name, gate = slow_module
    _use_stack(monkeypatch, routes_mod, name)
    gate.release.set()

    asyncio.run(routes_mod.load_training_stack())

    assert gate.finished.is_set()
    assert gate.thread != threading.get_ident()


def test_a_loaded_stack_needs_no_thread(
    routes_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_stack(monkeypatch, routes_mod, "sys", "json")

    def must_not_run(names: tuple[str, ...]) -> None:
        raise AssertionError(f"re-imported a loaded stack: {names}")

    monkeypatch.setattr(routes_mod, "_import_modules", must_not_run)

    asyncio.run(routes_mod.load_training_stack())


def test_detection_also_loads_its_trainer(
    routes_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent = f"_vf_absent_{uuid.uuid4().hex[:8]}"
    _use_stack(monkeypatch, routes_mod, "sys", absent)
    monkeypatch.setattr(routes_mod, "_DETECTION_STACK", ("json", absent + "_det"))
    monkeypatch.setitem(
        routes_mod._TASK_STACK, "detection", routes_mod._DETECTION_STACK
    )

    missing = routes_mod._missing_training_stack(*routes_mod._TASK_STACK["detection"])

    assert missing == (absent, absent + "_det")


def test_a_broken_import_is_left_to_the_job(
    routes_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_stack(monkeypatch, routes_mod, f"_vf_missing_{uuid.uuid4().hex[:8]}")
    started: list[bool] = []

    async def start() -> None:
        started.append(True)

    async def scenario() -> None:
        routes_mod._submit_job("broken", "broken", "regression", "simple", start)
        await _wait_until_idle(routes_mod)

    asyncio.run(scenario())

    # The executor imports it again and reports the failure as the run's own.
    assert started == [True]


def test_a_colab_job_does_not_load_the_local_stack(
    routes_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[tuple[str, ...]] = []
    monkeypatch.setattr(routes_mod, "_import_modules", loaded.append)
    _use_stack(monkeypatch, routes_mod, f"_vf_never_{uuid.uuid4().hex[:8]}")
    started: list[bool] = []

    async def start() -> None:
        started.append(True)

    async def scenario() -> None:
        routes_mod._submit_job(
            "remote", "remote", "regression", "simple", start, lane=colab_lane("a")
        )
        await _wait_until_idle(routes_mod)

    asyncio.run(scenario())

    assert started == [True]
    assert loaded == []


def _imports_training_code(node: ast.AST) -> bool:
    if not isinstance(node, ast.ImportFrom) or node.module is None:
        return False
    return node.module.startswith("visionforge.blocks.") or node.module in {
        "visionforge.tasks.runner",
        "visionforge.tasks.engine",
    }


def _awaits_loader(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == _LOADER
    )


def _is_route(fn: ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(d, ast.Call)
        and isinstance(d.func, ast.Attribute)
        and isinstance(d.func.value, ast.Name)
        and d.func.value.id == "router"
        for d in fn.decorator_list
    )


def _line(node: ast.AST) -> int:
    """Line of a statement or expression (``ast.AST`` itself declares none)."""
    return int(getattr(node, "lineno", 0))


def test_no_route_imports_the_training_code_before_the_loader() -> None:
    # A route that imports a block or runner directly would load torch on the
    # loop again; the queue's executors are covered by the queue itself.
    tree = ast.parse(_ROUTES_FILE.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef) or not _is_route(fn):
            continue
        loaded_at = min(
            (_line(n) for n in ast.walk(fn) if _awaits_loader(n)), default=None
        )
        offenders.extend(
            f"{fn.name}:{_line(node)}"
            for node in ast.walk(fn)
            if _imports_training_code(node)
            and (loaded_at is None or _line(node) < loaded_at)
        )
    assert offenders == []
