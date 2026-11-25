"""The child side of a server job, in this process (ADR-248).

``test_job_process.py`` starts real children; here the pieces of the worker are
run directly: how it reads the server's lines, how it hears a stop or the end
of input, how a run's end and error are worded, and that an empty stdin exits.
"""

from __future__ import annotations

import contextvars
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import visionforge
from visionforge.core.cancellation import CancellationToken
from visionforge.gui.api import job_worker
from visionforge.gui.api.job_worker import _Stdin, _watch_stdin
from visionforge.utils.messages import message, tr

SRC = Path(visionforge.__file__).resolve().parents[1]


@pytest.fixture
def pipe() -> Iterator[tuple[int, int]]:
    read_fd, write_fd = os.pipe()
    yield read_fd, write_fd
    for fd in (read_fd, write_fd):
        try:
            os.close(fd)
        except OSError:
            pass  # the test closed it already


def _close(fd: int) -> None:
    os.close(fd)


# ── reading the server's lines ──────────────────────────────────────────────


def test_lines_are_joined_across_reads_and_a_cut_last_line_is_dropped(
    pipe: tuple[int, int],
) -> None:
    read_fd, write_fd = pipe
    os.write(write_fd, b'{"type": "st')
    os.write(write_fd, b'op"}\nsecond\nthird without end')
    _close(write_fd)
    stdin = _Stdin(read_fd)
    assert stdin.line() == b'{"type": "stop"}'
    assert stdin.line() == b"second"
    assert stdin.line() is None


def test_a_peeked_pipe_is_read_only_when_bytes_wait(
    pipe: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    read_fd, write_fd = pipe
    monkeypatch.setattr(job_worker, "_PEEK_S", 0)
    os.write(write_fd, b"one\ntwo\n")
    # Nothing waiting twice, then 4 bytes, then the rest, then a closed pipe.
    answers = iter([0, 0, 4, 4, None])
    asked: list[int | None] = []

    def available() -> int | None:
        asked.append(next(answers))
        return asked[-1]

    stdin = _Stdin(read_fd, available)
    assert stdin.line() == b"one"
    assert asked == [0, 0, 4]
    assert stdin.line() == b"two"
    assert stdin.line() is None
    assert asked == [0, 0, 4, 4, None]


@pytest.mark.skipif(sys.platform != "win32", reason="PeekNamedPipe is Windows'")
def test_the_windows_peek_counts_waiting_bytes_and_sees_the_pipe_close(
    pipe: tuple[int, int],
) -> None:
    read_fd, write_fd = pipe
    peek = job_worker._pipe_peek(read_fd)
    assert peek() == 0
    os.write(write_fd, b"stop\n")
    assert peek() == 5
    _close(write_fd)
    assert _Stdin(read_fd, peek).line() == b"stop"
    assert peek() is None


# ── hearing a stop, or the server going away ─────────────────────────────────


class _Timers:
    """Stands in for ``threading.Timer``: notes what would fire, starts nothing."""

    def __init__(self) -> None:
        self.made: list[tuple[float, Any, tuple[Any, ...]]] = []
        self.started = 0

    def __call__(self, interval: float, function: Any, args: tuple[Any, ...]) -> Any:
        self.made.append((interval, function, args))
        return SimpleNamespace(daemon=False, start=self._start)

    def _start(self) -> None:
        self.started += 1


@pytest.fixture
def timers(monkeypatch: pytest.MonkeyPatch) -> _Timers:
    fake = _Timers()
    monkeypatch.setattr(job_worker.threading, "Timer", fake)
    return fake


def test_a_stop_line_cancels_and_other_lines_are_ignored(
    pipe: tuple[int, int], timers: _Timers
) -> None:
    read_fd, write_fd = pipe
    token = CancellationToken()
    finished = threading.Event()
    for line in (b"not json", b"\xff\xfe", b"[1, 2]", b'{"type": "hello"}'):
        os.write(write_fd, line + b"\n")
    os.write(write_fd, b'{"type": "stop"}\n')
    finished.set()  # the run ended: the close that follows is the normal way out
    _close(write_fd)
    _watch_stdin(_Stdin(read_fd), token, 5.0, finished)
    assert token.cancelled
    assert timers.made == []


def test_lines_without_a_stop_leave_the_run_alone(
    pipe: tuple[int, int], timers: _Timers
) -> None:
    read_fd, write_fd = pipe
    token = CancellationToken()
    finished = threading.Event()
    finished.set()
    os.write(write_fd, b'{"type": "hello"}\n')
    _close(write_fd)
    _watch_stdin(_Stdin(read_fd), token, 5.0, finished)
    assert not token.cancelled


def test_a_server_gone_mid_run_stops_the_block_and_arms_the_exit(
    pipe: tuple[int, int], timers: _Timers
) -> None:
    read_fd, write_fd = pipe
    token = CancellationToken()
    _close(write_fd)
    _watch_stdin(_Stdin(read_fd), token, 7.5, threading.Event())
    assert token.cancelled
    assert timers.made == [(7.5, os._exit, (job_worker.ORPHAN_EXIT,))]
    assert timers.started == 1


# ── the run's end ───────────────────────────────────────────────────────────


class _Lines:
    """Stands in for the protocol pipe."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send(self, line: dict[str, Any]) -> None:
        self.sent.append(line)


class _Config:
    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> dict[str, Any]:
        return data


class _Block:
    """Reports its run folder under ``train``; fails when the config says so."""

    # Wired by the worker after ``setup``, as on the real blocks.
    _progress_callback: Any
    _resume_dir: Path | None

    def setup(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self) -> None:
        if self.config.get("fail") == "empty":
            raise RuntimeError
        if self.config.get("fail") == "paging":
            raise _PagingFileError
        self._progress_callback({"type": "epoch_end", "epoch": 1})
        self._train_result = SimpleNamespace(model_path=Path("runs") / "a" / "best.pt")

    def report(self) -> dict[str, Any]:
        return {"train": {"run_dir": "runs/a"}, "resume": str(self._resume_dir)}


class _PagingFileError(OSError):
    winerror = 1455


def _spec(**overrides: Any) -> dict[str, Any]:
    return {
        "block": f"{__name__}:_Block",
        "config_type": f"{__name__}:_Config",
        "config": {"fail": overrides.pop("fail", None)},
        "resume_dir": None,
        "report_section": "train",
        "spawn_hint": False,
        "lang": None,
        **overrides,
    }


def _run(spec: dict[str, Any]) -> list[dict[str, Any]]:
    lines = _Lines()
    # Its own context: the language a spec binds must not leak into other tests.
    contextvars.copy_context().run(
        job_worker.run,
        spec,
        lines,  # type: ignore[arg-type]
        CancellationToken(),
    )
    return lines.sent


def test_a_run_sends_its_events_then_its_end_with_the_folder() -> None:
    sent = _run(_spec(resume_dir=str(Path("runs") / "old")))
    assert sent[0] == {"type": "event", "event": {"type": "epoch_end", "epoch": 1}}
    report = {"train": {"run_dir": "runs/a"}, "resume": str(Path("runs") / "old")}
    assert sent[1:] == [
        {"type": "job_end", "status": "ok", "report": report, "run_dir": "runs/a"}
    ]


def test_without_a_report_section_the_folder_is_the_checkpoints() -> None:
    sent = _run(_spec(report_section=None))
    assert sent[-1]["run_dir"] == str(Path("runs") / "a")


def test_a_report_section_with_no_folder_gives_none() -> None:
    sent = _run(_spec(report_section="eval"))
    assert sent[-1]["status"] == "ok"
    assert sent[-1]["run_dir"] is None


def test_an_error_without_a_message_is_worded_in_the_jobs_language() -> None:
    sent = _run(_spec(fail="empty", lang="en"))
    assert sent == [
        {
            "type": "job_end",
            "status": "error",
            "error": f"RuntimeError: {message('en', 'run.no_message')}",
        }
    ]


def test_the_paging_file_error_gets_its_hint_only_when_asked() -> None:
    hinted = _run(_spec(fail="paging", spawn_hint=True))
    assert hinted[-1]["error"] == tr("winerror.paging_file")
    plain = _run(_spec(fail="paging"))
    assert plain[-1]["error"].startswith("_PagingFileError: ")


def test_a_report_value_json_cannot_hold_is_made_plain() -> None:
    numpy = pytest.importorskip("numpy")
    assert job_worker._plain(numpy.float32(0.5)) == 0.5
    assert type(job_worker._plain(numpy.int64(3))) is int
    assert job_worker._plain(Path("runs") / "a") == str(Path("runs") / "a")


def test_a_child_whose_server_sends_nothing_exits_with_code_2(
    tmp_path: Path,
) -> None:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    done = subprocess.run(  # noqa: S603 - this Python and a module of this package
        [sys.executable, "-m", "visionforge.gui.api.job_worker"],
        input=b"",
        capture_output=True,
        env=env,
        cwd=tmp_path,
        timeout=60,
        check=False,
    )
    assert done.returncode == 2
    assert done.stdout == b""
