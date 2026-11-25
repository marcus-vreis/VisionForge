"""The child side of a server job (ADR-248): ``python -m visionforge.gui.api.job_worker``.

Started by :mod:`visionforge.gui.api.job_process`, never imported by the server.
It reads one JSON line on stdin (the block to run, its config, the resume
folder, the language), runs the block as the executors did in a thread, and
writes one JSON object per line on a pipe of its own:

- ``{"type": "event", "event": {...}}`` for each progress event, in order;
- ``{"type": "job_end", "status": "ok", "report": {...}, "run_dir": ...}`` or
  ``{"type": "job_end", "status": "error", "error": "Cls: message"}``, last.

The process's own stdout is sent to stderr before anything else runs, so a
library that prints (Ultralytics does) cannot be read as an event. Later lines
on stdin are ``{"type": "stop"}``: the block's cancellation token is set and it
stops at its stop point (ADR-111). When stdin closes, the server is gone: the
block is stopped the same way and the process exits by itself after the grace.
On Windows stdin is peeked, never left with a read pending (see ``_Stdin``).

The block and config classes are imported by the ``module:QualName`` the server
sent; this script is reached only through the server's pipe.

Windows starts DataLoader workers with ``spawn``, which imports this module
again as ``__mp_main__``: nothing runs at import time (ADR-030).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

# Exit code of a child that outlived its server by more than the grace.
ORPHAN_EXIT = 70
# How long the child waits, after its last line, for the server to close stdin.
_CLOSE_WAIT_S = 10.0
# How often an idle stdin pipe is peeked on Windows, and the most read at once.
_PEEK_S = 0.1
_CHUNK = 65536


class _Pipe:
    """The protocol stream: one JSON object per line, from any thread."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, default=_plain).encode("utf-8") + b"\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()


def _plain(value: Any) -> Any:
    """JSON for what a report may hold besides plain values: numpy numbers, paths."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


def _protocol_stream() -> IO[bytes]:
    """A private copy of stdout for the protocol; fd 1 then points at stderr."""
    sys.stdout.flush()
    private = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(private, "wb", buffering=0)


def _load(path: str) -> Any:
    module, _, name = path.partition(":")
    target: Any = importlib.import_module(module)
    for part in name.split("."):
        target = getattr(target, part)
    return target


if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    def _pipe_peek(fd: int) -> Callable[[], int | None]:
        """How many bytes wait in the pipe at ``fd``; None once it is closed."""
        handle = msvcrt.get_osfhandle(fd)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        peek = kernel32.PeekNamedPipe
        peek.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        peek.restype = wintypes.BOOL

        def available() -> int | None:
            count = wintypes.DWORD(0)
            if not peek(handle, None, 0, None, ctypes.byref(count), None):
                return None  # broken pipe: the server closed it
            return int(count.value)

        return available

else:

    def _pipe_peek(fd: int) -> Callable[[], int | None] | None:
        """POSIX reads block without holding anything another thread needs."""
        return None


class _Stdin:
    """The server's lines on fd 0, read without ever leaving a read pending on Windows.

    On Windows a thread blocked in a read of the stdin pipe also blocks every
    other call on that handle, and the start-up code of a DLL with its own C
    runtime asks for the type of the standard handles: numpy's OpenBLAS, which
    torch loads, then waits for ever under the loader lock. So the pipe is
    peeked and only read when bytes are waiting.
    """

    def __init__(self, fd: int = 0) -> None:
        self._fd = fd
        self._buffer = b""
        self._available = _pipe_peek(fd)

    def line(self) -> bytes | None:
        """The next line without its newline; None at the end of input."""
        while b"\n" not in self._buffer:
            chunk = self._read()
            if chunk is None:
                return None
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line

    def _read(self) -> bytes | None:
        if self._available is None:
            return os.read(self._fd, _CHUNK) or None
        while True:
            count = self._available()
            if count is None:
                return None
            if count:
                return os.read(self._fd, min(count, _CHUNK))
            time.sleep(_PEEK_S)


def _watch_stdin(
    stdin: _Stdin, token: Any, grace: float, finished: threading.Event
) -> None:
    """Set ``token`` on a stop line; on end of input also exit after ``grace``.

    The server closes stdin once it has read ``job_end``; that end of input is
    the normal way out and changes nothing.
    """
    while (raw := stdin.line()) is not None:
        try:
            message = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(message, dict) and message.get("type") == "stop":
            token.cancel()
    if finished.is_set():
        return
    # The server closed the pipe mid-run: nobody will read this run any more.
    token.cancel()
    timer = threading.Timer(grace, os._exit, (ORPHAN_EXIT,))
    timer.daemon = True
    timer.start()


def _run_dir(block: Any, report: dict[str, Any], section: str | None) -> str | None:
    """The folder of the one model the block trained, as the executors read it."""
    if section is not None:
        part = report.get(section)
        run_dir = part.get("run_dir") if isinstance(part, dict) else None
        return str(run_dir) if run_dir else None
    result = getattr(block, "_train_result", None)
    model_path = getattr(result, "model_path", None)
    return str(Path(model_path).parent) if model_path else None


def _error(exc: BaseException, spawn_hint: bool) -> str:
    """The run's error as the in-thread executors wrote it."""
    from visionforge.utils.messages import tr

    if spawn_hint:
        from visionforge.core.loader_lifecycle import describe_worker_spawn_failure

        hint = describe_worker_spawn_failure(exc)
        if hint:
            return hint
    return f"{type(exc).__name__}: {str(exc) or tr('run.no_message')}"


def run(spec: dict[str, Any], pipe: _Pipe, token: Any) -> None:
    """Run the block ``spec`` names and send its events and its end on ``pipe``."""
    from visionforge.utils.messages import set_lang

    if spec.get("lang"):
        set_lang(spec["lang"])
    try:
        config = _load(spec["config_type"]).model_validate(spec["config"])
        block = _load(spec["block"])()
        block.setup(config)
        block._progress_callback = lambda event: pipe.send(
            {"type": "event", "event": event}
        )
        block._cancel_token = token
        resume = spec.get("resume_dir")
        block._resume_dir = Path(resume) if resume else None
        block.run()
        report = block.report()
        run_dir = _run_dir(block, report, spec.get("report_section"))
    except Exception as exc:  # noqa: BLE001 - the server records it as the run's error
        from loguru import logger

        logger.exception("Job child: the run failed")
        error = _error(exc, bool(spec.get("spawn_hint")))
        pipe.send({"type": "job_end", "status": "error", "error": error})
        return
    pipe.send({"type": "job_end", "status": "ok", "report": report, "run_dir": run_dir})


def main() -> int:
    """Read the spec, run it, and exit."""
    pipe = _Pipe(_protocol_stream())
    stdin = _Stdin()
    first = stdin.line()
    if first is None:
        return 2  # no spec: the server went away before sending it
    spec = json.loads(first.decode("utf-8"))

    from visionforge.core.cancellation import CancellationToken

    token = CancellationToken()
    finished = threading.Event()
    watcher = threading.Thread(
        target=_watch_stdin,
        args=(stdin, token, float(spec.get("grace") or 0), finished),
        daemon=True,
    )
    watcher.start()
    try:
        run(spec, pipe, token)
    finally:
        finished.set()
    # Wait for the server to close stdin: a thread still blocked reading it
    # when the interpreter shuts down aborts the process.
    watcher.join(_CLOSE_WAIT_S)
    if watcher.is_alive():
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
