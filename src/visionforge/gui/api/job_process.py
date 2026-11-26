"""A server job's training in a process of its own (ADR-248, ADR-212 phase 1).

On a server every person's training used to be ``asyncio.to_thread(block.run)``:
a thread of the server, with its memory, its open files, its environment (the
tokens in it) and its Python state. Here the block runs in a child Python
started with ``python -m visionforge.gui.api.job_worker``:

- **Its own environment.** Only the variables a training needs are passed (the
  paths of the system, Python, CUDA and the caches); anything else, and any name
  that looks like a secret, is left out. ``VISIONFORGE_HOME`` points at an empty
  folder, so code that looks for the profile tokens finds none.
- **Its own working folder,** a fresh temporary folder removed when the job
  ends. The config's own paths are made absolute before they cross over, so a
  relative path still means what the server checked (ADR-217).
- **Events over a pipe.** The child writes one JSON object per line on a pipe of
  its own (its stdout is sent to stderr, so a library's prints cannot be taken
  for an event); the last one is ``job_end``. The events reach the run's log in
  the order they were written, as the trainer thread's did.
- **Stop, then kill.** A stop is a ``stop`` line on the child's stdin: the block
  stops at its stop point (ADR-111) and keeps what it earned. A child that has
  not ended ``grace`` seconds later is killed and the run marked failed.

What it does not do: a child of the same OS user can still open any file that
user can, the config folder included. Denying that takes a container or another
account (ADR-248). The isolation here is of memory, environment, Python state
and crashes: a child that dies takes only its own run with it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from loguru import logger
from pydantic import BaseModel

from visionforge.core.cancellation import CancellationToken
from visionforge.gui.api.config_scope import DATASET_FILES
from visionforge.gui.api.run_events import RunEventLog
from visionforge.utils.messages import bound_lang, tr

# The child's entry point: ``python -m`` this module.
WORKER_MODULE = "visionforge.gui.api.job_worker"

# How long a job may keep running after it was asked to stop, before it is
# killed. A stop is honoured at the end of an epoch, and an epoch can be long;
# killing sooner would throw away what the stop is meant to keep (ADR-111).
STOP_GRACE_S = 600.0
# How long a child may take to exit after its last line; it has nothing left
# to say, so killing it then loses nothing.
EXIT_GRACE_S = 30.0
# How long the reader may take to see the end of the pipe after the child exits
# (a forked DataLoader worker can hold it open for a moment).
_DRAIN_S = 5.0
_POLL_S = 0.1

# Variables passed to the child by name, compared in upper case.
_KEEP_NAMES = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "COMMONPROGRAMFILES",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "OS",
        "LANG",
        "LANGUAGE",
        "TZ",
        "USER",
        "USERNAME",
        "LOGNAME",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "YOLO_CONFIG_DIR",
        "YOLO_OFFLINE",
        "MPLCONFIGDIR",
        "VISIONFORGE_LANG",
        "VISIONFORGE_OFFLINE",
        "VISIONFORGE_CONTAINER",
        # Set only when the server itself runs under coverage.py (the test
        # suite, `patch = ["subprocess"]`): the child then measures itself too.
        "COVERAGE_PROCESS_CONFIG",
        "COVERAGE_PROCESS_START",
        "COVERAGE_CORE",
    }
)
# Families passed whole: locale, Python, CUDA and the math libraries' threads.
_KEEP_PREFIXES = (
    "LC_",
    "PYTHON",
    "CUDA_",
    "CUDNN_",
    "CUBLAS_",
    "NVIDIA_",
    "NCCL_",
    "PYTORCH_",
    "TORCH_",
    "OMP_",
    "MKL_",
    "KMP_",
)
# A name with one of these is never passed, even when it matches the above.
_SECRET_MARKS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "KEY", "CREDENTIAL", "AUTH")
_PROXIES = frozenset({"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"})


def child_env(environ: Mapping[str, str], home: Path) -> dict[str, str]:
    """The environment of a job's child: what a training needs, and no secrets.

    ``home`` becomes ``VISIONFORGE_HOME``: an empty folder, so code that looks
    for the profile tokens or the dataset providers' keys finds none.
    """
    env: dict[str, str] = {}
    for name, value in environ.items():
        upper = name.upper()
        if any(mark in upper for mark in _SECRET_MARKS):
            continue
        if upper in _PROXIES and "@" in value:
            continue  # a proxy address that carries a user and password
        if upper in _KEEP_NAMES or upper.startswith(_KEEP_PREFIXES):
            env[name] = value
    env["VISIONFORGE_HOME"] = str(home)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def pinned_config(config: BaseModel) -> dict[str, Any]:
    """``config`` as JSON with every path made absolute against the server's folder.

    The child works in another folder; a relative path must still name what the
    server checked when the run was submitted. A path the loader resolves
    against the dataset folder (``DATASET_FILES``) stays as it is.
    """
    data = config.model_dump(mode="json")
    _pin(config, data, "")
    return data


def _pin(model: BaseModel, data: dict[str, Any], prefix: str) -> None:
    for name in type(model).model_fields:
        value = getattr(model, name)
        dotted = f"{prefix}{name}"
        if isinstance(value, BaseModel) and isinstance(data.get(name), dict):
            _pin(value, data[name], f"{dotted}.")
        elif isinstance(value, Path) and dotted not in DATASET_FILES:
            data[name] = str(value.expanduser().absolute())


def import_path(obj: Callable[..., Any]) -> str:
    """``module:QualName`` of a class or function, which the child imports by that name."""
    return f"{obj.__module__}:{obj.__qualname__}"


def child_spec(
    block_cls: type[Any],
    config: BaseModel,
    resume_dir: Path | None,
    *,
    report_section: str | None,
    spawn_hint: bool = False,
    single_model: bool = True,
    grace: float = STOP_GRACE_S,
) -> dict[str, Any]:
    """What the child needs to run one block, as the first line it reads.

    ``report_section`` names the part of the report that holds the run folder;
    None reads it from the block's training result (classification).
    ``spawn_hint`` words a DataLoader worker-spawn failure as the classification
    executor does. ``single_model`` False is a block that trains several models
    (a K-fold, a search, a comparison): it has no one run folder, so none is
    read, as in the thread (ADR-251).
    """
    return {
        "block": import_path(block_cls),
        "config_type": import_path(type(config)),
        "config": pinned_config(config),
        "resume_dir": str(resume_dir.absolute()) if resume_dir else None,
        "report_section": report_section,
        "single_model": single_model,
        "spawn_hint": spawn_hint,
        "lang": bound_lang(),
        "grace": grace,
    }


@dataclass
class ChildOutcome:
    """How a child's run ended: the fields of the job's run dict."""

    status: str
    report: dict[str, Any] | None = None
    run_dir: str | None = None
    error: str | None = None
    returncode: int | None = None
    killed: bool = False
    # A strategy's call (ADR-252): what the function returned, as ``job_wire`` wrote it.
    result: Any = None

    def run(self, run_id: str) -> dict[str, Any]:
        """The run dict ``/experiment/status`` and ``/experiment/result`` read."""
        return {
            "run_id": run_id,
            "status": self.status,
            "error": self.error,
            "report": self.report,
            "run_dir": self.run_dir,
        }


@dataclass
class _Relay:
    """What the reader thread saw on the child's pipe."""

    put_event: Callable[[dict[str, Any]], None]
    end: dict[str, Any] | None = None
    drained: threading.Event = field(default_factory=threading.Event)

    def read(self, stream: IO[bytes]) -> None:
        try:
            for raw in stream:
                self._take(raw)
        finally:
            self.drained.set()

    def _take(self, raw: bytes) -> None:
        try:
            message = json.loads(raw.decode("utf-8"))
        except ValueError:  # a UnicodeDecodeError too
            logger.warning("Job child: unreadable line {!r}", raw[:200])
            return
        if not isinstance(message, dict):
            return
        if message.get("type") == "event" and isinstance(message.get("event"), dict):
            self.put_event(message["event"])
        elif message.get("type") == "job_end":
            self.end = message


class ChildJob:
    """One child process running one block; :meth:`supervise` blocks until it ends."""

    def __init__(
        self,
        spec: dict[str, Any],
        put_event: Callable[[dict[str, Any]], None],
        token: CancellationToken,
        *,
        grace: float = STOP_GRACE_S,
        exit_grace: float = EXIT_GRACE_S,
    ) -> None:
        self._spec = {**spec, "grace": grace}
        self._token = token
        self._grace = grace
        self._exit_grace = exit_grace
        self._relay = _Relay(put_event)
        self._proc: subprocess.Popen[bytes] | None = None
        self._killed = False
        self._lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        """The child's process id once started."""
        return self._proc.pid if self._proc is not None else None

    def supervise(self) -> ChildOutcome:
        """Start the child, relay its events, forward a stop, and report how it ended."""
        workdir = Path(tempfile.mkdtemp(prefix="vf-job-"))
        try:
            home = workdir / "home"
            home.mkdir()
            try:
                proc = self._start(workdir, home)
            except OSError as exc:
                return ChildOutcome("failed", error=tr("job.child_start", error=exc))
            reader = threading.Thread(
                target=self._relay.read, args=(proc.stdout,), daemon=True
            )
            reader.start()
            self._watch(proc)
            if self._relay.drained.wait(_DRAIN_S) and proc.stdout is not None:
                proc.stdout.close()
            self._close_stdin(proc)
            return self._outcome(proc)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def kill(self) -> None:
        """End the child now (the server is stopping or the queue was reset)."""
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                self._killed = True
                proc.kill()

    def _start(self, workdir: Path, home: Path) -> subprocess.Popen[bytes]:
        env = child_env(os.environ, home)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        # The command is this Python and a module of this package; the spec
        # goes on stdin, never on the command line.
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=workdir,
            env=env,
            creationflags=flags,
        )
        with self._lock:
            self._proc = proc
        self._send(proc, self._spec)
        return proc

    def _send(self, proc: subprocess.Popen[bytes], message: dict[str, Any]) -> None:
        if proc.stdin is None:
            return
        try:
            proc.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
            proc.stdin.flush()
        except OSError:
            pass  # the child is gone; its exit says how it ended

    @staticmethod
    def _close_stdin(proc: subprocess.Popen[bytes]) -> None:
        if proc.stdin is not None and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass  # already gone

    def _watch(self, proc: subprocess.Popen[bytes]) -> None:
        """Wait for the child; forward a stop; kill it past the grace."""
        deadline: float | None = None
        ended = False
        while True:
            try:
                proc.wait(timeout=_POLL_S)
                return
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if self._relay.end is not None and not ended:
                # It said everything; closing its stdin lets it exit.
                ended = True
                self._close_stdin(proc)
                deadline = now + self._exit_grace
            if deadline is None and self._token.cancelled:
                self._send(proc, {"type": "stop"})
                deadline = now + self._grace
            if deadline is not None and now >= deadline:
                if self._relay.end is None:
                    logger.warning(
                        "Job child {} ignored the stop; killing it.", proc.pid
                    )
                    with self._lock:
                        self._killed = True
                proc.kill()
                proc.wait()
                return

    def _outcome(self, proc: subprocess.Popen[bytes]) -> ChildOutcome:
        code = proc.returncode
        end = self._relay.end
        if end is not None and end.get("status") == "ok":
            report = end.get("report")
            return ChildOutcome(
                "completed",
                report=report if isinstance(report, dict) else {},
                run_dir=end.get("run_dir"),
                returncode=code,
                result=end.get("result"),
            )
        if end is not None:
            error = str(end.get("error") or tr("run.no_message"))
            return ChildOutcome("failed", error=error, returncode=code)
        if self._killed:
            error = tr("job.child_killed", seconds=round(self._grace))
            return ChildOutcome("failed", error=error, returncode=code, killed=True)
        return ChildOutcome(
            "failed", error=tr("job.child_died", code=code), returncode=code
        )


async def run_in_child(
    spec: dict[str, Any],
    put_event: Callable[[dict[str, Any]], None],
    token: CancellationToken,
    *,
    grace: float = STOP_GRACE_S,
) -> ChildOutcome:
    """Run one block or call in a child process from the event loop; see :class:`ChildJob`.

    If the awaiting task is cancelled (the server stops, the queue is reset) the
    child is killed rather than left training with nobody to read it.
    """
    job = ChildJob(spec, put_event, token, grace=grace)
    try:
        return await asyncio.to_thread(job.supervise)
    except asyncio.CancelledError:
        job.kill()
        raise


async def execute_in_child(
    run_id: str,
    spec: dict[str, Any],
    log: RunEventLog,
    token: CancellationToken,
) -> dict[str, Any]:
    """Run ``spec`` in a child, feeding its events to ``log``; return the run dict.

    The caller stores the run dict and then closes ``log``, in that order, as
    the in-thread executors do: a reader that sees the end of the stream finds
    the run already finished.
    """
    loop = asyncio.get_running_loop()

    def put_event(event: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(log.put(event), loop)

    try:
        outcome = await run_in_child(spec, put_event, token)
    except Exception as exc:  # noqa: BLE001 - a failure here is this run's failure
        logger.exception("GUI: job {} could not run in a child process", run_id)
        message = str(exc) or tr("run.no_message")
        outcome = ChildOutcome("failed", error=f"{type(exc).__name__}: {message}")
    return outcome.run(run_id)


__all__ = [
    "EXIT_GRACE_S",
    "STOP_GRACE_S",
    "WORKER_MODULE",
    "ChildJob",
    "ChildOutcome",
    "child_env",
    "child_spec",
    "execute_in_child",
    "import_path",
    "pinned_config",
    "run_in_child",
]
