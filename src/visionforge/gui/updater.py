"""Install a VisionForge release and start the GUI again (ADR-120).

Run by ``gui/api/updates.spawn_updater`` as a copy outside the package, in a
process of its own: the install replaces the package's files, and on Windows a
running program cannot replace its own. So this file imports nothing from
VisionForge and only the standard library, and takes everything it needs as one
JSON argument:

``host``/``port`` of the server and the ``pids`` of its processes to wait for,
``target`` version, ``install`` and ``relaunch`` argv lists, ``restore`` (the argv
that reinstalls the version that was running), ``cwd`` to relaunch in (the outputs
folder is relative to it), and the ``log`` and ``status`` files to write.

Waiting for the port is not enough on Windows: the server closes its socket and
lives on for seconds while it shuts down, and the ``visionforge.exe`` launcher that
started it stays until it does. pip, finding that file locked, had already removed
the package and did not put it back -- a failed update left no VisionForge at all.
So the updater waits for the processes, retries an install that still finds a
file locked, and, if the install fails anyway, reinstalls the previous version
before starting the GUI again.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PORT_WAIT_SECONDS = 60.0
PROCESS_WAIT_SECONDS = 60.0
INSTALL_TIMEOUT_SECONDS = 900.0
LOCKED_RETRIES = 5
LOCKED_RETRY_SECONDS = 5.0
# What pip and uv print when Windows refuses to replace a file another process
# holds ("os error 5" is uv's access-denied on the rename that follows).
LOCKED_MARKERS = ("WinError 32", "os error 32", "os error 5")


def port_in_use(host: str, port: int) -> bool:
    """Whether something still answers on ``host:port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_for_port_free(
    host: str, port: int, timeout: float = PORT_WAIT_SECONDS, poll: float = 0.5
) -> bool:
    """Wait until the old server has let go of its port; False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_in_use(host, port):
            return True
        time.sleep(poll)
    return False


def process_alive(pid: int) -> bool:
    """Whether process ``pid`` is still running."""
    if sys.platform == "win32":
        import ctypes

        synchronize, wait_timeout = 0x00100000, 0x00000102
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return bool(kernel32.WaitForSingleObject(handle, 0) == wait_timeout)
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_processes_gone(
    pids: list[int], timeout: float = PROCESS_WAIT_SECONDS, poll: float = 0.5
) -> bool:
    """Wait until none of ``pids`` runs any more; False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(process_alive(pid) for pid in pids):
            return True
        time.sleep(poll)
    return False


def run_install(command: list[str], log: Path) -> int:
    """Run the install, appending its output to ``log``; its return code."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n=== {datetime.now(UTC).isoformat()} {' '.join(command)}\n")
        out.flush()
        # This process has no console (it was started detached), so Windows
        # would open a new console window for pip/uv: an empty terminal the
        # user sees pop up and wonders about.
        creationflags: int = (
            subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            if sys.platform == "win32"
            else 0
        )
        try:
            # argv from the server, no shell
            completed = subprocess.run(  # noqa: S603
                command,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=INSTALL_TIMEOUT_SECONDS,
                check=False,
                creationflags=creationflags,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            out.write(f"install did not run: {exc}\n")
            return -1
    return completed.returncode


def install_with_retries(
    command: list[str],
    log: Path,
    retries: int = LOCKED_RETRIES,
    pause: float = LOCKED_RETRY_SECONDS,
) -> int:
    """``run_install``, again while the log says a file was still locked."""
    code = run_install(command, log)
    for _ in range(retries):
        tail = _tail(log)
        if code == 0 or not any(marker in tail for marker in LOCKED_MARKERS):
            break
        time.sleep(pause)
        code = run_install(command, log)
    return code


def _tail(log: Path, size: int = 4000) -> str:
    try:
        return log.read_text(encoding="utf-8", errors="replace")[-size:]
    except OSError:
        return ""


def relaunch(command: list[str], cwd: str) -> None:
    """Start the GUI again, detached from this short-lived process."""
    # The page that asked for the update reloads itself; no second tab.
    env = {**os.environ, "VISIONFORGE_NO_BROWSER": "1"}
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
    else:
        kwargs["start_new_session"] = True
    # argv from the server, no shell
    subprocess.Popen(command, **kwargs)  # noqa: S603


def main(args: dict[str, Any]) -> int:
    """Wait, install (or restore), record, relaunch. The return code is the install's."""
    log = Path(args["log"])
    status = Path(args["status"])
    gone = wait_for_processes_gone([int(p) for p in args.get("pids", [])])
    freed = gone and wait_for_port_free(args["host"], int(args["port"]))
    code = install_with_retries(list(args["install"]), log) if freed else -2
    restored: bool | None = None
    if freed and code != 0 and args.get("restore"):
        # The failed install may have removed the package already: put the
        # version that was running back before starting the GUI again.
        restored = install_with_retries(list(args["restore"]), log) == 0
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(
        json.dumps(
            {
                "target": args["target"],
                "ok": code == 0,
                "returncode": code,
                "port_freed": freed,
                "restored": restored,
                "finished_at": datetime.now(UTC).isoformat(),
                "log": str(log),
            }
        ),
        encoding="utf-8",
    )
    if freed:
        relaunch(list(args["relaunch"]), args["cwd"])
    _remove_own_copy()
    return code


def _remove_own_copy() -> None:
    """Delete the temporary folder this copy was run from, and nothing else."""
    import shutil

    folder = Path(__file__).resolve().parent
    if folder.name.startswith("vf-update-"):
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(json.loads(sys.argv[1])))
