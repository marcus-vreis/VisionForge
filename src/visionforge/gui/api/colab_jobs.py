"""Running one training job on a Colab VM (ADR-163, phase 2).

``ColabJobs.run_job`` allocates a runtime with ``colab new``, runs
``colab_remote_runner.py`` through ``colab exec`` and reads its stdout line by
line: training events go to the caller, which feeds the queue's SSE log, and the
final ``job_end`` decides the outcome. ``colab stop`` runs in a ``finally``, so
the VM is released whether the job ends, fails, or is stopped.

Two facts about google-colab-cli 0.7.5 shape this module (design doc,
2026-11-05): the exit code of ``exec`` is not the script's (a failing script
exits 0, a healthy one exits 1 once it outlives ``--timeout``), and ``--timeout``
is a fixed deadline rather than a silence limit. Hence the ``job_end`` event as
the only source of truth, and an ``--timeout`` longer than any job can last.

The job runs in the existing local queue for now: one job at a time, whichever
the hardware. Parallel queues are phase 4.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from visionforge.gui.api.colab_connection import (
    CLI_TIMEOUT_S,
    Popen,
    Run,
    cli_env,
    cli_path,
    profile_home,
)
from visionforge.gui.api.colab_drive import DriveConsent, consent_url
from visionforge.gui.api.colab_remote_runner import EVENT_PREFIX, REMOTE_ROOT
from visionforge.utils.messages import tr

# Longer than a Colab session can live (12 h on the free tier, 24 h on paid
# plans), so the deadline never cuts a job short and the CLI never spins past it.
EXEC_TIMEOUT_S = 26 * 3600
ALLOCATE_TIMEOUT_S = 300.0
CANCEL_POLL_S = 1.0
# A kernel that starts slowly can lose the first execute_request: in 0.7.5 the
# CLI sends it before its websocket is ready, and `exec` then waits forever
# (second real run, 2026-11-05). So a tiny warm-up `exec` must answer first...
WARMUP_MARKER = "vf-kernel-ready"
WARMUP_ATTEMPTS = 3
WARMUP_TIMEOUT_S = 60.0
# ...and, as a second defence, a runner that sends no event this long after its
# `exec` started is killed and sent again once. Its first event goes out before
# the pip install, so a silent minute means the code never ran.
STARTUP_TIMEOUT_S = 120.0
RUN_ATTEMPTS = 2
# The run comes back as one bundle (ADR-163, phase 3): a checkpoint and its
# records. The ceiling is refused before downloading, and the import checks it again.
MAX_BUNDLE_BYTES = 2 * 1024**3
DOWNLOAD_TIMEOUT_S = 1800.0
# The Drive mount asks for a browser consent on every runtime (ADR-164). Each
# attempt shows one URL; an Enter sent before the consent fails with a 400 and
# the next attempt shows a new URL. The person's time on the consent page has
# its own deadline, apart from the CLI's silence outside it, and neither touches
# STARTUP_TIMEOUT_S, which starts only with the runner's exec.
DRIVE_CONSENT_ATTEMPTS = 3
DRIVE_CONSENT_TIMEOUT_S = 600.0
DRIVE_SILENCE_TIMEOUT_S = 120.0
DRIVE_MOUNTED = "Mounted at"
DRIVE_REFUSED = "Error propagating"
_RUNNER = Path(__file__).with_name("colab_remote_runner.py")
# Session names VisionForge creates; anything else on the account is the user's.
SESSION_PREFIX = "vf-"
_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SESSION_LINE = re.compile(
    r"^\[(?P<name>[^\]]+)\] \S+ \| Hardware: (?P<hardware>[^|]+?) \|"
)

Outcome = Literal["completed", "failed", "stopped"]


@dataclass(frozen=True)
class RemoteResult:
    outcome: Outcome
    report: Any = None
    error: str | None = None
    # Where the runner left the run's bundle on the VM ({"path", "size"}), and the
    # local copy once `colab download` brought it here (ADR-163, phase 3).
    remote_bundle: dict[str, Any] | None = None
    bundle: Path | None = None


@dataclass(frozen=True)
class RemoteSession:
    name: str
    hardware: str

    @property
    def ours(self) -> bool:
        return self.name.startswith(SESSION_PREFIX)


def session_name(now: datetime | None = None) -> str:
    return SESSION_PREFIX + (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def is_session_name(name: str) -> bool:
    return bool(_SESSION_NAME.match(name))


def build_script(payload: dict[str, Any]) -> str:
    """The runner's text with its payload in front, as one file for ``colab exec -f``."""
    blob = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    header = (
        "import base64 as _b64, json as _json\n"
        f'PAYLOAD = _json.loads(_b64.b64decode("{blob}").decode("utf-8"))\n'
    )
    return header + _RUNNER.read_text(encoding="utf-8")


def parse_event(line: str) -> dict[str, Any] | None:
    """The event a runner line carries, or None for anything else the CLI prints."""
    start = line.find(EVENT_PREFIX)
    if start < 0:
        return None
    try:
        event = json.loads(line[start + len(EVENT_PREFIX) :])
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def parse_sessions(text: str) -> list[RemoteSession]:
    sessions = []
    for line in text.splitlines():
        match = _SESSION_LINE.match(line.strip())
        if match:
            sessions.append(RemoteSession(match["name"], match["hardware"].strip()))
    return sessions


class ColabJobs:
    """Allocate, run and release; the CLI calls are injected for the tests."""

    def __init__(
        self, run: Run = subprocess.run, popen: Popen = subprocess.Popen
    ) -> None:
        self._run = run
        self._popen = popen

    def run_job(
        self,
        slug: str,
        accelerator: str,
        payload: dict[str, Any],
        on_event: Callable[[dict[str, Any]], None],
        cancelled: Callable[[], bool],
        consent: DriveConsent | None = None,
    ) -> RemoteResult:
        name = session_name()
        script = profile_home(slug) / "jobs" / f"{name}.py"
        try:
            new = self._cli(
                slug, ALLOCATE_TIMEOUT_S, "new", "-s", name, "--gpu", accelerator
            )
            if new.returncode != 0:
                return RemoteResult(
                    "failed",
                    error=tr(
                        "colab.no_accelerator",
                        accelerator=accelerator,
                        detail=_tail(new),
                    ),
                )
            on_event(
                {
                    "event": "remote",
                    "stage": "allocated",
                    "session": name,
                    "accelerator": accelerator,
                }
            )
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text(build_script(payload), encoding="utf-8")
            warm = self._warm_up(
                slug, name, script.with_suffix(".warmup.py"), cancelled
            )
            if warm is not None:
                return warm
            on_event({"event": "remote", "stage": "kernel_ready", "session": name})
            if needs_drive(payload):
                mounted = self._mount_drive(
                    slug, name, on_event, cancelled, consent or DriveConsent()
                )
                if mounted is not None:
                    return mounted
            result = self._run_runner(slug, name, script, on_event, cancelled)
            if result.outcome != "completed" or result.remote_bundle is None:
                return result
            # Before the `finally`: once the VM stops, its files are gone.
            on_event({"event": "remote", "stage": "download", "session": name})
            return self._download(slug, name, result, script.with_suffix(".zip"))
        finally:
            stop = self._cli(slug, CLI_TIMEOUT_S, "stop", "-s", name)
            if stop.returncode != 0:
                logger.warning("colab stop -s {} failed: {}", name, _tail(stop))
            script.unlink(missing_ok=True)
            script.with_suffix(".warmup.py").unlink(missing_ok=True)

    def _download(
        self, slug: str, name: str, result: RemoteResult, local: Path
    ) -> RemoteResult:
        """Bring the run's bundle here; a failure fails the job with the reason."""
        remote = result.remote_bundle or {}
        path, size = remote.get("path"), remote.get("size")
        if (
            not isinstance(path, str)
            or not path.startswith(f"{REMOTE_ROOT.as_posix()}/")
            or ".." in path.split("/")
            or not isinstance(size, int)
        ):
            return RemoteResult(
                "failed", error=tr("colab.download_failed", detail=repr(path))
            )
        if size > MAX_BUNDLE_BYTES:
            return RemoteResult(
                "failed",
                error=tr(
                    "colab.bundle_too_large",
                    size_mb=size // 2**20,
                    limit_mb=MAX_BUNDLE_BYTES // 2**20,
                ),
            )
        local.unlink(missing_ok=True)
        proc = self._cli(
            slug, DOWNLOAD_TIMEOUT_S, "download", "-s", name, path, str(local)
        )
        # The file is the proof, as job_end is for exec: its exit code is not trusted.
        if not local.is_file() or local.stat().st_size != size:
            local.unlink(missing_ok=True)
            return RemoteResult(
                "failed", error=tr("colab.download_failed", detail=_tail(proc))
            )
        return replace(result, bundle=local)

    def _mount_drive(
        self,
        slug: str,
        name: str,
        on_event: Callable[[dict[str, Any]], None],
        cancelled: Callable[[], bool],
        consent: DriveConsent,
    ) -> RemoteResult | None:
        """Mount the Drive on the VM, asking for the consent through the GUI; None when mounted."""
        on_event({"event": "remote", "stage": "drive_mount", "session": name})
        for attempt in range(1, DRIVE_CONSENT_ATTEMPTS + 1):
            outcome, detail = self._drivemount(
                slug, name, attempt, on_event, cancelled, consent
            )
            if outcome == "mounted":
                on_event({"event": "remote", "stage": "drive_mounted", "session": name})
                return None
            if outcome == "cancelled":
                return RemoteResult("stopped", error=tr("colab.run_stopped"))
            if outcome == "failed":
                return RemoteResult(
                    "failed", error=tr("colab.drive_mount_failed", detail=detail)
                )
            logger.warning(
                "Drive consent for {} not given ({}/{}): {}",
                name,
                attempt,
                DRIVE_CONSENT_ATTEMPTS,
                outcome,
            )
        return RemoteResult(
            "failed",
            error=tr("colab.drive_consent_failed", attempts=DRIVE_CONSENT_ATTEMPTS),
        )

    def _drivemount(
        self,
        slug: str,
        name: str,
        attempt: int,
        on_event: Callable[[dict[str, Any]], None],
        cancelled: Callable[[], bool],
        consent: DriveConsent,
    ) -> tuple[str, str]:
        """One ``colab drivemount``: mounted, refused (no consent), cancelled or failed.

        The URL goes to the GUI only, never to the log: the ``state`` it carries
        holds a token.
        """
        proc = self._popen(
            [str(cli_path()), "drivemount", "-s", name],
            env=cli_env(slug),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        clock = {"last": time.monotonic(), "consent": False}
        killed = {"why": ""}
        done = threading.Event()

        def watch() -> None:
            while not done.wait(CANCEL_POLL_S):
                if clock["consent"]:
                    continue
                if cancelled():
                    killed["why"] = "cancelled"
                elif time.monotonic() - float(clock["last"]) > DRIVE_SILENCE_TIMEOUT_S:
                    killed["why"] = "silent"
                else:
                    continue
                proc.kill()
                return

        threading.Thread(target=watch, name=f"colab-drive-{name}", daemon=True).start()
        mounted, refused, other = False, False, ""
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                clock["last"] = time.monotonic()
                url = consent_url(line)
                if url is not None:
                    clock["consent"] = True
                    on_event(
                        {
                            "event": "drive_consent_required",
                            "url": url,
                            "attempt": attempt,
                            "attempts": DRIVE_CONSENT_ATTEMPTS,
                            "session": name,
                        }
                    )
                    answer = consent.wait(DRIVE_CONSENT_TIMEOUT_S, cancelled)
                    clock["last"], clock["consent"] = time.monotonic(), False
                    if answer != "granted":
                        killed["why"] = answer
                        proc.kill()
                        break
                    on_event({"event": "remote", "stage": "drive_consent_sent"})
                    _send_enter(proc)
                elif DRIVE_MOUNTED in line:
                    mounted = True
                elif DRIVE_REFUSED in line:
                    # Without the consent the CLI hangs after this line.
                    refused = True
                    proc.kill()
                    break
                elif line.strip() and not line.lstrip().startswith("<"):
                    other = line.strip()
            proc.wait()
        finally:
            done.set()
        if killed["why"] == "cancelled":
            return "cancelled", ""
        if mounted:
            return "mounted", ""
        if refused or killed["why"] == "timeout":
            return "refused", killed["why"] or "400"
        if killed["why"] == "silent":
            return "failed", f"no output in {DRIVE_SILENCE_TIMEOUT_S:.0f}s"
        return "failed", other or f"exit {proc.returncode}"

    def _warm_up(
        self, slug: str, name: str, path: Path, cancelled: Callable[[], bool]
    ) -> RemoteResult | None:
        """Wait until the kernel answers a print; None when it did."""
        path.write_text(f"print({WARMUP_MARKER!r}, flush=True)\n", encoding="utf-8")
        for attempt in range(1, WARMUP_ATTEMPTS + 1):
            if cancelled():
                return RemoteResult("stopped", error=tr("colab.run_stopped"))
            proc = self._cli(
                slug,
                WARMUP_TIMEOUT_S,
                "exec",
                "-s",
                name,
                "-f",
                str(path),
                "--timeout",
                str(int(WARMUP_TIMEOUT_S / 2)),
            )
            # The marker, not the exit code: exec's exit code is not the script's.
            if WARMUP_MARKER in (proc.stdout or ""):
                return None
            logger.warning(
                "Colab kernel of {} did not answer the warm-up ({}/{}): {}",
                name,
                attempt,
                WARMUP_ATTEMPTS,
                _tail(proc),
            )
        return RemoteResult(
            "failed", error=tr("colab.kernel_not_ready", attempts=WARMUP_ATTEMPTS)
        )

    def _run_runner(
        self,
        slug: str,
        name: str,
        script: Path,
        on_event: Callable[[dict[str, Any]], None],
        cancelled: Callable[[], bool],
    ) -> RemoteResult:
        for attempt in range(1, RUN_ATTEMPTS + 1):
            result = self._exec(slug, name, script, on_event, cancelled)
            if result is not None:
                return result
            logger.warning(
                "Colab runner on {} sent nothing in {:.0f}s ({}/{})",
                name,
                STARTUP_TIMEOUT_S,
                attempt,
                RUN_ATTEMPTS,
            )
            if attempt < RUN_ATTEMPTS:
                on_event({"event": "remote", "stage": "retry", "session": name})
        return RemoteResult(
            "failed",
            error=tr(
                "colab.run_no_start",
                seconds=int(STARTUP_TIMEOUT_S),
                attempts=RUN_ATTEMPTS,
            ),
        )

    def _exec(
        self,
        slug: str,
        name: str,
        script: Path,
        on_event: Callable[[dict[str, Any]], None],
        cancelled: Callable[[], bool],
    ) -> RemoteResult | None:
        """One `exec` of the runner; None when it never started (see STARTUP_TIMEOUT_S)."""
        proc = self._popen(
            [
                str(cli_path()),
                "exec",
                "-s",
                name,
                "-f",
                str(script),
                "--timeout",
                str(EXEC_TIMEOUT_S),
            ],
            env=cli_env(slug),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        stopped = threading.Event()
        stalled = threading.Event()
        started = threading.Event()
        done = threading.Event()
        launched = time.monotonic()

        def watch() -> None:
            # Stopping means releasing the VM now: the local exec is killed and
            # the `finally` of run_job stops the session.
            while not done.wait(CANCEL_POLL_S):
                if cancelled():
                    stopped.set()
                    proc.kill()
                    return
                if (
                    not started.is_set()
                    and time.monotonic() - launched > STARTUP_TIMEOUT_S
                ):
                    stalled.set()
                    proc.kill()
                    return

        threading.Thread(target=watch, name=f"colab-watch-{name}", daemon=True).start()
        end: dict[str, Any] | None = None
        other: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                event = parse_event(line)
                if event is not None:
                    started.set()
                if event is None:
                    if line.strip():
                        other = [*other[-19:], line.rstrip()]
                elif event.get("event") == "job_end":
                    end = event
                else:
                    on_event(event)
            proc.wait()
        finally:
            done.set()
        if stalled.is_set():
            return None
        return _result(end, stopped.is_set(), other)

    # ── sessions left behind ─────────────────────────────────────────────────

    def sessions(self, slug: str) -> list[RemoteSession]:
        proc = self._cli(slug, CLI_TIMEOUT_S, "sessions")
        return parse_sessions(proc.stdout) if proc.returncode == 0 else []

    def stop_session(self, slug: str, name: str) -> bool:
        if not is_session_name(name):
            return False
        return self._cli(slug, CLI_TIMEOUT_S, "stop", "-s", name).returncode == 0

    def _cli(
        self, slug: str, timeout: float, *args: str
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._run(
                [str(cli_path()), *args],
                env=cli_env(slug),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args, 124, stdout="", stderr=f"timeout after {timeout:.0f}s"
            )


def needs_drive(payload: dict[str, Any]) -> bool:
    """Whether the job mounts the Drive: a dataset in it, or the run saved to it."""
    source = payload.get("source") or {}
    return source.get("kind") == "drive" or bool(payload.get("drive_export"))


def _send_enter(proc: Any) -> None:
    """The Enter the CLI waits for after the consent (ADR-164)."""
    if proc.stdin is None:
        return
    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
    except (OSError, ValueError):
        logger.warning("Could not answer the Drive consent prompt: stdin closed")


def _result(
    end: dict[str, Any] | None, stopped: bool, other: list[str]
) -> RemoteResult:
    if stopped:
        return RemoteResult("stopped", error=tr("colab.run_stopped"))
    if end is None:
        detail = other[-1] if other else ""
        return RemoteResult("failed", error=tr("colab.run_no_end", detail=detail))
    if end.get("status") == "ok":
        bundle = end.get("bundle")
        return RemoteResult(
            "completed",
            report=end.get("report"),
            remote_bundle=bundle if isinstance(bundle, dict) else None,
        )
    error = str(end.get("error") or "")
    if end.get("detail"):
        error = f"{error} ({end['detail']})"
    return RemoteResult("failed", error=error or tr("colab.run_no_end", detail=""))


def _tail(proc: subprocess.CompletedProcess[str]) -> str:
    lines = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()
    return lines[-1] if lines else f"exit {proc.returncode}"


__all__ = [
    "DRIVE_CONSENT_ATTEMPTS",
    "EXEC_TIMEOUT_S",
    "ColabJobs",
    "RemoteResult",
    "RemoteSession",
    "build_script",
    "needs_drive",
    "parse_event",
    "parse_sessions",
    "session_name",
]
