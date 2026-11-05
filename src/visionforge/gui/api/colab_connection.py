"""The connection to Google Colab through the official CLI (ADR-163, phase 1).

VisionForge does not talk to Colab itself. It drives ``google-colab-cli``, pinned
to one version and installed into a venv VisionForge owns, in
``~/.visionforge/colab-cli/venv``: never the project environment, whose torch
build a pip resolve could replace.

Each profile logs in on its own. Every call runs with ``USERPROFILE`` and
``HOME`` pointing at ``~/.visionforge/colab/<profile>/``, so the CLI keeps the
token, its session list and its log there, and two profiles can use two Google
accounts. The token itself is never read here: whether a profile is logged in is
whether ``token.json`` exists, and the account is the e-mail ``colab whoami``
prints.

The login is Google's copy-and-paste flow. ``colab sessions`` (which allocates
nothing) prints an authorization URL and waits for a code on stdin. The GUI shows
the URL and sends back the code the person pasted, which goes to that stdin and
nowhere else.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, Protocol

from loguru import logger

from visionforge.utils.credentials import config_dir
from visionforge.utils.messages import tr

PACKAGE = "google-colab-cli"
PINNED_VERSION = "0.7.5"

INSTALL_TIMEOUT_S = 600.0
CLI_TIMEOUT_S = 60.0
# How long `colab sessions` may take to print the authorization URL.
LOGIN_URL_TIMEOUT_S = 30.0
# How long the CLI may take to exchange the code for a token.
LOGIN_FINISH_TIMEOUT_S = 60.0
MAX_CODE_LENGTH = 512

_AUTH_URL = re.compile(r"https://accounts\.google\.com/\S+")
_VERSION = re.compile(r"Version:\s*(\S+)")
_EMAIL = re.compile(r"^Email:\s*(\S+@\S+)\s*$", re.MULTILINE)
_BALANCE = re.compile(r"Current balance:\s*([0-9.]+)")
_RATE = re.compile(r"Usage rate:\s*([0-9.]+)")
_ASSIGNMENTS = re.compile(r"Active assignments:\s*(\d+)")

InstallState = Literal["idle", "installing", "failed"]
Run = Callable[..., subprocess.CompletedProcess[str]]


class ColabError(Exception):
    """A CLI step failed; the message is safe to show (no token, no code)."""


@dataclass(frozen=True)
class Usage:
    balance: float | None
    rate_per_hour: float | None
    active_assignments: int | None

    @property
    def hours_left(self) -> float | None:
        """Balance over the current rate; None while nothing is running."""
        if self.balance is None or not self.rate_per_hour:
            return None
        return self.balance / self.rate_per_hour


def cli_root() -> Path:
    return config_dir() / "colab-cli"


def venv_dir() -> Path:
    return cli_root() / "venv"


def _venv_bin(name: str) -> Path:
    if sys.platform == "win32":
        return venv_dir() / "Scripts" / f"{name}.exe"
    return venv_dir() / "bin" / name


def cli_path() -> Path:
    return _venv_bin("colab")


def profile_home(slug: str) -> Path:
    return config_dir() / "colab" / slug


def token_path(slug: str) -> Path:
    return profile_home(slug) / ".config" / "colab-cli" / "token.json"


def cli_env(slug: str) -> dict[str, str]:
    """The environment that confines the CLI's state to the profile's folder."""
    home = profile_home(slug)
    env = dict(os.environ)
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)
    # Piped output is block-buffered otherwise, and live events would arrive late.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def parse_version(text: str) -> str | None:
    match = _VERSION.search(text)
    return match.group(1) if match else None


def parse_email(text: str) -> str | None:
    match = _EMAIL.search(text)
    return match.group(1) if match else None


def parse_usage(text: str) -> Usage:
    def number(pattern: re.Pattern[str]) -> float | None:
        match = pattern.search(text)
        return float(match.group(1)) if match else None

    assignments = _ASSIGNMENTS.search(text)
    return Usage(
        balance=number(_BALANCE),
        rate_per_hour=number(_RATE),
        active_assignments=int(assignments.group(1)) if assignments else None,
    )


class ColabCli:
    """Install, query and log out through the pinned CLI.

    ``run`` is ``subprocess.run`` in production and a fake in the tests, which
    never reach Google.
    """

    def __init__(self, run: Run = subprocess.run, python: str = sys.executable) -> None:
        self._run = run
        self._python = python
        self._lock = threading.Lock()
        self._install_state: InstallState = "idle"
        self._install_error: str | None = None
        # slug -> (token mtime, e-mail): whoami is a network call, the token rarely changes.
        self._accounts: dict[str, tuple[float, str | None]] = {}
        self._version: tuple[float, str | None] | None = None

    # ── install ──────────────────────────────────────────────────────────────

    def installed_version(self) -> str | None:
        exe = cli_path()
        if not exe.is_file():
            return None
        # `colab version` takes about a second to start; the exe changes only on install.
        mtime = exe.stat().st_mtime
        if self._version is not None and self._version[0] == mtime:
            return self._version[1]
        proc = self._cli(None, "version")
        version = parse_version(proc.stdout + proc.stderr)
        self._version = (mtime, version)
        return version

    @property
    def install_state(self) -> InstallState:
        return self._install_state

    @property
    def install_error(self) -> str | None:
        return self._install_error

    def start_install(self) -> bool:
        """Install in a background thread; False when one is already running."""
        with self._lock:
            if self._install_state == "installing":
                return False
            self._install_state = "installing"
            self._install_error = None
        threading.Thread(
            target=self._install, name="colab-cli-install", daemon=True
        ).start()
        return True

    def _install(self) -> None:
        try:
            if not _venv_bin("python").is_file():
                self._step([self._python, "-m", "venv", str(venv_dir())])
            self._step(
                [
                    str(_venv_bin("python")),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "-q",
                    f"{PACKAGE}=={PINNED_VERSION}",
                ]
            )
            if self.installed_version() != PINNED_VERSION:
                raise ColabError(tr("colab.install_missing", version=PINNED_VERSION))
        except (ColabError, OSError, subprocess.SubprocessError) as exc:
            logger.warning("Colab CLI install failed: {}", exc)
            with self._lock:
                self._install_state = "failed"
                self._install_error = str(exc)
            return
        with self._lock:
            self._install_state = "idle"

    def _step(self, argv: list[str]) -> None:
        proc = self._run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=INSTALL_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
            raise ColabError(
                tr("colab.install_failed", detail=" | ".join(tail) or proc.returncode)
            )

    # ── per profile ──────────────────────────────────────────────────────────

    def logged_in(self, slug: str) -> bool:
        return token_path(slug).is_file()

    def account(self, slug: str) -> str | None:
        """The logged-in e-mail, or None when logged out or not reported."""
        token = token_path(slug)
        if not token.is_file() or not cli_path().is_file():
            return None
        mtime = token.stat().st_mtime
        cached = self._accounts.get(slug)
        if cached and cached[0] == mtime:
            return cached[1]
        proc = self._cli(slug, "whoami")
        email = parse_email(proc.stdout) if proc.returncode == 0 else None
        # whoami refreshes an expired access token, which rewrites token.json.
        if token.is_file():
            self._accounts[slug] = (token.stat().st_mtime, email)
        return email

    def usage(self, slug: str) -> Usage:
        proc = self._cli(slug, "usage")
        if proc.returncode != 0:
            raise ColabError(
                tr("colab.cli_failed", command="usage", detail=_last_line(proc))
            )
        return parse_usage(proc.stdout)

    def logout(self, slug: str) -> None:
        """Forget the profile's login. Sessions still running stay the account's."""
        token_path(slug).unlink(missing_ok=True)
        self._accounts.pop(slug, None)

    def _cli(self, slug: str | None, *args: str) -> subprocess.CompletedProcess[str]:
        env = cli_env(slug) if slug else None
        try:
            return self._run(
                [str(cli_path()), *args],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CLI_TIMEOUT_S,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise ColabError(tr("colab.cli_timeout", command=args[0])) from exc


def _last_line(proc: subprocess.CompletedProcess[str]) -> str:
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    return lines[-1] if lines else ""


class LoginProcess(Protocol):
    """The part of ``subprocess.Popen`` a waiting login uses (a fake in the tests)."""

    # Read-only, so a fake with narrower attribute types still matches.
    @property
    def stdin(self) -> IO[str] | None: ...
    @property
    def stdout(self) -> IO[str] | None: ...
    @property
    def returncode(self) -> int | None: ...
    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


Popen = Callable[..., LoginProcess]


class LoginFlows:
    """At most one pending login per profile, waiting for its code."""

    def __init__(self, popen: Popen = subprocess.Popen) -> None:
        self._popen = popen
        self._lock = threading.Lock()
        self._pending: dict[str, LoginProcess] = {}

    def pending(self, slug: str) -> bool:
        with self._lock:
            proc = self._pending.get(slug)
            return proc is not None and proc.poll() is None

    def start(self, slug: str) -> str:
        """Start ``colab sessions`` and return the authorization URL it prints."""
        self.cancel(slug)
        profile_home(slug).mkdir(parents=True, exist_ok=True)
        proc = self._popen(
            [str(cli_path()), "sessions"],
            env=cli_env(slug),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        url = _read_auth_url(proc, LOGIN_URL_TIMEOUT_S)
        if url is None:
            _kill(proc)
            raise ColabError(tr("colab.login_no_url"))
        with self._lock:
            self._pending[slug] = proc
        return url

    def finish(self, slug: str, code: str) -> None:
        """Hand the pasted code to the waiting CLI; raises when no token came of it."""
        code = code.strip()
        if not code or len(code) > MAX_CODE_LENGTH or any(c.isspace() for c in code):
            raise ColabError(tr("colab.login_bad_code"))
        with self._lock:
            proc = self._pending.pop(slug, None)
        if proc is None or proc.poll() is not None:
            raise ColabError(tr("colab.login_not_started"))
        assert proc.stdin is not None
        try:
            proc.stdin.write(code + "\n")
            proc.stdin.close()
            proc.wait(timeout=LOGIN_FINISH_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _kill(proc)
            raise ColabError(tr("colab.login_stuck")) from exc
        if proc.returncode != 0 or not token_path(slug).is_file():
            raise ColabError(tr("colab.login_rejected"))

    def cancel(self, slug: str) -> None:
        with self._lock:
            proc = self._pending.pop(slug, None)
        if proc is not None:
            _kill(proc)


def _read_auth_url(proc: LoginProcess, timeout: float) -> str | None:
    """Read the CLI's output until the URL shows up, without blocking forever."""
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
            if _AUTH_URL.search(line):
                break
        else:
            lines.put(None)

    threading.Thread(target=pump, name="colab-login-url", daemon=True).start()
    while True:
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            return None
        match = _AUTH_URL.search(line)
        if match:
            return match.group(0)


def _kill(proc: LoginProcess) -> None:
    if proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Colab CLI process {} did not exit", proc.pid)
