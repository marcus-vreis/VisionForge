"""Tell the page a newer release exists, and install it when asked (ADR-120).

The page asks ``GET /api/update`` once it has drawn. The server reads the latest
version from PyPI (cached, with a short timeout, silent when offline) and says
whether this install can update itself from the button, and if not, why: a
source checkout updates with ``git pull``; an environment with neither ``pip``
nor ``uv`` cannot install anything; a client on another machine must not
reinstall the server; a training in flight must not be killed.

Applying runs ``gui/updater.py`` in a process of its own, because Windows will
not let a running program replace its own files: the server exits, the updater
waits for the port to free, installs the exact version the page announced,
records the outcome, and starts the GUI again. Nothing updates on its own: a
researcher in the middle of a study keeps the version until they press the
button, and every run records the version it trained with.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from visionforge import __version__
from visionforge.utils.credentials import config_dir
from visionforge.utils.doctor import installed_from_source

DIST_NAME = "visionforge-studio"
PYPI_URL = f"https://pypi.org/pypi/{DIST_NAME}/json"
DISABLE_ENV = "VISIONFORGE_NO_UPDATE_CHECK"
_CACHE_SECONDS = 6 * 3600
_TIMEOUT_SECONDS = 3.0

# (monotonic time of the answer, the version PyPI named or None on failure)
_cache: tuple[float, str | None] | None = None


def parse_version(text: str | None) -> tuple[int, int, int] | None:
    """``"0.14.0"`` as a tuple, or None for anything that is not a plain X.Y.Z.

    Releases are bumped by bump-my-version as major.minor.patch; a pre-release or
    a local build (``0.0.0+unknown``) is never offered as an update.
    """
    if not text:
        return None
    parts = text.strip().split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def is_newer(latest: str | None, current: str | None) -> bool:
    """Whether ``latest`` is a release later than ``current``."""
    new, old = parse_version(latest), parse_version(current)
    return new is not None and old is not None and new > old


def fetch_latest(
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: float = _TIMEOUT_SECONDS,
) -> str | None:
    """The latest release PyPI lists, or None when it cannot be read.

    Any failure (no network, a proxy, a timeout, an unexpected body) is None: the
    page then says nothing, which is the right answer for a machine offline.
    """
    try:
        with opener(PYPI_URL, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        version = payload["info"]["version"]
    except Exception as exc:  # noqa: BLE001 - offline is not an error worth raising
        logger.debug("Update check failed: {}", exc)
        return None
    return version if parse_version(version) is not None else None


def latest_version(now: Callable[[], float] = time.monotonic) -> str | None:
    """``fetch_latest`` at most once every six hours per server process."""
    global _cache
    stamp = now()
    if _cache is not None and stamp - _cache[0] < _CACHE_SECONDS:
        return _cache[1]
    version = fetch_latest()
    _cache = (stamp, version)
    return version


def clear_cache() -> None:
    """Forget the cached answer. For tests."""
    global _cache
    _cache = None


def check_disabled() -> bool:
    """``VISIONFORGE_NO_UPDATE_CHECK`` set to anything but empty or 0."""
    return os.environ.get(DISABLE_ENV, "").strip() not in ("", "0")


def installer_command(version: str) -> list[str] | None:
    """The command that installs ``version`` into this very interpreter.

    ``pip`` of this Python first; an environment created by ``uv`` has no pip, so
    ``uv pip install --python <this python>`` next. None when neither exists.
    The version is pinned: the button installs what the page announced, not
    whatever is newest by the time the updater runs.
    """
    spec = f"{DIST_NAME}=={version}"
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "--upgrade", spec]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, "--upgrade", spec]
    return None


def manual_command(version: str | None) -> str:
    """What to type by hand: ``git pull`` for a checkout, else the pip line."""
    if installed_from_source():
        return "git pull"
    target = f"=={version}" if version else ""
    return f"pip install --upgrade {DIST_NAME}{target}"


def update_dir() -> Path:
    """Where the updater writes its log and its outcome."""
    return config_dir() / "update"


def last_attempt() -> dict[str, Any] | None:
    """The outcome the updater recorded last time, if any."""
    path = update_dir() / "status.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def update_status(*, client_is_local: bool, busy: bool) -> dict[str, Any]:
    """What ``GET /api/update`` answers.

    ``blocked`` is a code the page words itself: ``source``, ``no_installer``,
    ``remote``, ``busy``. It is only set when an update is available.
    """
    if check_disabled():
        return {
            "current": __version__,
            "latest": None,
            "available": False,
            "can_apply": False,
            "blocked": None,
            "command": None,
            "last_attempt": last_attempt(),
            "check_disabled": True,
        }
    latest = latest_version()
    available = is_newer(latest, __version__)
    blocked: str | None = None
    if available:
        assert latest is not None
        if installed_from_source():
            blocked = "source"
        elif installer_command(latest) is None:
            blocked = "no_installer"
        elif not client_is_local:
            blocked = "remote"
        elif busy:
            blocked = "busy"
    return {
        "current": __version__,
        "latest": latest,
        "available": available,
        "can_apply": available and blocked is None,
        "blocked": blocked,
        "command": manual_command(latest) if available else None,
        "last_attempt": last_attempt(),
        "check_disabled": False,
    }


def _relaunch_command(host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "visionforge",
        "gui",
        "--host",
        host,
        "--port",
        str(port),
    ]


def server_pids() -> list[int]:
    """The processes the updater must outlive: this one, and its launcher.

    ``visionforge gui`` on Windows runs through the ``visionforge.exe`` that pip
    wrote, which starts this Python and waits for it. That file is one the install
    replaces, so the updater waits for the launcher too. It is the parent only when
    this process was started through it (``argv[0]`` is the .exe); a server started
    with ``python -m`` from a shell has the shell as parent, which must not be
    waited for.
    """
    pids = [os.getpid()]
    if Path(sys.argv[0]).suffix.lower() == ".exe":
        pids.append(os.getppid())
    return pids


def spawn_updater(*, version: str, host: str, port: int) -> Path:
    """Start the updater in a process that outlives this server; return its log.

    The script is copied out of the package first: the install it runs replaces
    the package's own files, so it must not be executed from among them.
    """
    install = installer_command(version)
    if install is None:
        raise RuntimeError("No installer available for this environment.")
    folder = update_dir()
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "update.log"
    status = folder / "status.json"
    script = Path(tempfile.mkdtemp(prefix="vf-update-")) / "vf_updater.py"
    script.write_text(
        (Path(__file__).resolve().parent.parent / "updater.py").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    args = {
        "host": host,
        "port": port,
        "pids": server_pids(),
        "target": version,
        "install": install,
        "restore": installer_command(__version__),
        "relaunch": _relaunch_command(host, port),
        "cwd": str(Path.cwd()),
        "log": str(log),
        "status": str(status),
    }
    command = [sys.executable, "-I", str(script), json.dumps(args)]
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "cwd": str(Path.cwd()),
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)  # noqa: S603 - fixed argv, no shell
    logger.info("Update to {} started; log at {}", version, log)
    return log


__all__ = [
    "DISABLE_ENV",
    "PYPI_URL",
    "check_disabled",
    "clear_cache",
    "fetch_latest",
    "installer_command",
    "is_newer",
    "last_attempt",
    "latest_version",
    "manual_command",
    "parse_version",
    "spawn_updater",
    "update_dir",
    "update_status",
]
