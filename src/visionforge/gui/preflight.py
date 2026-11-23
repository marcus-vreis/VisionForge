"""``visionforge gui --check``: can the GUI start here? Answers without starting it.

The audit's quick win asked for a smoke test of the install. This one never
binds the port, opens a browser or runs the server: it imports what
``visionforge gui`` imports, looks at the SPA build the server would serve, calls
the health route in this process and asks whether the port is free. One line per
check, ``[OK]``/``[FAIL]`` like ``visionforge doctor``; the exit code is 0 only
when every line is ``[OK]``.

``--smoke`` adds the full test: the real app under uvicorn on an ephemeral
loopback port (never the chosen one, never a browser), one HTTP GET of
``/api/health``, then a clean shutdown. Its startup runs the same sweep of
orphaned filtered datasets (``outputs/models/_filtered`` under the current
folder) that ``visionforge gui`` runs; nothing else is written.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from visionforge.gui.security import is_wildcard
from visionforge.gui.updater import port_in_use

Check = tuple[bool, str]

_SMOKE_TIMEOUT = 15.0  # seconds, for startup and again for shutdown
_SMOKE_HOST = "127.0.0.1"


def _check_server_imports() -> Check:
    try:
        import visionforge.gui.server  # noqa: F401
    except Exception as exc:  # any import failure is the answer
        return False, f"the server does not import: {type(exc).__name__}: {exc}"
    return True, "the server module imports"


def _check_spa_build() -> Check:
    from visionforge.gui import server

    index = server.STATIC_DIR / server.INDEX_FILENAME
    if not index.is_file():
        return False, f"no SPA build at {index} (run `npm run build` in frontend/)"
    bundle = server._read_spa_bundle()
    if not bundle:
        return False, f"{index} does not reference an index-*.js bundle"
    if not (server.STATIC_DIR / "assets" / bundle).is_file():
        return False, f"{index} names assets/{bundle}, which is missing"
    return True, f"SPA build present ({bundle})"


def _check_health() -> Check:
    from visionforge import __version__
    from visionforge.gui import server

    try:
        body = asyncio.run(server.health())
    except Exception as exc:
        return False, f"/api/health failed: {type(exc).__name__}: {exc}"
    if body.get("version") != __version__:
        return False, f"/api/health reports version {body.get('version')!r}"
    return True, f"/api/health answers (version {__version__})"


def _port_check(host: str, port: int) -> Callable[[], Check]:
    def check() -> Check:
        # A wildcard is where the server listens, not an address to connect to.
        target = "127.0.0.1" if is_wildcard(host) else host
        if port_in_use(target, port):
            return False, f"port {port} on {host} is already in use (try --port)"
        return True, f"port {port} on {host} is free"

    return check


def _smoke_port(uv_server: Any) -> int | None:
    """The port uvicorn bound once started, or None while it has not."""
    if not uv_server.started or not uv_server.servers:
        return None
    sockets = uv_server.servers[0].sockets
    return int(sockets[0].getsockname()[1]) if sockets else None


def _check_smoke() -> Check:
    """Run the real app on an ephemeral loopback port and GET /api/health."""
    import uvicorn

    from visionforge import __version__
    from visionforge.gui import server

    uv_server = uvicorn.Server(
        uvicorn.Config(server.app, host=_SMOKE_HOST, port=0, log_level="warning")
    )
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            uv_server.run()
        except BaseException as exc:  # reported as the answer, not lost in the thread
            errors.append(exc)

    thread = threading.Thread(target=serve, name="gui-smoke", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + _SMOKE_TIMEOUT
        port = _smoke_port(uv_server)
        while port is None and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
            port = _smoke_port(uv_server)
        if port is None:
            why = f"{type(errors[0]).__name__}: {errors[0]}" if errors else "timed out"
            return False, f"smoke: the server did not start ({why})"
        return _smoke_get(port, __version__)
    finally:
        uv_server.should_exit = True
        thread.join(_SMOKE_TIMEOUT)


def _smoke_get(port: int, version: str) -> Check:
    url = f"http://{_SMOKE_HOST}:{port}/api/health"  # NOSONAR python:S5332 - loopback
    try:
        with urllib.request.urlopen(url, timeout=_SMOKE_TIMEOUT) as response:  # noqa: S310
            status = response.status
            body = json.loads(response.read())
    except Exception as exc:
        return False, f"smoke: GET {url} failed: {type(exc).__name__}: {exc}"
    if status != 200:
        return False, f"smoke: GET {url} answered {status}"
    reported = body.get("version") if isinstance(body, dict) else None
    if reported != version:
        return False, f"smoke: GET {url} reports version {reported!r}"
    return True, f"smoke: the server answered GET {url} (version {version}) and stopped"


def run_gui_check(host: str, port: int, smoke: bool = False) -> int:
    """Print one line per check; return 0 when all pass, else 1.

    ``smoke`` adds the full server test (see the module docstring).
    """
    # The rest needs the server module: skip it when the import itself failed.
    ok_import, message = _check_server_imports()
    print(f"[{'OK' if ok_import else 'FAIL'}] {message}")
    failed = not ok_import
    if ok_import:
        checks = [_check_spa_build, _check_health, _port_check(host, port)]
        if smoke:
            checks.append(_check_smoke)
        for check in checks:
            ok, message = check()
            print(f"[{'OK' if ok else 'FAIL'}] {message}")
            failed = failed or not ok
    return 1 if failed else 0
