"""``visionforge gui --check``: can the GUI start here? Answers without starting it.

The audit's quick win asked for a smoke test of the install. This one never
binds the port, opens a browser or runs the server: it imports what
``visionforge gui`` imports, looks at the SPA build the server would serve, calls
the health route in this process and asks whether the port is free. One line per
check, ``[OK]``/``[FAIL]`` like ``visionforge doctor``; the exit code is 0 only
when every line is ``[OK]``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from visionforge.gui.security import is_wildcard
from visionforge.gui.updater import port_in_use

Check = tuple[bool, str]


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


def run_gui_check(host: str, port: int) -> int:
    """Print one line per check; return 0 when all pass, else 1."""
    # The rest needs the server module: skip it when the import itself failed.
    ok_import, message = _check_server_imports()
    print(f"[{'OK' if ok_import else 'FAIL'}] {message}")
    failed = not ok_import
    if ok_import:
        for check in (_check_spa_build, _check_health, _port_check(host, port)):
            ok, message = check()
            print(f"[{'OK' if ok else 'FAIL'}] {message}")
            failed = failed or not ok
    return 1 if failed else 0
