"""Open a folder in the operating system's file manager.

``visionforge gui --host`` can expose the server beyond the machine it runs on.
A file manager opened by the server appears on the *server's* desktop, never on
the desktop of whoever clicked, so the action is only meaningful when the
browser and the server share a machine. That is decided here from the client
address, and the route refuses (403) and the page hides the button otherwise.

The folder is chosen by the caller from a path it already resolved on the
server (``routes._find_run_dir``); nothing here accepts a path from the client.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


class FolderOpenError(RuntimeError):
    """The file manager could not be started on this machine."""


def is_loopback_host(host: str | None) -> bool:
    """Whether ``host`` (a request's client address) is this very machine.

    Covers ``127.0.0.0/8``, ``::1`` and the IPv4-mapped form some dual-stack
    servers report (``::ffff:127.0.0.1``), plus the literal ``localhost``.
    A missing address is not loopback: refusing is the safe answer.
    """
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def is_loopback_origin(origin: str) -> bool:
    """Whether a request's ``Origin`` header names this very machine.

    A browser puts the page's origin on every cross-site POST, so this tells the
    app's own page (``http://127.0.0.1:8000``, the Vite dev server on
    ``localhost``) from a page of another site that fired the request at the
    user's local server. Only the host counts, so a lookalike such as
    ``127.0.0.1.evil.example`` or ``127.0.0.1@evil.example`` is refused, and so
    are ``null`` (a sandboxed frame, a ``file://`` page) and anything that does
    not parse. The request's own ``Host`` is deliberately not trusted as "the
    server's host": a DNS-rebinding page controls both it and the ``Origin``.
    """
    try:
        host = urlsplit(origin).hostname
    except ValueError:
        return False
    return is_loopback_host(host)


def opener_available() -> bool:
    """Whether this machine has a way to open a folder at all.

    Windows and macOS always do. On Linux ``xdg-open`` is a package, absent from
    minimal installs and containers, so its presence is what is checked.
    """
    if sys.platform == "win32" or sys.platform == "darwin":
        return True
    return shutil.which("xdg-open") is not None


def can_reveal(client_host: str | None) -> bool:
    """Whether the run folder can be opened for a client at ``client_host``."""
    return is_loopback_host(client_host) and opener_available()


def open_folder(path: Path) -> None:
    """Show ``path`` in the file manager of the platform this server runs on.

    The command is an argument list, never a shell string, so a folder name
    cannot become a command. ``path`` is made absolute first: a leading ``-``
    would otherwise read as an option to ``open``/``xdg-open``.
    """
    target = str(path.resolve())
    try:
        if sys.platform == "win32":
            # Windows has no executable for this: ShellExecute does it.
            os.startfile(target)
        elif sys.platform == "darwin":
            subprocess.Popen(
                ["open", target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["xdg-open", target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except OSError as exc:
        raise FolderOpenError(str(exc)) from exc
