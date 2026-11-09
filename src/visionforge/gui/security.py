"""Refuse requests whose ``Host`` or ``Origin`` does not name this server.

A page of another site can make the user's own browser talk to the local
server. CORS only hides the answer from that page; it does not stop the request.
Two attacks follow, and each needs its own check:

- **DNS rebinding.** The attacker's name first resolves to their machine, which
  serves the page, then to ``127.0.0.1``. The browser now treats the local server
  as the attacker's own site, so it reads every answer and sends any request.
  What it cannot hide is the name it asked for: ``Host`` carries the attacker's
  name. Only the names this server is reached by are accepted.
- **Cross-site requests.** A page of another site posts to ``localhost``. Its
  ``Host`` is right, but the browser names the page in ``Origin``. A request that
  changes state and carries a foreign ``Origin`` is refused. No ``Origin`` is a
  same-origin navigation or a client that is not a browser: allowed.

Bound to loopback (the default), the names are ``localhost``, the loopback
addresses (``127.0.0.1``, ``[::1]``) and each ``--allowed-host``. Bound to
another address for a lab (ADR-114), the machine's own names and addresses are
added, so the people who already reach it by name or IP keep doing so.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from visionforge.gui.api.folder_opener import is_loopback_host, is_loopback_origin
from visionforge.utils.messages import LANG_HEADER, message, resolve_lang

# Methods a browser lets any page send without asking the server first; they
# must not change anything, so a foreign Origin on them is harmless.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

# WebSocket close code for a policy violation (RFC 6455, 7.4.1).
_POLICY_VIOLATION = 1008


def _canonical(name: str) -> str | None:
    """A host name or address in the one spelling the policy compares."""
    name = name.strip().lower().rstrip(".")
    if not name:
        return None
    try:
        return str(ipaddress.ip_address(name))
    except ValueError:
        return name


def _bracketed(value: str) -> str | None:
    """The address inside ``[v6]`` or ``[v6]:port``; ``None`` if malformed."""
    end = value.find("]")
    if end < 0:
        return None
    rest = value[end + 1 :]
    if rest and not (rest.startswith(":") and rest[1:].isdigit()):
        return None
    try:
        return str(ipaddress.IPv6Address(value[1:end]))
    except ValueError:
        return None


def normalize_host(value: str | None) -> str | None:
    """The host of a ``Host`` header or ``--allowed-host`` value, port removed.

    ``[::1]:8000`` is ``::1`` and ``LOCALHOST:8000`` is ``localhost``. A bare
    IPv6 address (``::1``) is accepted too, as people type it on the command
    line. Anything that does not parse (an unclosed bracket, a port that is not a
    number) is ``None``, which no policy accepts.
    """
    if not value:
        return None
    value = value.strip()
    if value.startswith("["):
        return _bracketed(value)
    if value.count(":") > 1:
        try:
            return str(ipaddress.IPv6Address(value))
        except ValueError:
            return None
    name, sep, port = value.partition(":")
    if sep and not port.isdigit():
        return None
    return _canonical(name)


def machine_names() -> set[str]:
    """This machine's names and the addresses they resolve to.

    What a colleague types to reach a lab server: its hostname, its fully
    qualified name, or one of its addresses. A lookup that fails adds nothing:
    ``--allowed-host`` is the way to name what is missing.
    """
    names = {socket.gethostname(), socket.getfqdn()}
    addresses: set[str] = set()
    for name in names:
        try:
            infos = socket.getaddrinfo(name, None)
        except OSError:
            continue
        addresses.update(str(info[4][0]) for info in infos)
    return names | addresses


def is_wildcard(bind_host: str) -> bool:
    """Whether ``bind_host`` means "every interface" (``0.0.0.0``, ``::``)."""
    try:
        return ipaddress.ip_address(bind_host).is_unspecified
    except ValueError:
        return False


@dataclass(frozen=True)
class HostPolicy:
    """The names this server answers to: loopback always, plus ``extra``."""

    extra: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def for_bind(
        cls,
        bind_host: str,
        allowed_hosts: Iterable[str] = (),
        *,
        names_of_machine: Iterable[str] | None = None,
    ) -> HostPolicy:
        """The policy for a server bound to ``bind_host``.

        Loopback accepts only loopback and ``allowed_hosts``. Any other address
        also accepts the machine's names and addresses (``names_of_machine``,
        looked up when not given) and the address it was bound to, unless that
        is a wildcard such as ``0.0.0.0``.
        """
        names = list(allowed_hosts)
        if not is_loopback_host(bind_host):
            if names_of_machine is None:
                names_of_machine = machine_names()
            names.extend(names_of_machine)
            if not is_wildcard(bind_host):
                names.append(bind_host)
        normalized = {normalize_host(name) for name in names}
        return cls(extra=frozenset(name for name in normalized if name is not None))

    def allows_host(self, name: str | None) -> bool:
        """Whether ``name`` (already normalized) is one this server answers to."""
        if name is None:
            return False
        return is_loopback_host(name) or name in self.extra

    def allows_origin(self, origin: str) -> bool:
        """Whether a request's ``Origin`` is a page this server delivered."""
        if is_loopback_origin(origin):
            return True
        try:
            host = urlsplit(origin).hostname
        except ValueError:
            return False
        return host is not None and _canonical(host) in self.extra


# What a server answers to when nobody configured it (``uvicorn ...:app``, tests).
DEFAULT_POLICY = HostPolicy()


def policy_of(app: Any) -> HostPolicy:
    """The policy ``start_server`` stored on ``app``, or the loopback default."""
    policy = getattr(getattr(app, "state", None), "host_policy", None)
    return policy if isinstance(policy, HostPolicy) else DEFAULT_POLICY


class HostOriginGuard:
    """ASGI middleware applying :class:`HostPolicy` to every request.

    A foreign ``Host`` is 400 on any method: the request did not ask for this
    server. A foreign ``Origin`` is 403 on a method that changes state, and on a
    WebSocket handshake, which CORS never covers.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        policy = policy_of(scope.get("app"))
        headers = Headers(scope=scope)
        if not policy.allows_host(normalize_host(headers.get("host"))):
            await _refuse(scope, receive, send, 400, "guard.foreign_host")
            return
        origin = headers.get("origin")
        changes_state = (
            scope["type"] == "websocket" or scope["method"] not in SAFE_METHODS
        )
        if changes_state and origin is not None and not policy.allows_origin(origin):
            await _refuse(scope, receive, send, 403, "guard.foreign_origin")
            return
        await self.app(scope, receive, send)


async def _refuse(
    scope: Scope, receive: Receive, send: Send, status: int, key: str
) -> None:
    if scope["type"] == "websocket":
        await WebSocketClose(code=_POLICY_VIOLATION)(scope, receive, send)
        return
    lang = resolve_lang(Headers(scope=scope).get(LANG_HEADER))
    response = JSONResponse({"detail": message(lang, key)}, status_code=status)
    await response(scope, receive, send)


__all__ = [
    "DEFAULT_POLICY",
    "SAFE_METHODS",
    "HostOriginGuard",
    "HostPolicy",
    "is_wildcard",
    "machine_names",
    "normalize_host",
    "policy_of",
]
