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

:class:`SecurityHeaders` adds the Content-Security-Policy and the fixed
security headers to every response (SEC-2): what a page this server delivered
may load, and that no other site may frame it.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import re
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send
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


# --- Response headers (SEC-2) ---------------------------------------
#
# The policy names this server and nothing else. The SPA is one module script
# and one stylesheet from /assets, plus the inline script in index.html that
# paints the theme before the bundle loads (ADR-125); that one is allowed by its
# hash. React writes the ``style`` prop through the CSSOM (``el.style.x = y``),
# which CSP does not govern, and the bundle injects no <style>, so ``style-src``
# needs no 'unsafe-inline'. Dataset crops arrive as ``data:`` images and
# downloads are built as ``blob:`` URLs.

_INLINE_SCRIPT = re.compile(
    r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script\s*>", re.IGNORECASE | re.DOTALL
)
_SRC_ATTRIBUTE = re.compile(r"\bsrc\s*=", re.IGNORECASE)


def inline_script_hashes(html: str) -> list[str]:
    """CSP source expressions (``'sha256-…'``) for each inline ``<script>``.

    Line endings are normalised first, as the browser's HTML parser does before
    it hashes: a Windows checkout may write ``index.html`` with CRLF.
    """
    html = html.replace("\r\n", "\n").replace("\r", "\n")
    hashes = []
    for match in _INLINE_SCRIPT.finditer(html):
        if _SRC_ATTRIBUTE.search(match["attrs"]):
            continue
        digest = hashlib.sha256(match["body"].encode("utf-8")).digest()
        hashes.append(f"'sha256-{base64.b64encode(digest).decode('ascii')}'")
    return hashes


# CSP keywords keep their single quotes inside the header value.
_SELF = "'self'"
_NONE = "'none'"


def _policy(directives: dict[str, str]) -> str:
    return "; ".join(f"{name} {value}" for name, value in directives.items())


def content_security_policy(script_hashes: Iterable[str]) -> str:
    """The SPA's policy: this server only, inline scripts only by hash."""
    script_src = " ".join([_SELF, *script_hashes])
    return _policy(
        {
            "default-src": _SELF,
            "script-src": script_src,
            "style-src": _SELF,
            "img-src": "'self' data: blob:",
            "font-src": _SELF,
            "connect-src": _SELF,
            "object-src": _NONE,
            "base-uri": _SELF,
            "form-action": _SELF,
            "frame-ancestors": _NONE,
        }
    )


# FastAPI's Swagger UI and ReDoc pages load their code from cdn.jsdelivr.net,
# run an inline bootstrap script and (ReDoc) Google Fonts and blob: workers. They
# exist only with VISIONFORGE_DOCS=1, and only their own responses get this.
DOCS_CONTENT_SECURITY_POLICY = _policy(
    {
        "default-src": _SELF,
        "script-src": "'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "style-src": (
            "'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com"
        ),
        "font-src": "'self' https://fonts.gstatic.com",
        "img-src": "'self' data: blob: https://fastapi.tiangolo.com https://cdn.redoc.ly",
        "worker-src": "'self' blob:",
        "connect-src": _SELF,
        "object-src": _NONE,
        "base-uri": _SELF,
        "form-action": _SELF,
        "frame-ancestors": _NONE,
    }
)


class SpaContentSecurityPolicy:
    """The SPA's policy, following the ``index.html`` on disk.

    Static files are read per request, so a rebuild reaches the browser at once;
    the hash of its inline script must follow, or the theme script is blocked.
    The file is hashed again only when its size or modification time changes.
    """

    def __init__(self, index: Path) -> None:
        self.index = index
        self._key: tuple[int, int] | None = None
        self._policy = content_security_policy([])

    def __call__(self) -> str:
        try:
            stat = self.index.stat()
        except OSError:
            return content_security_policy([])
        key = (stat.st_mtime_ns, stat.st_size)
        if key != self._key:
            html = self.index.read_text(encoding="utf-8")
            self._policy = content_security_policy(inline_script_hashes(html))
            self._key = key
        return self._policy


# ``same-origin``, not ``no-referrer``: both send nothing to any other site, but
# under ``no-referrer`` the Fetch standard ("serializing a request origin") has a
# POST carry ``Origin: null``, which HostOriginGuard refuses as foreign. Chromium
# sends the real origin on same-origin fetches anyway (checked in the e2e suite);
# other engines were not checked, and ``same-origin`` does not depend on it.
STATIC_HEADERS: tuple[tuple[str, str], ...] = (
    ("x-frame-options", "DENY"),
    ("x-content-type-options", "nosniff"),
    ("referrer-policy", "same-origin"),
)


class SecurityHeaders:
    """ASGI middleware adding the CSP and the fixed security headers to every
    HTTP response, refusals of :class:`HostOriginGuard` included.

    ``spa_policy`` gives the policy at response time; ``docs_paths`` (empty
    unless the API docs are on) get :data:`DOCS_CONTENT_SECURITY_POLICY`.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        spa_policy: Callable[[], str],
        docs_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.spa_policy = spa_policy
        self.docs_paths = frozenset(docs_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        csp = (
            DOCS_CONTENT_SECURITY_POLICY
            if scope["path"] in self.docs_paths
            else self.spa_policy()
        )

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["content-security-policy"] = csp
                for name, value in STATIC_HEADERS:
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_headers)


__all__ = [
    "DEFAULT_POLICY",
    "DOCS_CONTENT_SECURITY_POLICY",
    "SAFE_METHODS",
    "STATIC_HEADERS",
    "HostOriginGuard",
    "HostPolicy",
    "SecurityHeaders",
    "SpaContentSecurityPolicy",
    "content_security_policy",
    "inline_script_hashes",
    "is_wildcard",
    "machine_names",
    "normalize_host",
    "policy_of",
]
