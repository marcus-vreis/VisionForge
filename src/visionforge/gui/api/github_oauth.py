"""Connect a profile to GitHub through the operator's OAuth App (ADR-250).

Pasting a personal token is the way a profile gets a GitHub token (ADR-175).
When the operator registers a GitHub OAuth App and gives the server its
credentials, a person can instead press "Conectar com GitHub", authorize the
app on github.com and come back with the token saved as their profile's GitHub
token, in the same file ``profile_tokens`` writes; the export reads it as before.

- The credentials come from the environment only, never from the command line
  (which other users of the machine can list) and never into a log:
  ``VISIONFORGE_GITHUB_CLIENT_ID`` and ``VISIONFORGE_GITHUB_CLIENT_SECRET``.
  Without both the feature is absent: the routes answer 404 and the session
  says ``github_oauth: false``.
- ``VISIONFORGE_PUBLIC_URL`` is the address people open the GUI at. Behind a
  proxy the server cannot know it (the scheme is lost and the ``Host`` is the
  one the proxy chose), so without it no ``redirect_uri`` is sent and GitHub
  uses the callback URL registered in the app.
- ``POST /oauth/start`` opens a request for the profile asking: a random
  ``state`` and a PKCE verifier kept in this process for ten minutes, and the
  GitHub address the page then opens. It is a POST, read with ``fetch``, because
  only a ``fetch`` carries the profile header on a local install and only a
  POST has its ``Origin`` checked by the Host and Origin guard.
- ``GET /oauth/callback`` is where GitHub sends the browser back. The ``state``
  must be one this process issued, unexpired and unused (it is spent on first
  sight), and on a server it must belong to the signed-in person; otherwise 400.
  The code is then exchanged for the token and the browser goes back to the
  page with ``?github_oauth=connected``, ``denied`` (refused on GitHub) or
  ``failed`` (GitHub gave no usable token): never the code nor the token.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from loguru import logger

from visionforge.gui.api.http_responses import BAD_REQUEST_TEXT, NOT_FOUND_TEXT
from visionforge.gui.api.profile_tokens import (
    ProfileHeader,
    token_file,
    valid_token,
    write_token,
)
from visionforge.gui.api.profiles import Profile
from visionforge.gui.server_mode import mode_of
from visionforge.utils.messages import tr

CLIENT_ID_ENV = "VISIONFORGE_GITHUB_CLIENT_ID"
CLIENT_SECRET_ENV = "VISIONFORGE_GITHUB_CLIENT_SECRET"  # noqa: S105 - a name
PUBLIC_URL_ENV = "VISIONFORGE_PUBLIC_URL"
AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105
CALLBACK_PATH = "/api/github/oauth/callback"
# The export writes to a private repository and refuses a public one (ADR-175);
# ``public_repo`` reaches public repositories only, so ``repo`` is the one scope
# with which a connected profile can export at all.
SCOPE = "repo"
# GitHub's code itself expires after ten minutes.
STATE_TTL_S = 600.0
# Requests a profile may have open at once; a new one drops the oldest.
MAX_PENDING = 5
RESULT_PARAM = "github_oauth"
_TIMEOUT_S = 30
_MAX_ANSWER = 64 * 1024
# GitHub's error codes (``bad_verification_code``): safe to log, unlike its text.
_ERROR_CODE = re.compile(r"[a-z_]{1,64}")
_warned: set[str] = set()


@dataclass(frozen=True)
class OAuthApp:
    """The operator's GitHub OAuth App, as the environment gives it."""

    client_id: str
    client_secret: str = field(repr=False)
    # Where the GUI is opened, without a trailing slash; None when unset.
    public_url: str | None = None

    @property
    def redirect_uri(self) -> str | None:
        """The callback GitHub returns to; None leaves the registered one."""
        return None if self.public_url is None else self.public_url + CALLBACK_PATH

    def page(self, result: str) -> str:
        """The GUI's own address with the outcome of the connection."""
        query = urllib.parse.urlencode({RESULT_PARAM: result})
        return f"{self.public_url or ''}/?{query}"


def oauth_app() -> OAuthApp | None:
    """The configured OAuth App, or None when the feature is off.

    Read on each call, so the environment is the one source of truth; an
    unusable ``VISIONFORGE_PUBLIC_URL`` turns the feature off with a warning
    rather than sending GitHub an address it would refuse.
    """
    client_id = os.environ.get(CLIENT_ID_ENV, "").strip()
    secret = os.environ.get(CLIENT_SECRET_ENV, "").strip()
    if not client_id or not secret:
        return None
    raw = os.environ.get(PUBLIC_URL_ENV, "").strip()
    public = _public_url(raw) if raw else None
    if raw and public is None:
        if raw not in _warned:
            _warned.add(raw)
            logger.warning(
                "{} must be an http(s) address without query or fragment; "
                "connecting with GitHub stays off",
                PUBLIC_URL_ENV,
            )
        return None
    return OAuthApp(client_id=client_id, client_secret=secret, public_url=public)


def _public_url(raw: str) -> str | None:
    """``raw`` without its trailing slash, or None when it is not a page address."""
    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    if parts.query or parts.fragment or parts.username or parts.password:
        return None
    return raw.rstrip("/")


@dataclass(frozen=True)
class Pending:
    """One connection request: whose it is and how to finish it."""

    profile: Profile
    verifier: str = field(repr=False)
    expires: float


def _owner(profile: Profile) -> Path:
    """What tells two profiles apart: the file their GitHub token lives in."""
    return token_file("github", profile)


class OAuthStates:
    """The open connection requests of this process, by ``state``."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = Lock()
        self._pending: dict[str, Pending] = {}

    def open(self, profile: Profile) -> tuple[str, str]:
        """A new ``state`` and PKCE verifier for ``profile``."""
        now = self._clock()
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(48)
        owner = _owner(profile)
        with self._lock:
            self._pending = {
                key: entry
                for key, entry in self._pending.items()
                if entry.expires > now
            }
            mine = [
                key
                for key, entry in self._pending.items()
                if _owner(entry.profile) == owner
            ]
            for key in mine[: max(0, len(mine) - MAX_PENDING + 1)]:
                del self._pending[key]
            self._pending[state] = Pending(profile, verifier, now + STATE_TTL_S)
        return state, verifier

    def take(self, state: str) -> Pending | None:
        """The request of ``state``, spent; None when unknown or expired."""
        with self._lock:
            entry = self._pending.pop(state, None)
        if entry is None or entry.expires <= self._clock():
            return None
        return entry

    def clear(self) -> None:
        """Forget every open request."""
        with self._lock:
            self._pending.clear()


STATES = OAuthStates()


def challenge(verifier: str) -> str:
    """The PKCE ``S256`` challenge of ``verifier`` (RFC 7636)."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def authorize_url(app: OAuthApp, state: str, verifier: str) -> str:
    """The GitHub page where the person authorizes the operator's app."""
    query = {
        "client_id": app.client_id,
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge(verifier),
        "code_challenge_method": "S256",
    }
    if app.redirect_uri is not None:
        query["redirect_uri"] = app.redirect_uri
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(query)}"


def exchange_code(app: OAuthApp, code: str, verifier: str) -> str | None:
    """GitHub's access token for ``code``, or None; nothing secret is logged."""
    fields = {
        "client_id": app.client_id,
        "client_secret": app.client_secret,
        "code": code,
        "code_verifier": verifier,
    }
    if app.redirect_uri is not None:
        fields["redirect_uri"] = app.redirect_uri
    request = urllib.request.Request(  # noqa: S310 - a fixed https URL
        ACCESS_TOKEN_URL,
        data=urllib.parse.urlencode(fields).encode("ascii"),
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "VisionForge",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
            answer = json.loads(response.read(_MAX_ANSWER))
    except (OSError, ValueError) as exc:
        # URLError and HTTPError are OSErrors; only the kind is logged.
        logger.warning(
            "GitHub OAuth: the code exchange failed ({})", type(exc).__name__
        )
        return None
    token = answer.get("access_token") if isinstance(answer, dict) else None
    if isinstance(token, str) and valid_token(token.strip()):
        return token.strip()
    error = answer.get("error") if isinstance(answer, dict) else None
    shown = error if isinstance(error, str) and _ERROR_CODE.fullmatch(error) else "?"
    logger.warning("GitHub OAuth: GitHub gave no token ({})", shown)
    return None


def add_oauth_routes(
    router: APIRouter, resolve: Callable[[str | None], Profile]
) -> None:
    """Mount ``POST /oauth/start`` and ``GET /oauth/callback`` on ``router``."""

    def configured() -> OAuthApp:
        app = oauth_app()
        if app is None:
            raise HTTPException(404, tr("github.oauth_off"))
        return app

    @router.post("/oauth/start", responses={404: {"description": NOT_FOUND_TEXT}})
    def start(header: ProfileHeader = None) -> dict[str, str]:
        """Open a connection request and give the GitHub page to open."""
        app = configured()
        state, verifier = STATES.open(resolve(header))
        return {"url": authorize_url(app, state, verifier)}

    @router.get(
        "/oauth/callback",
        response_class=RedirectResponse,
        status_code=303,
        responses={
            400: {"description": BAD_REQUEST_TEXT},
            404: {"description": NOT_FOUND_TEXT},
        },
    )
    def callback(
        request: Request,
        state: str | None = None,
        code: str | None = None,
        error: str | None = None,
        header: ProfileHeader = None,
    ) -> RedirectResponse:
        """Check the request, trade the code for the token and go back to the page."""
        app = configured()
        pending = STATES.take(state) if state else None
        if pending is None:
            raise HTTPException(400, tr("github.oauth_state_invalid"))
        profile = pending.profile
        # On a server the gate names the signed-in person; the request must be
        # theirs. A local install has no signed-in person: the browser's top
        # level navigation carries no profile header, so the request's own
        # profile is the one it was opened for.
        if mode_of(request.app).enabled and _owner(resolve(header)) != _owner(profile):
            raise HTTPException(400, tr("github.oauth_state_invalid"))
        if error is not None or not code:
            # The person refused on GitHub (``error=access_denied``), or GitHub
            # sent no code: nothing to exchange.
            return _back(app, "denied")
        token = exchange_code(app, code, pending.verifier)
        if token is None:
            return _back(app, "failed")
        write_token("github", profile, token)
        logger.info("GitHub OAuth: a GitHub token was saved for a profile")
        return _back(app, "connected")


def _back(app: OAuthApp, result: str) -> RedirectResponse:
    """Back to the page, with nothing of the exchange in the address."""
    return RedirectResponse(
        app.page(result), status_code=303, headers={"Cache-Control": "no-store"}
    )
