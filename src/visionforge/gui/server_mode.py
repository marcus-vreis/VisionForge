"""Server mode: who is asking, and which routes answer (ADR-212, ADR-213).

``visionforge gui --server`` puts the GUI behind a reverse proxy that signs
people in (oauth2-proxy, Authelia). The proxy tells who the person is in a
header; nothing else may. With the mode on, :class:`ServerModeGate`:

- answers only the proxy's address (``--trusted-proxy``): a client that
  reaches the port directly could write the user header itself;
- turns the user the proxy names into that person's profile and overwrites
  ``X-VF-Profile`` with it, so every route that works in a profile (ADR-114)
  works in the signed-in person's, whatever the browser sent;
- binds the folders the person may read datasets from: their own and the ones
  the operator shares (``--datasets-dir``, ADR-214);
- answers 404 for every route that is not ready for several people.

Every route has a class in :data:`ROUTE_CLASSES`; a test compares the table
with the routes the app serves, so a new route without one fails the suite. A
request the table does not know is refused: the gate fails closed.

With the mode off (the default) the gate passes every request through.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from visionforge.gui.action_limit import DEFAULT_MAX_ACTIONS, ActionSlots, heavy
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.profiles import (
    PROFILE_HEADER,
    Profile,
    ProfileError,
    ensure_profile,
    slugify,
)
from visionforge.gui.disk_quota import GIB, DiskUsage, grows, size_text
from visionforge.utils.messages import LANG_HEADER, message, resolve_lang

# The header oauth2-proxy sets on the request it forwards (``--pass-user-headers``).
DEFAULT_USER_HEADER = "X-Forwarded-User"

# An account name is an e-mail address or a login; nothing longer is one.
MAX_USER_LEN = 254

# The proxy's own liveness probe carries no user.
HEALTH_ROUTE = "GET /api/health"

# The SPA fallback: every page, and every asset of the bundle.
SPA_ROUTE = "GET /{path:path}"
_API_DOCUMENTS = frozenset(
    {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}
)

_SLUG_BASE_LEN = 29
_DIGEST_LEN = 10

Network = ipaddress.IPv4Network | ipaddress.IPv6Network

# What each person's folder holds, for the quota (ADR-225).
USAGE = DiskUsage()

# The heavy actions running now, outside the queue (ADR-226).
SLOTS = ActionSlots()

RouteClass = Literal["open", "owner", "confined", "off"]
"""What a route needs before it may answer in server mode.

- ``open``: reads nothing personal.
- ``owner``: works in the profile of the signed-in person, or on a run or job
  that is theirs.
- ``confined``: takes a dataset path, which must sit inside that person's own
  dataset folder.
- ``off``: acts on the server's own desktop, runs code a user brings, or holds
  state every user would share; it has no place in server mode.
"""

ROUTE_CLASSES: dict[str, RouteClass] = {
    "POST /api/anomaly/compare": "owner",
    "POST /api/anomaly/dataset/stats": "confined",
    "POST /api/anomaly/replicated-comparison": "owner",
    "POST /api/anomaly/replicates": "owner",
    "POST /api/anomaly/run": "owner",
    "GET /api/anomaly/schema": "open",
    "POST /api/anomaly/sweep": "owner",
    "GET /api/artifacts/{file_path:path}": "owner",
    "POST /api/checkpoint/pick": "off",
    "POST /api/classification/replicated-comparison": "owner",
    "POST /api/classification/replicates": "owner",
    "GET /api/code-tour": "open",
    "GET /api/code-tour/{task}": "open",
    "POST /api/colab/drive/continue": "off",
    "GET /api/colab/drive/folders": "off",
    "POST /api/colab/drive/open": "off",
    "POST /api/colab/experiment/run": "off",
    "POST /api/colab/install": "off",
    "POST /api/colab/login/cancel": "off",
    "POST /api/colab/login/finish": "off",
    "POST /api/colab/login/start": "off",
    "POST /api/colab/logout": "off",
    "GET /api/colab/sessions": "off",
    "POST /api/colab/sessions/stop": "off",
    "GET /api/colab/status": "off",
    "GET /api/colab/usage": "off",
    "POST /api/comet/runs/{run_id}/retry": "owner",
    "DELETE /api/comet/token": "owner",
    "GET /api/comet/token": "owner",
    "PUT /api/comet/token": "owner",
    "GET /api/credentials": "off",
    "POST /api/credentials": "off",
    "DELETE /api/credentials/{provider}": "off",
    "DELETE /api/custom/{key}": "off",
    "POST /api/custom/{key}/hide": "off",
    "POST /api/custom/{key}/replicated-comparison": "off",
    "POST /api/custom/{key}/replicates": "off",
    "POST /api/custom/{key}/run": "off",
    "GET /api/custom/{key}/schema": "off",
    "POST /api/custom/{key}/sweep": "off",
    "POST /api/custom/{key}/unhide": "off",
    "POST /api/dataset/audit": "confined",
    "POST /api/dataset/detect": "confined",
    "POST /api/dataset/download": "confined",
    "GET /api/dataset/file": "confined",
    "POST /api/dataset/pick": "off",
    "GET /api/datasets/own": "owner",
    "DELETE /api/datasets/own/{name}": "owner",
    "POST /api/dataset/preview_augment": "confined",
    "POST /api/dataset/preview_preprocess": "confined",
    "POST /api/dataset/samples": "confined",
    "POST /api/dataset/stats": "confined",
    "POST /api/detection/compare": "owner",
    "POST /api/detection/dataset/pick_yaml": "off",
    "POST /api/detection/dataset/samples": "confined",
    "POST /api/detection/dataset/stats": "confined",
    "POST /api/detection/replicated-comparison": "owner",
    "POST /api/detection/replicates": "owner",
    "POST /api/detection/run": "owner",
    "GET /api/detection/schema": "open",
    "POST /api/detection/sweep": "owner",
    "GET /api/device/info": "open",
    # DVC runs the server's own git and DVC on a folder of the server's disk
    # (ADR-221): it has no meaning for a person at a browser elsewhere.
    "POST /api/dvc/export": "off",
    "POST /api/dvc/prepare": "off",
    "GET /api/dvc/preview": "off",
    "GET /api/experiment/events": "owner",
    "GET /api/experiment/result/{run_id}": "owner",
    "POST /api/experiment/run": "owner",
    "GET /api/experiment/status": "owner",
    "POST /api/explorer/coco/export/preview": "confined",
    "GET /api/explorer/coco/export/{job_id}": "confined",
    "POST /api/explorer/coco/export/{job_id}/apply": "confined",
    "POST /api/explorer/coco/export/{job_id}/cancel": "confined",
    "POST /api/explorer/coco/export/{job_id}/plan": "confined",
    "POST /api/explorer/coco/import/preview": "confined",
    "GET /api/explorer/coco/import/{job_id}": "confined",
    "POST /api/explorer/coco/import/{job_id}/apply": "confined",
    "POST /api/explorer/coco/import/{job_id}/cancel": "confined",
    "POST /api/explorer/coco/import/{job_id}/plan": "confined",
    "POST /api/explorer/images": "confined",
    "POST /api/explorer/incorporate/preview": "confined",
    "GET /api/explorer/incorporate/{job_id}": "confined",
    "POST /api/explorer/incorporate/{job_id}/apply": "confined",
    "POST /api/explorer/incorporate/{job_id}/cancel": "confined",
    "POST /api/explorer/incorporate/{job_id}/plan": "confined",
    "POST /api/explorer/merge/preview": "confined",
    "GET /api/explorer/merge/{job_id}": "confined",
    "POST /api/explorer/merge/{job_id}/apply": "confined",
    "POST /api/explorer/merge/{job_id}/cancel": "confined",
    "POST /api/explorer/merge/{job_id}/plan": "confined",
    "POST /api/explorer/resplit/preview": "confined",
    "GET /api/explorer/resplit/{job_id}": "confined",
    "POST /api/explorer/resplit/{job_id}/apply": "confined",
    "POST /api/explorer/resplit/{job_id}/cancel": "confined",
    "POST /api/explorer/resplit/{job_id}/plan": "confined",
    "POST /api/explorer/scan": "confined",
    "GET /api/explorer/scan/{job_id}": "confined",
    "POST /api/explorer/scan/{job_id}/cancel": "confined",
    "POST /api/explorer/scan/saved": "confined",
    "POST /api/explorer/summary": "confined",
    "GET /api/explorer/thumb": "confined",
    "POST /api/github/export": "owner",
    "GET /api/github/preview": "owner",
    "DELETE /api/github/token": "owner",
    "GET /api/github/token": "owner",
    "PUT /api/github/token": "owner",
    "GET /api/health": "open",
    "POST /api/hub/export": "owner",
    "GET /api/hub/preview": "owner",
    "DELETE /api/hub/token": "owner",
    "GET /api/hub/token": "owner",
    "PUT /api/hub/token": "owner",
    "POST /api/model/defaults": "open",
    "POST /api/model/learning-rate-plan": "open",
    "GET /api/profiles": "owner",
    "POST /api/profiles": "off",
    "PUT /api/profiles/{slug}": "owner",
    "GET /api/queue": "owner",
    "DELETE /api/queue/{run_id}": "owner",
    "POST /api/regression/compare": "owner",
    "POST /api/regression/cv": "owner",
    "POST /api/regression/dataset/stats": "confined",
    "POST /api/regression/replicated-comparison": "owner",
    "POST /api/regression/replicates": "owner",
    "POST /api/regression/run": "owner",
    "GET /api/regression/schema": "open",
    "POST /api/regression/sweep": "owner",
    "GET /api/runs": "owner",
    "DELETE /api/runs/{run_id}": "owner",
    "GET /api/runs/{run_id}": "owner",
    "POST /api/runs/{run_id}/batch_predict": "confined",
    "GET /api/runs/{run_id}/bundle": "owner",
    "GET /api/runs/{run_id}/export_md": "owner",
    "POST /api/runs/{run_id}/export_onnx": "owner",
    "POST /api/runs/{run_id}/gradcam": "confined",
    "POST /api/runs/{run_id}/mlflow/retry": "off",
    "POST /api/runs/{run_id}/resume": "owner",
    "POST /api/runs/{run_id}/reveal": "off",
    "POST /api/runs/{run_id}/test": "confined",
    "POST /api/sample-dataset": "confined",
    "GET /api/schema": "open",
    "POST /api/segmentation/compare": "owner",
    "POST /api/segmentation/cv": "owner",
    "POST /api/segmentation/dataset/stats": "confined",
    "POST /api/segmentation/replicated-comparison": "owner",
    "POST /api/segmentation/replicates": "owner",
    "POST /api/segmentation/run": "owner",
    "GET /api/segmentation/schema": "open",
    "POST /api/segmentation/sweep": "owner",
    "GET /api/session": "open",
    "GET /api/system/info": "open",
    "GET /api/tasks": "open",
    "GET /api/update": "open",
    "POST /api/update/apply": "off",
    "POST /api/wandb/runs/{run_id}/retry": "owner",
    "DELETE /api/wandb/token": "owner",
    "GET /api/wandb/token": "owner",
    "PUT /api/wandb/token": "owner",
    "POST /api/zenodo/export": "owner",
    "GET /api/zenodo/preview": "owner",
    "DELETE /api/zenodo/sandbox/token": "owner",
    "GET /api/zenodo/sandbox/token": "owner",
    "PUT /api/zenodo/sandbox/token": "owner",
    "DELETE /api/zenodo/token": "owner",
    "GET /api/zenodo/token": "owner",
    "PUT /api/zenodo/token": "owner",
    "GET /{path:path}": "open",
}

# The ``owner`` and ``confined`` routes whose handler was read: an ``owner`` one
# works only inside the profile it is given, a ``confined`` one types every path
# of its request with ``DatasetDir`` or ``DatasetPart`` (ADR-214). The rest
# answer 404 in server mode until they are read.
READY: frozenset[str] = frozenset(
    {
        "GET /api/artifacts/{file_path:path}",
        "POST /api/anomaly/run",
        "POST /api/detection/run",
        "POST /api/experiment/run",
        "POST /api/regression/run",
        "POST /api/segmentation/run",
        # The strategies: each confines its base config and what it rewrites
        # per trial (ADR-217).
        "POST /api/anomaly/compare",
        "POST /api/anomaly/replicated-comparison",
        "POST /api/anomaly/replicates",
        "POST /api/anomaly/sweep",
        "POST /api/classification/replicated-comparison",
        "POST /api/classification/replicates",
        "POST /api/detection/compare",
        "POST /api/detection/replicated-comparison",
        "POST /api/detection/replicates",
        "POST /api/detection/sweep",
        "POST /api/regression/compare",
        "POST /api/regression/cv",
        "POST /api/regression/replicated-comparison",
        "POST /api/regression/replicates",
        "POST /api/regression/sweep",
        "POST /api/segmentation/compare",
        "POST /api/segmentation/cv",
        "POST /api/segmentation/replicated-comparison",
        "POST /api/segmentation/replicates",
        "POST /api/segmentation/sweep",
        "GET /api/experiment/events",
        "GET /api/experiment/result/{run_id}",
        "GET /api/experiment/status",
        "GET /api/queue",
        "DELETE /api/queue/{run_id}",
        "POST /api/anomaly/dataset/stats",
        # Read a dataset in the scope; a preview writes its images inside the
        # person's own profile (ADR-222).
        "POST /api/dataset/audit",
        "POST /api/dataset/preview_augment",
        "POST /api/dataset/preview_preprocess",
        "POST /api/dataset/detect",
        # Writes a dataset, into the person's own folder only (ADR-219).
        # The Laboratory: each reads a dataset in the scope, its jobs belong
        # to whoever started them, and what it writes goes into the person's
        # own folder (ADR-220).
        "GET /api/explorer/coco/export/{job_id}",
        "GET /api/explorer/coco/import/{job_id}",
        "GET /api/explorer/incorporate/{job_id}",
        "GET /api/explorer/merge/{job_id}",
        "GET /api/explorer/resplit/{job_id}",
        "GET /api/explorer/scan/{job_id}",
        "GET /api/explorer/thumb",
        "POST /api/explorer/coco/export/preview",
        "POST /api/explorer/coco/export/{job_id}/apply",
        "POST /api/explorer/coco/export/{job_id}/cancel",
        "POST /api/explorer/coco/export/{job_id}/plan",
        "POST /api/explorer/coco/import/preview",
        "POST /api/explorer/coco/import/{job_id}/apply",
        "POST /api/explorer/coco/import/{job_id}/cancel",
        "POST /api/explorer/coco/import/{job_id}/plan",
        "POST /api/explorer/images",
        "POST /api/explorer/incorporate/preview",
        "POST /api/explorer/incorporate/{job_id}/apply",
        "POST /api/explorer/incorporate/{job_id}/cancel",
        "POST /api/explorer/incorporate/{job_id}/plan",
        "POST /api/explorer/merge/preview",
        "POST /api/explorer/merge/{job_id}/apply",
        "POST /api/explorer/merge/{job_id}/cancel",
        "POST /api/explorer/merge/{job_id}/plan",
        "POST /api/explorer/resplit/preview",
        "POST /api/explorer/resplit/{job_id}/apply",
        "POST /api/explorer/resplit/{job_id}/cancel",
        "POST /api/explorer/resplit/{job_id}/plan",
        "POST /api/explorer/scan",
        "POST /api/explorer/scan/{job_id}/cancel",
        # The last finished scan, kept inside the person's profile (ADR-231).
        "POST /api/explorer/scan/saved",
        "POST /api/explorer/summary",
        "POST /api/dataset/download",
        "POST /api/sample-dataset",
        # What the person's own folder holds, and deleting one of them: how
        # a full folder gets under the quota again (ADR-227).
        "GET /api/datasets/own",
        "DELETE /api/datasets/own/{name}",
        "GET /api/dataset/file",
        "POST /api/dataset/samples",
        "POST /api/dataset/stats",
        "POST /api/detection/dataset/samples",
        "POST /api/detection/dataset/stats",
        "POST /api/regression/dataset/stats",
        "POST /api/segmentation/dataset/stats",
        "DELETE /api/comet/token",
        "GET /api/comet/token",
        "PUT /api/comet/token",
        # The person's own profile, and nobody else's: where they turn W&B
        # and Comet on, each sent with their own token (ADR-223).
        "GET /api/profiles",
        "PUT /api/profiles/{slug}",
        "POST /api/comet/runs/{run_id}/retry",
        "POST /api/wandb/runs/{run_id}/retry",
        "DELETE /api/github/token",
        "GET /api/github/token",
        "PUT /api/github/token",
        "DELETE /api/hub/token",
        "GET /api/hub/token",
        "PUT /api/hub/token",
        "GET /api/runs",
        "DELETE /api/runs/{run_id}",
        "GET /api/runs/{run_id}",
        # How a run leaves the server: each takes a run id, packages files from
        # inside that run's folder and uses its owner's own token (ADR-221).
        "GET /api/runs/{run_id}/bundle",
        "GET /api/runs/{run_id}/export_md",
        "POST /api/github/export",
        "GET /api/github/preview",
        "POST /api/hub/export",
        "GET /api/hub/preview",
        "POST /api/zenodo/export",
        "GET /api/zenodo/preview",
        # What a finished run offers: each loads the run's own checkpoint and
        # reads its dataset path through ``DatasetDir`` (ADR-218).
        "POST /api/runs/{run_id}/batch_predict",
        "POST /api/runs/{run_id}/export_onnx",
        "POST /api/runs/{run_id}/gradcam",
        "POST /api/runs/{run_id}/resume",
        "POST /api/runs/{run_id}/test",
        "DELETE /api/wandb/token",
        "GET /api/wandb/token",
        "PUT /api/wandb/token",
        "DELETE /api/zenodo/sandbox/token",
        "GET /api/zenodo/sandbox/token",
        "PUT /api/zenodo/sandbox/token",
        "DELETE /api/zenodo/token",
        "GET /api/zenodo/token",
        "PUT /api/zenodo/token",
    }
)


class ServerModeError(ValueError):
    """The server-mode options cannot be used as given."""


@dataclass(frozen=True)
class ServerMode:
    """The options of ``--server``; the default is the mode off."""

    enabled: bool = False
    trusted: tuple[Network, ...] = ()
    user_header: str = DEFAULT_USER_HEADER
    # Folders the operator shares with everyone, read-only (``--datasets-dir``).
    datasets: tuple[Path, ...] = ()
    # The most each person's folder may hold (``--quota-gb``); None is no limit.
    quota_bytes: int | None = None
    # How many heavy actions run at once outside the queue (``--max-actions``).
    max_actions: int = DEFAULT_MAX_ACTIONS

    @classmethod
    def from_options(
        cls,
        *,
        enabled: bool,
        trusted_proxies: Iterable[str] = (),
        user_header: str = DEFAULT_USER_HEADER,
        datasets_dirs: Iterable[str | Path] = (),
        quota_gb: float | None = None,
        max_actions: int | None = None,
    ) -> ServerMode:
        """Validate what the command line gave.

        Raises:
            ServerModeError: ``--trusted-proxy`` or ``--user-header`` without
                ``--server``; server mode with no trusted proxy; an address that
                is not an IP or a CIDR block; an empty header name; a
                ``--datasets-dir`` that is not a folder; ``--quota-gb`` without
                ``--server`` or not above zero; ``--max-actions`` without
                ``--server`` or below one.
        """
        proxies = [p.strip() for p in trusted_proxies if p.strip()]
        header = user_header.strip()
        shared = [Path(d) for d in datasets_dirs]
        if not enabled:
            given = proxies or shared or header != DEFAULT_USER_HEADER or quota_gb
            if given or max_actions is not None:
                raise ServerModeError(
                    "--trusted-proxy, --user-header, --datasets-dir, --quota-gb and "
                    "--max-actions need --server."
                )
            return cls()
        if max_actions is not None and max_actions < 1:
            raise ServerModeError(f"--max-actions {max_actions!r} must be 1 or more.")
        if quota_gb is not None and not quota_gb > 0:
            raise ServerModeError(f"--quota-gb {quota_gb!r} must be above zero.")
        if not proxies:
            raise ServerModeError(
                "--server needs --trusted-proxy <address>: the address of the "
                "reverse proxy that signs people in. Without it anyone who "
                "reaches the port could name themselves in the user header."
            )
        if not header or header.lower() == PROFILE_HEADER.lower():
            raise ServerModeError(f"--user-header cannot be {user_header!r}.")
        networks: list[Network] = []
        for proxy in proxies:
            try:
                networks.append(ipaddress.ip_network(proxy, strict=False))
            except ValueError as exc:
                raise ServerModeError(
                    f"--trusted-proxy {proxy!r} is not an IP address or a CIDR block."
                ) from exc
        for folder in shared:
            if not folder.is_dir():
                raise ServerModeError(
                    f"--datasets-dir {str(folder)!r} is not a folder."
                )
        return cls(
            enabled=True,
            trusted=tuple(networks),
            user_header=header,
            datasets=tuple(folder.resolve() for folder in shared),
            quota_bytes=None if quota_gb is None else int(quota_gb * GIB),
            max_actions=DEFAULT_MAX_ACTIONS if max_actions is None else max_actions,
        )

    def trusts(self, client_host: str | None) -> bool:
        """True when ``client_host`` is one of the proxies' addresses."""
        if client_host is None:
            return False
        try:
            address = ipaddress.ip_address(client_host)
        except ValueError:
            return False
        mapped = getattr(address, "ipv4_mapped", None)
        if mapped is not None:
            address = mapped
        return any(
            address.version == network.version and address in network
            for network in self.trusted
        )


OFF = ServerMode()


def mode_of(app: Any) -> ServerMode:
    """The mode ``start_server`` stored on ``app``, or off."""
    mode = getattr(getattr(app, "state", None), "server_mode", None)
    return mode if isinstance(mode, ServerMode) else OFF


def normalize_user(raw: str | None) -> str | None:
    """The account name the proxy sent, in one spelling; ``None`` if unusable.

    Case is folded: an identity provider that sends ``Ana@lab.org`` today and
    ``ana@lab.org`` tomorrow means the same account.
    """
    if raw is None:
        return None
    user = raw.strip().casefold()
    if not user or len(user) > MAX_USER_LEN:
        return None
    if any(not ch.isprintable() or ch.isspace() for ch in user):
        return None
    return user


def user_slug(user: str) -> str:
    """The profile folder of the account ``user`` (already normalised).

    The readable part comes from the name and the tail is a digest of the whole
    name: ``a.b@lab.org`` and ``a-b@lab.org`` fold to the same readable part and
    must not share a folder.
    """
    digest = hashlib.sha256(user.encode("utf-8")).hexdigest()[:_DIGEST_LEN]
    base = slugify(user)[:_SLUG_BASE_LEN].rstrip("-_") or "u"
    return f"{base}-{digest}"


_PARAM = re.compile(r"{[^{}:]+(:path)?}")


def _pattern(template: str) -> re.Pattern[str]:
    """``/api/runs/{run_id}`` as a pattern; ``{x:path}`` also takes slashes."""
    parts: list[str] = []
    end = 0
    for found in _PARAM.finditer(template):
        parts.append(re.escape(template[end : found.start()]))
        parts.append(".*" if found.group(1) else "[^/]+")
        end = found.end()
    parts.append(re.escape(template[end:]))
    return re.compile("".join(parts))


def _specificity(key: str) -> tuple[int, int, int]:
    """Sort key: a literal path before one with parameters, the catch-all last."""
    template = key.split(" ", 1)[1]
    return (template.count(":path}"), template.count("{"), -len(template))


def _matchers() -> list[tuple[str, re.Pattern[str], str]]:
    rows = []
    for key in sorted(ROUTE_CLASSES, key=_specificity):
        method, template = key.split(" ", 1)
        rows.append((method, _pattern(template), key))
    return rows


# The table is matched here, not through the app's router: how a router keeps
# the routes of an included one changes between FastAPI versions, and a gate
# that cannot find a route must not depend on that.
_MATCHERS = _matchers()


def route_key(method: str, path: str) -> str | None:
    """The ``ROUTE_CLASSES`` key of the route that answers ``method path``."""
    asked = "GET" if method == "HEAD" else method
    for route_method, pattern, key in _MATCHERS:
        if route_method == asked and pattern.fullmatch(path):
            return None if key == SPA_ROUTE and _not_a_page(path) else key
    return None


def _not_a_page(path: str) -> bool:
    """True for a path the SPA fallback matches and that is no page of the SPA:
    an ``/api`` address no route serves, and the API documents."""
    return path == "/api" or path.startswith("/api/") or path in _API_DOCUMENTS


def answers(key: str) -> bool:
    """True when the route ``key`` answers in server mode today."""
    kind = ROUTE_CLASSES.get(key)
    return kind == "open" or (kind in ("owner", "confined") and key in READY)


def closed_routes() -> list[str]:
    """The routes that do not answer in server mode today, for the screen.

    The GUI reads this from ``GET /api/session`` and hides what would answer
    404, so the list of what a server offers lives here and nowhere else.
    """
    return sorted(key for key in ROUTE_CLASSES if not answers(key))


class ServerModeGate:
    """ASGI middleware applying :class:`ServerMode` to every request."""

    def __init__(self, app: ASGIApp, profiles_dir: Callable[[], Path]) -> None:
        self.app = app
        self._profiles_dir = profiles_dir

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        mode = mode_of(scope.get("app"))
        if not mode.enabled:
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        if not mode.trusts(client[0] if client else None):
            await _refuse(scope, receive, send, 403, "server.untrusted_client")
            return
        if scope["type"] == "websocket":
            await _refuse(scope, receive, send, 404, "server.route_off")
            return
        key = route_key(scope["method"], scope["path"])
        if key is None or not answers(key):
            await _refuse(scope, receive, send, 404, "server.route_off")
            return
        raw = Headers(scope=scope).get(mode.user_header)
        if raw is None and key == HEALTH_ROUTE:
            await self.app(_with_profile(scope, None), receive, send)
            return
        user = normalize_user(raw)
        if user is None:
            await _refuse(
                scope, receive, send, 401, "server.no_user", header=mode.user_header
            )
            return
        try:
            profile = ensure_profile(user_slug(user), user, self._profiles_dir())
        except ProfileError as exc:
            response = JSONResponse({"detail": str(exc)}, status_code=500)
            await response(scope, receive, send)
            return
        signed = _with_profile(scope, profile.slug)
        signed.setdefault("state", {})["vf_user"] = user
        try:
            own = own_datasets_dir(profile)
        except OSError as exc:
            response = JSONResponse({"detail": str(exc)}, status_code=500)
            await response(scope, receive, send)
            return
        # Everything of this person sits beside their dataset folder.
        signed["state"]["vf_root"] = own.parent
        if await _over_quota(mode, key, own.parent, scope, receive, send):
            return
        counted = heavy(key)
        if counted and not SLOTS.take(mode.max_actions):
            await _refuse(scope, receive, send, 429, "server.busy")
            return
        try:
            with dataset_scope(
                DatasetScope(
                    readable=(own, *mode.datasets), owner=profile.slug, own=own
                )
            ):
                await self.app(signed, receive, send)
        finally:
            if counted:
                SLOTS.give()
        if scope["method"] == "DELETE":
            # Something of theirs may be gone: measure again on the next ask.
            USAGE.forget(own.parent)


async def _over_quota(
    mode: ServerMode, key: str, root: Path, scope: Scope, receive: Receive, send: Send
) -> bool:
    """Refuse (507) an action that writes when the person's folder is full."""
    if mode.quota_bytes is None or not grows(key):
        return False
    used = await asyncio.to_thread(USAGE.of, root)
    if used < mode.quota_bytes:
        return False
    lang = resolve_lang(Headers(scope=scope).get(LANG_HEADER))
    await _refuse(
        scope,
        receive,
        send,
        507,
        "server.quota_full",
        used=size_text(used, lang),
        limit=size_text(mode.quota_bytes, lang),
    )
    return True


def own_datasets_dir(profile: Profile) -> Path:
    """The signed-in person's own dataset folder, made when missing."""
    root = profile.root if profile.root is not None else profile.models_dir.parent
    folder = (root / "datasets").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _with_profile(scope: Scope, slug: str | None) -> Scope:
    """``scope`` with ``X-VF-Profile`` set to ``slug`` and the client's dropped."""
    name = PROFILE_HEADER.lower().encode("latin-1")
    headers = [(k, v) for k, v in scope["headers"] if k.lower() != name]
    if slug is not None:
        headers.append((name, slug.encode("latin-1")))
    return {**scope, "headers": headers}


async def _refuse(
    scope: Scope, receive: Receive, send: Send, status: int, key: str, **params: str
) -> None:
    if scope["type"] == "websocket":
        await WebSocketClose(code=1008)(scope, receive, send)
        return
    lang = resolve_lang(Headers(scope=scope).get(LANG_HEADER))
    response = JSONResponse(
        {"detail": message(lang, key, **params)}, status_code=status
    )
    await response(scope, receive, send)


__all__ = [
    "DEFAULT_USER_HEADER",
    "READY",
    "ROUTE_CLASSES",
    "RouteClass",
    "ServerMode",
    "ServerModeError",
    "SLOTS",
    "ServerModeGate",
    "USAGE",
    "answers",
    "mode_of",
    "normalize_user",
    "route_key",
    "user_slug",
]
