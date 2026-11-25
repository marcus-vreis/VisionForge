"""Check server mode behind a real login proxy (ADR-247).

Runs against the stack of ``deploy/server/docker-compose.yml``: a mock OpenID
Connect provider, oauth2-proxy and ``visionforge gui --server``. It signs in two
accounts through the provider's login page, the way a browser does, and checks
what the e2e suite cannot (ADR-224): that the proxy, not the client, names the
user.

    python scripts/check_server_proxy.py \\
        --proxy http://localhost:4180 --provider http://localhost:9080 \\
        --direct http://localhost:8000

Standard library only, so the CI runner needs no install. Exits 1 at the first
check that fails, naming it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.cookiejar import CookieJar
from typing import Any

USER_HEADER = "X-Forwarded-User"
ALICE = "alice"
BOB = "bob"
# A run's status once it no longer holds the queue.
FINISHED = frozenset({"completed", "failed", "error", "stopped"})


class CheckFailed(Exception):
    """A check did not hold; the message says which."""


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def location(self) -> str:
        return self.headers.get("location", "")

    def json(self) -> Any:
        return json.loads(self.body or b"null")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hand every redirect back to the caller, which checks where it points."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


@dataclass
class Client:
    """One browser: its own cookie jar, redirects not followed."""

    base: str
    jar: CookieJar = field(default_factory=CookieJar)
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._opener = urllib.request.build_opener(
            _NoRedirect, urllib.request.HTTPCookieProcessor(self.jar)
        )

    def send(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Response:
        """``url`` may be a path, resolved against ``base``."""
        full = urllib.parse.urljoin(self.base + "/", url)
        merged = {**self.extra_headers, **(headers or {})}
        request = urllib.request.Request(full, data=body, headers=merged, method=method)
        try:
            with self._opener.open(request, timeout=timeout) as reply:
                return Response(reply.status, _lower(reply.headers), reply.read())
        except urllib.error.HTTPError as error:
            return Response(error.code, _lower(error.headers), error.read())

    def get_json(self, path: str, **headers: str) -> Response:
        return self.send("GET", path, headers={"Accept": "application/json", **headers})

    def post_json(self, path: str, payload: dict[str, Any]) -> Response:
        return self.send(
            "POST",
            path,
            body=json.dumps(payload).encode(),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )


def _lower(headers: Any) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def expect(condition: bool, what: str, detail: object = "") -> None:
    if not condition:
        raise CheckFailed(f"{what}: {detail}" if detail != "" else what)
    print(f"ok    {what}")


def wait_until_ready(proxy: str, timeout: float) -> None:
    """The proxy answers and lets ``GET /api/health`` through without a login."""
    client = Client(proxy)
    deadline = time.monotonic() + timeout
    last: object = "no answer"
    while time.monotonic() < deadline:
        try:
            reply = client.get_json("/api/health")
            if reply.status == 200:
                expect(True, "GET /api/health passes the proxy without a login")
                return
            last = reply.status
        except OSError as error:
            last = error
        time.sleep(2)
    raise CheckFailed(f"stack not ready after {timeout:.0f} s: {last}")


def check_anonymous(proxy: str, provider: str) -> None:
    """No login: a page goes to the provider, the API is 401, never VisionForge."""
    client = Client(proxy)
    page = client.send("GET", "/", headers={"Accept": "text/html"})
    expect(
        page.status == 302 and page.location.startswith(provider),
        "anonymous page request is sent to the provider's login",
        f"{page.status} {page.location!r}",
    )
    for headers in ({}, {USER_HEADER: ALICE}):
        api = client.get_json("/api/session", **headers)
        label = " with a forged user header" if headers else ""
        expect(
            api.status in (302, 401) and b"server_mode" not in api.body,
            f"anonymous GET /api/session{label} is refused before VisionForge",
            api.status,
        )


def login(proxy: str, provider: str, username: str) -> Client:
    """Sign ``username`` in through the provider's form, as a browser would."""
    client = Client(proxy)
    start = client.send("GET", "/", headers={"Accept": "text/html"})
    if start.status != 302 or not start.location.startswith(provider):
        raise CheckFailed(f"{username}: no redirect to the provider ({start.status})")
    form = client.send("GET", start.location, headers={"Accept": "text/html"})
    if form.status != 200 or b'name="username"' not in form.body:
        raise CheckFailed(
            f"{username}: the provider showed no login form ({form.status})"
        )
    claims = {"email": f"{username}@lab.test", "email_verified": True}
    fields = urllib.parse.urlencode(
        {"username": username, "claims": json.dumps(claims)}
    )
    signed = client.send(
        "POST",
        start.location,
        body=fields.encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if signed.status != 302 or "/oauth2/callback" not in signed.location:
        raise CheckFailed(
            f"{username}: the provider did not send the code back ({signed.status})"
        )
    back = client.send("GET", signed.location, headers={"Accept": "text/html"})
    # The session cookie, split into _0, _1... when large; not the CSRF one.
    session_cookie = any(
        c.name.startswith("_oauth2_proxy") and "csrf" not in c.name for c in client.jar
    )
    if back.status != 302 or not session_cookie:
        raise CheckFailed(
            f"{username}: the proxy set no session cookie ({back.status} {back.body[:300]!r})"
        )
    expect(True, f"{username} signed in through the provider")
    return client


def check_identity(client: Client, username: str, impostor: str) -> None:
    """The proxy names the account; a header the client sends changes nothing."""
    session = client.get_json("/api/session")
    expect(
        session.status == 200 and session.json().get("user") == username,
        f"/api/session names {username}",
        f"{session.status} {session.body[:300]!r}",
    )
    expect(session.json().get("server_mode") is True, "VisionForge runs in server mode")
    for name in (USER_HEADER, USER_HEADER.lower()):
        forged = client.get_json("/api/session", **{name: impostor})
        expect(
            forged.status == 200 and forged.json().get("user") == username,
            f"{username} sending '{name}: {impostor}' is still {username}",
            f"{forged.status} {forged.body[:300]!r}",
        )


def train_one_epoch(client: Client, timeout: float) -> tuple[str, str]:
    """Train the sample classification dataset for one epoch.

    Returns the run's id in the history and the sample dataset's folder name.
    The run's name is unique, so a second check on a kept volume finds its own.
    """
    name = f"proxy_check_{int(time.time())}"
    sample = client.post_json("/api/sample-dataset", {"task": "classification"})
    expect(sample.status in (200, 201), "the sample dataset is written", sample.status)
    data = sample.json()
    config = {
        "name": name,
        "task": "multiclass",
        "block": "classification",
        "model": {
            "name": "resnet18",
            # A sample folder that already exists answers without its classes
            # (ADR-206); the sample always has two.
            "num_classes": len(data["classes"]) or 2,
            "pretrained": False,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
            "seed": 0,
        },
        "data": {
            "base_dir": data["path"],
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {"image_size": 32},
        },
        "device": {"kind": "cpu"},
    }
    started = client.post_json("/api/experiment/run", config)
    expect(started.status == 200, "the training is accepted", started.body[:300])
    job = started.json().get("run_id")
    deadline = time.monotonic() + timeout
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status = client.get_json("/api/experiment/status").json() or {}
        if status.get("run_id") == job and status.get("status") in FINISHED:
            break
        time.sleep(2)
    expect(
        status.get("run_id") == job and status.get("status") == "completed",
        "the training completed",
        status,
    )
    runs = client.get_json("/api/runs").json()
    mine = [run for run in runs if run.get("experiment_name") == name]
    expect(len(mine) == 1, "the run is in its owner's history", runs)
    folder = data["path"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return str(mine[0]["run_id"]), folder


def check_isolation(owner: Client, other: Client, run_id: str, dataset: str) -> None:
    """``other`` sees neither the run nor the dataset of ``owner``."""
    theirs = other.get_json("/api/runs")
    ids = [run.get("run_id") for run in theirs.json()] if theirs.status == 200 else None
    expect(
        ids is not None and run_id not in ids,
        "the other account's history does not list the run",
        f"{theirs.status} {ids}",
    )
    direct = other.get_json(f"/api/runs/{run_id}")
    expect(
        direct.status == 404, "the other account gets 404 for the run", direct.status
    )
    own = [d["name"] for d in owner.get_json("/api/datasets/own").json()["datasets"]]
    other_own = [
        d["name"] for d in other.get_json("/api/datasets/own").json()["datasets"]
    ]
    expect(
        dataset in own and dataset not in other_own,
        "the sample dataset is only in its owner's folder",
        f"{own} / {other_own}",
    )


def check_direct(urls: list[str]) -> None:
    """VisionForge's own port is not reachable from outside the proxy's network."""
    for url in urls:
        client = Client(url)
        try:
            reply = client.get_json("/api/session", **{USER_HEADER: ALICE})
        except OSError as error:
            expect(True, f"{url} is not exposed ({type(error).__name__})")
            continue
        expect(
            reply.status in (401, 403),
            f"{url} refuses a request that skipped the proxy",
            reply.status,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--proxy", default="http://localhost:4180")
    parser.add_argument("--provider", default="http://localhost:9080")
    parser.add_argument(
        "--direct",
        action="append",
        default=[],
        help="an address of VisionForge's own port, expected unreachable or 401/403",
    )
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--train-timeout", type=float, default=240.0)
    args = parser.parse_args(argv)
    proxy = args.proxy.rstrip("/")
    provider = args.provider.rstrip("/")
    try:
        wait_until_ready(proxy, args.ready_timeout)
        check_anonymous(proxy, provider)
        alice = login(proxy, provider, ALICE)
        bob = login(proxy, provider, BOB)
        check_identity(alice, ALICE, BOB)
        check_identity(bob, BOB, ALICE)
        run_id, dataset = train_one_epoch(bob, args.train_timeout)
        check_isolation(bob, alice, run_id, dataset)
        check_direct(args.direct)
    except CheckFailed as failure:
        print(f"FAIL  {failure}", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
