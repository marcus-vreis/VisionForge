"""Server mode: only the login proxy is answered, and it says who is asking.

With ``--server`` the user comes from the proxy's header and nothing the
browser sends can change it; a route that is not ready for several people is
404 (ADR-212, ADR-213). With the mode off nothing changes.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from visionforge.gui.api import routes
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.profiles import is_valid_slug
from visionforge.gui.api.schemas import DatasetStatsRequest
from visionforge.gui.server_mode import (
    READY,
    ROUTE_CLASSES,
    ServerMode,
    ServerModeError,
    answers,
    normalize_user,
    route_key,
    user_slug,
)

PROXY = "10.0.0.5"
USER = {"X-Forwarded-User": "ana@lab.org"}


def test_off_by_default_and_the_proxy_options_need_the_flag() -> None:
    assert ServerMode.from_options(enabled=False) == ServerMode()
    with pytest.raises(ServerModeError, match="need --server"):
        ServerMode.from_options(enabled=False, trusted_proxies=["10.0.0.5"])
    with pytest.raises(ServerModeError, match="need --server"):
        ServerMode.from_options(enabled=False, user_header="Remote-User")


def test_server_mode_refuses_to_start_without_a_trusted_proxy() -> None:
    with pytest.raises(ServerModeError, match="--trusted-proxy"):
        ServerMode.from_options(enabled=True)
    with pytest.raises(ServerModeError, match="not an IP address"):
        ServerMode.from_options(enabled=True, trusted_proxies=["proxy.lab"])


def test_the_user_header_cannot_be_the_profile_header() -> None:
    with pytest.raises(ServerModeError, match="--user-header"):
        ServerMode.from_options(
            enabled=True, trusted_proxies=[PROXY], user_header="x-vf-profile"
        )


def test_only_the_proxy_addresses_are_trusted() -> None:
    mode = ServerMode.from_options(
        enabled=True, trusted_proxies=["10.0.0.5", "172.18.0.0/16", "::1"]
    )
    assert mode.trusts("10.0.0.5")
    assert mode.trusts("172.18.3.9")
    assert mode.trusts("::1")
    assert mode.trusts("::ffff:10.0.0.5")
    assert not mode.trusts("10.0.0.6")
    assert not mode.trusts("127.0.0.1")
    assert not mode.trusts("testclient")
    assert not mode.trusts(None)


def test_the_account_name_has_one_spelling() -> None:
    assert normalize_user("  Ana@Lab.org ") == "ana@lab.org"
    assert normalize_user(None) is None
    assert normalize_user("   ") is None
    assert normalize_user("ana lab") is None
    assert normalize_user("ana\x00") is None
    assert normalize_user("a" * 255) is None


def test_two_accounts_never_share_a_folder() -> None:
    dotted, dashed = user_slug("a.b@lab.org"), user_slug("a-b@lab.org")
    assert dotted != dashed
    assert dotted.startswith("a-b-lab-org-")
    assert user_slug("a.b@lab.org") == dotted
    long = user_slug("x" * 200 + "@lab.org")
    for slug in (dotted, dashed, long, user_slug("日本")):
        assert is_valid_slug(slug)


def test_every_route_of_the_app_has_a_class() -> None:
    from visionforge.gui.server import app

    # The OpenAPI document lists every route however the routers are split;
    # ``app.routes`` hides an included router's children on newer FastAPI.
    served = {
        f"{method.upper()} {path}"
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }
    # OpenAPI drops the ``:path`` converter.
    classed = {key.replace(":path}", "}") for key in ROUTE_CLASSES}
    assert served - classed == set(), "routes with no server-mode class"
    assert classed - served == set(), "classes of routes that are gone"


def test_a_request_finds_its_route_in_the_table() -> None:
    assert route_key("GET", "/api/runs") == "GET /api/runs"
    assert route_key("HEAD", "/api/health") == "GET /api/health"
    assert route_key("DELETE", "/api/runs/20261118_1") == "DELETE /api/runs/{run_id}"
    assert route_key("GET", "/api/detection/schema") == "GET /api/detection/schema"
    assert route_key("GET", "/api/custom/mine/schema") == "GET /api/custom/{key}/schema"
    assert (
        route_key("GET", "/api/artifacts/outputs/a/b.png")
        == "GET /api/artifacts/{file_path:path}"
    )
    assert route_key("GET", "/assets/index.js") == "GET /{path:path}"
    assert route_key("GET", "/") == "GET /{path:path}"
    assert route_key("POST", "/api/explorer/scan") == "POST /api/explorer/scan"
    assert route_key("OPTIONS", "/api/runs") is None
    assert route_key("POST", "/api/nothing") is None
    assert route_key("GET", "/api/nothing") is None
    assert route_key("GET", "/openapi.json") is None


def test_only_owner_and_confined_routes_are_marked_as_read() -> None:
    assert {ROUTE_CLASSES[key] for key in READY} == {"owner", "confined"}


def test_a_route_the_table_does_not_know_is_refused() -> None:
    assert not answers("GET /api/brand-new")
    assert answers("GET /api/health")
    assert answers("GET /api/runs")
    assert answers("GET /api/queue")
    assert answers("POST /api/experiment/run")
    assert answers("POST /api/classification/replicates")
    assert answers("POST /api/runs/{run_id}/resume")
    assert answers("GET /api/runs/{run_id}/bundle")
    assert not answers("POST /api/runs/{run_id}/mlflow/retry")
    assert answers("POST /api/dataset/stats")
    assert answers("POST /api/dataset/download")
    assert answers("POST /api/dataset/audit")
    assert not answers("POST /api/dataset/pick")


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The real app in server mode, its profiles under ``tmp_path``."""
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY]
    )
    yield app
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _client(app: Any, host: str = PROXY) -> TestClient:
    return TestClient(app, base_url="http://localhost", client=(host, 50000))


def test_a_local_install_reports_no_session() -> None:
    from visionforge.gui.server import app

    body = TestClient(app, base_url="http://localhost").get("/api/session").json()
    assert body == {
        "server_mode": False,
        "user": None,
        "profile": None,
        "dataset_dirs": [],
        "closed": [],
        "closed_blocks": [],
        "quota": None,
    }


def test_a_client_that_is_not_the_proxy_is_refused(served: Any) -> None:
    response = _client(served, "10.0.0.6").get("/api/session", headers=USER)
    assert response.status_code == 403
    assert "proxy" in response.json()["detail"]


def test_a_request_without_a_user_is_refused(served: Any) -> None:
    response = _client(served).get("/api/session")
    assert response.status_code == 401
    assert "X-Forwarded-User" in response.json()["detail"]
    assert (
        _client(served).get("/", headers={"X-Forwarded-User": " "}).status_code == 401
    )


def test_the_proxy_health_probe_needs_no_user(served: Any) -> None:
    assert _client(served).get("/api/health").status_code == 200


def test_the_user_the_proxy_names_gets_their_own_profile(
    served: Any, tmp_path: Path
) -> None:
    body = _client(served).get("/api/session", headers=USER).json()
    slug = user_slug("ana@lab.org")
    assert (body["server_mode"], body["user"], body["profile"]) == (
        True,
        "ana@lab.org",
        slug,
    )
    assert (tmp_path / "profiles" / slug / "models").is_dir()
    assert _client(served).get("/api/runs", headers=USER).json() == []


def test_the_browser_cannot_pick_another_profile(served: Any, tmp_path: Path) -> None:
    client = _client(served)
    client.get("/api/session", headers={"X-Forwarded-User": "bia@lab.org"})
    theirs = user_slug("bia@lab.org")
    run = tmp_path / "profiles" / theirs / "models" / "exp" / "20261118_120000_000001"
    run.mkdir(parents=True)

    forged = {**USER, "X-VF-Profile": theirs}
    assert client.get("/api/session", headers=forged).json()["profile"] == user_slug(
        "ana@lab.org"
    )
    assert client.get(f"/api/runs/{run.name}", headers=forged).status_code == 404
    assert client.delete(f"/api/runs/{run.name}", headers=forged).status_code == 404
    assert run.is_dir()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/dataset/pick"),
        ("POST", "/api/update/apply"),
        ("POST", "/api/profiles"),
        ("POST", "/api/runs/some_run/mlflow/retry"),
        ("GET", "/api/colab/status"),
        ("GET", "/api/brand-new"),
        ("OPTIONS", "/api/runs"),
        ("GET", "/openapi.json"),
    ],
)
def test_a_route_not_ready_for_several_people_is_404(
    served: Any, method: str, path: str
) -> None:
    response = _client(served).request(method, path, headers=USER)
    assert response.status_code == 404
    assert "servidor" in response.json()["detail"]


# --- Datasets: each person reads only their own folder and the shared ones (ADR-214)


def _png(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (200, 30, 30)).save(path)
    return path


@pytest.fixture
def shared(served: Any, tmp_path: Path) -> Path:
    """A dataset the operator shares: ``beans`` with two classes in ``train``."""
    root = tmp_path / "shared"
    _png(root / "beans" / "train" / "healthy" / "a.png")
    _png(root / "beans" / "train" / "healthy" / "b.png")
    _png(root / "beans" / "train" / "rust" / "c.png")
    served.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], datasets_dirs=[root]
    )
    return root.resolve()


def test_without_a_scope_the_path_types_accept_what_a_string_accepts() -> None:
    req = DatasetStatsRequest(base_dir="relative/folder", train_dir="../elsewhere")
    assert (req.base_dir, req.train_dir) == ("relative/folder", "../elsewhere")


def test_a_dataset_folder_must_sit_in_the_scope(tmp_path: Path) -> None:
    inside = tmp_path / "mine" / "beans"
    inside.mkdir(parents=True)
    (tmp_path / "other").mkdir()
    with dataset_scope(DatasetScope(readable=((tmp_path / "mine").resolve(),))):
        assert DatasetStatsRequest(base_dir=str(inside)).base_dir == str(
            inside.resolve()
        )
        sneaky = str(tmp_path / "mine" / ".." / "other")
        for outside in (str(tmp_path / "other"), sneaky, "mine/beans", "~", "  "):
            with pytest.raises(ValidationError):
                DatasetStatsRequest(base_dir=outside)


@pytest.mark.parametrize(
    "part", ["../other", "a/../../b", "/etc", "C:\\Windows", "\\\\host\\share", "~/x"]
)
def test_a_name_inside_the_dataset_may_only_go_down(tmp_path: Path, part: str) -> None:
    with dataset_scope(DatasetScope(readable=(tmp_path.resolve(),))):
        assert DatasetStatsRequest(base_dir=str(tmp_path), train_dir="a/b").train_dir
        with pytest.raises(ValidationError):
            DatasetStatsRequest(base_dir=str(tmp_path), train_dir=part)


def test_every_path_of_an_open_dataset_route_is_typed() -> None:
    from visionforge.gui.api import (
        coco_routes,
        evidence_routes,
        explorer_routes,
        incorporate_routes,
        merge_routes,
        resplit_routes,
        schemas,
    )
    from visionforge.gui.server import app

    # Where a request model may live; FastAPI names the Laboratory's by module
    # when two share a name (``PreviewRequest``).
    homes = (
        schemas,
        explorer_routes,
        merge_routes,
        resplit_routes,
        incorporate_routes,
        coco_routes,
        evidence_routes,
    )
    components = app.openapi()["components"]["schemas"]

    def model_of(ref: str) -> Any:
        title = components[ref]["title"]
        found = [getattr(home, title) for home in homes if hasattr(home, title)]
        assert found, f"no request model named {title}"
        return found[0]

    # Text fields that name a column, not a place on the disk.
    # A dataset id at its provider and a task key are not places either.
    not_paths = {"image_column", "dataset", "task"}
    paths = app.openapi()["paths"]
    checked = 0
    for key in sorted(READY):
        method, template = key.split(" ", 1)
        if ROUTE_CLASSES[key] != "confined" or method != "POST":
            continue
        if "requestBody" not in paths[template]["post"]:
            continue  # a cancel: it names a job, and a job has an owner
        body = paths[template]["post"]["requestBody"]["content"]["application/json"]
        model = model_of(body["schema"]["$ref"].rsplit("/", 1)[1])
        for name, field in model.model_fields.items():
            if field.annotation is str and name not in not_paths:
                assert field.metadata, f"{model.__name__}.{name} is a plain str"
        checked += 1
    assert checked == 43


def test_the_session_says_where_the_datasets_may_live(
    served: Any, shared: Path, tmp_path: Path
) -> None:
    body = _client(served).get("/api/session", headers=USER).json()
    own = (tmp_path / "profiles" / user_slug("ana@lab.org") / "datasets").resolve()
    assert body["dataset_dirs"] == [str(own), str(shared)]
    assert own.is_dir()


def test_the_session_lists_the_routes_that_do_not_answer(served: Any) -> None:
    client = _client(served)
    closed = client.get("/api/session", headers=USER).json()["closed"]
    assert closed == sorted(closed)
    assert set(closed) == {key for key in ROUTE_CLASSES if not answers(key)}
    # What the screen hides is what the gate refuses, and nothing else.
    assert "POST /api/dataset/pick" in closed
    assert "POST /api/runs/{run_id}/mlflow/retry" in closed
    assert "POST /api/classification/replicates" not in closed
    assert "POST /api/experiment/run" not in closed
    assert "GET /api/session" not in closed
    assert client.post("/api/dataset/pick", headers=USER).status_code == 404


def test_a_shared_dataset_can_be_read(served: Any, shared: Path) -> None:
    client = _client(served)
    stats = client.post(
        "/api/dataset/stats", json={"base_dir": str(shared / "beans")}, headers=USER
    )
    assert stats.status_code == 200
    train = stats.json()["splits"]["train"]
    assert (train["total_images"], train["classes"]) == (3, {"healthy": 2, "rust": 1})
    image = shared / "beans" / "train" / "rust" / "c.png"
    served_file = client.get(
        "/api/dataset/file", params={"path": str(image)}, headers=USER
    )
    assert served_file.status_code == 200


def test_a_folder_outside_the_scope_cannot_be_read(
    served: Any, shared: Path, tmp_path: Path
) -> None:
    client = _client(served)
    client.get("/api/session", headers={"X-Forwarded-User": "bia@lab.org"})
    theirs = tmp_path / "profiles" / user_slug("bia@lab.org") / "datasets"
    secret = _png(tmp_path / "secret" / "train" / "x" / "s.png")
    for base in (theirs, tmp_path / "secret", tmp_path / "profiles"):
        for route in ("/api/dataset/stats", "/api/dataset/detect"):
            refused = client.post(route, json={"base_dir": str(base)}, headers=USER)
            assert refused.status_code == 422, (route, base)
    climbing = client.post(
        "/api/dataset/stats",
        json={"base_dir": str(shared / "beans"), "train_dir": "../../secret/train"},
        headers=USER,
    )
    assert climbing.status_code == 422
    assert secret.is_file()


def test_a_folder_someone_else_opened_is_not_served(
    served: Any, shared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.gui.api import datasets_routes

    secret = _png(tmp_path / "secret" / "s.png")
    monkeypatch.setattr(
        datasets_routes, "_ALLOWED_DATASET_ROOTS", {secret.parent.resolve()}
    )
    response = _client(served).get(
        "/api/dataset/file", params={"path": str(secret.resolve())}, headers=USER
    )
    assert response.status_code == 403


def test_the_shared_folders_need_the_flag_and_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ServerModeError, match="need --server"):
        ServerMode.from_options(enabled=False, datasets_dirs=[tmp_path])
    with pytest.raises(ServerModeError, match="not a folder"):
        ServerMode.from_options(
            enabled=True, trusted_proxies=[PROXY], datasets_dirs=[tmp_path / "nope"]
        )


def _run_cli(argv: list[str]) -> dict[str, Any]:
    from visionforge.__main__ import main

    seen: dict[str, Any] = {}
    with (
        patch.object(sys, "argv", ["visionforge", *argv]),
        patch("visionforge.gui.server.start_server", lambda **kw: seen.update(kw)),
    ):
        main()
    return seen


def test_the_cli_refuses_server_mode_without_a_proxy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stop:
        _run_cli(["gui", "--server"])
    assert stop.value.code == 2
    assert "--trusted-proxy" in capsys.readouterr().err


def test_the_cli_hands_the_server_options_to_the_server() -> None:
    seen = _run_cli(
        [
            "gui",
            "--server",
            "--trusted-proxy",
            "10.0.0.5",
            "--user-header",
            "Remote-User",
        ]
    )
    mode = seen["server_mode"]
    assert mode.enabled
    assert mode.trusts("10.0.0.5")
    assert mode.user_header == "Remote-User"
    assert not _run_cli(["gui"])["server_mode"].enabled


def test_every_route_the_screen_asks_about_is_one_the_server_knows() -> None:
    """The GUI hides a button by the key of its route (ADR-216): a key that is
    not in the table would never be closed, and the button would never hide."""
    source = (
        Path(__file__).parents[2] / "frontend" / "src" / "lib" / "session.ts"
    ).read_text(encoding="utf-8")
    asked = set(re.findall(r'"((?:GET|POST|PUT|DELETE) /api/[^"]+)"', source))
    assert len(asked) > 25
    assert asked <= set(ROUTE_CLASSES), sorted(asked - set(ROUTE_CLASSES))

    # A strategy's route is built from the task and one of these endings.
    endings = set(re.findall(r'^  \w+: "([\w-]+)",$', source, flags=re.MULTILINE))
    assert endings == {"cv", "sweep", "compare", "replicates", "replicated-comparison"}
    built = {
        key
        for key in ROUTE_CLASSES
        if key.startswith("POST /api/") and key.rsplit("/", 1)[-1] in endings
    }
    tasks = {key.split("/")[2] for key in built}
    assert {"classification", "detection", "segmentation", "regression", "anomaly"} <= (
        tasks
    )
