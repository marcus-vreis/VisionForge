"""Server mode: the previews and the audit read the scope and keep to their owner (ADR-222).

A preview writes a handful of PNGs. On a local install they go to one cache
folder; on a server that folder would be everyone's, so each person's go
inside their own profile, which is also where the artifact route serves from.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import routes
from visionforge.gui.api.dataset_scope import (
    DatasetScope,
    dataset_scope,
    preview_cache,
)
from visionforge.gui.api.sample_dataset import create_sample_dataset
from visionforge.gui.server_mode import ServerMode, user_slug

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}


def test_where_a_previews_images_go(tmp_path: Path) -> None:
    local = Path("outputs/preview_cache")
    own = tmp_path / "profiles" / "ana" / "datasets"
    assert preview_cache(local) == local
    with dataset_scope(DatasetScope(readable=(own,), owner="ana", own=own)):
        assert preview_cache(local) == tmp_path / "profiles" / "ana" / "preview_cache"
    with dataset_scope(DatasetScope(readable=(), owner="ana")):
        assert preview_cache(local) == local  # no folder of theirs: unchanged


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    shared = (tmp_path / "shared").resolve()
    beans = create_sample_dataset(shared, "classification").path
    hidden = create_sample_dataset(
        (tmp_path / "secret").resolve(), "classification"
    ).path
    return {"beans": beans, "hidden": hidden, "shared": shared, "root": tmp_path}


@pytest.fixture
def client(
    world: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], datasets_dirs=[world["shared"]]
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _profile(world: dict[str, Path], account: str) -> Path:
    return (world["root"] / "profiles" / user_slug(account)).resolve()


@pytest.mark.parametrize(
    ("route", "extra", "images"),
    [
        ("preview_augment", {"num_variants": 2}, "variants"),
        ("preview_preprocess", {"steps": [{"kind": "grayscale"}]}, "steps"),
    ],
)
def test_a_preview_of_a_shared_dataset_is_written_inside_the_persons_profile(
    client: TestClient,
    world: dict[str, Path],
    route: str,
    extra: dict[str, object],
    images: str,
) -> None:
    body = {"base_dir": str(world["beans"]), **extra}
    mine = client.post(f"/api/dataset/{route}", json=body, headers=ANA)
    assert mine.status_code == 200, mine.text
    original = Path(mine.json()["original"]).resolve()
    assert original.is_file()
    assert original.is_relative_to(_profile(world, "ana@lab.org") / "preview_cache")
    assert len(mine.json()[images]) >= 1

    # Another person's preview goes to their own folder and leaves hers alone:
    # a shared cache would have erased it.
    theirs = client.post(f"/api/dataset/{route}", json=body, headers=BIA)
    assert theirs.status_code == 200, theirs.text
    other = Path(theirs.json()["original"]).resolve()
    assert other.is_relative_to(_profile(world, "bia@lab.org") / "preview_cache")
    assert original.is_file()
    # Nothing was written into the shared dataset or a cache of the server's.
    assert not (world["shared"] / "preview_cache").exists()


@pytest.mark.parametrize("route", ["preview_augment", "preview_preprocess", "audit"])
def test_a_dataset_outside_the_scope_is_not_read(
    client: TestClient, world: dict[str, Path], route: str
) -> None:
    refused = client.post(
        f"/api/dataset/{route}", json={"base_dir": str(world["hidden"])}, headers=ANA
    )
    assert refused.status_code == 422, refused.text
    relative = client.post(
        f"/api/dataset/{route}", json={"base_dir": "exemplo"}, headers=ANA
    )
    assert relative.status_code == 422


def test_a_shared_dataset_can_be_audited(
    client: TestClient, world: dict[str, Path]
) -> None:
    audit = client.post(
        "/api/dataset/audit", json={"base_dir": str(world["beans"])}, headers=ANA
    )
    assert audit.status_code == 200, audit.text
    assert sum(audit.json()["image_counts"].values()) > 0
    # A split that climbs out of the dataset is reported, never read.
    climbing = client.post(
        "/api/dataset/audit",
        json={"base_dir": str(world["beans"]), "splits": {"train": "../../secret"}},
        headers=ANA,
    )
    assert climbing.status_code in (200, 400), climbing.text
    if climbing.status_code == 200:
        assert climbing.json()["image_counts"] == {"train": 0}


def test_the_session_no_longer_lists_these_routes_as_closed(
    client: TestClient,
) -> None:
    closed = client.get("/api/session", headers=ANA).json()["closed"]
    for name in ("audit", "preview_augment", "preview_preprocess"):
        assert f"POST /api/dataset/{name}" not in closed
