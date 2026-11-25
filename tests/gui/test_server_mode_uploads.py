"""Server mode: a person sends a dataset of their own as a .zip, in parts (ADR-246).

Each test runs the real app behind the trusted proxy, with a small part size
so a few kilobytes make several parts.
"""

from __future__ import annotations

import io
import os
import stat
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import routes, upload_routes
from visionforge.gui.server_mode import (
    ROUTE_CLASSES,
    ServerMode,
    ServerModeError,
    answers,
    user_slug,
)

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}
PART = 1024
MIB = 1024 * 1024
ROUTES = (
    "POST /api/uploads",
    "GET /api/uploads/{upload_id}",
    "DELETE /api/uploads/{upload_id}",
    "POST /api/uploads/{upload_id}/complete",
    "PUT /api/uploads/{upload_id}/parts/{part}",
)


def _serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **options: Any
) -> Iterator[TestClient]:
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(upload_routes, "PART_SIZE", PART)
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], **options
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _serve(tmp_path, monkeypatch)


@pytest.fixture
def tight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The same server with one mebibyte per person."""
    yield from _serve(tmp_path, monkeypatch, quota_gb=1 / 1024)


def _own(client: TestClient, tmp_path: Path, headers: dict[str, str]) -> Path:
    client.get("/api/profiles", headers=headers)  # makes the profile and its folder
    return tmp_path / "profiles" / user_slug(headers["X-Forwarded-User"]) / "datasets"


def _zip(entries: dict[str, bytes], compression: int = zipfile.ZIP_STORED) -> bytes:
    """An archive with exactly these names, written as given (no clean-up)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, data in entries.items():
            info = zipfile.ZipInfo("placeholder")
            # Set after construction: ZipInfo turns ``\\`` into ``/`` on Windows.
            info.filename = name
            info.compress_type = compression
            archive.writestr(info, data)
    return buffer.getvalue()


def _dataset_zip() -> bytes:
    """A zipped dataset folder: one top folder, two splits, random images."""
    return _zip(
        {
            "beans/train/a/1.jpg": os.urandom(1500),
            "beans/train/b/2.jpg": os.urandom(1500),
            "beans/val/a/3.jpg": os.urandom(700),
            "__MACOSX/beans/._1.jpg": b"mac metadata",
        }
    )


def _start(
    client: TestClient, data: bytes, headers: dict[str, str], name: str = "beans"
) -> dict[str, Any]:
    body = {"name": name, "filename": f"{name}.zip", "size": len(data)}
    started = client.post("/api/uploads", json=body, headers=headers)
    assert started.status_code == 200, started.text
    return dict(started.json())


def _send(
    client: TestClient,
    upload: dict[str, Any],
    data: bytes,
    headers: dict[str, str],
    parts: list[int] | None = None,
) -> None:
    size = upload["part_size"]
    for part in range(upload["parts"]) if parts is None else parts:
        sent = client.put(
            f"/api/uploads/{upload['id']}/parts/{part}",
            content=data[part * size : (part + 1) * size],
            headers=headers,
        )
        assert sent.status_code == 200, sent.text


def _upload(
    client: TestClient, data: bytes, headers: dict[str, str], name: str = "beans"
) -> Any:
    upload = _start(client, data, headers, name)
    _send(client, upload, data, headers)
    return client.post(f"/api/uploads/{upload['id']}/complete", headers=headers)


def _staging(own: Path) -> Path:
    return own.parent / "uploads"


def test_the_route_classes() -> None:
    for key in ROUTES:
        assert ROUTE_CLASSES[key] == "owner"
        assert answers(key)


def test_the_size_limit_needs_server_mode_and_a_size_above_zero() -> None:
    with pytest.raises(ServerModeError, match="--max-upload-gb"):
        ServerMode.from_options(enabled=False, max_upload_gb=1)
    for bad in (0, -2, float("nan")):
        with pytest.raises(ServerModeError, match="above zero"):
            ServerMode.from_options(
                enabled=True, trusted_proxies=[PROXY], max_upload_gb=bad
            )
    mode = ServerMode.from_options(enabled=True, trusted_proxies=[PROXY])
    assert mode.max_upload_bytes == 5 * 1024**3


def test_a_zip_sent_in_parts_becomes_a_dataset_of_their_own(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    data = _dataset_zip()
    upload = _start(client, data, ANA)
    assert upload["parts"] == -(-len(data) // PART) > 2
    assert upload["received"] == []

    _send(client, upload, data, ANA)
    done = client.post(f"/api/uploads/{upload['id']}/complete", headers=ANA)

    assert done.status_code == 200, done.text
    # The zipped top folder is the dataset's root; macOS metadata is left out.
    assert done.json() == {
        "name": "beans",
        "path": str(own / "beans"),
        "size_bytes": 3700,
    }
    assert sorted(p.relative_to(own).as_posix() for p in own.rglob("*.jpg")) == [
        "beans/train/a/1.jpg",
        "beans/train/b/2.jpg",
        "beans/val/a/3.jpg",
    ]
    assert not (own / "beans" / "__MACOSX").exists()
    # The staging is gone and the dataset is listed like any other.
    assert list(_staging(own).iterdir()) == []
    listed = client.get("/api/datasets/own", headers=ANA).json()["datasets"]
    assert [d["name"] for d in listed] == ["beans"]


def test_a_missing_part_is_found_and_sent_again(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    data = _dataset_zip()
    upload = _start(client, data, ANA)
    last = upload["parts"] - 1
    _send(client, upload, data, ANA, parts=[p for p in range(last + 1) if p != 1])

    early = client.post(f"/api/uploads/{upload['id']}/complete", headers=ANA)
    assert early.status_code == 409, early.text
    assert "(1)" in early.json()["detail"]
    status = client.get(f"/api/uploads/{upload['id']}", headers=ANA).json()
    assert status["received"] == [p for p in range(last + 1) if p != 1]

    # A part sent twice rewrites the same bytes.
    _send(client, upload, data, ANA, parts=[1, 1, 0])
    assert client.get(f"/api/uploads/{upload['id']}", headers=ANA).json()[
        "received"
    ] == list(range(last + 1))
    done = client.post(f"/api/uploads/{upload['id']}/complete", headers=ANA)
    assert done.status_code == 200, done.text
    assert (own / "beans" / "val" / "a" / "3.jpg").is_file()


def test_starting_again_after_a_reload_resumes_the_same_upload(
    client: TestClient,
) -> None:
    data = _dataset_zip()
    first = _start(client, data, ANA)
    _send(client, first, data, ANA, parts=[0])
    again = _start(client, data, ANA)
    assert again["id"] == first["id"]
    assert again["received"] == [0]
    # Another file for the same folder is refused while the first is open.
    other = {"name": "beans", "filename": "other.zip", "size": len(data)}
    busy = client.post("/api/uploads", json=other, headers=ANA)
    assert busy.status_code == 409, busy.text


def test_a_part_of_the_wrong_length_or_number_is_refused(client: TestClient) -> None:
    data = _dataset_zip()
    upload = _start(client, data, ANA)
    url = f"/api/uploads/{upload['id']}/parts"
    short = client.put(f"{url}/0", content=b"x" * (PART - 1), headers=ANA)
    assert short.status_code == 400, short.text
    long = client.put(f"{url}/0", content=b"x" * (PART + 1), headers=ANA)
    assert long.status_code == 400, long.text
    beyond = client.put(f"{url}/{upload['parts']}", content=b"x", headers=ANA)
    assert beyond.status_code == 400, beyond.text
    assert client.put(f"{url}/-1", content=b"x", headers=ANA).status_code == 400
    assert (
        client.get(f"/api/uploads/{upload['id']}", headers=ANA).json()["received"] == []
    )


def test_another_account_cannot_see_send_finish_or_cancel_it(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    _own(client, tmp_path, BIA)
    data = _dataset_zip()
    upload = _start(client, data, ANA)
    url = f"/api/uploads/{upload['id']}"

    assert client.get(url, headers=BIA).status_code == 404
    assert (
        client.put(f"{url}/parts/0", content=data[:PART], headers=BIA).status_code
        == 404
    )
    assert client.post(f"{url}/complete", headers=BIA).status_code == 404
    assert client.delete(url, headers=BIA).status_code == 404
    # A made-up id, or one that is not an id, is the same answer.
    assert client.get("/api/uploads/" + "0" * 32, headers=ANA).status_code == 404
    assert client.get("/api/uploads/not-an-id", headers=ANA).status_code == 404

    # Ana's upload is untouched and still hers.
    assert client.get(url, headers=ANA).json()["received"] == []
    _send(client, upload, data, ANA)
    assert client.post(f"{url}/complete", headers=ANA).status_code == 200
    assert (own / "beans").is_dir()


def test_cancelling_drops_every_part(client: TestClient, tmp_path: Path) -> None:
    own = _own(client, tmp_path, ANA)
    data = _dataset_zip()
    upload = _start(client, data, ANA)
    _send(client, upload, data, ANA, parts=[0])
    gone = client.delete(f"/api/uploads/{upload['id']}", headers=ANA)
    assert gone.status_code == 200, gone.text
    assert gone.json() == {"id": upload["id"], "status": "cancelled"}
    assert list(_staging(own).iterdir()) == []
    assert client.get(f"/api/uploads/{upload['id']}", headers=ANA).status_code == 404


def test_an_existing_dataset_is_never_replaced(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    (own / "beans").mkdir()
    (own / "beans" / "mine.txt").write_text("keep", encoding="utf-8")
    data = _dataset_zip()
    body = {"name": "beans", "filename": "beans.zip", "size": len(data)}
    refused = client.post("/api/uploads", json=body, headers=ANA)
    assert refused.status_code == 409, refused.text
    assert "beans" in refused.json()["detail"]

    # A folder made after the upload started is not replaced either.
    upload = _start(client, data, ANA, name="apples")
    _send(client, upload, data, ANA)
    (own / "apples").mkdir()
    (own / "apples" / "mine.txt").write_text("keep", encoding="utf-8")
    late = client.post(f"/api/uploads/{upload['id']}/complete", headers=ANA)
    assert late.status_code == 409, late.text
    assert [p.name for p in (own / "apples").iterdir()] == ["mine.txt"]
    assert [p.name for p in (own / "beans").iterdir()] == ["mine.txt"]


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("..", 422),
        (".hidden", 422),
        ("a/b", 422),
        ("a\\b", 422),
        ("C:", 422),
        ("", 422),
    ],
)
def test_the_folder_name_is_one_plain_name(
    client: TestClient, tmp_path: Path, name: str, status: int
) -> None:
    own = _own(client, tmp_path, ANA)
    body = {"name": name, "filename": "d.zip", "size": 10}
    assert client.post("/api/uploads", json=body, headers=ANA).status_code == status
    assert not _staging(own).exists() or list(_staging(own).iterdir()) == []


def test_only_a_zip_within_the_size_limit_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for small in _serve(tmp_path, monkeypatch, max_upload_gb=1 / 1024):
        body = {"name": "d", "filename": "d.tar", "size": 10}
        assert small.post("/api/uploads", json=body, headers=ANA).status_code == 422
        body = {"name": "d", "filename": "d.ZIP", "size": MIB + 1}
        refused = small.post("/api/uploads", json=body, headers=ANA)
        assert refused.status_code == 413, refused.text
        body["size"] = MIB
        assert small.post("/api/uploads", json=body, headers=ANA).status_code == 200


def test_the_quota_is_checked_at_the_start_and_before_unpacking(
    tight: TestClient, tmp_path: Path
) -> None:
    own = _own(tight, tmp_path, ANA)
    body = {"name": "big", "filename": "big.zip", "size": 2 * MIB}
    refused = tight.post("/api/uploads", json=body, headers=ANA)
    assert refused.status_code == 507, refused.text
    assert not _staging(own).exists() or list(_staging(own).iterdir()) == []

    # It fits zipped, and not once unpacked beside the archive.
    data = _zip({"d/a.bin": os.urandom(600 * 1024)})
    upload = _start(tight, data, ANA, name="d")
    _send(tight, upload, data, ANA)
    full = tight.post(f"/api/uploads/{upload['id']}/complete", headers=ANA)
    assert full.status_code == 507, full.text
    assert not (own / "d").exists()
    # The parts are kept: free some space and finish.
    assert tight.get(f"/api/uploads/{upload['id']}", headers=ANA).json()[
        "received"
    ] == list(range(upload["parts"]))


def test_a_folder_over_the_quota_takes_no_more_parts(
    tight: TestClient, tmp_path: Path
) -> None:
    own = _own(tight, tmp_path, ANA)
    data = _dataset_zip()
    upload = _start(tight, data, ANA)
    (own / "grown.bin").write_bytes(b"\0" * (2 * MIB))
    from visionforge.gui.server_mode import USAGE

    USAGE.forget(own.parent)
    sent = tight.put(
        f"/api/uploads/{upload['id']}/parts/0", content=data[:PART], headers=ANA
    )
    assert sent.status_code == 507, sent.text


def _refused(client: TestClient, own: Path, data: bytes, word: str) -> None:
    answer = _upload(client, data, ANA)
    assert answer.status_code == 422, answer.text
    assert word in answer.json()["detail"]
    # Nothing was unpacked, and the upload, useless now, is dropped.
    assert list(own.iterdir()) == []
    assert list(_staging(own).iterdir()) == []


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.txt",
        "beans/../../escaped.txt",
        # Read as ``../`` on Windows, refused for its ``\\`` elsewhere.
        "..\\escaped.txt",
        "/abs.txt",
        "C:/x.txt",
    ],
)
def test_a_name_that_leaves_the_folder_unpacks_nothing(
    client: TestClient, tmp_path: Path, name: str
) -> None:
    own = _own(client, tmp_path, ANA)
    data = _zip({"beans/train/a/1.jpg": os.urandom(800), name: b"escape"})
    _refused(client, own, data, "caminho")
    assert not (own.parent / "escaped.txt").exists()
    assert not (own.parent.parent / "escaped.txt").exists()


def test_a_symbolic_link_unpacks_nothing(client: TestClient, tmp_path: Path) -> None:
    own = _own(client, tmp_path, ANA)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("beans/train/a/1.jpg", os.urandom(800))
        link = zipfile.ZipInfo("beans/train/a/link.jpg")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "/etc/passwd")
    _refused(client, own, buffer.getvalue(), "link simbólico")


def test_a_zip_bomb_unpacks_nothing(client: TestClient, tmp_path: Path) -> None:
    own = _own(client, tmp_path, ANA)
    data = _zip({"beans/bomb.bin": b"\0" * (4 * MIB)}, zipfile.ZIP_DEFLATED)
    assert len(data) < 64 * 1024
    _refused(client, own, data, "bomba de compressão")


def test_what_is_not_a_zip_unpacks_nothing(client: TestClient, tmp_path: Path) -> None:
    own = _own(client, tmp_path, ANA)
    _refused(client, own, os.urandom(3000), ".zip")


def test_an_idle_upload_expires_on_the_next_start(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    data = _dataset_zip()
    old = _start(client, data, ANA)
    record = _staging(own) / old["id"] / "upload.json"
    day_ago = time.time() - upload_routes.EXPIRE_AFTER_S - 60
    os.utime(record, (day_ago, day_ago))

    fresh = _start(client, data, ANA, name="apples")

    assert [p.name for p in _staging(own).iterdir()] == [fresh["id"]]
    assert client.get(f"/api/uploads/{old['id']}", headers=ANA).status_code == 404


def test_a_person_has_a_few_uploads_open_at_most(client: TestClient) -> None:
    data = _dataset_zip()
    for index in range(upload_routes.MAX_OPEN):
        _start(client, data, ANA, name=f"d{index}")
    body = {"name": "one-more", "filename": "x.zip", "size": len(data)}
    refused = client.post("/api/uploads", json=body, headers=ANA)
    assert refused.status_code == 409, refused.text
    # Somebody else's count is theirs.
    _start(client, data, BIA, name="d0")


def test_a_local_install_takes_no_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.gui.server import app

    monkeypatch.chdir(tmp_path)
    local = TestClient(app, base_url="http://localhost")
    body = {"name": "d", "filename": "d.zip", "size": 10}
    assert local.post("/api/uploads", json=body).status_code == 404
    assert local.get("/api/uploads/" + "0" * 32).status_code == 404
    assert list(tmp_path.iterdir()) == []
