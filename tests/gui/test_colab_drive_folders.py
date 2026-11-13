"""Picking the Colab dataset folder in this computer's "My Drive" (ADR-164).

The GUI browses the local Google Drive for desktop folder; the server lists the
sub-folders of one folder under it and says where that folder is on the VM.
Nothing outside the "My Drive" root is ever listed or opened.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from fastapi.testclient import TestClient

import visionforge.gui.api.colab_run_routes as colab_run_mod
from visionforge.gui.api import colab_drive as cd


@pytest.fixture(autouse=True)
def vf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    return tmp_path


def _my_drive(tmp_path: Path) -> Path:
    root = tmp_path / "Meu Drive"
    for folder in ("datasets/flores/train", "datasets/gatos", "Zeta", "alfa", ".cache"):
        (root / folder).mkdir(parents=True)
    (root / "notas.txt").write_text("x", encoding="utf-8")
    (root / "datasets" / "leia.md").write_text("x", encoding="utf-8")
    return root


def _link_outside(link: Path, target: Path) -> None:
    """A folder link inside the root that leads out of it."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def _hide(path: Path) -> None:
    import ctypes

    assert ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x2)  # type: ignore[attr-defined]


# ── finding "My Drive" ────────────────────────────────────────────────────────


def test_an_existing_manual_root_wins(tmp_path: Path) -> None:
    manual = tmp_path / "Mi unidad"
    manual.mkdir()
    (tmp_path / "G" / "My Drive").mkdir(parents=True)
    found = cd.find_my_drive_root(
        str(manual),
        lambda: None,
        home=tmp_path / "h",
        drives=lambda: [str(tmp_path / "G")],
    )
    assert found == manual


def test_a_manual_root_that_does_not_exist_falls_back_to_the_drive_letter(
    tmp_path: Path,
) -> None:
    (tmp_path / "G" / "Meu Drive").mkdir(parents=True)
    found = cd.find_my_drive_root(
        str(tmp_path / "nope"),
        lambda: None,
        home=tmp_path / "h",
        drives=lambda: [str(tmp_path / "G")],
    )
    assert found == tmp_path / "G" / "Meu Drive"


def test_the_registry_mount_point_is_tried_before_the_defaults(tmp_path: Path) -> None:
    mount = tmp_path / "mnt"
    (mount / "My Drive").mkdir(parents=True)
    (tmp_path / "h" / "Meu Drive").mkdir(parents=True)
    found = cd.find_my_drive_root(
        None, lambda: str(mount), home=tmp_path / "h", drives=lambda: []
    )
    assert found == mount / "My Drive"


def test_mirror_mode_under_the_home_folder_is_found(tmp_path: Path) -> None:
    (tmp_path / "h" / "My Drive").mkdir(parents=True)
    found = cd.find_my_drive_root(
        None, lambda: None, home=tmp_path / "h", drives=lambda: []
    )
    assert found == tmp_path / "h" / "My Drive"


def test_no_drive_for_desktop_finds_nothing(tmp_path: Path) -> None:
    (tmp_path / "G" / "Documents").mkdir(parents=True)
    found = cd.find_my_drive_root(
        "", lambda: None, home=tmp_path / "h", drives=lambda: [str(tmp_path / "G")]
    )
    assert found is None


# ── listing one folder ────────────────────────────────────────────────────────


def test_the_root_lists_its_visible_folders_sorted_without_files(
    tmp_path: Path,
) -> None:
    root = _my_drive(tmp_path)
    listing = cd.list_drive_folders(root, "")
    assert listing.folders == ["alfa", "datasets", "Zeta"]
    assert listing.truncated is False
    assert listing.parts == ()
    assert listing.path == root
    assert listing.remote == PurePosixPath("/content/drive/MyDrive")


def test_a_nested_folder_gives_its_path_here_and_on_the_vm(tmp_path: Path) -> None:
    root = _my_drive(tmp_path)
    listing = cd.list_drive_folders(root, "datasets/flores")
    assert listing.folders == ["train"]
    assert listing.parts == ("datasets", "flores")
    assert listing.path == root / "datasets" / "flores"
    assert listing.remote == PurePosixPath("/content/drive/MyDrive/datasets/flores")
    # The browser may send Windows separators too.
    assert cd.list_drive_folders(root, "datasets\\flores").parts == (
        "datasets",
        "flores",
    )


def test_a_long_folder_is_cut_and_says_so(tmp_path: Path) -> None:
    root = tmp_path / "Meu Drive"
    for name in ("c", "a", "d", "b"):
        (root / name).mkdir(parents=True)
    listing = cd.list_drive_folders(root, "", limit=2)
    assert listing.folders == ["a", "b"]
    assert listing.truncated is True
    assert cd.list_drive_folders(root, "", limit=4).truncated is False


@pytest.mark.skipif(sys.platform != "win32", reason="hidden attribute is Windows only")
def test_folders_windows_marks_hidden_are_left_out(tmp_path: Path) -> None:
    root = _my_drive(tmp_path)
    _hide(root / "Zeta")
    assert cd.list_drive_folders(root, "").folders == ["alfa", "datasets"]


@pytest.mark.parametrize(
    "rel",
    [
        "..",
        "datasets/../..",
        "../Meu Drive",
        "/etc",
        "\\Windows",
        "C:/Windows",
        "a/./b",
    ],
)
def test_a_path_that_climbs_or_jumps_out_is_refused(tmp_path: Path, rel: str) -> None:
    root = _my_drive(tmp_path)
    with pytest.raises(cd.DriveFolderError) as caught:
        cd.list_drive_folders(root, rel)
    assert caught.value.reason == "outside"


def test_a_link_out_of_the_drive_is_neither_listed_nor_entered(tmp_path: Path) -> None:
    root = _my_drive(tmp_path)
    secret = tmp_path / "secret"
    (secret / "inner").mkdir(parents=True)
    _link_outside(root / "atalho", secret)
    assert "atalho" not in cd.list_drive_folders(root, "").folders
    with pytest.raises(cd.DriveFolderError) as caught:
        cd.list_drive_folders(root, "atalho")
    assert caught.value.reason == "outside"


def test_a_missing_folder_or_a_file_is_missing(tmp_path: Path) -> None:
    root = _my_drive(tmp_path)
    for rel in ("nope", "notas.txt"):
        with pytest.raises(cd.DriveFolderError) as caught:
            cd.list_drive_folders(root, rel)
        assert caught.value.reason == "missing"


# ── the routes ────────────────────────────────────────────────────────────────


def _client(host: str = "127.0.0.1") -> TestClient:
    from visionforge.gui.server import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=(host, 50000))


@pytest.fixture
def drive_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _my_drive(tmp_path)
    seen: list[str | None] = []

    def find(manual: str | None) -> Path | None:
        seen.append(manual)
        return root

    monkeypatch.setattr(colab_run_mod, "find_my_drive_root", find)
    return root


def test_the_route_lists_the_root_and_its_colab_path(drive_root: Path) -> None:
    response = _client().get("/api/colab/drive/folders")
    assert response.status_code == 200
    body = response.json()
    assert body["root"] == str(drive_root)
    assert body["rel"] == ""
    assert body["path"] == str(drive_root)
    assert body["remote"] == "/content/drive/MyDrive"
    assert body["folders"] == ["alfa", "datasets", "Zeta"]
    assert body["truncated"] is False


def test_the_route_enters_a_folder(drive_root: Path) -> None:
    response = _client().get(
        "/api/colab/drive/folders", params={"rel": "datasets/flores"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rel"] == "datasets/flores"
    assert body["path"] == str(drive_root / "datasets" / "flores")
    assert body["remote"] == "/content/drive/MyDrive/datasets/flores"
    assert body["folders"] == ["train"]


def test_the_route_passes_the_typed_root_to_the_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str | None] = []
    monkeypatch.setattr(
        colab_run_mod, "find_my_drive_root", lambda manual: asked.append(manual)
    )
    response = _client().get(
        "/api/colab/drive/folders", params={"my_drive_root": "X:\\Mi unidad"}
    )
    assert asked == ["X:\\Mi unidad"]
    assert response.status_code == 200
    # No "My Drive" here: the page explains Drive for desktop instead of browsing.
    assert response.json() == {
        "root": None,
        "rel": "",
        "path": None,
        "remote": None,
        "folders": [],
        "truncated": False,
        "can_open": False,
    }


def test_the_route_refuses_a_path_out_of_the_drive(drive_root: Path) -> None:
    response = _client().get("/api/colab/drive/folders", params={"rel": "../.."})
    assert response.status_code == 400
    assert "folders" not in response.json()


def test_the_route_says_a_missing_folder_is_missing(drive_root: Path) -> None:
    response = _client().get("/api/colab/drive/folders", params={"rel": "nope"})
    assert response.status_code == 404


@pytest.mark.parametrize("host", ["192.168.1.20", "testclient"])
def test_the_route_refuses_another_machine(drive_root: Path, host: str) -> None:
    response = _client(host).get("/api/colab/drive/folders")
    assert response.status_code == 403


def test_the_route_refuses_a_page_of_another_site(drive_root: Path) -> None:
    response = _client().get(
        "/api/colab/drive/folders", headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 403
    allowed = _client().get(
        "/api/colab/drive/folders", headers={"Origin": "http://127.0.0.1:8000"}
    )
    assert allowed.status_code == 200


class _Opened:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> None:
        self.paths.append(path)


def test_open_shows_the_drive_folder_in_the_file_manager(
    drive_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = _Opened()
    monkeypatch.setattr(colab_run_mod, "open_folder", opened)
    response = _client().post("/api/colab/drive/open", json={"rel": "datasets"})
    assert response.status_code == 200
    assert opened.paths == [drive_root / "datasets"]
    assert response.json() == {"path": str(drive_root / "datasets")}


def test_open_refuses_out_of_the_drive_and_other_machines(
    drive_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = _Opened()
    monkeypatch.setattr(colab_run_mod, "open_folder", opened)
    assert (
        _client().post("/api/colab/drive/open", json={"rel": ".."}).status_code == 400
    )
    assert (
        _client("10.0.0.5").post("/api/colab/drive/open", json={"rel": ""}).status_code
        == 403
    )
    foreign = _client().post(
        "/api/colab/drive/open",
        json={"rel": ""},
        headers={"Origin": "http://127.0.0.1.evil.example"},
    )
    assert foreign.status_code == 403
    assert opened.paths == []


def test_open_without_a_drive_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _Opened()
    monkeypatch.setattr(colab_run_mod, "open_folder", opened)
    monkeypatch.setattr(colab_run_mod, "find_my_drive_root", lambda manual: None)
    assert _client().post("/api/colab/drive/open", json={}).status_code == 404
    assert opened.paths == []


# ── the picked folder reaches the Colab run ───────────────────────────────────


def _dataset(root: Path) -> Path:
    folder = root / "datasets" / "flores"
    (folder / "train" / "a").mkdir(parents=True, exist_ok=True)
    for i in range(2):
        (folder / "train" / "a" / f"{i}.png").write_bytes(b"x")
    return folder


def test_a_picked_folder_wins_over_the_form_folder(tmp_path: Path) -> None:
    root = _my_drive(tmp_path)
    folder = _dataset(root)
    config: dict[str, Any] = {"data": {"base_dir": str(tmp_path / "elsewhere")}}
    source = colab_run_mod._drive_source(config, str(root), str(folder))
    assert source == {
        "kind": "drive",
        "path": "/content/drive/MyDrive/datasets/flores",
        "files": 2,
    }


def test_without_a_picked_folder_the_form_folder_is_used(tmp_path: Path) -> None:
    root = _my_drive(tmp_path)
    folder = _dataset(root)
    source = colab_run_mod._drive_source(
        {"data": {"base_dir": str(folder)}}, str(root), None
    )
    assert source["path"] == "/content/drive/MyDrive/datasets/flores"
