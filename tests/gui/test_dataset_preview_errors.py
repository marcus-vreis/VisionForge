"""Bad input to the dataset previews and the regression stats is a 4xx, never a 500.

A transform setting out of range, a corrupt first image and a manifest that is
not UTF-8 text are the person's data, not a fault of the server: each answer
says what is wrong instead of an internal error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from visionforge.gui.api import dataset_tools_routes, datasets_routes
    from visionforge.gui.server import app

    cache = tmp_path / "preview_cache"
    monkeypatch.setattr(dataset_tools_routes, "_PREVIEW_CACHE_DIR", cache)
    monkeypatch.setattr(datasets_routes, "_AUGMENT_CACHE_DIR", cache / "augment")
    monkeypatch.setattr(datasets_routes, "_ALLOWED_DATASET_ROOTS", set())
    # A 500 must come back as a response, not as the test's own exception.
    return TestClient(app, raise_server_exceptions=False)


def _imagefolder(root: Path, *, corrupt_first: bool = False) -> Path:
    folder = root / "train" / "good"
    folder.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(30, 200, 30)).save(folder / "b.png")
    if corrupt_first:
        (folder / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n not really an image")
    return root


def test_an_out_of_range_transform_is_named_not_a_server_error(
    tmp_path: Path, client: TestClient
) -> None:
    root = _imagefolder(tmp_path / "ds")
    resp = client.post(
        "/api/dataset/preview_augment",
        json={"base_dir": str(root), "transforms": {"rotation_degrees": -1}},
    )
    assert resp.status_code == 422, resp.text
    assert "rotation_degrees" in resp.json()["detail"]


@pytest.mark.parametrize("route", ["preview_augment", "preview_preprocess"])
def test_a_corrupt_first_image_is_named_not_a_server_error(
    tmp_path: Path, client: TestClient, route: str
) -> None:
    root = _imagefolder(tmp_path / "ds", corrupt_first=True)
    resp = client.post(f"/api/dataset/{route}", json={"base_dir": str(root)})
    assert resp.status_code == 400, resp.text
    assert "a.png" in resp.json()["detail"]


@pytest.mark.parametrize(
    "content",
    [
        "image,target\nimá.png,1.0\n".encode("latin-1"),
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe",
    ],
    ids=["latin-1", "binary"],
)
def test_a_manifest_that_is_not_utf8_text_is_named_not_a_server_error(
    tmp_path: Path, client: TestClient, content: bytes
) -> None:
    base = tmp_path / "reg"
    (base / "images").mkdir(parents=True)
    (base / "train.csv").write_bytes(content)
    resp = client.post("/api/regression/dataset/stats", json={"base_dir": str(base)})
    assert resp.status_code == 400, resp.text
    assert "train.csv" in resp.json()["detail"]
