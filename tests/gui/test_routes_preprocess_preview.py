"""POST /api/dataset/preview_preprocess — the filter strip in the data panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    from visionforge.gui.api import routes as routes_mod
    from visionforge.gui.server import app

    cache = tmp_path / "preview_cache"
    monkeypatch.setattr(routes_mod, "_PREVIEW_CACHE_DIR", cache)
    return TestClient(app), cache


def _imagefolder(root: Path) -> Path:
    for cls, color in (("bad", (200, 30, 30)), ("good", (30, 200, 30))):
        d = root / "train" / cls
        d.mkdir(parents=True)
        Image.new("RGB", (48, 48), color=color).save(d / "a.png")
    return root


def _post(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post("/api/dataset/preview_preprocess", json=body)


class TestPreprocessPreview:
    def test_renders_the_original_each_step_and_the_final(
        self, tmp_path: Path, client_and_cache: tuple[TestClient, Path]
    ) -> None:
        client, cache = client_and_cache
        root = _imagefolder(tmp_path / "ds")

        resp = _post(
            client,
            {
                "base_dir": str(root),
                "steps": [{"kind": "grayscale"}, {"kind": "equalize"}],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert [s["kind"] for s in body["steps"]] == ["grayscale", "equalize"]
        for path in [
            body["original"],
            body["final"],
            *(s["artifact"] for s in body["steps"]),
        ]:
            assert Path(path).is_file()
            assert Path(path).parent.parent == cache
        # No class asked for: the first one alphabetically is used.
        assert Path(body["source_image"]).parent.name == "bad"
        assert "grayscale" in body["available_kinds"]

    def test_the_requested_class_is_used(
        self, tmp_path: Path, client_and_cache: tuple[TestClient, Path]
    ) -> None:
        client, _ = client_and_cache
        root = _imagefolder(tmp_path / "ds")

        body = _post(
            client, {"base_dir": str(root), "class_name": "good", "steps": []}
        ).json()

        assert Path(body["source_image"]).parent.name == "good"
        # Without steps the final image is the original itself.
        assert body["final"] == body["original"]

    def test_an_unknown_filter_is_bad_input_not_a_server_error(
        self, tmp_path: Path, client_and_cache: tuple[TestClient, Path]
    ) -> None:
        client, _ = client_and_cache
        root = _imagefolder(tmp_path / "ds")

        resp = _post(client, {"base_dir": str(root), "steps": [{"kind": "sepia"}]})

        assert resp.status_code == 422
        # The message lists what does exist (ADR-078).
        assert "grayscale" in resp.json()["detail"]

    def test_a_missing_split_is_a_message_not_an_error(
        self, tmp_path: Path, client_and_cache: tuple[TestClient, Path]
    ) -> None:
        client, _ = client_and_cache
        root = _imagefolder(tmp_path / "ds")

        body = _post(
            client, {"base_dir": str(root), "split": "val", "steps": []}
        ).json()

        assert body["original"] == ""
        assert "val" in body["message"]

    def test_a_split_without_classes_says_so(
        self, tmp_path: Path, client_and_cache: tuple[TestClient, Path]
    ) -> None:
        client, _ = client_and_cache
        (tmp_path / "ds" / "train").mkdir(parents=True)

        body = _post(client, {"base_dir": str(tmp_path / "ds"), "steps": []}).json()

        assert body["original"] == ""
        assert "classe" in body["message"].lower()

    def test_a_class_without_images_says_so(
        self, tmp_path: Path, client_and_cache: tuple[TestClient, Path]
    ) -> None:
        client, _ = client_and_cache
        (tmp_path / "ds" / "train" / "empty").mkdir(parents=True)

        body = _post(client, {"base_dir": str(tmp_path / "ds"), "steps": []}).json()

        assert body["original"] == ""
        assert "Sem imagens" in body["message"]

    def test_repeated_previews_do_not_accumulate_files(
        self, tmp_path: Path, client_and_cache: tuple[TestClient, Path]
    ) -> None:
        client, cache = client_and_cache
        root = _imagefolder(tmp_path / "ds")
        body = {"base_dir": str(root), "steps": [{"kind": "grayscale"}]}

        _post(client, body)
        _post(client, body)

        # original + one step + final, from the last call only.
        assert len(list((cache / "bad").glob("*.png"))) == 3
