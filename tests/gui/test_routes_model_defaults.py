"""POST /api/model/defaults — what the form fills in when a model is chosen."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client() -> TestClient:
    from visionforge.gui.server import app

    return TestClient(app)


def _images(root: Path, side: int, n: int = 6) -> Path:
    d = root / "train" / "a"
    d.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (side, side)).save(d / f"{i}.png")
    return root


class TestModelDefaults:
    def test_a_cnn_gets_adam_at_1e_3_and_no_dataset_fields(
        self, client: TestClient
    ) -> None:
        resp = client.post("/api/model/defaults", json={"architecture": "resnet18"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["optimizer"] == "adam"
        assert body["learning_rate"] == pytest.approx(1e-3)
        assert body["image_size"] is None
        assert body["dataset_median_side"] is None
        assert body["collapse_prone"] is False
        assert body["note"] is None

    def test_an_attention_model_gets_adamw_at_1e_4(self, client: TestClient) -> None:
        resp = client.post(
            "/api/model/defaults", json={"architecture": "convnext_tiny"}
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["optimizer"] == "adamw"
        assert body["learning_rate"] == pytest.approx(1e-4)

    def test_the_measured_collapse_is_explained(self, client: TestClient) -> None:
        resp = client.post("/api/model/defaults", json={"architecture": "vgg16"})

        assert resp.status_code == 200
        body = resp.json()
        # The pair measured to collapse is Adam at 1e-3; the suggestion is the
        # same optimizer at the unnormalised-network rate of 1e-4.
        assert body["optimizer"] == "adam"
        assert body["learning_rate"] == pytest.approx(1e-4)
        assert body["collapse_prone"] is True
        assert "Adam a 1e-3" in body["note"]

    def test_the_size_follows_the_images(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        # 64px images: min(64, 224) rounded to the 32px stride is 64, no note.
        root = _images(tmp_path / "ds", side=64)

        resp = client.post(
            "/api/model/defaults",
            json={"architecture": "resnet18", "base_dir": str(root)},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["dataset_median_side"] == 64
        assert body["image_size"] == 64
        assert body["note"] is None

    def test_images_below_the_floor_are_upscaled_and_the_note_says_so(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        # 32px images: the 64px floor wins, so training enlarges them.
        root = _images(tmp_path / "ds", side=32)

        resp = client.post(
            "/api/model/defaults",
            json={"architecture": "resnet18", "base_dir": str(root)},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["image_size"] == 64
        assert "32px" in body["note"]

    def test_a_fixed_input_model_keeps_224_and_the_collapse_note_wins(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        # ViT is both fixed-input and collapse-prone at Adam 1e-3; the route
        # reports the collapse first (if/elif), because it is the failure that
        # was measured.
        root = _images(tmp_path / "ds", side=64)

        resp = client.post(
            "/api/model/defaults",
            json={"architecture": "vit_b_16", "base_dir": str(root)},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["image_size"] == 224
        assert body["dataset_median_side"] == 64
        assert body["collapse_prone"] is True
        assert "Adam a 1e-3" in body["note"]

    def test_without_pretrained_weights_the_size_is_not_capped_at_224(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        # 300px images. Pretrained weights expect ~224, so that caps the
        # suggestion; from scratch nothing does, and the images' own size
        # (300, rounded to the 32px stride: 288) is used.
        root = _images(tmp_path / "ds", side=300)
        request = {"architecture": "resnet18", "base_dir": str(root)}

        pretrained = client.post("/api/model/defaults", json=request)
        scratch = client.post(
            "/api/model/defaults", json={**request, "pretrained": False}
        )

        assert pretrained.status_code == 200
        assert scratch.status_code == 200
        assert pretrained.json()["image_size"] == 224
        assert scratch.json()["image_size"] == 288
        assert scratch.json()["dataset_median_side"] == 300
