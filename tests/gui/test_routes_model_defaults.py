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
        body = client.post(
            "/api/model/defaults", json={"architecture": "resnet18"}
        ).json()

        assert body["optimizer"] == "adam"
        assert body["learning_rate"] == pytest.approx(1e-3)
        assert body["image_size"] is None
        assert body["dataset_median_side"] is None
        assert body["collapse_prone"] is False
        assert body["note"] is None

    def test_an_attention_model_gets_adamw_at_1e_4(self, client: TestClient) -> None:
        body = client.post(
            "/api/model/defaults", json={"architecture": "convnext_tiny"}
        ).json()

        assert body["optimizer"] == "adamw"
        assert body["learning_rate"] == pytest.approx(1e-4)

    def test_the_measured_collapse_is_explained(self, client: TestClient) -> None:
        body = client.post("/api/model/defaults", json={"architecture": "vgg16"}).json()

        assert body["collapse_prone"] is True
        assert "Adam a 1e-3" in body["note"]

    def test_the_size_follows_the_images(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        # 64px images: min(64, 224) rounded to the 32px stride is 64, no note.
        root = _images(tmp_path / "ds", side=64)

        body = client.post(
            "/api/model/defaults",
            json={"architecture": "resnet18", "base_dir": str(root)},
        ).json()

        assert body["dataset_median_side"] == 64
        assert body["image_size"] == 64
        assert body["note"] is None

    def test_images_below_the_floor_are_upscaled_and_the_note_says_so(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        # 32px images: the 64px floor wins, so training enlarges them.
        root = _images(tmp_path / "ds", side=32)

        body = client.post(
            "/api/model/defaults",
            json={"architecture": "resnet18", "base_dir": str(root)},
        ).json()

        assert body["image_size"] == 64
        assert "32px" in body["note"]

    def test_a_fixed_input_model_keeps_224_and_the_collapse_note_wins(
        self, tmp_path: Path, client: TestClient
    ) -> None:
        # ViT is both fixed-input and collapse-prone at Adam 1e-3; the route
        # reports the collapse first (if/elif), because it is the failure that
        # was measured.
        root = _images(tmp_path / "ds", side=64)

        body = client.post(
            "/api/model/defaults",
            json={"architecture": "vit_b_16", "base_dir": str(root)},
        ).json()

        assert body["image_size"] == 224
        assert body["dataset_median_side"] == 64
        assert body["collapse_prone"] is True
        assert "Adam a 1e-3" in body["note"]
