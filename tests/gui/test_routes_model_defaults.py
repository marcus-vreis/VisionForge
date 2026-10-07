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

    @pytest.mark.parametrize("arch", ["vgg16", "alexnet"])
    def test_a_measured_collapse_names_the_model_and_the_number(
        self, client: TestClient, arch: str
    ) -> None:
        body = client.post("/api/model/defaults", json={"architecture": arch}).json()

        # No recovery was measured on this setup: VGG16 at 1e-4 was only run on
        # a two-class problem (ADR-099), and AlexNet at 1e-4 never was.
        assert body["collapse_evidence"] == {
            "measured_on": arch,
            "accuracy": 0.25,
            "outcome": "collapse",
            "recovered_accuracy": None,
        }
        assert body["note"] == (
            f"{arch} previu uma classe só com Adam a 1e-3 "
            f"(acurácia 0.25 em 4 classes). Sugerimos adam a 0.0001."
        )
        # A suggestion is not a promise: no claim that the new setting works.
        assert "treina normal" not in body["note"]

    def test_a_sibling_of_the_measured_model_says_it_was_not_the_one_measured(
        self, client: TestClient
    ) -> None:
        # Only vgg16 was run. vgg19 gets the same suggestion, but the note must
        # say whose number this is instead of implying vgg19 was measured.
        body = client.post("/api/model/defaults", json={"architecture": "vgg19"}).json()

        assert body["collapse_prone"] is True
        assert body["collapse_evidence"]["measured_on"] == "vgg16"
        assert body["note"] == (
            "vgg19: o vgg16, da mesma família, previu uma classe só com Adam a "
            "1e-3 (acurácia 0.25 em 4 classes); este modelo não foi medido. "
            "Sugerimos adam a 0.0001."
        )

    def test_vit_failed_to_learn_it_did_not_collapse(self, client: TestClient) -> None:
        body = client.post(
            "/api/model/defaults", json={"architecture": "vit_b_16"}
        ).json()

        assert body["collapse_evidence"] == {
            "measured_on": "vit_b_16",
            "accuracy": 0.41,
            "outcome": "fails_to_learn",
            "recovered_accuracy": 0.85,
        }
        # AdamW at 1e-4 was run on the same data (ADR-100), so the recovery is
        # stated as a number, not as a promise.
        assert body["note"] == (
            "vit_b_16 aprendeu pouco com Adam a 1e-3 (acurácia 0.41 em 4 "
            "classes). Com adamw a 0.0001, a acurácia foi 0.85 nas mesmas "
            "condições."
        )
        # One-class guessing on 4 classes gives 0.25, so 0.41 is "little", not
        # "nothing".
        assert "não aprendeu" not in body["note"]
        # 0.41 is above the 0.25 of a one-class prediction: not what happened.
        assert "uma classe só" not in body["note"]
        assert "0.25" not in body["note"]

    def test_a_sibling_never_gets_the_recovery_claim(self, client: TestClient) -> None:
        # vit_b_16 recovered to 0.85; vit_l_16 was never run, so its note says
        # only what is suggested.
        body = client.post(
            "/api/model/defaults", json={"architecture": "vit_l_16"}
        ).json()

        assert body["note"] == (
            "vit_l_16: o vit_b_16, da mesma família, aprendeu pouco com Adam a "
            "1e-3 (acurácia 0.41 em 4 classes); este modelo não foi medido. "
            "Sugerimos adamw a 0.0001."
        )
        assert "0.85" not in body["note"]

    def test_swin_collapsed(self, client: TestClient) -> None:
        body = client.post(
            "/api/model/defaults", json={"architecture": "swin_t"}
        ).json()

        assert body["collapse_evidence"]["outcome"] == "collapse"
        assert body["collapse_evidence"]["recovered_accuracy"] == 0.88
        assert body["note"] == (
            "swin_t previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 "
            "classes). Com adamw a 0.0001, a acurácia foi 0.88 nas mesmas "
            "condições."
        )

    def test_convnext_tiny_was_measured_in_adr_100(self, client: TestClient) -> None:
        # ADR-100's table has convnext_tiny at 0.25 with Adam 1e-3 (collapse).
        body = client.post(
            "/api/model/defaults", json={"architecture": "convnext_tiny"}
        ).json()

        assert body["collapse_evidence"] == {
            "measured_on": "convnext_tiny",
            "accuracy": 0.25,
            "outcome": "collapse",
            "recovered_accuracy": 0.91,
        }
        assert body["note"] == (
            "convnext_tiny previu uma classe só com Adam a 1e-3 (acurácia 0.25 "
            "em 4 classes). Com adamw a 0.0001, a acurácia foi 0.91 nas mesmas "
            "condições."
        )

    def test_a_family_never_measured_quotes_no_number(self, client: TestClient) -> None:
        # maxvit gets the attention families' remedy but was never run.
        body = client.post(
            "/api/model/defaults", json={"architecture": "maxvit_t"}
        ).json()

        assert body["collapse_prone"] is True
        assert body["collapse_evidence"] is None
        # The only numbers are the settings (1e-3, the suggested 0.0001), never
        # an accuracy or a class count.
        assert body["note"] == (
            "maxvit_t: Adam a 1e-3 não foi medido para esta família; sugerimos "
            "adamw a 0.0001, o mesmo das famílias de atenção medidas."
        )

    def test_an_architecture_that_is_not_collapse_prone_has_no_evidence(
        self, client: TestClient
    ) -> None:
        body = client.post(
            "/api/model/defaults", json={"architecture": "resnet50"}
        ).json()

        assert body["collapse_prone"] is False
        assert body["collapse_evidence"] is None
        assert body["note"] is None

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
