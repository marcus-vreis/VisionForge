from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.gui.api.routes import _render_augment_preview
from visionforge.gui.api.schemas import AugmentPreviewRequest


def _make_dataset(root: Path, n_classes: int = 2) -> Path:
    for ci in range(n_classes):
        cdir = root / "train" / f"class_{ci}"
        cdir.mkdir(parents=True)
        arr = np.random.default_rng(ci).integers(0, 255, (48, 48, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(cdir / "img.png")
    return root


class TestRenderAugmentPreview:
    def test_generates_requested_variants(self, tmp_path: Path) -> None:
        base = _make_dataset(tmp_path)
        resp = _render_augment_preview(
            AugmentPreviewRequest(
                base_dir=str(base),
                transforms={"image_size": 32, "horizontal_flip": True},
                num_variants=4,
            )
        )
        assert Path(resp.original).is_file()
        assert len(resp.variants) == 4
        for v in resp.variants:
            assert Path(v).is_file()
        assert "flip" in resp.active

    def test_no_augmentations_still_returns_variants(self, tmp_path: Path) -> None:
        base = _make_dataset(tmp_path)
        resp = _render_augment_preview(
            AugmentPreviewRequest(
                base_dir=str(base),
                transforms={
                    "image_size": 32,
                    "horizontal_flip": False,
                    "rotation_degrees": 0,
                    "color_jitter": False,
                },
                num_variants=3,
            )
        )
        assert len(resp.variants) == 3
        assert resp.active == []

    def test_active_lists_each_augmentation(self, tmp_path: Path) -> None:
        base = _make_dataset(tmp_path)
        resp = _render_augment_preview(
            AugmentPreviewRequest(
                base_dir=str(base),
                transforms={
                    "image_size": 32,
                    "horizontal_flip": True,
                    "rotation_degrees": 15,
                    "color_jitter": True,
                },
            )
        )
        assert set(resp.active) == {"flip", "rotation", "jitter"}

    def test_missing_split_returns_message(self, tmp_path: Path) -> None:
        base = _make_dataset(tmp_path)
        resp = _render_augment_preview(
            AugmentPreviewRequest(base_dir=str(base), split="val")
        )
        assert resp.variants == []
        assert resp.message is not None

    def test_picks_named_class(self, tmp_path: Path) -> None:
        base = _make_dataset(tmp_path, n_classes=2)
        resp = _render_augment_preview(
            AugmentPreviewRequest(
                base_dir=str(base),
                class_name="class_1",
                transforms={"image_size": 32},
            )
        )
        assert "class_1" in resp.source_image
