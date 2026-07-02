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

    def test_missing_split_falls_back_to_any_dataset_image(
        self, tmp_path: Path
    ) -> None:
        # Non-ImageFolder layouts (CSV manifest, MVTec, paired masks) have no
        # val/ class dirs — the preview degrades to any image under base_dir
        # instead of failing.
        base = _make_dataset(tmp_path)
        resp = _render_augment_preview(
            AugmentPreviewRequest(base_dir=str(base), split="val")
        )
        assert len(resp.variants) > 0
        assert Path(resp.original).is_file()

    def test_flat_split_dir_without_class_subdirs(self, tmp_path: Path) -> None:
        # Regression-style images/ folder: loose files, no class subdirs.
        img_dir = tmp_path / "train"
        img_dir.mkdir(parents=True)
        arr = np.random.default_rng(0).integers(0, 255, (48, 48, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(img_dir / "sample.png")
        resp = _render_augment_preview(
            AugmentPreviewRequest(base_dir=str(tmp_path), transforms={"image_size": 32})
        )
        assert len(resp.variants) > 0
        assert "sample" in resp.source_image

    def test_csv_manifest_layout_recursive_fallback(self, tmp_path: Path) -> None:
        # No train/ split at all — images live under images/<sub>/, as in the
        # regression CSV-manifest layout.
        nested = tmp_path / "images" / "batch_a"
        nested.mkdir(parents=True)
        arr = np.random.default_rng(1).integers(0, 255, (48, 48, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(nested / "deep.png")
        resp = _render_augment_preview(
            AugmentPreviewRequest(base_dir=str(tmp_path), transforms={"image_size": 32})
        )
        assert len(resp.variants) > 0
        assert "deep" in resp.source_image

    def test_dataset_without_any_image_returns_message(self, tmp_path: Path) -> None:
        (tmp_path / "train").mkdir()
        resp = _render_augment_preview(AugmentPreviewRequest(base_dir=str(tmp_path)))
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
