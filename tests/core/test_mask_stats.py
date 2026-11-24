"""Mask statistics of a segmentation dataset (ADR-234), on hand-computed masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.core.cancellation import CancellationToken
from visionforge.core.mask_stats import analyze_masks, list_pairs, pairs_signature

SPLITS = {"train": "train", "val": "val"}


def _pair(
    root: Path,
    split: str,
    name: str,
    mask: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> None:
    """An RGB image and its mask; ``image_size`` is (width, height)."""
    h, w = mask.shape
    images, masks = root / split / "images", root / split / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size or (w, h)).save(images / f"{name}.jpg")
    Image.fromarray(mask.astype(np.uint8), mode="L").save(masks / f"{name}.png")


def _analyze(root: Path, **kw: object) -> dict:
    return analyze_masks(root, SPLITS, num_classes=3, ignore_index=255, **kw)  # type: ignore[arg-type]


def test_pixel_shares_per_class_and_split(tmp_path: Path) -> None:
    # train/a: 70 px of class 0, 20 of class 1, 10 void. val/b: 50 of 0, 50 of 2.
    a = np.zeros((10, 10), np.uint8)
    a[0:2, :] = 1
    a[9, :] = 255
    b = np.zeros((10, 10), np.uint8)
    b[5:, :] = 2
    _pair(tmp_path, "train", "a", a)
    _pair(tmp_path, "val", "b", b)
    report = _analyze(tmp_path)
    by_id = {c["id"]: c for c in report["classes"]}
    assert {i: c["pixels"] for i, c in by_id.items()} == {0: 120, 1: 20, 2: 50, 255: 10}
    assert {i: c["share"] for i, c in by_id.items()} == {
        0: 60.0,
        1: 10.0,
        2: 25.0,
        255: 5.0,
    }
    assert by_id[0]["shares"] == {"train": 70.0, "val": 50.0}
    assert by_id[255]["ignore"] is True
    assert report["absent_classes"] == []
    # Coverage (not class 0, not void): a 20%, b 50%.
    counts = {(b_["lo"], b_["hi"]): b_["count"] for b_ in report["coverage_bins"]}
    assert counts[(0.1, 0.25)] == 1
    assert counts[(0.5, 0.75)] == 1
    assert report["empty_masks"] == 0


def test_an_id_outside_the_classes_is_reported(tmp_path: Path) -> None:
    mask = np.zeros((10, 10), np.uint8)
    mask[0, 0:3] = 7
    _pair(tmp_path, "train", "a", mask)
    report = _analyze(tmp_path)
    assert report["out_of_range_pixels"] == 3
    assert report["out_of_range_ids"] == [7]
    assert report["examples"]["out_of_range"] == [
        {"split": "train", "path": "train/masks/a.png", "ids": [7]}
    ]
    # Classes 1 and 2 never appear.
    assert report["absent_classes"] == [1, 2]


def test_tiny_regions_are_counted_per_class(tmp_path: Path) -> None:
    # 200 x 200 = 40 000 px: a region below 0.01% is below 4 px.
    mask = np.zeros((200, 200), np.uint8)
    mask[5, 5] = 1  # 1 px: tiny
    mask[50:52, 50:52] = 1  # 4 px: not below the limit
    mask[100, 100] = 2  # 1 px: tiny
    _pair(tmp_path, "train", "a", mask)
    report = _analyze(tmp_path)
    assert report["tiny_regions"] == [{"id": 1, "regions": 1}, {"id": 2, "regions": 1}]
    assert report["tiny_images"] == 1


def test_pairing_size_and_color_problems(tmp_path: Path) -> None:
    ok = np.zeros((10, 10), np.uint8)
    _pair(tmp_path, "train", "same", ok)
    _pair(tmp_path, "train", "wide", ok, image_size=(20, 10))
    (tmp_path / "train/images/lonely.jpg").write_bytes(
        (tmp_path / "train/images/same.jpg").read_bytes()
    )
    Image.new("L", (4, 4)).save(tmp_path / "train/masks/orphan.png")
    Image.new("RGB", (10, 10)).save(tmp_path / "train/images/rgb.jpg")
    Image.new("RGB", (10, 10), (255, 0, 0)).save(tmp_path / "train/masks/rgb.png")
    report = _analyze(tmp_path)
    train = report["splits"]["train"]
    assert (train["no_mask"], train["no_image"]) == (["lonely"], ["orphan"])
    assert report["size_mismatch"] == 1
    assert report["examples"]["size_mismatch"][0]["image"] == [20, 10]
    assert report["color_masks"] == 1
    assert report["empty_masks"] == 2  # same and wide are all background
    assert report["splits"]["val"]["present"] is False


def test_sampling_and_cancelling(tmp_path: Path) -> None:
    for i in range(4):
        _pair(tmp_path, "train", f"m{i}", np.zeros((4, 4), np.uint8))
    sampled = _analyze(tmp_path, max_images=2)
    assert (sampled["total_pairs"], sampled["scanned"], sampled["sampled"]) == (
        4,
        2,
        True,
    )
    token = CancellationToken()
    token.cancel()
    stopped = _analyze(tmp_path, token=token)
    assert (stopped["scanned"], stopped["cancelled"]) == (0, True)


def test_the_signature_changes_when_a_mask_changes(tmp_path: Path) -> None:
    _pair(tmp_path, "train", "a", np.zeros((4, 4), np.uint8))
    before = pairs_signature(list_pairs(tmp_path, SPLITS, "images", "masks")[0])
    _pair(tmp_path, "train", "a", np.ones((5, 5), np.uint8))
    after = pairs_signature(list_pairs(tmp_path, SPLITS, "images", "masks")[0])
    assert before != after


def test_the_images_brightness_contrast_and_sharpness(tmp_path: Path) -> None:
    """ADR-240: the images of the pairs, not the masks, are measured."""
    from tests.quality_images import flagged, save_planted

    names = save_planted(tmp_path / "train" / "images")
    (tmp_path / "train" / "masks").mkdir(parents=True)
    for name in names:
        Image.new("L", (64, 64)).save(tmp_path / "train" / "masks" / name)
    report = _analyze(tmp_path)
    quality = report["quality"]
    assert quality["count"] == 27
    assert "dark" in flagged(quality, "dark")
    assert "flat" in flagged(quality, "flat")
    assert "blurry" in flagged(quality, "blurry")
    assert not any(n.startswith("n") for n in flagged(quality, "blurry"))
    assert quality["outliers"]["dark"][0]["path"] == "train/images/dark.png"
