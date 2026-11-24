"""Anomaly dataset statistics (ADR-236), on small built datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.core.anomaly_stats import (
    analyze_anomaly,
    images_signature,
    list_images,
    matches,
)
from visionforge.core.cancellation import CancellationToken


def _noise(seed: int, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8), "RGB")


def _dataset(root: Path) -> Path:
    """train/good: n0..n2 (64 px). test/good: a copy of n0 in grayscale, and n9.
    test/scratch: n1 enlarged to 80 px, and n7 at 80 px. train/bad: an extra folder.
    ground_truth/scratch: a mask for one of the two defects."""
    for name, seed in (("n0", 0), ("n1", 1), ("n2", 2)):
        (root / "train/good").mkdir(parents=True, exist_ok=True)
        _noise(seed).save(root / f"train/good/{name}.png")
    (root / "train/bad").mkdir(parents=True)
    (root / "test/good").mkdir(parents=True)
    _noise(0).convert("L").save(root / "test/good/copy.png")
    _noise(9).convert("L").save(root / "test/good/other.png")
    (root / "test/scratch").mkdir(parents=True)
    _noise(1).resize((80, 80)).save(root / "test/scratch/s1.png")
    _noise(7, 80).save(root / "test/scratch/s2.png")
    (root / "ground_truth/scratch").mkdir(parents=True)
    Image.new("L", (80, 80)).save(root / "ground_truth/scratch/s1_mask.png")
    return root


def test_counts_layout_and_masks(tmp_path: Path) -> None:
    report = analyze_anomaly(_dataset(tmp_path))
    assert report["counts"] == {"defect:scratch": 2, "test_normal": 2, "train": 3}
    assert report["defect_kinds"] == ["scratch"]
    assert report["extra_train_folders"] == ["bad"]
    assert report["missing_masks"] == {"count": 1, "paths": ["test/scratch/s2.png"]}
    assert report["scanned"] == 7


def test_sizes_and_modes_that_differ_from_training(tmp_path: Path) -> None:
    report = analyze_anomaly(_dataset(tmp_path))
    assert report["size_mismatch"] == ["defect:scratch"]
    assert report["mode_mismatch"] == ["test_normal"]
    train = next(g for g in report["groups"] if g["group"] == "train")
    assert train["sizes"] == [{"size": "64×64", "count": 3}]


def test_a_defect_in_training_and_a_test_image_seen_in_training(tmp_path: Path) -> None:
    report = analyze_anomaly(_dataset(tmp_path))
    # n1 is in train/good and, enlarged, in test/scratch.
    assert [m["path"] for m in report["train_like_defect"]] == ["train/good/n1.png"]
    assert report["train_like_defect"][0]["like"] == "test/scratch/s1.png"
    # test/good/copy.png is n0 in grayscale.
    assert [(m["path"], m["like"]) for m in report["test_like_train"]] == [
        ("test/good/copy.png", "train/good/n0.png")
    ]


def test_no_ground_truth_folder_means_no_mask_check(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    for p in (root / "ground_truth/scratch").iterdir():
        p.unlink()
    (root / "ground_truth/scratch").rmdir()
    (root / "ground_truth").rmdir()
    assert analyze_anomaly(root)["missing_masks"] is None


def test_unreadable_sampled_and_cancelled(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    (root / "train/good/broken.png").write_bytes(b"not an image")
    report = analyze_anomaly(root)
    assert report["unreadable_count"] == 1
    assert report["unreadable"] == [{"group": "train", "path": "train/good/broken.png"}]
    sampled = analyze_anomaly(root, max_images=4)
    assert sampled["sampled"] is True
    token = CancellationToken()
    token.cancel()
    assert analyze_anomaly(root, token=token)["cancelled"] is True


def test_flat_images_never_match() -> None:
    assert matches([("a", 0)], [("b", 0)], 6) == []
    assert matches([("a", 0b1011)], [("b", 0b1010), ("c", 0b0)], 6) == [
        {"path": "a", "like": "b", "distance": 1}
    ]


def test_the_signature_changes_when_an_image_changes(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    before = images_signature(list_images(root, "train", "test", "good")[0])
    _noise(5).save(root / "train/good/n2.png")
    assert images_signature(list_images(root, "train", "test", "good")[0]) != before


def test_the_images_brightness_contrast_and_sharpness(tmp_path: Path) -> None:
    """ADR-240: measured per group, the planted images stand out in training."""
    from tests.quality_images import flagged, save_planted

    save_planted(tmp_path / "train" / "good")
    (tmp_path / "test" / "good").mkdir(parents=True)
    report = analyze_anomaly(tmp_path)
    quality = report["quality"]
    assert quality["count"] == 27
    assert set(quality["metrics"]["brightness"]["median_by_split"]) == {"train"}
    assert {"dark"} <= flagged(quality, "dark")
    assert {"flat"} <= flagged(quality, "flat")
    assert {"blurry"} <= flagged(quality, "blurry")
    assert quality["outliers"]["flat"][0]["path"] == "train/good/flat.png"
