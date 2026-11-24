"""Visual near-duplicates across splits (issue 351, ADR-228)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionforge.core.dataset_audit import audit_dataset
from visionforge.core.near_duplicates import (
    MAX_DISTANCE,
    NearImage,
    difference_hash,
    hamming,
    near_duplicate_groups,
)


def _gradient(seed: int, size: int = 64) -> np.ndarray:
    """A smooth random field: its structure survives small edits."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return np.asarray(
        Image.fromarray(coarse).resize((size, size), Image.Resampling.BICUBIC)
    )


def _save(root: Path, relative: str, pixels: np.ndarray, fmt: str = "PNG") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format=fmt)
    return path


def test_the_hash_ignores_a_reencoding_and_a_small_brightness_change(
    tmp_path: Path,
) -> None:
    pixels = _gradient(1)
    original = difference_hash(_save(tmp_path, "a.png", pixels))
    brighter = np.clip(pixels.astype(int) + 6, 0, 255).astype(np.uint8)
    edited = difference_hash(_save(tmp_path, "b.jpg", brighter, "JPEG"))
    other = difference_hash(_save(tmp_path, "c.png", _gradient(2)))

    assert hamming(original, edited) <= MAX_DISTANCE
    assert hamming(original, other) > MAX_DISTANCE


def test_hamming_counts_the_differing_bits() -> None:
    assert hamming(0b1011, 0b0010) == 2
    assert hamming(2**64 - 1, 0) == 64


def _near(split: str, path: str, value: int, sha: str) -> NearImage:
    return NearImage(split=split, path=path, dhash=value, sha256=sha)


def test_a_pair_within_the_distance_in_two_splits_is_one_group() -> None:
    base = 0x0123_4567_89AB_CDEF
    groups = near_duplicate_groups(
        [
            _near("train", "train/a.png", base, "s1"),
            _near("test", "test/b.png", base ^ 0b111, "s2"),  # 3 bits apart
            _near("test", "test/far.png", base ^ (2**64 - 1), "s3"),
        ]
    )
    assert groups == [
        {"splits": ["test", "train"], "paths": ["test/b.png", "train/a.png"]}
    ]


def test_every_pair_within_the_distance_is_found_whatever_bits_differ() -> None:
    # 8 bands of 8 bits: 7 or fewer differing bits leave one band equal, so
    # spreading them over the bands must not hide a pair.
    base = 0xF0F0_F0F0_F0F0_F0F0
    spread = sum(1 << (8 * band) for band in range(MAX_DISTANCE))
    groups = near_duplicate_groups(
        [_near("train", "a", base, "s1"), _near("val", "b", base ^ spread, "s2")]
    )
    assert len(groups) == 1


def test_pairs_within_one_split_and_byte_copies_are_left_out() -> None:
    base = 0x0F0F_0F0F_0F0F_0F0F
    assert (
        near_duplicate_groups(
            [_near("train", "a", base, "s1"), _near("train", "b", base ^ 1, "s2")]
        )
        == []
    )
    # Identical bytes across splits are already in duplicate_groups.
    assert (
        near_duplicate_groups(
            [_near("train", "a", base, "same"), _near("val", "b", base, "same")]
        )
        == []
    )


def test_flat_images_have_no_structure_to_compare() -> None:
    assert (
        near_duplicate_groups(
            [_near("train", "red", 0, "s1"), _near("val", "blue", 0, "s2")]
        )
        == []
    )


# ── in the audit ─────────────────────────────────────────────────────────────


def test_the_audit_finds_an_edited_copy_in_another_split(tmp_path: Path) -> None:
    pixels = _gradient(7)
    _save(tmp_path, "train/cat/a.png", pixels)
    _save(tmp_path, "test/cat/a.jpg", pixels, "JPEG")
    _save(tmp_path, "val/cat/z.png", _gradient(8))

    off = audit_dataset(tmp_path)
    on = audit_dataset(tmp_path, near_duplicates=True)

    assert off["visual_similarity_checked"] is False
    assert off["near_duplicate_groups"] == []
    assert off["ok"] is True
    assert on["visual_similarity_checked"] is True
    assert on["near_duplicate_groups"] == [
        {"splits": ["test", "train"], "paths": ["test/cat/a.jpg", "train/cat/a.png"]}
    ]
    assert on["near_duplicate_max_distance"] == MAX_DISTANCE
    assert on["ok"] is False


def test_a_byte_copy_stays_an_exact_duplicate_only(tmp_path: Path) -> None:
    source = _save(tmp_path, "train/a.png", _gradient(3))
    (tmp_path / "val").mkdir()
    shutil.copyfile(source, tmp_path / "val/a.png")
    _save(tmp_path, "test/b.png", _gradient(4))

    report = audit_dataset(tmp_path, near_duplicates=True)

    assert len(report["duplicate_groups"]) == 1
    assert report["near_duplicate_groups"] == []


def test_a_truncated_image_is_unreadable_and_not_hashed(tmp_path: Path) -> None:
    good = _save(tmp_path, "train/a.png", _gradient(5))
    data = good.read_bytes()
    truncated = tmp_path / "train/b.png"
    truncated.write_bytes(data[: len(data) // 2])
    (tmp_path / "val").mkdir()
    (tmp_path / "test").mkdir()

    report = audit_dataset(tmp_path, near_duplicates=True)

    assert "train/b.png" in [i["path"] for i in report["unreadable"]]


@pytest.mark.parametrize("near", [False, True])
def test_the_report_always_names_the_near_duplicate_fields(
    tmp_path: Path, near: bool
) -> None:
    for split in ("train", "val", "test"):
        _save(tmp_path, f"{split}/x.png", _gradient(10 + len(split)))
    report = audit_dataset(tmp_path, near_duplicates=near)
    assert set(report) >= {
        "visual_similarity_checked",
        "near_duplicate_groups",
        "near_duplicate_max_distance",
    }


def test_the_screen_states_the_same_distance_as_the_server() -> None:
    source = Path(__file__).parents[2] / "frontend/src/components/DatasetAudit.tsx"
    match = re.search(r"NEAR_MAX_DISTANCE = (\d+);", source.read_text(encoding="utf-8"))
    assert match is not None
    assert int(match.group(1)) == MAX_DISTANCE
