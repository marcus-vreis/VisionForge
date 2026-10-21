from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.dataset_audit import audit_dataset, load_group_manifest


def _image(root: Path, relative: str, color: str = "red") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def test_reports_only_cross_split_byte_duplicates(tmp_path: Path) -> None:
    source = _image(tmp_path, "train/a/a.png")
    shutil.copyfile(source, _image(tmp_path, "val/a/b.png", "blue"))
    shutil.copyfile(source, tmp_path / "train/a/c.png")
    _image(tmp_path, "test/a/d.png", "green")
    (tmp_path / "train/labels.txt").write_text("label")
    (tmp_path / "val/labels.txt").write_text("label")

    report = audit_dataset(tmp_path)

    assert report["ok"] is False
    assert report["image_counts"] == {"train": 2, "val": 1, "test": 1}
    assert report["duplicate_groups"][0]["paths"] == [
        "train/a/a.png",
        "train/a/c.png",
        "val/a/b.png",
    ]
    assert len(report["duplicate_groups"]) == 1
    assert report["visual_similarity_checked"] is False
    assert json.loads(json.dumps(report)) == report
    assert audit_dataset(tmp_path) == report


def test_corrupt_images_are_bounded_and_nonimages_ignored(tmp_path: Path) -> None:
    for index in range(3):
        path = tmp_path / "train" / f"bad{index}.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"not an image")
    (tmp_path / "train/labels.csv").write_text("path,label")

    report = audit_dataset(tmp_path, {"train": "train"}, max_issues=1)

    assert report["unreadable_count"] == 3
    assert len(report["unreadable"]) == 1
    assert report["image_counts"] == {"train": 0}
    assert report["ok"] is False


def test_png_checksum_error_is_reported(tmp_path: Path) -> None:
    path = _image(tmp_path, "train/broken.png")
    data = bytearray(path.read_bytes())
    data[data.index(b"IDAT") + 5] ^= 1
    path.write_bytes(data)

    report = audit_dataset(tmp_path, {"train": "train"})

    assert report["unreadable_count"] == 1
    assert report["image_counts"] == {"train": 0}
    assert report["ok"] is False


@pytest.mark.parametrize("relative", ["../outside", "C:/outside", "/outside"])
def test_rejects_unsafe_split_paths(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        audit_dataset(tmp_path, {"train": relative})


def test_manifest_detects_cross_split_groups(tmp_path: Path) -> None:
    first = _image(tmp_path, "train/a/a.png")
    second = _image(tmp_path, "val/a/b.png", "blue")
    manifest = tmp_path / "groups.csv"
    manifest.write_text("path,group\ntrain/a/a.png,patient1\nval/a/b.png,patient1\n")

    assert load_group_manifest(tmp_path, manifest, [first, second]) == [
        "patient1",
        "patient1",
    ]
    report = audit_dataset(
        tmp_path,
        {"train": "train", "val": "val"},
        group_manifest=manifest,
        strict_groups=True,
    )

    assert report["group_overlaps"] == [
        {
            "group": "patient1",
            "splits": ["train", "val"],
            "paths": ["train/a/a.png", "val/a/b.png"],
        }
    ]
    assert report["ok"] is False


@pytest.mark.parametrize(
    "rows,match",
    [
        ("path,group\ntrain/a/a.png,g\ntrain/a/a.png,h\n", "Duplicate"),
        ("path,group\nmissing.png,g\n", "Unknown"),
        ("path,group\n", "Missing"),
        ("path,group\n../outside.png,g\n", "relative"),
        ("path,group\ntrain/a/a.png,\n", "group"),
        ("path,label\ntrain/a/a.png,g\n", "path,group"),
    ],
)
def test_manifest_rejects_invalid_rows(tmp_path: Path, rows: str, match: str) -> None:
    path = _image(tmp_path, "train/a/a.png")
    manifest = tmp_path / "groups.csv"
    manifest.write_text(rows)
    with pytest.raises(ValueError, match=match):
        load_group_manifest(tmp_path, manifest, [path])


def test_strict_groups_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest"):
        audit_dataset(tmp_path, strict_groups=True)


def test_external_directory_link_is_never_read(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "private"
    _image(outside, "secret.png")
    (root / "train").mkdir()
    link = root / "train" / "linked"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(outside, target_is_directory=True)

    report = audit_dataset(root, {"train": "train"})

    assert report["image_counts"] == {"train": 0}
    assert report["unreadable_count"] == 1
    assert report["unreadable"][0]["path"] == "train/linked"
    assert report["ok"] is False


def test_missing_split_and_empty_mapping(tmp_path: Path) -> None:
    report = audit_dataset(tmp_path)
    assert report["missing_splits"] == ["train", "val", "test"]
    assert report["ok"] is False
    with pytest.raises(ValueError, match="one split"):
        audit_dataset(tmp_path, {})
    with pytest.raises(ValueError, match="non-negative"):
        audit_dataset(tmp_path, max_issues=-1)


def test_present_empty_splits_fail_audit(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        (tmp_path / split).mkdir()

    report = audit_dataset(tmp_path)

    assert report["empty_splits"] == ["train", "val", "test"]
    assert report["missing_splits"] == []
    assert report["ok"] is False


def test_group_ids_ignore_surrounding_whitespace(tmp_path: Path) -> None:
    _image(tmp_path, "train/a.png")
    _image(tmp_path, "val/b.png", "blue")
    manifest = tmp_path / "groups.csv"
    manifest.write_text("path,group\ntrain/a.png,plant1\nval/b.png, plant1 \n")

    report = audit_dataset(
        tmp_path,
        {"train": "train", "val": "val"},
        group_manifest=manifest,
        strict_groups=True,
    )

    assert report["group_overlaps"] == [
        {
            "group": "plant1",
            "splits": ["train", "val"],
            "paths": ["train/a.png", "val/b.png"],
        }
    ]
    assert report["ok"] is False


def test_group_overlap_is_advisory_without_strict_mode(tmp_path: Path) -> None:
    _image(tmp_path, "train/a.png")
    _image(tmp_path, "val/b.png", "blue")
    manifest = tmp_path / "groups.csv"
    manifest.write_text("path,group\ntrain/a.png,subject\nval/b.png,subject\n")
    report = audit_dataset(
        tmp_path, {"train": "train", "val": "val"}, group_manifest=manifest
    )
    assert len(report["group_overlaps"]) == 1
    assert report["ok"] is True


def test_identical_pixels_with_different_bytes_are_not_duplicates(
    tmp_path: Path,
) -> None:
    _image(tmp_path, "train/a.png")
    destination = tmp_path / "val/b.png"
    destination.parent.mkdir()
    Image.new("RGB", (8, 8), "red").save(destination, compress_level=0)
    report = audit_dataset(tmp_path, {"train": "train", "val": "val"})
    assert report["duplicate_groups"] == []
    assert report["ok"] is True
