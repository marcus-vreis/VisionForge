"""The image audit reads each task's own layout (ADR-230)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.dataset_audit_layouts import (
    audit_task_dataset,
    csv_split_images,
    yolo_split_folders,
)


def _image(root: Path, relative: str, color: str = "red") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def _csv(root: Path, name: str, *images: str, column: str = "image") -> None:
    rows = "".join(f"{image},1.0\n" for image in images)
    (root / name).write_text(f"{column},target\n{rows}", encoding="utf-8")


@pytest.mark.parametrize("layout", ["images/{split}", "{split}/images"])
def test_yolo_finds_either_layout_and_ignores_labels(
    tmp_path: Path, layout: str
) -> None:
    source = _image(tmp_path, layout.format(split="train") + "/a.jpg")
    shutil.copyfile(source, _image(tmp_path, layout.format(split="val") + "/b.jpg"))
    _image(tmp_path, layout.format(split="test") + "/c.jpg", "blue")
    labels = tmp_path / layout.format(split="train").replace("images", "labels")
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    report = audit_task_dataset(tmp_path, None, "yolo")

    assert report["image_counts"] == {"train": 1, "val": 1, "test": 1}
    assert [group["splits"] for group in report["duplicate_groups"]] == [
        ["train", "val"]
    ]


def test_a_yolo_split_without_images_is_missing(tmp_path: Path) -> None:
    _image(tmp_path, "images/train/a.jpg")
    _image(tmp_path, "images/val/b.jpg", "blue")

    assert yolo_split_folders(tmp_path, {"test": "test"}) == {"test": "images/test"}
    report = audit_task_dataset(tmp_path, None, "yolo")

    assert report["missing_splits"] == ["test"]
    assert report["ok"] is False


def test_segmentation_folders_skip_the_masks(tmp_path: Path) -> None:
    # Every all-background mask is the same file: auditing the split folder
    # whole would report those as leaks; the images folder alone does not.
    for split, color in (("train", "red"), ("val", "blue")):
        _image(tmp_path, f"{split}/images/x.png", color)
        _image(tmp_path, f"{split}/masks/x.png", "black")

    report = audit_task_dataset(
        tmp_path, {"train": "train/images", "val": "val/images"}, "folders"
    )

    assert report["ok"] is True
    assert report["image_counts"] == {"train": 1, "val": 1}


def test_anomaly_finds_a_test_image_copied_into_training(tmp_path: Path) -> None:
    source = _image(tmp_path, "test/crack/a.png", "black")
    shutil.copyfile(source, tmp_path / "test/crack/b.png")
    _image(tmp_path, "train/good/n.png", "white")
    shutil.copyfile(source, tmp_path / "train/good/leak.png")

    report = audit_task_dataset(tmp_path, {"train": "train", "test": "test"}, "folders")

    assert report["duplicate_groups"][0]["paths"] == [
        "test/crack/a.png",
        "test/crack/b.png",
        "train/good/leak.png",
    ]


def test_csv_splits_find_an_image_listed_in_two_splits(tmp_path: Path) -> None:
    _image(tmp_path, "images/a.png")
    _image(tmp_path, "images/sub/b.png", "blue")
    _image(tmp_path, "images/unlisted.png", "green")
    _csv(tmp_path, "train.csv", "a.png", "sub/b.png", "a.png")
    _csv(tmp_path, "val.csv", "a.png")

    report = audit_task_dataset(
        tmp_path, {"train": "train.csv", "val": "val.csv", "test": "test.csv"}, "csv"
    )

    # a.png listed twice in train is one image; unlisted.png is in no split.
    assert report["image_counts"] == {"train": 2, "val": 1, "test": 0}
    assert report["duplicate_groups"] == [
        {
            "sha256": report["duplicate_groups"][0]["sha256"],
            "splits": ["train", "val"],
            "paths": ["images/a.png", "images/a.png"],
        }
    ]
    assert report["missing_splits"] == ["test"]


def test_csv_rows_that_name_no_readable_image_are_reported(tmp_path: Path) -> None:
    _image(tmp_path, "images/a.png")
    outside = _image(tmp_path.parent, f"{tmp_path.name}-outside.png", "blue")
    _csv(tmp_path, "train.csv", "a.png", "gone.png", f"../../{outside.name}")

    report = audit_task_dataset(tmp_path, {"train": "train.csv"}, "csv")

    assert report["image_counts"] == {"train": 1}
    reasons = {item["reason"] for item in report["unreadable"]}
    assert reasons == {"FileNotFoundError", "path escapes dataset root"}
    assert report["unreadable_count"] == 2


def test_csv_without_the_image_column_is_refused(tmp_path: Path) -> None:
    _csv(tmp_path, "train.csv", "a.png", column="file")

    with pytest.raises(ValueError, match="no column 'image'"):
        csv_split_images(tmp_path, {"train": "train.csv"}, "images", "image")


def test_a_csv_outside_the_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    _csv(tmp_path, "train.csv", "a.png")

    with pytest.raises(ValueError, match="outside the dataset root"):
        csv_split_images(root, {"train": "../train.csv"}, "images", "image")


def test_csv_near_duplicates_cross_splits(tmp_path: Path) -> None:
    pixels = [(x * 30 % 256, x * 7 % 256, 90) for x in range(64)]
    first = Image.new("RGB", (8, 8))
    first.putdata(pixels)
    (tmp_path / "images").mkdir()
    first.resize((64, 64)).save(tmp_path / "images/a.png")
    first.resize((48, 48)).save(tmp_path / "images/b.jpg", quality=90)
    _csv(tmp_path, "train.csv", "a.png")
    _csv(tmp_path, "val.csv", "b.jpg")

    report = audit_task_dataset(
        tmp_path,
        {"train": "train.csv", "val": "val.csv"},
        "csv",
        near_duplicates=True,
    )

    assert report["duplicate_groups"] == []
    assert [group["paths"] for group in report["near_duplicate_groups"]] == [
        ["images/a.png", "images/b.jpg"]
    ]


def test_an_absent_optional_split_is_no_finding(tmp_path: Path) -> None:
    _image(tmp_path, "images/a.png")
    _image(tmp_path, "images/b.png", "blue")
    _csv(tmp_path, "train.csv", "a.png")
    _csv(tmp_path, "val.csv", "b.png")
    splits = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}

    optional = audit_task_dataset(tmp_path, splits, "csv", optional_splits=["test"])
    required = audit_task_dataset(tmp_path, splits, "csv")

    assert optional["ok"] is True
    assert optional["image_counts"] == {"train": 1, "val": 1}
    assert optional["absent_optional_splits"] == ["test"]
    assert optional["missing_splits"] == []
    assert required["ok"] is False
    assert required["missing_splits"] == ["test"]
    assert required["absent_optional_splits"] == []


def test_a_present_optional_split_is_still_audited(tmp_path: Path) -> None:
    source = _image(tmp_path, "train/images/a.png")
    _image(tmp_path, "val/images/b.png", "blue")
    (tmp_path / "test/images").mkdir(parents=True)
    shutil.copyfile(source, tmp_path / "test/images/a.png")
    splits = {s: f"{s}/images" for s in ("train", "val", "test")}

    report = audit_task_dataset(tmp_path, splits, "folders", optional_splits=["test"])

    assert report["duplicate_groups"][0]["splits"] == ["test", "train"]
    assert report["absent_optional_splits"] == []
