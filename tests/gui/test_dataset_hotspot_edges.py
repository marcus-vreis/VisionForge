"""Behavioral edges retained while extracting API domain helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from visionforge.gui.api.datasets_routes import (
    _collect_detection_dataset_samples,
    _collect_regression_dataset_stats,
    _collect_segmentation_dataset_stats,
    _count_split_instances,
    _detect_dataset_layout,
)
from visionforge.gui.api.schemas import (
    DetectionDatasetSamplesRequest,
    RegressionDatasetStatsRequest,
    SegmentationDatasetStatsRequest,
)


def test_detection_counts_only_valid_class_rows(tmp_path: Path) -> None:
    """Only five-field labels with numeric class IDs count as instances."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "a.jpg").write_bytes(b"not decoded by stats")
    (images / "b.png").write_bytes(b"not decoded by stats")
    (images / "ignore.txt").write_text("x", encoding="utf-8")
    (labels / "a.txt").write_text(
        "1.9 0.5 0.5 0.2 0.2\ninvalid 0.5 0.5 0.2 0.2\n2 0 0 0\n",
        encoding="utf-8",
    )
    assert _count_split_instances(images, labels) == (2, {1: 1}, 1)


def test_detection_samples_skip_invalid_labels_and_cap_per_class(
    tmp_path: Path,
) -> None:
    """A malformed row cannot prevent a valid crop from the same image."""
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (20, 20), "red").save(images / "a.png")
    (labels / "a.txt").write_text(
        "bad 0.5 0.5 0.8 0.8\n0 0.5 0.5 0.8 0.8\n0 0.5 0.5 0.8 0.8\n",
        encoding="utf-8",
    )
    (tmp_path / "classes.txt").write_text("cat\n", encoding="utf-8")
    response = _collect_detection_dataset_samples(
        DetectionDatasetSamplesRequest(base_dir=str(tmp_path), per_class=1)
    )
    assert list(response.crops) == ["cat"]
    assert len(response.crops["cat"]) == 1
    assert response.crops["cat"][0].startswith("data:image/png;base64,")
    assert response.message is None


def test_segmentation_skips_corrupt_mask_and_samples_valid_one(tmp_path: Path) -> None:
    """An unreadable train mask does not suppress IDs from later masks."""
    masks = tmp_path / "train" / "masks"
    masks.mkdir(parents=True)
    (masks / "a.png").write_bytes(b"broken")
    Image.new("L", (2, 2), 7).save(masks / "b.png")
    response = _collect_segmentation_dataset_stats(
        SegmentationDatasetStatsRequest(base_dir=str(tmp_path))
    )
    assert response.splits["train"].masks == 2
    assert response.mask_class_ids == [7]


def test_regression_keeps_missing_column_and_numeric_target(tmp_path: Path) -> None:
    """A missing image column leaves image checks empty but target stats work."""
    with (tmp_path / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["target"])
        writer.writeheader()
        writer.writerow({"target": "3"})
        writer.writerow({"target": "bad"})
    response = _collect_regression_dataset_stats(
        RegressionDatasetStatsRequest(base_dir=str(tmp_path))
    )
    train = response.splits["train"]
    assert train.rows == 2
    assert train.missing_columns == ["image"]
    assert train.checked_images == 0
    assert train.targets["target"].count == 1
    assert train.targets["target"].mean == 3.0


def test_dataset_layout_requires_all_three_roles(tmp_path: Path) -> None:
    """Two recognized directories return a partial message and their names."""
    (tmp_path / "Train").mkdir()
    (tmp_path / "validation").mkdir()
    response = _detect_dataset_layout(str(tmp_path))
    assert response.detected is False
    assert response.train_dir == "Train"
    assert response.val_dir == "validation"
    assert response.test_dir is None
    assert response.candidates == ["Train", "validation"]
    assert response.message is not None


def test_segmentation_limit_counts_decodable_masks(tmp_path: Path) -> None:
    """Corrupt masks do not consume the twenty-mask budget; later IDs are ignored."""
    masks = tmp_path / "train" / "masks"
    masks.mkdir(parents=True)
    (masks / "00.png").write_bytes(b"broken")
    (masks / "01.txt").write_text("ignored", encoding="utf-8")
    (masks / "02.png").mkdir()
    for index in range(21):
        Image.new("L", (2, 2), index).save(masks / f"mask_{index:02}.png")
    response = _collect_segmentation_dataset_stats(
        SegmentationDatasetStatsRequest(base_dir=str(tmp_path))
    )
    assert response.mask_class_ids == list(range(20))
    assert response.splits["train"].masks == 22


def test_regression_preserves_row_and_image_check_limits(tmp_path: Path) -> None:
    """Rows after 50,000 and image paths after 500 never enter aggregates."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "present.png").write_bytes(b"stats only check existence")
    with (tmp_path / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        stream.write("image,target\n")
        stream.write("present.png,2\n" * 500)
        stream.write("missing.png,2\n" * 49_500)
        stream.write("missing.png,999\n")
    train = _collect_regression_dataset_stats(
        RegressionDatasetStatsRequest(base_dir=str(tmp_path))
    ).splits["train"]
    assert train.rows == 50_000
    assert train.checked_images == 500
    assert train.missing_images == 0
    assert train.targets["target"].model_dump() == {
        "count": 50_000,
        "mean": 2.0,
        "min": 2.0,
        "max": 2.0,
    }


def test_detection_sample_fields_differ_from_count_fields(tmp_path: Path) -> None:
    """Crops accept trailing fields and unknown IDs; counts require five fields."""
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (20, 20), "red").save(images / "a.png")
    (labels / "a.txt").write_text(
        "0 0 0\n0 bad 0.5 0.8 0.8\n-1 0.5 0.5 0.8 0.8 extra\n",
        encoding="utf-8",
    )
    response = _collect_detection_dataset_samples(
        DetectionDatasetSamplesRequest(base_dir=str(tmp_path))
    )
    assert list(response.crops) == ["class_-1"]
    assert len(response.crops["class_-1"]) == 1
    assert _count_split_instances(images, labels) == (1, {0: 1}, 0)


@pytest.mark.parametrize(
    ("folders", "roles", "detected", "message_key"),
    [
        (
            ["Train", "validation", "TEST"],
            ("Train", "validation", "TEST"),
            True,
            "detect.found",
        ),
        (["validation", "TEST"], (None, "validation", "TEST"), False, "detect.partial"),
        (["Train", "TEST"], ("Train", None, "TEST"), False, "detect.partial"),
        (["Train"], (None, None, None), False, "detect.unrecognized"),
    ],
)
def test_layout_responses_preserve_roles_and_messages(
    tmp_path: Path,
    folders: list[str],
    roles: tuple[str | None, ...],
    detected: bool,
    message_key: str,
) -> None:
    """Partial and unrecognized layouts retain distinct role reporting rules."""
    from visionforge.utils.messages import tr

    for folder in folders:
        (tmp_path / folder).mkdir()
    response = _detect_dataset_layout(str(tmp_path))
    assert (response.train_dir, response.val_dir, response.test_dir) == roles
    assert response.detected is detected
    assert response.candidates == sorted(folders)
    if message_key == "detect.found":
        expected = tr(message_key, train=roles[0], val=roles[1], test=roles[2])
    elif message_key == "detect.partial":
        missing = "detect.role_train" if roles[0] is None else "detect.role_val"
        expected = tr(message_key, missing=tr(missing))
    else:
        expected = tr(message_key, found=", ".join(sorted(folders)))
    assert response.message == expected
