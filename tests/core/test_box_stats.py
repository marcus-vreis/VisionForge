"""Box statistics of a detection dataset (ADR-233), on hand-computed cases."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.box_stats import BoxTally, box_problem, coco_size, parse_row
from visionforge.core.dataset_explorer import build_index, scan_integrity

# A 640x480 image. Pixel sizes of the three valid boxes below:
#   0.05 x 0.05 -> 32 x 24  = 768 px²   (< 32² = 1024: small)
#   0.10 x 0.10 -> 64 x 48  = 3072 px²  (< 96² = 9216: medium)
#   0.20 x 0.20 -> 128 x 96 = 12288 px² (large)
# All three are 4:3, aspect 1.33, in the [1.25, 2.0) bin.
SIZE = (640, 480)
VALID = [
    "0 0.5 0.5 0.05 0.05",  # cell (row 5, col 5)
    "1 0.9 0.1 0.1 0.1",  # cell (row 1, col 9)
    "1 0.25 0.75 0.2 0.2",  # cell (row 7, col 2)
]


def _label(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / f"{name}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestRows:
    @pytest.mark.parametrize(
        "line",
        [
            "0 0.5 0.5",
            "0 0.5 0.5 0.1 0.1 9",
            "a 0.5 0.5 0.1 0.1",
            "0.5 0.5 0.5 0.1 0.1",
            "0 nan 0.5 0.1 0.1",
        ],
    )
    def test_not_five_numbers_with_an_integer_class_is_malformed(
        self, line: str
    ) -> None:
        assert parse_row(line) is None

    @pytest.mark.parametrize(
        ("line", "reason"),
        [
            ("2 0.5 0.5 0.1 0.1", "class_out_of_range"),
            ("-1 0.5 0.5 0.1 0.1", "class_out_of_range"),
            ("0 0.5 0.5 0 0.1", "zero_area"),
            ("0 0.5 0.5 0.1 -0.1", "zero_area"),
            ("0 0.95 0.5 0.2 0.2", "outside_image"),  # right edge at 1.05
            ("0 0.5 0.02 0.1 0.1", "outside_image"),  # top edge at -0.03
            ("0 0.905 0.5 0.2 0.2", None),  # right edge at 1.005: within tolerance
        ],
    )
    def test_box_problem(self, line: str, reason: str | None) -> None:
        row = parse_row(line)
        assert row is not None
        assert box_problem(row, n_classes=2) == reason


@pytest.mark.parametrize(
    ("area", "bucket"),
    [(1023.9, "small"), (1024, "medium"), (9215.9, "medium"), (9216, "large")],
)
def test_coco_size_limits_are_half_open(area: float, bucket: str) -> None:
    # COCO: small is area < 32², medium 32² <= area < 96², large the rest.
    assert coco_size(area) == bucket


class TestTally:
    def test_sizes_aspect_and_heatmap(self, tmp_path: Path) -> None:
        tally = BoxTally(["cat", "dog"], 2)
        tally.add("train", "a.png", _label(tmp_path, "a", VALID), SIZE)
        report = tally.report()
        assert report["boxes"] == 3
        assert report["sizes"] == {"small": 1, "medium": 1, "large": 1}
        assert report["class_sizes"] == [
            {"name": "cat", "small": 1, "medium": 0, "large": 0},
            {"name": "dog", "small": 0, "medium": 1, "large": 1},
        ]
        assert [b["count"] for b in report["aspect_bins"]] == [0, 0, 0, 0, 3, 0, 0]
        cells = {
            (r, c): n
            for r, row in enumerate(report["heatmap"])
            for c, n in enumerate(row)
            if n
        }
        assert cells == {(5, 5): 1, (1, 9): 1, (7, 2): 1}

    def test_boxes_per_image_and_images_without_boxes(self, tmp_path: Path) -> None:
        tally = BoxTally(["cat", "dog"], 2)
        tally.add("train", "a.png", _label(tmp_path, "a", VALID), SIZE)
        tally.add("train", "b.png", _label(tmp_path, "b", ["0 0.5 0.5 0.1 0.1"]), SIZE)
        tally.add("train", "c.png", _label(tmp_path, "c", []), SIZE)
        tally.add("val", "d.png", None, SIZE)
        report = tally.report()
        assert report["images"] == 4
        assert report["boxes"] == 4
        assert report["mean_per_image"] == 1.0
        assert report["max_per_image"] == 3
        # Bins 0, 1, 2-3, 4-7, ...: two empty, one with 1, one with 3.
        assert [b["count"] for b in report["per_image_bins"]][:4] == [2, 1, 1, 0]
        assert (report["no_label_file"], report["empty_label"]) == (1, 1)

    def test_invalid_rows_are_counted_and_located(self, tmp_path: Path) -> None:
        tally = BoxTally(["cat", "dog"], 2)
        lines = [
            "0 0.5 0.5 0.1 0.1",
            "7 0.5 0.5 0.1 0.1",
            "",
            "0 0.5 0.5 0 0.1",
            "0 0.5",
        ]
        tally.add("val", "x.png", _label(tmp_path, "x", lines), SIZE)
        report = tally.report()
        assert report["boxes"] == 1
        assert report["invalid_count"] == 3
        assert report["invalid"] == {
            "class_out_of_range": 1,
            "malformed": 1,
            "zero_area": 1,
        }
        assert [(e["line"], e["reason"]) for e in report["invalid_examples"]] == [
            (2, "class_out_of_range"),
            (4, "zero_area"),
            (5, "malformed"),
        ]

    def test_an_image_with_only_invalid_rows_counts_as_empty(
        self, tmp_path: Path
    ) -> None:
        tally = BoxTally(["cat"], 1)
        tally.add("train", "x.png", _label(tmp_path, "x", ["0 0.5 0.5 0 0"]), SIZE)
        assert tally.report()["empty_label"] == 1

    def test_classes_seen_together(self, tmp_path: Path) -> None:
        tally = BoxTally(["cat", "dog", "fox"], 3)
        both = ["0 0.5 0.5 0.1 0.1", "1 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1"]
        tally.add("train", "a.png", _label(tmp_path, "a", both), SIZE)
        tally.add("train", "b.png", _label(tmp_path, "b", both), SIZE)
        tally.add(
            "train",
            "c.png",
            _label(tmp_path, "c", ["0 0.5 0.5 0.1 0.1", "2 0.5 0.5 0.1 0.1"]),
            SIZE,
        )
        assert tally.report()["pairs"] == [
            {"a": "cat", "b": "dog", "images": 2},
            {"a": "cat", "b": "fox", "images": 1},
        ]

    def test_an_image_that_did_not_open_has_no_pixel_size(self, tmp_path: Path) -> None:
        tally = BoxTally(["cat", "dog"], 2)
        tally.add("train", "a.png", _label(tmp_path, "a", VALID), None)
        report = tally.report()
        assert report["boxes"] == 3
        assert report["unsized"] == 3
        assert report["sizes"] == {"small": 0, "medium": 0, "large": 0}


class TestInTheScan:
    def test_detection_scan_reports_boxes(self, tmp_path: Path) -> None:
        for split, name, lines in (
            ("train", "a", VALID),
            ("val", "b", ["0 0.5 0.5 0.05 0.05", "9 0.5 0.5 0.1 0.1"]),
        ):
            image = tmp_path / "images" / split / f"{name}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", SIZE, (len(name), 0, 0)).save(image)
            labels = tmp_path / "labels" / split
            labels.mkdir(parents=True, exist_ok=True)
            (labels / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        (tmp_path / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
        boxes = scan_integrity(build_index(tmp_path, "detection"))["boxes"]
        assert boxes["images"] == 2
        assert boxes["sizes"] == {"small": 2, "medium": 1, "large": 1}
        assert boxes["invalid"] == {"class_out_of_range": 1}
        assert boxes["invalid_examples"][0]["path"] == "images/val/b.png"

    def test_classification_scan_has_no_boxes(self, tmp_path: Path) -> None:
        image = tmp_path / "train" / "cat" / "a.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(image)
        assert scan_integrity(build_index(tmp_path, "classification"))["boxes"] is None
