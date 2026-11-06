"""Dataset explorer core (ADR-165): index, counts, gallery, thumbnails and the scan.

Every dataset is a few tiny images built in ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_explorer import (
    build_index,
    contained_path,
    read_yolo_class_names,
    render_thumbnail,
    scan_integrity,
    select_images,
    summarize,
)


def _png(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _folder_dataset(root: Path) -> Path:
    """cat/dog in train and val; val/cat/a.png is a byte copy of train/cat/a.png."""
    _png(root / "train/cat/a.png", (255, 0, 0))
    _png(root / "train/cat/b.png", (0, 255, 0))
    _png(root / "train/dog/c.png", (0, 0, 255))
    _png(root / "val/cat/a.png", (255, 0, 0))  # same bytes as train/cat/a.png
    _png(root / "val/dog/d.png", (9, 9, 9), (16, 8))
    return root


def _yolo_dataset(root: Path) -> Path:
    for split, name, color, lines in (
        ("train", "x", (1, 2, 3), ["0 0.5 0.5 0.4 0.4", "1 0.2 0.2 0.1 0.1"]),
        ("train", "y", (4, 5, 6), ["1 0.5 0.5 0.3 0.3"]),
        ("train", "empty", (7, 8, 9), []),
        ("val", "z", (1, 2, 3), ["0 0.5 0.5 0.4 0.4"]),  # same pixels as train/x
    ):
        _png(root / "images" / split / f"{name}.png", color, (20, 10))
        labels = root / "labels" / split
        labels.mkdir(parents=True, exist_ok=True)
        if lines:
            (labels / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    (root / "classes.txt").write_text("apple\npear\n", encoding="utf-8")
    return root


class TestIndexClassification:
    def test_counts_per_class_and_split(self, tmp_path: Path) -> None:
        index = build_index(_folder_dataset(tmp_path), "classification")
        summary = summarize(index)
        assert summary["class_names"] == ["cat", "dog"]
        assert summary["total_images"] == 5
        assert summary["splits"]["train"]["per_class"] == {"cat": 2, "dog": 1}
        assert summary["splits"]["val"]["per_class"] == {"cat": 1, "dog": 1}
        assert summary["splits"]["test"]["present"] is False
        assert summary["splits"]["test"]["images"] == 0

    def test_custom_split_folder_names(self, tmp_path: Path) -> None:
        _png(tmp_path / "treino/a/1.png", (1, 1, 1))
        index = build_index(tmp_path, "classification", {"train": "treino"})
        assert summarize(index)["splits"]["train"]["images"] == 1

    def test_ignores_non_images(self, tmp_path: Path) -> None:
        _png(tmp_path / "train/a/1.png", (1, 1, 1))
        (tmp_path / "train/a/notes.txt").write_text("x", encoding="utf-8")
        assert summarize(build_index(tmp_path, "classification"))["total_images"] == 1

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a directory"):
            build_index(tmp_path / "nope", "classification")


class TestIndexDetection:
    def test_boxes_per_class_and_unlabeled(self, tmp_path: Path) -> None:
        index = build_index(_yolo_dataset(tmp_path), "detection")
        train = summarize(index)["splits"]["train"]
        assert index.class_names == ["apple", "pear"]
        assert train["images"] == 3
        assert train["boxes"] == {"apple": 1, "pear": 2}
        assert train["per_class"] == {"apple": 1, "pear": 2}  # images with the class
        assert train["unlabeled"] == 1

    def test_class_names_from_data_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "data.yaml").write_text(
            "names:\n  0: a\n  1: b\n", encoding="utf-8"
        )
        assert read_yolo_class_names(tmp_path) == ["a", "b"]

    def test_unnamed_classes_are_generated(self, tmp_path: Path) -> None:
        _png(tmp_path / "images/train/a.png", (1, 1, 1))
        (tmp_path / "labels/train").mkdir(parents=True)
        (tmp_path / "labels/train/a.txt").write_text("2 .5 .5 .1 .1", encoding="utf-8")
        assert build_index(tmp_path, "detection").class_names == [
            "class_0",
            "class_1",
            "class_2",
        ]

    def test_transposed_layout(self, tmp_path: Path) -> None:
        _png(tmp_path / "train/images/a.png", (1, 1, 1))
        index = build_index(tmp_path, "detection")
        assert index.present["train"] and not index.present["val"]


class TestGallery:
    def test_filters_and_pages(self, tmp_path: Path) -> None:
        index = build_index(_folder_dataset(tmp_path), "classification")
        total, page = select_images(index, split="train", class_name="cat")
        assert total == 2
        assert [im.rel for im in page] == ["train/cat/a.png", "train/cat/b.png"]
        total, page = select_images(index, offset=3, limit=10)
        assert total == 5
        assert len(page) == 2

    def test_detection_class_filter_matches_images_with_the_box(
        self, tmp_path: Path
    ) -> None:
        index = build_index(_yolo_dataset(tmp_path), "detection")
        total, page = select_images(index, split="train", class_name="apple")
        assert total == 1
        assert page[0].rel == "images/train/x.png"

    def test_limit_is_capped(self, tmp_path: Path) -> None:
        index = build_index(_folder_dataset(tmp_path), "classification")
        assert select_images(index, limit=10_000)[0] == 5


class TestThumbnail:
    def test_is_a_small_jpeg(self, tmp_path: Path) -> None:
        big = tmp_path / "big.png"
        _png(big, (10, 20, 30), (400, 200))
        data = render_thumbnail(big, 64)
        assert data[:2] == b"\xff\xd8"
        import io

        with Image.open(io.BytesIO(data)) as thumb:
            assert max(thumb.size) <= 64

    def test_draws_boxes(self, tmp_path: Path) -> None:
        import io

        img = tmp_path / "a.png"
        _png(img, (0, 0, 0), (100, 100))
        label = tmp_path / "a.txt"
        label.write_text("0 0.5 0.5 0.5 0.5", encoding="utf-8")
        with Image.open(io.BytesIO(render_thumbnail(img, 100, label))) as drawn:
            pixel = drawn.convert("RGB").getpixel((25, 50))
        assert isinstance(pixel, tuple)
        red, green, blue = pixel
        assert red > 150 and green < 100 and blue < 100  # the box edge

    def test_contained_path_refuses_escapes(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (tmp_path / "secret.png").write_bytes(b"x")
        with pytest.raises(ValueError):
            contained_path(root, "../secret.png")
        with pytest.raises(ValueError):
            contained_path(root, str(tmp_path / "secret.png"))


class TestScan:
    def test_flags_corrupted_images(self, tmp_path: Path) -> None:
        _folder_dataset(tmp_path)
        (tmp_path / "train/cat/broken.png").write_bytes(b"not an image")
        (tmp_path / "train/dog/empty.jpg").write_bytes(b"")
        report = scan_integrity(build_index(tmp_path, "classification"))
        paths = {item["path"] for item in report["corrupted"]}
        assert paths == {"train/cat/broken.png", "train/dog/empty.jpg"}
        assert report["corrupted_count"] == 2
        assert report["scanned"] == 7

    def test_finds_exact_duplicates_and_leakage(self, tmp_path: Path) -> None:
        _folder_dataset(tmp_path)
        _png(tmp_path / "train/cat/a_copy.png", (255, 0, 0))  # in-split duplicate
        report = scan_integrity(build_index(tmp_path, "classification"))
        assert report["duplicate_group_count"] == 1
        group = report["duplicate_groups"][0]
        assert {f["path"] for f in group["files"]} == {
            "train/cat/a.png",
            "train/cat/a_copy.png",
            "val/cat/a.png",
        }
        assert report["leakage_group_count"] == 1
        assert report["leakage_groups"][0]["splits"] == ["train", "val"]
        assert report["duplicate_file_count"] == 2

    def test_same_split_duplicate_is_not_leakage(self, tmp_path: Path) -> None:
        _png(tmp_path / "train/cat/a.png", (1, 2, 3))
        _png(tmp_path / "train/cat/b.png", (1, 2, 3))
        _png(tmp_path / "val/cat/c.png", (9, 9, 9))
        report = scan_integrity(build_index(tmp_path, "classification"))
        assert report["duplicate_group_count"] == 1
        assert report["leakage_group_count"] == 0

    def test_label_conflict_when_same_image_has_two_classes(
        self, tmp_path: Path
    ) -> None:
        _png(tmp_path / "train/cat/a.png", (1, 2, 3))
        _png(tmp_path / "train/dog/a.png", (1, 2, 3))
        report = scan_integrity(build_index(tmp_path, "classification"))
        assert report["duplicate_groups"][0]["label_conflict"] is True

    def test_clean_dataset_reports_nothing(self, tmp_path: Path) -> None:
        _png(tmp_path / "train/cat/a.png", (1, 2, 3))
        _png(tmp_path / "val/cat/b.png", (4, 5, 6))
        report = scan_integrity(build_index(tmp_path, "classification"))
        assert report["corrupted_count"] == 0
        assert report["duplicate_group_count"] == 0
        assert report["leakage_group_count"] == 0
        assert report["complete"] is True

    def test_detection_leakage(self, tmp_path: Path) -> None:
        report = scan_integrity(build_index(_yolo_dataset(tmp_path), "detection"))
        assert report["leakage_group_count"] == 1
        assert {f["split"] for f in report["leakage_groups"][0]["files"]} == {
            "train",
            "val",
        }

    def test_size_distribution(self, tmp_path: Path) -> None:
        _folder_dataset(tmp_path)  # four 8x8 images and one 16x8
        sizes = scan_integrity(build_index(tmp_path, "classification"))["sizes"]
        assert sizes["count"] == 5
        assert sizes["distinct_sizes"] == 2
        assert sizes["top_sizes"][0] == {"width": 8, "height": 8, "count": 4}
        assert sizes["width"]["max"] == 16
        assert sum(b["count"] for b in sizes["side_bins"]) == 5
        # 8x8 -> aspect 1.0, 16x8 -> aspect 2.0
        assert sum(b["count"] for b in sizes["aspect_bins"]) == 5

    def test_limit_samples_evenly_and_says_so(self, tmp_path: Path) -> None:
        for i in range(10):
            _png(tmp_path / f"train/a/{i}.png", (i, i, i))
        report = scan_integrity(build_index(tmp_path, "classification"), max_images=5)
        assert report["scanned"] == 5
        assert report["sampled"] is True
        assert report["complete"] is False
        assert report["total_images"] == 10

    def test_cancel_stops_early_and_keeps_partial_result(self, tmp_path: Path) -> None:
        for i in range(10):
            _png(tmp_path / f"train/a/{i}.png", (i, i, i))
        token = CancellationToken()

        def progress(done: int, total: int) -> None:
            if done == 3:
                token.cancel()

        report = scan_integrity(
            build_index(tmp_path, "classification"), token=token, on_progress=progress
        )
        assert report["cancelled"] is True
        assert report["scanned"] == 3
        assert report["complete"] is False

    def test_progress_reaches_total(self, tmp_path: Path) -> None:
        _folder_dataset(tmp_path)
        seen: list[tuple[int, int]] = []
        scan_integrity(
            build_index(tmp_path, "classification"),
            on_progress=lambda d, t: seen.append((d, t)),
        )
        assert seen[0] == (0, 5)
        assert seen[-1] == (5, 5)
