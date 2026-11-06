"""Converting detection datasets between YOLO and COCO (ADR-169)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_coco import (
    CocoChangedError,
    CocoError,
    analyze_coco_export,
    build_coco_export,
    coco_bbox_to_yolo,
    parse_yolo_label,
    suggest_export_name,
    yolo_to_coco_bbox,
)
from visionforge.core.dataset_coco_import import (
    SKIP,
    analyze_coco_import,
    build_coco_import,
    detect_split,
    plan_import,
    suggest_import_name,
)

Json = dict[str, Any]


def _png(path: Path, size: tuple[int, int] = (100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)


def _tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _label(root: Path, split: str, stem: str, text: str) -> None:
    path = root / "labels" / split / f"{stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def yolo(tmp_path: Path) -> Path:
    """Two labelled images (100x50 and 200x100), one background, classes cat/dog."""
    root = tmp_path / "ds"
    _png(root / "images" / "train" / "a.png", (100, 50))
    _png(root / "images" / "train" / "bg.png", (40, 40))
    _png(root / "images" / "val" / "c.png", (200, 100))
    _label(root, "train", "a", "0 0.5 0.5 0.2 0.4\n1 0.25 0.5 0.1 0.2\n")
    _label(root, "train", "bg", "")
    _label(root, "val", "c", "1 0.75 0.25 0.5 0.5\n")
    (root / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    return root


def _export(root: Path, name: str = "ds-coco") -> dict[str, Any]:
    analysis = analyze_coco_export(root)
    assert analysis is not None
    result = build_coco_export(analysis, name)
    assert result is not None
    return result


def _read(path: Path) -> Json:
    data: Json = json.loads(path.read_text(encoding="utf-8"))
    return data


# ── the conversion ────────────────────────────────────────────────────────────


def test_yolo_to_coco_uses_top_left_corner_and_absolute_pixels() -> None:
    assert yolo_to_coco_bbox(0.5, 0.5, 0.2, 0.4, 100, 50) == (40.0, 15.0, 20.0, 20.0)


def test_coco_to_yolo_is_the_inverse() -> None:
    box, clipped = coco_bbox_to_yolo((40.0, 15.0, 20.0, 20.0), 100, 50)
    assert box == pytest.approx((0.5, 0.5, 0.2, 0.4))
    assert clipped is False


def test_parse_yolo_label_accepts_blank_lines_and_float_ids() -> None:
    assert parse_yolo_label("\n2.0 0.5 0.5 0.1 0.1\n\n") == [(2, 0.5, 0.5, 0.1, 0.1)]


@pytest.mark.parametrize(
    "text",
    [
        "0 0.5 0.5 0.2",
        "0 0.5 0.5 0.2 0.2 0.9",
        "x 0.5 0.5 0.2 0.2",
        "0 0.5 abc 0.2 0.2",
        "-1 0.5 0.5 0.2 0.2",
        "1.5 0.5 0.5 0.2 0.2",
        "0 0.5 0.5 0 0.2",
        "0 0.5 nan 0.2 0.2",
    ],
)
def test_parse_yolo_label_rejects_malformed_lines(text: str) -> None:
    with pytest.raises(ValueError, match="line 1"):
        parse_yolo_label(text)


# ── export ────────────────────────────────────────────────────────────────────


def test_export_writes_every_coco_field(yolo: Path) -> None:
    result = _export(yolo)
    out = yolo.parent / "ds-coco"
    assert result["path"] == str(out)
    train = _read(out / "annotations" / "instances_train.json")
    assert train["licenses"] == []
    assert train["info"]["visionforge_version"]
    assert train["info"]["source_fingerprint"]
    assert train["categories"] == [
        {"id": 1, "name": "cat", "supercategory": ""},
        {"id": 2, "name": "dog", "supercategory": ""},
    ]
    assert train["images"] == [
        {"id": 1, "file_name": "a.png", "width": 100, "height": 50},
        {"id": 2, "file_name": "bg.png", "width": 40, "height": 40},
    ]
    assert train["annotations"] == [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [40.0, 15.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
            "segmentation": [],
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 2,
            "bbox": [20.0, 20.0, 10.0, 10.0],
            "area": 100.0,
            "iscrowd": 0,
            "segmentation": [],
        },
    ]


def test_export_ids_continue_across_splits(yolo: Path) -> None:
    _export(yolo)
    val = _read(yolo.parent / "ds-coco" / "annotations" / "instances_val.json")
    assert val["images"] == [
        {"id": 3, "file_name": "c.png", "width": 200, "height": 100}
    ]
    assert val["annotations"] == [
        {
            "id": 3,
            "image_id": 3,
            "category_id": 2,
            "bbox": [100.0, 0.0, 100.0, 50.0],
            "area": 5000.0,
            "iscrowd": 0,
            "segmentation": [],
        }
    ]


def test_export_keeps_background_images_without_annotations(yolo: Path) -> None:
    _export(yolo)
    train = _read(yolo.parent / "ds-coco" / "annotations" / "instances_train.json")
    background = next(i for i in train["images"] if i["file_name"] == "bg.png")
    assert [a for a in train["annotations"] if a["image_id"] == background["id"]] == []


def test_export_layout_has_images_and_one_json_per_existing_split(yolo: Path) -> None:
    _export(yolo)
    out = yolo.parent / "ds-coco"
    assert sorted(_tree(out)) == [
        "annotations/instances_train.json",
        "annotations/instances_val.json",
        "images/train/a.png",
        "images/train/bg.png",
        "images/val/c.png",
        "visionforge_dataset.json",
    ]
    assert (out / "images" / "train" / "a.png").read_bytes() == (
        yolo / "images" / "train" / "a.png"
    ).read_bytes()


def test_export_excludes_malformed_label_and_unreadable_image(yolo: Path) -> None:
    _label(yolo, "val", "bad", "0 0.5 0.5 0.2\n")
    _png(yolo / "images" / "val" / "bad.png")
    (yolo / "images" / "val" / "broken.png").write_bytes(b"not an image")
    analysis = analyze_coco_export(yolo)
    assert analysis is not None
    summary = analysis.summary()
    assert summary["excluded_by_reason"] == {
        "malformed_label": 1,
        "unreadable_image": 1,
    }
    assert sorted(e["path"] for e in summary["excluded_listed"]) == [
        "images/val/bad.png",
        "images/val/broken.png",
    ]
    assert summary["images"] == 3
    build_coco_export(analysis, "ds-coco")
    val = _read(yolo.parent / "ds-coco" / "annotations" / "instances_val.json")
    assert [i["file_name"] for i in val["images"]] == ["c.png"]
    assert not (yolo.parent / "ds-coco" / "images" / "val" / "bad.png").exists()


def test_export_unnamed_classes_get_a_placeholder(yolo: Path) -> None:
    (yolo / "classes.txt").write_text("cat\n", encoding="utf-8")
    analysis = analyze_coco_export(yolo)
    assert analysis is not None
    assert analysis.classes == ["cat", "class_1"]


def test_export_registry_records_source_counts_and_exclusions(yolo: Path) -> None:
    _png(yolo / "images" / "val" / "broken.png")
    (yolo / "images" / "val" / "broken.png").write_bytes(b"nope")
    _export(yolo)
    registry = _read(yolo.parent / "ds-coco" / "visionforge_dataset.json")
    assert registry["schema_version"] == 1
    assert registry["kind"] == "coco_export"
    assert registry["derived_from"]["path"] == str(yolo.resolve())
    assert registry["derived_from"]["fingerprint"]["digest"]
    assert registry["classes"] == ["cat", "dog"]
    assert registry["counts"]["images"] == 3
    assert registry["counts"]["boxes"] == 3
    assert registry["counts"]["per_split"]["train"] == {"images": 2, "boxes": 2}
    assert registry["counts"]["excluded_by_reason"] == {"unreadable_image": 1}


def test_export_leaves_the_original_untouched(yolo: Path) -> None:
    before = _tree(yolo)
    _export(yolo)
    assert _tree(yolo) == before


def test_export_falls_back_to_copy_when_linking_fails(yolo: Path) -> None:
    def no_link(src: Path, dst: Path) -> None:
        raise PermissionError("no hardlinks here")

    analysis = analyze_coco_export(yolo)
    assert analysis is not None
    result = build_coco_export(analysis, "ds-coco", link=no_link)
    assert result is not None
    assert (result["files_linked"], result["files_copied"]) == (0, 3)


def test_export_cancel_leaves_nothing_behind(yolo: Path) -> None:
    analysis = analyze_coco_export(yolo)
    assert analysis is not None
    token = CancellationToken()
    token.cancel()
    assert build_coco_export(analysis, "ds-coco", token=token) is None
    assert sorted(p.name for p in yolo.parent.iterdir()) == ["ds"]


def test_export_error_leaves_nothing_behind(yolo: Path) -> None:
    def boom(src: Path, dst: Path) -> None:
        raise RuntimeError("disk on fire")

    analysis = analyze_coco_export(yolo)
    assert analysis is not None
    with pytest.raises(RuntimeError, match="disk on fire"):
        build_coco_export(analysis, "ds-coco", link=boom)
    assert sorted(p.name for p in yolo.parent.iterdir()) == ["ds"]


def test_export_refuses_when_the_source_changed(yolo: Path) -> None:
    analysis = analyze_coco_export(yolo)
    assert analysis is not None
    _png(yolo / "images" / "val" / "late.png")
    with pytest.raises(CocoChangedError):
        build_coco_export(analysis, "ds-coco")
    assert sorted(p.name for p in yolo.parent.iterdir()) == ["ds"]


def test_export_refuses_a_bad_or_taken_destination(yolo: Path) -> None:
    analysis = analyze_coco_export(yolo)
    assert analysis is not None
    with pytest.raises(CocoError) as bad:
        build_coco_export(analysis, "a/b")
    assert bad.value.code == "coco.bad_dest_name"
    (yolo.parent / "taken").mkdir()
    with pytest.raises(CocoError) as taken:
        build_coco_export(analysis, "taken")
    assert taken.value.code == "coco.dest_exists"


def test_export_of_a_dataset_with_nothing_readable_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "train" / "x.png").write_bytes(b"nope")
    with pytest.raises(CocoError) as err:
        analyze_coco_export(root)
    assert err.value.code == "coco.nothing_to_export"


def test_suggested_names_skip_existing_folders(yolo: Path) -> None:
    assert suggest_export_name(yolo) == "ds-coco"
    (yolo.parent / "ds-coco").mkdir()
    assert suggest_export_name(yolo) == "ds-coco-2"
    assert suggest_import_name(yolo) == "ds-yolo"


# ── import ────────────────────────────────────────────────────────────────────


def _coco(
    root: Path,
    json_name: str,
    images: list[Json],
    annotations: list[Json],
    categories: list[Json] | None = None,
) -> None:
    path = root / json_name
    path.parent.mkdir(parents=True, exist_ok=True)
    cats = categories or [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": cats}),
        encoding="utf-8",
    )


def _img(image_id: int, name: str, w: int = 100, h: int = 50) -> Json:
    return {"id": image_id, "file_name": name, "width": w, "height": h}


def _ann(ann_id: int, image_id: int, cat: int, bbox: list[Any], **extra: Any) -> Json:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": cat,
        "bbox": bbox,
        **extra,
    }


@pytest.fixture
def coco(tmp_path: Path) -> Path:
    """The classic layout: ``val2017/`` next to ``annotations/instances_val2017.json``."""
    root = tmp_path / "cocoset"
    _png(root / "val2017" / "one.png", (100, 50))
    _png(root / "val2017" / "two.png", (100, 50))
    _coco(
        root,
        "annotations/instances_val2017.json",
        [_img(1, "one.png"), _img(2, "two.png")],
        [_ann(1, 1, 1, [10, 5, 20, 10])],
    )
    return root


def _import(
    root: Path, splits: dict[str, str] | None = None, name: str = "cocoset-yolo"
) -> dict[str, Any]:
    analysis = analyze_coco_import(root)
    assert analysis is not None
    plan = plan_import(analysis, splits)
    result = build_coco_import(analysis, plan, name)
    assert result is not None
    return result


@pytest.mark.parametrize(
    ("name", "split"),
    [
        ("instances_val2017.json", "val"),
        ("instances_train2017.json", "train"),
        ("train.json", "train"),
        ("valid.json", "val"),
        ("annotations/validation.json", "val"),
        ("test-dev2017.json", "test"),
        ("instances_default.json", None),
        ("labels.json", None),
    ],
)
def test_split_is_detected_from_the_file_name(name: str, split: str | None) -> None:
    assert detect_split(name)[0] == split


def test_import_converts_boxes_with_hand_computed_coordinates(coco: Path) -> None:
    _import(coco)
    out = coco.parent / "cocoset-yolo"
    # [10, 5, 20, 10] on 100x50: centre (20, 10) -> (0.2, 0.2), size (0.2, 0.2).
    assert (out / "labels" / "val" / "one.txt").read_text() == (
        "0 0.200000 0.200000 0.200000 0.200000\n"
    )
    assert (out / "labels" / "val" / "two.txt").read_text() == ""
    assert (out / "images" / "val" / "one.png").is_file()


def test_import_writes_class_files_and_yaml(coco: Path) -> None:
    _import(coco)
    out = coco.parent / "cocoset-yolo"
    assert (out / "classes.txt").read_text() == "cat\ndog\n"
    yaml_text = (out / "data.yaml").read_text()
    assert f"path: {out}" in yaml_text
    assert "val: images/val" in yaml_text
    assert "nc: 2" in yaml_text
    assert "0: cat" in yaml_text


def test_import_remaps_non_contiguous_category_ids_in_ascending_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "c"
    _png(root / "train" / "a.png")
    cats = [{"id": 90, "name": "z"}, {"id": 1, "name": "a"}, {"id": 3, "name": "m"}]
    _coco(
        root,
        "train.json",
        [_img(1, "a.png")],
        [
            _ann(1, 1, 90, [0, 0, 10, 10]),
            _ann(2, 1, 1, [0, 0, 10, 10]),
            _ann(3, 1, 3, [0, 0, 10, 10]),
        ],
        cats,
    )
    result = _import(root, name="c-yolo")
    out = tmp_path / "c-yolo"
    assert result["classes"] == ["a", "m", "z"]
    ids = [
        line.split()[0]
        for line in (out / "labels/train/a.txt").read_text().splitlines()
    ]
    assert ids == ["2", "0", "1"]
    registry = _read(out / "visionforge_dataset.json")
    assert [(m["coco_id"], m["yolo_id"]) for m in registry["mapping"]] == [
        (1, 0),
        (3, 1),
        (90, 2),
    ]


def test_import_skips_crowd_boxes_and_reports_them(coco: Path) -> None:
    _coco(
        coco,
        "annotations/instances_val2017.json",
        [_img(1, "one.png")],
        [
            _ann(1, 1, 1, [10, 5, 20, 10], iscrowd=1),
            _ann(2, 1, 2, [0, 0, 10, 10], iscrowd=0),
        ],
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    source = analysis.sources[0]
    assert source.report.skipped == {"crowd": 1}
    assert source.boxes == 1


def test_import_clips_boxes_to_the_image(coco: Path) -> None:
    _coco(
        coco,
        "annotations/instances_val2017.json",
        [_img(1, "one.png")],
        [_ann(1, 1, 1, [-10, -5, 30, 15])],
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    # Clipped to x 0..20, y 0..10 on 100x50.
    assert analysis.sources[0].images[0].boxes[0] == pytest.approx(
        (1, 0.1, 0.1, 0.2, 0.2)
    )
    assert analysis.sources[0].report.adjusted == {"clipped_box": 1}


def test_import_drops_boxes_with_no_area_after_clipping(coco: Path) -> None:
    _coco(
        coco,
        "annotations/instances_val2017.json",
        [_img(1, "one.png")],
        [
            _ann(1, 1, 1, [200, 0, 10, 10]),
            _ann(2, 1, 1, [10, 10, 0, 5]),
            _ann(3, 1, 1, [10, 10, 5, -2]),
        ],
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    source = analysis.sources[0]
    assert source.report.skipped == {"degenerate_box": 3}
    assert len(source.images) == 1
    assert source.images[0].boxes == ()


def test_import_reports_bad_boxes(coco: Path) -> None:
    _coco(
        coco,
        "annotations/instances_val2017.json",
        [_img(1, "one.png")],
        [
            _ann(1, 1, 1, [1, 2, 3]),
            _ann(2, 1, 1, ["a", 0, 1, 1]),
        ],
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    assert analysis.sources[0].report.skipped == {"bad_bbox": 2}


def test_import_excludes_missing_and_unreadable_images(coco: Path) -> None:
    (coco / "val2017" / "broken.png").write_bytes(b"nope")
    _coco(
        coco,
        "annotations/instances_val2017.json",
        [_img(1, "one.png"), _img(2, "ghost.png"), _img(3, "broken.png")],
        [_ann(1, 2, 1, [0, 0, 5, 5])],
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    source = analysis.sources[0]
    assert source.report.skipped == {"missing_image": 1, "unreadable_image": 1}
    assert [i.file_name for i in source.images] == ["one.png"]
    assert {e["item"] for e in source.report.listed} == {"ghost.png", "broken.png"}


def test_import_refuses_file_names_that_leave_the_folder(coco: Path) -> None:
    _coco(
        coco,
        "annotations/instances_val2017.json",
        [_img(1, "one.png"), _img(2, "../one.png"), _img(3, "/etc/passwd")],
        [],
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    assert analysis.sources[0].report.skipped == {"unsafe_path": 2}


def test_import_uses_the_real_size_when_the_json_disagrees(coco: Path) -> None:
    _coco(
        coco,
        "annotations/instances_val2017.json",
        [_img(1, "one.png", 640, 480)],
        [_ann(1, 1, 1, [10, 5, 20, 10])],
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    source = analysis.sources[0]
    assert source.report.adjusted == {"size_mismatch": 1}
    # Normalized by the real 100x50, not by 640x480.
    assert source.images[0].boxes[0] == pytest.approx((1, 0.2, 0.2, 0.2, 0.2))


def test_import_skips_annotations_of_unknown_images_and_categories(coco: Path) -> None:
    _coco(
        coco,
        "annotations/instances_val2017.json",
        [_img(1, "one.png")],
        [
            _ann(1, 99, 1, [0, 0, 5, 5]),
            _ann(2, 1, 77, [0, 0, 5, 5]),
            _ann(3, 1, 1, [0, 0, 5, 5]),
        ],
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    source = analysis.sources[0]
    assert source.report.skipped == {"unknown_image_id": 1, "unknown_category": 1}
    assert source.boxes == 1


def test_import_keeps_duplicate_category_names_and_suffixes_the_later(
    tmp_path: Path,
) -> None:
    root = tmp_path / "c"
    _png(root / "train" / "a.png")
    cats = [
        {"id": 2, "name": "thing"},
        {"id": 5, "name": "thing"},
        {"id": 7, "name": "thing"},
    ]
    _coco(root, "train.json", [_img(1, "a.png")], [], cats)
    analysis = analyze_coco_import(root)
    assert analysis is not None
    plan = plan_import(analysis)
    assert plan.classes == ["thing", "thing_5", "thing_7"]
    assert [m["coco_id"] for m in plan.summary()["renamed"]] == [5, 7]


def test_import_unknown_json_name_needs_a_split(coco: Path) -> None:
    (coco / "annotations" / "instances_val2017.json").rename(
        coco / "annotations" / "stuff.json"
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    plan = plan_import(analysis)
    assert plan.ready is False
    assert plan.unassigned == ["annotations/stuff.json"]
    with pytest.raises(CocoError) as err:
        build_coco_import(analysis, plan, "cocoset-yolo")
    assert err.value.code == "coco.unassigned_split"
    assigned = plan_import(analysis, {"annotations/stuff.json": "test"})
    assert assigned.ready is True
    assert assigned.summary()["per_split"]["test"]["images"] == 2


def test_import_can_skip_a_json(coco: Path) -> None:
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    plan = plan_import(analysis, {"annotations/instances_val2017.json": SKIP})
    assert plan.summary()["sources"][0]["status"] == "skipped"
    with pytest.raises(CocoError) as err:
        build_coco_import(analysis, plan, "cocoset-yolo")
    assert err.value.code == "coco.nothing_selected"


def test_import_refuses_two_jsons_on_the_same_split(coco: Path) -> None:
    _coco(coco, "annotations/val.json", [_img(1, "one.png")], [])
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    with pytest.raises(CocoError) as err:
        plan_import(analysis)
    assert err.value.code == "coco.split_taken"


def test_import_refuses_an_unknown_split_or_json(coco: Path) -> None:
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    with pytest.raises(CocoError) as bad_split:
        plan_import(analysis, {"annotations/instances_val2017.json": "dev"})
    assert bad_split.value.code == "coco.bad_split"
    with pytest.raises(CocoError) as bad_json:
        plan_import(analysis, {"other.json": "val"})
    assert bad_json.value.code == "coco.unknown_json"


def test_import_refuses_one_category_id_with_two_names(coco: Path) -> None:
    _png(coco / "train2017" / "t.png")
    _coco(
        coco,
        "annotations/instances_train2017.json",
        [_img(1, "t.png")],
        [],
        [{"id": 1, "name": "bird"}],
    )
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    with pytest.raises(CocoError) as err:
        plan_import(analysis)
    assert err.value.code == "coco.category_conflict"


def test_import_finds_images_in_the_other_layouts(tmp_path: Path) -> None:
    for layout, rel in (("images/train", "x.png"), ("images", "x.png"), (".", "x.png")):
        root = tmp_path / f"c-{layout.replace('/', '-').replace('.', 'root')}"
        _png(root / layout / rel)
        _coco(root, "train.json", [_img(1, rel)], [])
        analysis = analyze_coco_import(root)
        assert analysis is not None
        assert len(analysis.sources[0].images) == 1, layout


def test_import_reports_when_no_images_folder_is_found(tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    _coco(root, "train.json", [_img(1, "gone.png")], [])
    analysis = analyze_coco_import(root)
    assert analysis is not None
    assert analysis.sources[0].images_dir is None
    assert analysis.sources[0].report.skipped == {"missing_image": 1}


def test_import_ignores_json_files_that_are_not_coco(coco: Path) -> None:
    (coco / "notes.json").write_text('{"hello": 1}', encoding="utf-8")
    (coco / "broken.json").write_text("{nope", encoding="utf-8")
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    assert analysis.ignored == [
        {"name": "broken.json", "reason": "not_json"},
        {"name": "notes.json", "reason": "not_coco"},
    ]


def test_import_without_any_annotation_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(CocoError) as err:
        analyze_coco_import(tmp_path / "empty")
    assert err.value.code == "coco.no_annotations"


def test_import_leaves_the_original_untouched(coco: Path) -> None:
    before = _tree(coco)
    _import(coco)
    assert _tree(coco) == before


def test_import_cancel_and_error_leave_nothing_behind(coco: Path) -> None:
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    plan = plan_import(analysis)
    token = CancellationToken()
    token.cancel()
    assert build_coco_import(analysis, plan, "cocoset-yolo", token=token) is None

    def boom(src: Path, dst: Path) -> None:
        raise RuntimeError("disk on fire")

    with pytest.raises(RuntimeError, match="disk on fire"):
        build_coco_import(analysis, plan, "cocoset-yolo", link=boom)
    assert sorted(p.name for p in coco.parent.iterdir()) == ["cocoset"]


def test_import_refuses_when_the_source_changed(coco: Path) -> None:
    analysis = analyze_coco_import(coco)
    assert analysis is not None
    plan = plan_import(analysis)
    _png(coco / "val2017" / "late.png")
    with pytest.raises(CocoChangedError):
        build_coco_import(analysis, plan, "cocoset-yolo")
    assert sorted(p.name for p in coco.parent.iterdir()) == ["cocoset"]


def test_import_flattens_subfolders_without_overwriting(tmp_path: Path) -> None:
    root = tmp_path / "c"
    _png(root / "train" / "a" / "x.png")
    _png(root / "train" / "b" / "x.png")
    _coco(
        root,
        "train.json",
        [_img(1, "a/x.png"), _img(2, "b/x.png")],
        [_ann(1, 1, 1, [0, 0, 10, 10]), _ann(2, 2, 2, [0, 0, 20, 10])],
    )
    _import(root, name="c-yolo")
    out = tmp_path / "c-yolo"
    names = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert len(names) == 2
    assert len({Path(n).stem for n in names}) == 2
    for name in names:
        assert (out / "labels" / "train" / f"{Path(name).stem}.txt").read_text() != ""


def test_import_registry_records_source_mapping_and_everything_skipped(
    coco: Path,
) -> None:
    _coco(
        coco,
        "annotations/instances_val2017.json",
        [_img(1, "one.png"), _img(2, "ghost.png")],
        [_ann(1, 1, 1, [10, 5, 20, 10], iscrowd=1), _ann(2, 1, 2, [0, 0, 10, 10])],
        [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
    )
    _import(coco)
    registry = _read(coco.parent / "cocoset-yolo" / "visionforge_dataset.json")
    assert registry["schema_version"] == 1
    assert registry["kind"] == "coco_import"
    assert registry["derived_from"]["path"] == str(coco.resolve())
    assert registry["derived_from"]["fingerprint"]["digest"]
    assert registry["segmentation"] == "ignored"
    assert registry["sources"] == [
        {
            "json": "annotations/instances_val2017.json",
            "split": "val",
            "images_dir": "val2017",
            "images_listed": 2,
            "images_kept": 1,
            "annotations": 2,
            "boxes": 1,
        }
    ]
    assert registry["counts"]["skipped_by_reason"] == {"crowd": 1, "missing_image": 1}
    assert [m["name"] for m in registry["mapping"]] == ["cat", "dog"]


# ── round trip ────────────────────────────────────────────────────────────────


def _boxes(root: Path, split: str, stem: str) -> list[list[float]]:
    text = (root / "labels" / split / f"{stem}.txt").read_text()
    return [[float(p) for p in line.split()] for line in text.splitlines()]


def test_round_trip_keeps_class_names_and_boxes(yolo: Path) -> None:
    _export(yolo)
    out = yolo.parent / "ds-coco"
    _import(out, name="ds-coco-yolo")
    back = yolo.parent / "ds-coco-yolo"
    assert (back / "classes.txt").read_text() == (yolo / "classes.txt").read_text()
    for split, stem in (("train", "a"), ("train", "bg"), ("val", "c")):
        original = _boxes(yolo, split, stem)
        again = _boxes(back, split, stem)
        assert len(again) == len(original)
        for left, right in zip(again, original, strict=True):
            assert left == pytest.approx(right, abs=1e-5)
