"""Merging two datasets into a new one with a class mapping (ADR-167).

Every dataset is a few tiny images built in ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_derived import REGISTRY_NAME
from visionforge.core.dataset_explorer import ExplorerTask, build_index
from visionforge.core.dataset_merge import (
    ALL_BOXES_DISCARDED,
    DISCARDED_CLASS,
    MergeAnalysis,
    MergeChangedError,
    MergeError,
    analyze_merge,
    build_merged_dataset,
    plan_merge,
    suggest_merge_name,
)


def _png(path: Path, seed: int) -> None:
    """A tiny image whose bytes are unique per ``seed``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (seed % 256, (seed // 256) % 256, 9)).save(path)


def _tree_hash(root: Path) -> str:
    """Hash of every path and content under ``root``."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(path.relative_to(root).as_posix().encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _folder_set(root: Path, per_split: dict[str, dict[str, int]], start: int) -> Path:
    """``{split: {class: count}}`` of unique tiny images."""
    n = start
    for split, classes in per_split.items():
        for cls, count in classes.items():
            for i in range(count):
                n += 1
                _png(root / split / cls / f"{cls}{i}.png", n)
    return root


@pytest.fixture
def ds_a(tmp_path: Path) -> Path:
    """cat/dog, train 3 and val 1 per class, no test split."""
    split = {"cat": 3, "dog": 3}
    val = {"cat": 1, "dog": 1}
    return _folder_set(tmp_path / "A", {"train": split, "val": val}, 0)


@pytest.fixture
def ds_b(tmp_path: Path) -> Path:
    """cat/bird, train 2 and test 1 per class."""
    return _folder_set(
        tmp_path / "B",
        {"train": {"cat": 2, "bird": 2}, "test": {"cat": 1, "bird": 1}},
        1000,
    )


def _analyze(a: Path, b: Path, task: ExplorerTask = "classification") -> MergeAnalysis:
    analysis = analyze_merge(a, task, b)
    assert analysis is not None
    return analysis


def _merge(
    a: Path,
    b: Path,
    mapping: dict[str, dict[str, str | None]] | None = None,
    name: str = "merged",
    task: ExplorerTask = "classification",
) -> Path:
    analysis = _analyze(a, b, task)
    plan = plan_merge(analysis, mapping)
    result = build_merged_dataset(analysis, plan, name)
    assert result is not None
    return Path(result["path"])


def _counts(root: Path) -> dict[str, dict[str, int]]:
    index = build_index(root, "classification")
    out: dict[str, dict[str, int]] = {}
    for image in index.images:
        cls = out.setdefault(image.classes[0], {})
        cls[image.split] = cls.get(image.split, 0) + 1
    return out


class TestClassification:
    def test_same_named_classes_merge_by_default(self, ds_a: Path, ds_b: Path) -> None:
        merged = _merge(ds_a, ds_b)
        assert _counts(merged) == {
            "cat": {"train": 3 + 2, "val": 1, "test": 1},
            "dog": {"train": 3, "val": 1},
            "bird": {"train": 2, "test": 1},
        }

    def test_class_order_is_a_then_new_from_b(self, ds_a: Path, ds_b: Path) -> None:
        plan = plan_merge(_analyze(ds_a, ds_b), None)
        assert plan.classes == ["cat", "dog", "bird"]

    def test_rename_and_merge_by_same_target(self, ds_a: Path, ds_b: Path) -> None:
        mapping: dict[str, dict[str, str | None]] = {
            "a": {"dog": "pet"},
            "b": {"bird": "pet", "cat": "cat"},
        }
        plan = plan_merge(_analyze(ds_a, ds_b), mapping)
        assert plan.classes == ["cat", "pet"]
        merged = _merge(ds_a, ds_b, mapping)
        assert _counts(merged)["pet"] == {"train": 3 + 2, "val": 1, "test": 1}
        assert "dog" not in _counts(merged)

    def test_discard_removes_the_class_and_reports_it(
        self, ds_a: Path, ds_b: Path
    ) -> None:
        mapping: dict[str, dict[str, str | None]] = {"b": {"bird": None}}
        plan = plan_merge(_analyze(ds_a, ds_b), mapping)
        assert plan.classes == ["cat", "dog"]
        assert [d.reason for d in plan.dropped] == [DISCARDED_CLASS] * 3
        assert plan.summary()["dropped_by_reason"] == {DISCARDED_CLASS: 3}
        assert "bird" not in _counts(_merge(ds_a, ds_b, mapping))

    def test_each_image_keeps_its_split_and_splits_are_the_union(
        self, ds_a: Path, ds_b: Path
    ) -> None:
        plan = plan_merge(_analyze(ds_a, ds_b), None)
        summary = plan.summary()
        assert {s: v["present"] for s, v in summary["splits"].items()} == {
            "train": True,
            "val": True,
            "test": True,
        }
        assert summary["only_in"] == {"a": ["val"], "b": ["test"]}
        assert summary["splits"]["test"]["from_a"] == 0
        assert summary["splits"]["test"]["from_b"] == 2
        merged = _merge(ds_a, ds_b)
        assert (merged / "test" / "bird").is_dir()

    def test_a_target_may_not_be_empty_or_a_bad_folder_name(
        self, ds_a: Path, ds_b: Path
    ) -> None:
        analysis = _analyze(ds_a, ds_b)
        with pytest.raises(MergeError) as empty:
            plan_merge(analysis, {"a": {"cat": "   "}})
        assert empty.value.code == "merge.empty_target"
        with pytest.raises(MergeError) as bad:
            plan_merge(analysis, {"a": {"cat": "a/b"}})
        assert bad.value.code == "merge.bad_target"

    def test_targets_that_differ_only_by_case_are_refused(
        self, ds_a: Path, ds_b: Path
    ) -> None:
        analysis = _analyze(ds_a, ds_b)
        with pytest.raises(MergeError) as clash:
            plan_merge(analysis, {"a": {"cat": "Pet", "dog": "pet"}})
        assert clash.value.code == "merge.targets_differ_by_case"

    def test_targets_are_stripped_before_they_are_compared(
        self, ds_a: Path, ds_b: Path
    ) -> None:
        plan = plan_merge(
            _analyze(ds_a, ds_b),
            {"a": {"cat": " pet ", "dog": "pet"}, "b": {"cat": "pet"}},
        )
        assert plan.classes == ["pet", "bird"]

    def test_unknown_class_or_side_is_refused(self, ds_a: Path, ds_b: Path) -> None:
        analysis = _analyze(ds_a, ds_b)
        with pytest.raises(MergeError) as unknown:
            plan_merge(analysis, {"b": {"dog": "x"}})
        assert unknown.value.code == "merge.unknown_class"
        with pytest.raises(MergeError) as side:
            plan_merge(analysis, {"c": {}})
        assert side.value.code == "merge.bad_mapping"

    def test_discarding_everything_leaves_nothing_to_write(
        self, ds_a: Path, ds_b: Path
    ) -> None:
        mapping: dict[str, dict[str, str | None]] = {
            "a": {"cat": None, "dog": None},
            "b": {"cat": None, "bird": None},
        }
        analysis = _analyze(ds_a, ds_b)
        with pytest.raises(MergeError) as empty:
            plan_merge(analysis, mapping)
        assert empty.value.code == "merge.nothing_left"

    def test_names_that_differ_by_case_or_spacing_are_hinted_not_merged(
        self, tmp_path: Path, ds_a: Path
    ) -> None:
        other = _folder_set(tmp_path / "B2", {"train": {"Cat": 1, "big_dog": 1}}, 50)
        analysis = _analyze(ds_a, other)
        assert analysis.similar_names == [["Cat", "cat"]]
        # Distinct targets stay distinct names; on a case-folding file system the
        # two folders would be one, so the plan refuses instead of merging them.
        with pytest.raises(MergeError) as clash:
            plan_merge(analysis, None)
        assert clash.value.code == "merge.targets_differ_by_case"

    def test_filename_collisions_get_the_hash_suffix(self, tmp_path: Path) -> None:
        a = _folder_set(tmp_path / "A", {"train": {"cat": 2}}, 0)
        b = tmp_path / "B"
        _png(b / "train" / "cat" / "cat0.png", 777)  # same name, other bytes
        analysis = _analyze(a, b)
        merged = _merge(a, b)
        sha = analysis.images["b"][0].sha256[:8]
        assert _files(merged / "train" / "cat") == {
            "cat0.png",
            "cat1.png",
            f"cat0-{sha}.png",
        }

    def test_nested_class_subfolders_are_kept_under_the_renamed_class(
        self, tmp_path: Path
    ) -> None:
        a = tmp_path / "A"
        _png(a / "train" / "cat" / "sub" / "x.png", 1)
        _png(a / "train" / "dog" / "y.png", 2)
        b = _folder_set(tmp_path / "B", {"train": {"fox": 1}}, 10)
        merged = _merge(a, b, {"a": {"cat": "feline"}})
        assert "train/feline/sub/x.png" in _files(merged)

    def test_files_that_are_not_dataset_content_are_not_carried(
        self, ds_a: Path, ds_b: Path
    ) -> None:
        (ds_a / "train" / "cat" / "Thumbs.db").write_bytes(b"junk")
        (ds_a / "README.txt").write_text("hi", encoding="utf-8")
        merged = _merge(ds_a, ds_b, {"a": {"cat": "feline"}})
        assert "Thumbs.db" not in {Path(f).name for f in _files(merged)}
        assert "README.txt" not in _files(merged)


class TestDuplicates:
    def _with_shared(self, tmp_path: Path) -> tuple[Path, Path]:
        a = _folder_set(tmp_path / "A", {"train": {"cat": 2}}, 0)
        b = _folder_set(tmp_path / "B", {"val": {"cat": 1}}, 500)
        # B's val image is A's train image: same bytes, other split.
        (b / "val" / "cat" / "cat0.png").write_bytes(
            (a / "train" / "cat" / "cat0.png").read_bytes()
        )
        return a, b

    def test_a_copy_wins_and_the_leak_is_reported(self, tmp_path: Path) -> None:
        a, b = self._with_shared(tmp_path)
        plan = plan_merge(_analyze(a, b), None)
        assert len(plan.duplicates) == 1
        dup = plan.duplicates[0]
        assert (dup.a_split, dup.b_split, dup.cross_split) == ("train", "val", True)
        assert plan.summary()["duplicates_cross_split"] == 1
        merged = _merge(a, b)
        assert _counts(merged) == {"cat": {"train": 2}}
        assert not (merged / "val").exists()

    def test_same_split_duplicate_is_not_flagged_as_a_leak(
        self, tmp_path: Path
    ) -> None:
        a = _folder_set(tmp_path / "A", {"train": {"cat": 2}}, 0)
        b = tmp_path / "B"
        (b / "train" / "cat").mkdir(parents=True)
        (b / "train" / "cat" / "z.png").write_bytes(
            (a / "train" / "cat" / "cat1.png").read_bytes()
        )
        _png(b / "train" / "cat" / "w.png", 99)
        plan = plan_merge(_analyze(a, b), None)
        assert plan.summary()["duplicates_dropped"] == 1
        assert plan.summary()["duplicates_cross_split"] == 0

    def test_duplicates_inside_one_dataset_are_kept_and_counted(
        self, tmp_path: Path
    ) -> None:
        a = _folder_set(tmp_path / "A", {"train": {"cat": 2}}, 0)
        (a / "train" / "cat" / "copy.png").write_bytes(
            (a / "train" / "cat" / "cat0.png").read_bytes()
        )
        b = _folder_set(tmp_path / "B", {"train": {"dog": 1}}, 500)
        analysis = _analyze(a, b)
        assert analysis.repeated_within == {"a": 1, "b": 0}
        assert len(plan_merge(analysis, None).placements) == 4

    def test_b_copy_is_kept_when_a_copy_is_discarded(self, tmp_path: Path) -> None:
        a = _folder_set(tmp_path / "A", {"train": {"cat": 1, "dog": 1}}, 0)
        b = tmp_path / "B"
        (b / "train" / "pet").mkdir(parents=True)
        (b / "train" / "pet" / "x.png").write_bytes(
            (a / "train" / "cat" / "cat0.png").read_bytes()
        )
        plan = plan_merge(_analyze(a, b), {"a": {"cat": None}})
        assert plan.duplicates == []
        assert sorted(p.image.rel for p in plan.placements if p.image.side == "b") == [
            "train/pet/x.png"
        ]


class TestSafety:
    def test_both_sources_are_untouched(
        self, tmp_path: Path, ds_a: Path, ds_b: Path
    ) -> None:
        before = (_tree_hash(ds_a), _tree_hash(ds_b))
        _merge(ds_a, ds_b, {"a": {"cat": "feline"}, "b": {"bird": None}})
        after = (_tree_hash(ds_a), _tree_hash(ds_b))
        assert after == before

    def test_the_result_is_a_sibling_of_a(
        self, tmp_path: Path, ds_a: Path, ds_b: Path
    ) -> None:
        assert _merge(ds_a, ds_b) == (tmp_path / "merged").resolve()

    def test_cancelling_the_write_leaves_nothing(
        self, tmp_path: Path, ds_a: Path, ds_b: Path
    ) -> None:
        analysis = _analyze(ds_a, ds_b)
        plan = plan_merge(analysis, None)
        token = CancellationToken()

        def stop_midway(done: int, _total: int) -> None:
            if done >= 5:
                token.cancel()

        before = sorted(p.name for p in tmp_path.iterdir())
        result = build_merged_dataset(
            analysis, plan, "merged", token=token, on_progress=stop_midway
        )
        assert result is None
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_cancelling_the_analysis_returns_nothing(
        self, tmp_path: Path, ds_a: Path, ds_b: Path
    ) -> None:
        token = CancellationToken()
        token.cancel()
        assert analyze_merge(ds_a, "classification", ds_b, token=token) is None

    def test_an_error_while_writing_leaves_nothing(
        self, tmp_path: Path, ds_a: Path, ds_b: Path
    ) -> None:
        analysis = _analyze(ds_a, ds_b)
        plan = plan_merge(analysis, None)
        calls = 0

        def boom(_src: Path, _dst: Path) -> None:
            nonlocal calls
            calls += 1
            if calls > 3:
                raise RuntimeError("disk full")
            os.link(_src, _dst)

        before = sorted(p.name for p in tmp_path.iterdir())
        with pytest.raises(RuntimeError):
            build_merged_dataset(analysis, plan, "merged", link=boom)
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    @pytest.mark.parametrize("side", ["a", "b"])
    def test_a_source_that_changed_since_the_preview_is_refused(
        self, tmp_path: Path, ds_a: Path, ds_b: Path, side: str
    ) -> None:
        analysis = _analyze(ds_a, ds_b)
        plan = plan_merge(analysis, None)
        _png((ds_a if side == "a" else ds_b) / "train" / "cat" / "late.png", 4242)
        before = sorted(p.name for p in tmp_path.iterdir())
        with pytest.raises(MergeChangedError) as changed:
            build_merged_dataset(analysis, plan, "merged")
        assert changed.value.code == f"merge.{side}_changed"
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_an_image_replaced_since_the_preview_is_refused(
        self, tmp_path: Path, ds_a: Path, ds_b: Path
    ) -> None:
        analysis = _analyze(ds_a, ds_b)
        plan = plan_merge(analysis, None)
        victim = ds_b / "train" / "bird" / "bird0.png"
        victim.write_bytes(victim.read_bytes() + b"\0" * 7)
        with pytest.raises(MergeChangedError):
            build_merged_dataset(analysis, plan, "merged")
        assert not (tmp_path / "merged").exists()

    def test_a_destination_that_exists_or_is_not_a_name_is_refused(
        self, tmp_path: Path, ds_a: Path, ds_b: Path
    ) -> None:
        analysis = _analyze(ds_a, ds_b)
        plan = plan_merge(analysis, None)
        (tmp_path / "taken").mkdir()
        with pytest.raises(MergeError) as exists:
            build_merged_dataset(analysis, plan, "taken")
        assert exists.value.code == "merge.dest_exists"
        with pytest.raises(MergeError) as bad:
            build_merged_dataset(analysis, plan, "../escape")
        assert bad.value.code == "merge.bad_dest_name"

    def test_the_two_datasets_must_be_apart_and_have_images(
        self, tmp_path: Path, ds_a: Path
    ) -> None:
        with pytest.raises(MergeError) as same:
            analyze_merge(ds_a, "classification", ds_a)
        assert same.value.code == "merge.datasets_overlap"
        with pytest.raises(MergeError) as inside:
            analyze_merge(ds_a, "classification", ds_a / "train")
        assert inside.value.code == "merge.datasets_overlap"
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(MergeError) as none:
            analyze_merge(ds_a, "classification", empty)
        assert none.value.code == "merge.empty"
        with pytest.raises(MergeError) as missing:
            analyze_merge(ds_a, "classification", tmp_path / "nope")
        assert missing.value.code == "merge.b_missing"

    def test_suggested_name_is_a_dash_b_and_skips_taken_ones(
        self, tmp_path: Path, ds_a: Path, ds_b: Path
    ) -> None:
        assert suggest_merge_name(ds_a, ds_b) == "A-B"
        (tmp_path / "A-B").mkdir()
        (tmp_path / "A-B-2").mkdir()
        assert suggest_merge_name(ds_a, ds_b) == "A-B-3"


class TestRegistry:
    def test_registry_records_sources_mapping_and_numbers(
        self, ds_a: Path, ds_b: Path
    ) -> None:
        mapping: dict[str, dict[str, str | None]] = {
            "a": {"dog": "pet"},
            "b": {"bird": "pet"},
        }
        merged = _merge(ds_a, ds_b, mapping)
        registry = json.loads((merged / REGISTRY_NAME).read_text(encoding="utf-8"))
        assert registry["schema_version"] == 1
        assert registry["kind"] == "merged"
        assert registry["task"] == "classification"
        assert registry["classes"] == ["cat", "pet"]
        assert registry["mapping"]["a"] == {"cat": "cat", "dog": "pet"}
        assert registry["mapping"]["b"] == {"bird": "pet", "cat": "cat"}
        assert registry["sources"]["a"]["path"] == str(ds_a.resolve())
        assert registry["sources"]["b"]["path"] == str(ds_b.resolve())
        assert registry["sources"]["a"]["fingerprint"]["digest"]
        counts = registry["counts"]
        assert counts["found"] == {"a": 8, "b": 6}
        assert counts["images"] == 14
        assert counts["per_class"]["pet"]["images"] == {"train": 5, "val": 1, "test": 1}
        assert counts["duplicates_dropped"] == 0
        assert registry["files_linked"] + registry["files_copied"] == 14


# ── detection ─────────────────────────────────────────────────────────────────


def _yolo(
    root: Path,
    names: list[str],
    images: dict[str, list[tuple[str, list[int]]]],
    start: int,
    flat: bool = False,
) -> Path:
    """``{split: [(stem, [class ids])]}``; no ids = a background image."""
    n = start
    for split, items in images.items():
        images_dir = root / split / "images" if flat else root / "images" / split
        labels_dir = root / split / "labels" if flat else root / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        for stem, ids in items:
            n += 1
            _png(images_dir / f"{stem}.png", n)
            body = "".join(f"{i} 0.5 0.5 0.2 0.2\n" for i in ids)
            (labels_dir / f"{stem}.txt").write_text(
                body, encoding="utf-8", newline="\n"
            )
    (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def yolo_a(tmp_path: Path) -> Path:
    return _yolo(
        tmp_path / "YA",
        ["apple", "pear"],
        {"train": [("a1", [0]), ("a2", [1]), ("a3", [0, 1])], "val": [("a4", [1])]},
        0,
    )


@pytest.fixture
def yolo_b(tmp_path: Path) -> Path:
    # B numbers pear 0 and kiwi 1: its ids mean something else than A's.
    return _yolo(
        tmp_path / "YB",
        ["pear", "kiwi"],
        {
            "train": [("b1", [0]), ("b2", [1]), ("b3", [1, 0]), ("bg", [])],
            "test": [("b4", [1])],
        },
        900,
    )


def _labels(root: Path, split: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for label in sorted((root / "labels" / split).glob("*.txt")):
        text = label.read_text(encoding="utf-8")
        out[label.stem] = [int(line.split()[0]) for line in text.splitlines() if line]
    return out


class TestDetection:
    def test_ids_are_remapped_to_the_final_order(
        self, yolo_a: Path, yolo_b: Path
    ) -> None:
        merged = _merge(yolo_a, yolo_b, task="detection")
        assert (merged / "classes.txt").read_text(encoding="utf-8") == (
            "apple\npear\nkiwi\n"
        )
        labels = _labels(merged, "train")
        assert labels["a1"] == [0]
        assert labels["a3"] == [0, 1]
        assert labels["b1"] == [1]  # B's pear -> A's pear
        assert labels["b2"] == [2]  # B's kiwi -> new id 2
        assert labels["b3"] == [2, 1]

    def test_label_is_byte_identical_when_no_id_moves(
        self, yolo_a: Path, yolo_b: Path
    ) -> None:
        merged = _merge(yolo_a, yolo_b, task="detection")
        for stem in ("a1", "a2", "a3"):
            same = (merged / "labels" / "train" / f"{stem}.txt").read_bytes()
            assert same == (yolo_a / "labels" / "train" / f"{stem}.txt").read_bytes()

    def test_discarded_boxes_go_and_ids_follow(
        self, yolo_a: Path, yolo_b: Path
    ) -> None:
        merged = _merge(
            yolo_a,
            yolo_b,
            {"a": {"apple": None}, "b": {"kiwi": None}},
            task="detection",
        )
        assert (merged / "classes.txt").read_text(encoding="utf-8") == "pear\n"
        labels = _labels(merged, "train")
        assert labels["a3"] == [0]  # its apple box is gone, the pear box is id 0
        assert labels["b3"] == [0]

    def test_image_that_lost_all_its_boxes_is_dropped_but_background_stays(
        self, yolo_a: Path, yolo_b: Path
    ) -> None:
        mapping: dict[str, dict[str, str | None]] = {
            "a": {"apple": None},
            "b": {"kiwi": None},
        }
        analysis = _analyze(yolo_a, yolo_b, "detection")
        plan = plan_merge(analysis, mapping)
        dropped = {(d.side, Path(d.path).stem): d.reason for d in plan.dropped}
        assert dropped == {
            ("a", "a1"): ALL_BOXES_DISCARDED,
            ("b", "b2"): ALL_BOXES_DISCARDED,
            ("b", "b4"): ALL_BOXES_DISCARDED,
        }
        merged = _merge(yolo_a, yolo_b, mapping, task="detection")
        assert not (merged / "images" / "train" / "a1.png").exists()
        assert (merged / "images" / "train" / "bg.png").is_file()
        assert (merged / "labels" / "train" / "bg.txt").read_text(
            encoding="utf-8"
        ) == ""
        assert plan.summary()["background"]["train"] == 1

    def test_split_only_in_b_uses_the_standard_layout(
        self, yolo_a: Path, yolo_b: Path
    ) -> None:
        merged = _merge(yolo_a, yolo_b, task="detection")
        assert (merged / "images" / "test" / "b4.png").is_file()
        assert (merged / "labels" / "test" / "b4.txt").is_file()

    def test_a_layout_with_split_folders_first_is_followed(
        self, tmp_path: Path, yolo_b: Path
    ) -> None:
        flat = _yolo(
            tmp_path / "Flat",
            ["apple", "pear"],
            {"train": [("f1", [0])], "val": [("f2", [1])]},
            300,
            flat=True,
        )
        merged = _merge(flat, yolo_b, task="detection")
        assert (merged / "train" / "images" / "b1.png").is_file()
        assert (merged / "train" / "labels" / "b1.txt").is_file()
        assert (merged / "images" / "test" / "b4.png").is_file()

    def test_data_yaml_gets_the_new_names_nc_and_path(
        self, tmp_path: Path, yolo_b: Path
    ) -> None:
        import yaml

        a = _yolo(tmp_path / "YA", ["apple", "pear"], {"train": [("a1", [0])]}, 0)
        (a / "classes.txt").unlink()
        (a / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(a.resolve()),
                    "train": "images/train",
                    "names": ["apple", "pear"],
                    "nc": 2,
                }
            ),
            encoding="utf-8",
        )
        merged = _merge(a, yolo_b, task="detection")
        data = yaml.safe_load((merged / "data.yaml").read_text(encoding="utf-8"))
        assert data["names"] == ["apple", "pear", "kiwi"]
        assert data["nc"] == 3
        assert Path(data["path"]) == merged
        assert data["test"] == "images/test"
        assert not (merged / "classes.txt").exists()
        # The source's own file is untouched.
        assert yaml.safe_load((a / "data.yaml").read_text(encoding="utf-8"))["nc"] == 2

    def test_label_stem_collision_renames_image_and_label_together(
        self, tmp_path: Path
    ) -> None:
        a = _yolo(tmp_path / "YA", ["x"], {"train": [("same", [0])]}, 0)
        b = _yolo(tmp_path / "YB", ["x"], {"train": [("same", [0])]}, 800)
        analysis = _analyze(a, b, "detection")
        merged = _merge(a, b, task="detection")
        sha = analysis.images["b"][0].sha256[:8]
        assert _files(merged / "images" / "train") == {"same.png", f"same-{sha}.png"}
        assert _files(merged / "labels" / "train") == {"same.txt", f"same-{sha}.txt"}

    def test_unchanged_labels_are_linked_not_copied_when_possible(
        self, yolo_a: Path, yolo_b: Path
    ) -> None:
        merged = _merge(yolo_a, yolo_b, task="detection")
        original = yolo_a / "labels" / "train" / "a1.txt"
        result = merged / "labels" / "train" / "a1.txt"
        registry = json.loads((merged / REGISTRY_NAME).read_text(encoding="utf-8"))
        if registry["files_linked"]:
            assert os.path.samefile(original, result)
        # A file that changes is never a link to the original.
        rewritten = merged / "labels" / "train" / "b1.txt"
        assert not os.path.samefile(yolo_b / "labels" / "train" / "b1.txt", rewritten)

    def test_both_originals_are_untouched(self, yolo_a: Path, yolo_b: Path) -> None:
        before = (_tree_hash(yolo_a), _tree_hash(yolo_b))
        _merge(yolo_a, yolo_b, {"a": {"apple": "fruit"}}, task="detection")
        after = (_tree_hash(yolo_a), _tree_hash(yolo_b))
        assert after == before

    def test_renamed_class_to_the_same_name_in_both_merges_boxes(
        self, yolo_a: Path, yolo_b: Path
    ) -> None:
        plan = plan_merge(
            _analyze(yolo_a, yolo_b, "detection"), {"b": {"kiwi": "apple"}}
        )
        assert plan.classes == ["apple", "pear"]
        row = next(c for c in plan.summary()["classes"] if c["name"] == "apple")
        assert row["boxes"] == 2 + 3  # A: a1, a3; B kiwi boxes: b2, b3, b4
