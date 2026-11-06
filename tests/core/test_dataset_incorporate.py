"""Incorporating new images into a dataset as a new dataset (ADR-166).

Every dataset is a few tiny images built in ``tmp_path``.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_explorer import build_index
from visionforge.core.dataset_incorporate import (
    BAD_LABEL,
    DUPLICATE_EXISTING,
    DUPLICATE_NEW,
    NO_CLASS_FOLDER,
    NO_LABEL,
    REGISTRY_NAME,
    UNKNOWN_CLASS,
    UNREADABLE,
    Analysis,
    DatasetChangedError,
    IncorporateError,
    analyze_incorporation,
    build_incorporated_dataset,
    link_or_copy,
    plan_incorporation,
    suggest_name,
)


def _png(path: Path, seed: int) -> None:
    """A tiny image whose bytes are unique per ``seed``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (seed % 256, (seed // 256) % 256, 7)).save(path)


def _tree_hash(root: Path) -> str:
    """Hash of every path, size and content under ``root``."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*")):
        digest.update(path.relative_to(root).as_posix().encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _classification(root: Path, per_split: tuple[int, int, int] = (14, 4, 2)) -> Path:
    """cat/dog dataset: 14/4/2 images per class in train/val/test."""
    n = 0
    for split, count in zip(("train", "val", "test"), per_split, strict=True):
        for cls in ("cat", "dog"):
            for i in range(count):
                n += 1
                _png(root / split / cls / f"{cls}{i}.png", n)
    return root


def _new_folder(root: Path, per_class: dict[str, int], start: int = 1000) -> Path:
    n = start
    for cls, count in per_class.items():
        for i in range(count):
            n += 1
            _png(root / cls / f"n{i}.png", n)
    return root


def _yolo(root: Path) -> Path:
    n = 0
    for split, count in (("train", 6), ("val", 2), ("test", 2)):
        for i in range(count):
            n += 1
            _png(root / "images" / split / f"i{n}.png", n)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split / f"i{n}.txt").write_text(
                f"{i % 2} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    (root / "classes.txt").write_text("apple\npear\n", encoding="utf-8")
    return root


def _yolo_new(root: Path, rows: dict[str, list[str] | None], start: int = 2000) -> Path:
    """``rows``: image name -> label lines (None = no label file)."""
    for i, (name, lines) in enumerate(rows.items()):
        _png(root / f"{name}.png", start + i)
        if lines is not None:
            (root / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    return root


def _analyze(base: Path, source: Path, task: str = "classification") -> Analysis:
    analysis = analyze_incorporation(base, task, source)  # type: ignore[arg-type]
    assert analysis is not None
    return analysis


def _reasons(analysis: Analysis) -> dict[str, str]:
    return {e.path: e.reason for e in analysis.excluded}


class TestClassificationChecks:
    def test_clean_folder_is_all_candidates(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 3, "dog": 2})
        analysis = _analyze(base, source)
        assert len(analysis.candidates) == 5
        assert analysis.excluded == []
        assert analysis.new_classes == []

    def test_duplicate_of_an_existing_image_is_excluded(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 1})
        # Same bytes as train/cat/cat0.png, under another name and class folder.
        (source / "dog").mkdir()
        (source / "dog" / "copy.png").write_bytes(
            (base / "train/cat/cat0.png").read_bytes()
        )
        analysis = _analyze(base, source)
        assert _reasons(analysis) == {"dog/copy.png": DUPLICATE_EXISTING}
        assert analysis.excluded[0].detail == "train/cat/cat0.png"
        assert len(analysis.candidates) == 1

    def test_duplicate_among_the_new_images_keeps_the_first(
        self, tmp_path: Path
    ) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 1})
        (source / "dog").mkdir()
        (source / "dog" / "same.png").write_bytes((source / "cat/n0.png").read_bytes())
        analysis = _analyze(base, source)
        assert _reasons(analysis) == {"dog/same.png": DUPLICATE_NEW}
        assert [c.rel for c in analysis.candidates] == ["cat/n0.png"]

    def test_corrupted_image_is_excluded(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 1})
        (source / "cat" / "broken.png").write_bytes(b"not an image")
        analysis = _analyze(base, source)
        assert _reasons(analysis) == {"cat/broken.png": UNREADABLE}

    def test_image_outside_a_class_folder_is_excluded(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 1})
        _png(source / "loose.png", 5000)
        assert _reasons(_analyze(base, source)) == {"loose.png": NO_CLASS_FOLDER}

    def test_new_class_is_listed_and_refused_by_default(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 2, "bird": 3})
        analysis = _analyze(base, source)
        assert [(c.name, c.images) for c in analysis.new_classes] == [("bird", 3)]
        plan = plan_incorporation(analysis)
        assert len(plan.assignments) == 2
        assert {e.reason for e in plan.excluded} == {UNKNOWN_CLASS}
        assert sorted(e.path for e in plan.excluded) == [
            "bird/n0.png",
            "bird/n1.png",
            "bird/n2.png",
        ]
        assert plan.classes == ["cat", "dog"]

    def test_new_class_accepted_gets_its_own_folder(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"bird": 10})
        analysis = _analyze(base, source)
        plan = plan_incorporation(analysis, accept_classes=["bird"])
        assert plan.classes == ["cat", "dog", "bird"]
        build_incorporated_dataset(analysis, plan, "ds-v2")
        result = build_index(tmp_path / "ds-v2", "classification")
        assert result.class_names == ["bird", "cat", "dog"]
        assert sum(1 for i in result.images if i.classes == ("bird",)) == 10

    def test_accepting_a_class_that_is_not_new_is_an_error(
        self, tmp_path: Path
    ) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 1}))
        with pytest.raises(IncorporateError) as caught:
            plan_incorporation(analysis, accept_classes=["cat"])
        assert caught.value.code == "incorporate.not_a_new_class"

    def test_source_inside_the_dataset_is_refused(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        with pytest.raises(IncorporateError) as caught:
            analyze_incorporation(base, "classification", base / "train")
        assert caught.value.code == "incorporate.source_overlaps_base"


class TestDetectionChecks:
    def test_label_ids_in_range_are_candidates(self, tmp_path: Path) -> None:
        base = _yolo(tmp_path / "ds")
        source = _yolo_new(
            tmp_path / "new", {"a": ["0 0.5 0.5 0.2 0.2"], "b": ["1 0.1 0.1 0.2 0.2"]}
        )
        analysis = _analyze(base, source, "detection")
        assert len(analysis.candidates) == 2
        assert analysis.new_classes == []

    def test_image_without_label_and_bad_rows_are_excluded(
        self, tmp_path: Path
    ) -> None:
        base = _yolo(tmp_path / "ds")
        source = _yolo_new(
            tmp_path / "new",
            {
                "nolabel": None,
                "short": ["0 0.5 0.5"],
                "negative": ["-1 0.5 0.5 0.2 0.2"],
                "word": ["x 0.5 0.5 0.2 0.2"],
                "ok": ["0 0.5 0.5 0.2 0.2"],
            },
        )
        reasons = _reasons(_analyze(base, source, "detection"))
        assert reasons == {
            "nolabel.png": NO_LABEL,
            "short.png": BAD_LABEL,
            "negative.png": BAD_LABEL,
            "word.png": BAD_LABEL,
        }

    def test_id_beyond_the_class_list_is_a_new_class(self, tmp_path: Path) -> None:
        base = _yolo(tmp_path / "ds")
        source = _yolo_new(
            tmp_path / "new",
            {
                "a": ["0 0.5 0.5 0.2 0.2", "2 0.1 0.1 0.2 0.2"],
                "b": ["1 0.5 0.5 0.2 0.2"],
            },
        )
        analysis = _analyze(base, source, "detection")
        assert [(c.name, c.images, c.source_id) for c in analysis.new_classes] == [
            ("class_2", 1, 2)
        ]
        refused = plan_incorporation(analysis)
        assert [e.reason for e in refused.excluded] == [UNKNOWN_CLASS]
        assert len(refused.assignments) == 1

    def test_accepted_new_id_extends_classes_without_renumbering(
        self, tmp_path: Path
    ) -> None:
        base = _yolo(tmp_path / "ds")
        source = _yolo_new(tmp_path / "new", {"a": ["2 0.5 0.5 0.2 0.2"]})
        analysis = _analyze(base, source, "detection")
        plan = plan_incorporation(analysis, accept_classes=["class_2"])
        assert plan.classes == ["apple", "pear", "class_2"]
        build_incorporated_dataset(analysis, plan, "ds-v2")
        out = tmp_path / "ds-v2"
        assert (out / "classes.txt").read_text(encoding="utf-8").split() == [
            "apple",
            "pear",
            "class_2",
        ]
        # The original classes.txt is untouched (it was rewritten, not linked).
        assert (base / "classes.txt").read_text(encoding="utf-8").split() == [
            "apple",
            "pear",
        ]
        added = [p for p in (out / "labels").rglob("*.txt") if p.name.startswith("a")]
        assert [p.read_text(encoding="utf-8").split()[0] for p in added] == ["2"]

    def test_source_class_list_is_remapped_by_name(self, tmp_path: Path) -> None:
        base = _yolo(tmp_path / "ds")  # apple=0, pear=1
        source = _yolo_new(
            tmp_path / "new",
            {"a": ["0 0.5 0.5 0.2 0.2"], "b": ["1 0.5 0.5 0.2 0.2"]},
        )
        # The new folder numbers its classes the other way around, plus a new one.
        (source / "classes.txt").write_text("pear\napple\nplum\n", encoding="utf-8")
        analysis = _analyze(base, source, "detection")
        assert [c.name for c in analysis.new_classes] == []
        plan = plan_incorporation(analysis)
        build_incorporated_dataset(analysis, plan, "ds-v2")
        by_stem = {
            p.stem: p.read_text(encoding="utf-8").split()[0]
            for p in (tmp_path / "ds-v2/labels").rglob("*.txt")
        }
        assert by_stem["a"] == "1"  # "pear" in the new folder, id 1 in the dataset
        assert by_stem["b"] == "0"

    def test_data_yaml_names_are_extended(self, tmp_path: Path) -> None:
        base = _yolo(tmp_path / "ds")
        (base / "classes.txt").unlink()
        (base / "data.yaml").write_text(
            f"path: {base.as_posix()}\nnc: 2\nnames: [apple, pear]\n", encoding="utf-8"
        )
        source = _yolo_new(tmp_path / "new", {"a": ["2 0.5 0.5 0.2 0.2"]})
        analysis = _analyze(base, source, "detection")
        build_incorporated_dataset(
            analysis, plan_incorporation(analysis, accept_classes=["class_2"]), "ds-v2"
        )
        import yaml

        data = yaml.safe_load(
            (tmp_path / "ds-v2/data.yaml").read_text(encoding="utf-8")
        )
        assert data["names"] == ["apple", "pear", "class_2"]
        assert data["nc"] == 3
        assert Path(data["path"]) == (tmp_path / "ds-v2")
        assert "nc: 2" in (base / "data.yaml").read_text(encoding="utf-8")

    def test_new_labels_land_in_the_split_layout_of_the_dataset(
        self, tmp_path: Path
    ) -> None:
        base = _yolo(tmp_path / "ds")
        source = _yolo_new(
            tmp_path / "new",
            {f"n{i}": ["0 0.5 0.5 0.2 0.2"] for i in range(10)},
        )
        analysis = _analyze(base, source, "detection")
        build_incorporated_dataset(analysis, plan_incorporation(analysis), "ds-v2")
        out = build_index(tmp_path / "ds-v2", "detection")
        assert len(out.images) == 20
        for image in out.images:
            assert image.label_rel is not None
            assert (tmp_path / "ds-v2" / image.label_rel).is_file()


class TestPlan:
    def test_ratio_and_stratification_hold_per_class(self, tmp_path: Path) -> None:
        # Base is 14/4/2 per class = 70% / 20% / 10%.
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 10, "dog": 20})
        analysis = _analyze(base, source)
        plan = plan_incorporation(analysis)
        assert plan.ratios == pytest.approx({"train": 0.7, "val": 0.2, "test": 0.1})
        table = plan.counts()
        assert table["train"] == {"cat": 7, "dog": 14}
        assert table["val"] == {"cat": 2, "dog": 4}
        assert table["test"] == {"cat": 1, "dog": 2}

    def test_custom_ratios(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 10}))
        plan = plan_incorporation(analysis, ratios={"train": 1, "val": 1, "test": 0})
        assert plan.summary()["added_per_split"] == {"train": 5, "val": 5, "test": 0}

    def test_remainders_are_spread_not_dumped_on_one_split(
        self, tmp_path: Path
    ) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 7}))
        per_split = plan_incorporation(analysis).summary()["added_per_split"]
        assert sum(per_split.values()) == 7
        assert per_split == {"train": 5, "val": 1, "test": 1}

    def test_same_seed_same_plan_and_other_seed_differs(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 30, "dog": 30}))

        def assignment(seed: int) -> list[tuple[str, str]]:
            plan = plan_incorporation(analysis, seed=seed)
            return [(c.rel, s) for c, s in plan.assignments]

        first = assignment(3)
        assert assignment(3) == first
        assert assignment(3) != assignment(4)

    def test_same_seed_gives_the_same_dataset_on_disk(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 20, "dog": 20})
        for name in ("one", "two"):
            analysis = _analyze(base, source)
            build_incorporated_dataset(
                analysis, plan_incorporation(analysis, seed=11), name
            )
        assert _files(tmp_path / "one") - {REGISTRY_NAME} == _files(
            tmp_path / "two"
        ) - {REGISTRY_NAME}

    def test_dataset_without_test_never_gets_one(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds", per_split=(8, 2, 0))
        assert not (base / "test").exists()
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 10}))
        plan = plan_incorporation(analysis)
        assert plan.summary()["added_per_split"] == {"train": 8, "val": 2, "test": 0}
        build_incorporated_dataset(analysis, plan, "ds-v2")
        assert not (tmp_path / "ds-v2" / "test").exists()

    def test_ratio_for_a_missing_split_is_refused(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds", per_split=(8, 2, 0))
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 4}))
        with pytest.raises(IncorporateError) as caught:
            plan_incorporation(analysis, ratios={"train": 8, "val": 1, "test": 1})
        assert caught.value.code == "incorporate.ratio_split_absent"

    @pytest.mark.parametrize(
        "ratios", [{"train": -1.0}, {"train": 0.0, "val": 0.0}, {"holdout": 1.0}]
    )
    def test_bad_ratios_are_refused(
        self, tmp_path: Path, ratios: dict[str, float]
    ) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 4}))
        with pytest.raises(IncorporateError) as caught:
            plan_incorporation(analysis, ratios=ratios)
        assert caught.value.code == "incorporate.bad_ratios"


class TestBuild:
    def test_original_is_byte_identical_and_existing_images_stay_put(
        self, tmp_path: Path
    ) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 9, "dog": 9})
        before_tree = _tree_hash(base)
        before_files = _files(base)
        analysis = _analyze(base, source)
        build_incorporated_dataset(analysis, plan_incorporation(analysis), "ds-v2")
        assert _tree_hash(base) == before_tree
        out = tmp_path / "ds-v2"
        # Every original file is in the result at the same path, with the same bytes.
        assert before_files <= _files(out)
        for rel in before_files:
            assert (out / rel).read_bytes() == (base / rel).read_bytes()
        assert len(_files(out)) == len(before_files) + 18 + 1  # + registry

    def test_original_is_untouched_when_classes_are_rewritten(
        self, tmp_path: Path
    ) -> None:
        base = _yolo(tmp_path / "ds")
        source = _yolo_new(tmp_path / "new", {"a": ["5 0.5 0.5 0.2 0.2"]})
        before = _tree_hash(base)
        analysis = _analyze(base, source, "detection")
        plan = plan_incorporation(analysis, accept_classes=["class_5"])
        build_incorporated_dataset(analysis, plan, "ds-v2")
        assert _tree_hash(base) == before
        assert plan.classes[-1] == "class_5"
        assert len(plan.classes) == 6  # the gap is filled with placeholders

    def test_registry_records_origin_fingerprint_and_counts(
        self, tmp_path: Path
    ) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 10, "bird": 2})
        analysis = _analyze(base, source)
        build_incorporated_dataset(
            analysis, plan_incorporation(analysis, seed=42), "ds-v2"
        )
        record = json.loads((tmp_path / "ds-v2" / REGISTRY_NAME).read_text("utf-8"))
        assert record["derived_from"]["path"] == str(base.resolve())
        assert (
            record["derived_from"]["fingerprint"]["digest"]
            == (analysis.base_fingerprint["digest"])
        )
        assert record["incorporated_from"] == str(source.resolve())
        assert record["seed"] == 42
        assert record["ratios"]["train"] == pytest.approx(0.7)
        assert record["counts"]["added"] == 10
        assert record["counts"]["excluded_by_reason"] == {UNKNOWN_CLASS: 2}

    def test_the_explorer_opens_the_result_unchanged(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 10}))
        build_incorporated_dataset(analysis, plan_incorporation(analysis), "ds-v2")
        index = build_index(tmp_path / "ds-v2", "classification")
        assert len(index.images) == 40 + 10
        assert Counter(i.split for i in index.images) == {
            "train": 28 + 7,
            "val": 8 + 2,
            "test": 4 + 1,
        }

    def test_name_clash_keeps_both_files(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds", per_split=(4, 0, 0))
        source = tmp_path / "new"
        _png(source / "cat" / "cat0.png", 9000)  # same name as an existing image
        analysis = _analyze(base, source)
        build_incorporated_dataset(analysis, plan_incorporation(analysis), "ds-v2")
        names = sorted(p.name for p in (tmp_path / "ds-v2/train/cat").iterdir())
        assert len(names) == 5
        assert "cat0.png" in names
        assert sum(n.startswith("cat0-") for n in names) == 1

    @pytest.mark.parametrize("name", ["", "a/b", "..", "con", "x."])
    def test_unsafe_destination_name_is_refused(
        self, tmp_path: Path, name: str
    ) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 2}))
        plan = plan_incorporation(analysis)
        with pytest.raises(IncorporateError) as caught:
            build_incorporated_dataset(analysis, plan, name)
        assert caught.value.code == "incorporate.bad_dest_name"

    def test_existing_destination_is_never_touched(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 2}))
        (tmp_path / "taken").mkdir()
        (tmp_path / "taken" / "keep.txt").write_text("x", encoding="utf-8")
        plan = plan_incorporation(analysis)
        with pytest.raises(IncorporateError) as caught:
            build_incorporated_dataset(analysis, plan, "taken")
        assert caught.value.code == "incorporate.dest_exists"
        assert _files(tmp_path / "taken") == {"keep.txt"}

    def test_dataset_changed_since_the_preview_is_refused(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 2}))
        _png(base / "train/cat/extra.png", 7777)
        plan = plan_incorporation(analysis)
        with pytest.raises(DatasetChangedError) as caught:
            build_incorporated_dataset(analysis, plan, "ds-v2")
        assert caught.value.code == "incorporate.base_changed"
        assert not (tmp_path / "ds-v2").exists()

    def test_new_image_changed_since_the_preview_leaves_nothing(
        self, tmp_path: Path
    ) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 3})
        analysis = _analyze(base, source)
        _png(source / "cat" / "n1.png", 31337)  # replaced after the preview
        plan = plan_incorporation(analysis)
        with pytest.raises(DatasetChangedError):
            build_incorporated_dataset(analysis, plan, "ds-v2")
        assert not (tmp_path / "ds-v2").exists()
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []

    def test_cancelling_the_write_leaves_nothing_at_the_destination(
        self, tmp_path: Path
    ) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 10, "dog": 10}))
        plan = plan_incorporation(analysis)
        token = CancellationToken()

        def stop_midway(done: int, total: int) -> None:
            if done >= 25:  # after some files were already linked
                token.cancel()

        before = _tree_hash(base)
        result = build_incorporated_dataset(
            analysis, plan, "ds-v2", token=token, on_progress=stop_midway
        )
        assert result is None
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ds", "new"]
        assert _tree_hash(base) == before

    def test_cancelling_the_analysis_returns_nothing(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds")
        source = _new_folder(tmp_path / "new", {"cat": 5})
        token = CancellationToken()
        token.cancel()
        assert (
            analyze_incorporation(base, "classification", source, token=token) is None
        )
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ds", "new"]

    def test_failure_while_writing_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _classification(tmp_path / "ds")
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 4}))

        def boom(*_a: object, **_k: object) -> None:
            raise OSError(errno.ENOSPC, "disk full")

        monkeypatch.setattr("visionforge.core.dataset_incorporate.shutil.copy2", boom)
        plan = plan_incorporation(analysis)
        with pytest.raises(OSError):
            build_incorporated_dataset(
                analysis,
                plan,
                "ds-v2",
                link=lambda _s, _d: (_ for _ in ()).throw(OSError(errno.EXDEV, "x")),
            )
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ds", "new"]

    def test_suggest_name_skips_taken(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds", per_split=(1, 0, 0))
        assert suggest_name(base) == "ds-v2"
        (tmp_path / "ds-v2").mkdir()
        assert suggest_name(base) == "ds-v3"


class TestLinkOrCopy:
    def test_same_volume_makes_a_hardlink(self, tmp_path: Path) -> None:
        src = tmp_path / "a.bin"
        src.write_bytes(b"data")
        assert link_or_copy(src, tmp_path / "b.bin") == "link"
        assert os.stat(src).st_nlink == 2
        assert os.path.samefile(src, tmp_path / "b.bin")

    def test_other_volume_falls_back_to_a_copy(self, tmp_path: Path) -> None:
        src = tmp_path / "a.bin"
        src.write_bytes(b"data")

        def cross_device(_s: Path, _d: Path) -> None:
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        assert link_or_copy(src, tmp_path / "b.bin", link=cross_device) == "copy"
        assert (tmp_path / "b.bin").read_bytes() == b"data"
        assert os.stat(src).st_nlink == 1
        assert not os.path.samefile(src, tmp_path / "b.bin")

    def test_unsupported_filesystem_also_copies(self, tmp_path: Path) -> None:
        src = tmp_path / "a.bin"
        src.write_bytes(b"data")

        def denied(_s: Path, _d: Path) -> None:
            raise PermissionError(errno.EPERM, "not permitted")

        assert link_or_copy(src, tmp_path / "b.bin", link=denied) == "copy"

    def test_build_reports_how_many_files_were_copied(self, tmp_path: Path) -> None:
        base = _classification(tmp_path / "ds", per_split=(2, 1, 0))
        analysis = _analyze(base, _new_folder(tmp_path / "new", {"cat": 1}))

        def cross_device(_s: Path, _d: Path) -> None:
            raise OSError(errno.EXDEV, "x")

        result = build_incorporated_dataset(
            analysis, plan_incorporation(analysis), "ds-v2", link=cross_device
        )
        assert result is not None
        assert result["link_methods"] == {"copy": 6}
