"""Re-splitting a dataset into a new one (ADR-168).

Every dataset is a few tiny images built in ``tmp_path``; each image has unique
bytes unless a test copies one on purpose.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_derived import REGISTRY_NAME
from visionforge.core.dataset_explorer import ExplorerTask, build_index
from visionforge.core.dataset_resplit import (
    DUPLICATE_OF_TEST,
    GROUP_IN_TEST,
    Analysis,
    DatasetChangedError,
    Plan,
    ResplitError,
    analyze_resplit,
    build_resplit_dataset,
    plan_resplit,
    suggest_name,
)

Spec = dict[str, dict[str, int]]
NameFn = Callable[[str, int, int], str]


def _png(path: Path, seed: int) -> None:
    """A tiny image whose bytes are unique per ``seed``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (seed % 256, (seed // 256) % 256, 9)).save(path)


def _plain_name(cls: str, i: int, _n: int) -> str:
    return f"{cls}{i}.png"


def _shuffled_name(_cls: str, _i: int, n: int) -> str:
    return f"z{9999 - n}.png"


def _folder_set(
    root: Path, per_split: Spec, name_fn: NameFn = _plain_name, start: int = 0
) -> Path:
    """``{split: {class: count}}`` of unique tiny images in ImageFolder layout."""
    n = start
    for split, classes in per_split.items():
        for cls, count in classes.items():
            for i in range(count):
                n += 1
                _png(root / split / cls / name_fn(cls, i, n), n)
    return root


FULL: Spec = {
    "train": {"cat": 8, "dog": 8},
    "val": {"cat": 2, "dog": 2},
    "test": {"cat": 2, "dog": 2},
}
NO_TEST: Spec = {"train": {"cat": 8, "dog": 8}, "val": {"cat": 2, "dog": 2}}


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    """cat/dog: train 8+8, val 2+2, test 2+2 (24 images)."""
    return _folder_set(tmp_path / "ds", FULL)


@pytest.fixture
def ds_no_test(tmp_path: Path) -> Path:
    return _folder_set(tmp_path / "nt", NO_TEST)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(path.relative_to(root).as_posix().encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _analyze(
    root: Path, task: ExplorerTask = "classification", csv: Path | None = None
) -> Analysis:
    analysis = analyze_resplit(root, task, group_csv=str(csv) if csv else None)
    assert analysis is not None
    return analysis


def _by_sha(plan: Plan, analysis: Analysis) -> dict[str, str]:
    """``{sha256: split}`` of the result, independent of file names."""
    shas = {i.rel: i.sha256 for i in analysis.items}
    return {shas[rel]: split for rel, split in plan.final.items()}


def _counts(plan: Plan, split: str) -> dict[str, int]:
    counts: dict[str, int] = plan.counts(after=True)[split]
    return counts


def _write_csv(path: Path, rows: dict[str, str]) -> Path:
    lines = ["path,group", *(f"{rel},{group}" for rel, group in rows.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _result_counts(dest: Path, task: ExplorerTask = "classification") -> dict[str, int]:
    index = build_index(dest, task)
    return {
        s: sum(1 for i in index.images if i.split == s)
        for s in ("train", "val", "test")
    }


# ── stratified ratios ─────────────────────────────────────────────────────────


def test_ratios_are_respected_per_class(ds: Path) -> None:
    plan = plan_resplit(_analyze(ds), ratios={"train": 3, "val": 2})
    # Each class has 10 pooled images: 6 + 4.
    assert _counts(plan, "train") == {"cat": 6, "dog": 6}
    assert _counts(plan, "val") == {"cat": 4, "dog": 4}


def test_default_ratios_keep_the_current_proportion_of_the_pool(ds: Path) -> None:
    plan = plan_resplit(_analyze(ds))
    assert plan.ratios == {"train": 0.8, "val": 0.2, "test": 0.0}
    assert _counts(plan, "train") == {"cat": 8, "dog": 8}
    assert _counts(plan, "val") == {"cat": 2, "dog": 2}


def test_achieved_ratios_are_reported(ds: Path) -> None:
    plan = plan_resplit(_analyze(ds), ratios={"train": 3, "val": 2})
    summary = plan.summary()
    assert summary["achieved"] == {"train": 0.6, "val": 0.4}
    assert summary["ratios"] == {"train": 0.6, "val": 0.4, "test": 0.0}
    assert summary["moved"] == plan.moved


def test_summary_counts_before_and_after(ds: Path) -> None:
    summary = plan_resplit(_analyze(ds), ratios={"train": 1, "val": 1}).summary()
    assert summary["before_totals"] == {"train": 16, "val": 4, "test": 4}
    assert summary["after_totals"] == {"train": 10, "val": 10, "test": 4}
    assert summary["before"]["train"] == {"cat": 8, "dog": 8}
    assert summary["after"]["val"] == {"cat": 5, "dog": 5}


# ── determinism ───────────────────────────────────────────────────────────────


def test_same_seed_gives_the_same_result_whatever_the_file_order(
    ds: Path, tmp_path: Path
) -> None:
    other = _folder_set(tmp_path / "other", FULL, _shuffled_name)
    first = _analyze(ds)
    second = _analyze(other)
    plan_a = plan_resplit(first, ratios={"train": 1, "val": 1}, seed=7)
    plan_b = plan_resplit(second, ratios={"train": 1, "val": 1}, seed=7)
    assert _by_sha(plan_a, first) == _by_sha(plan_b, second)


def test_a_different_seed_gives_a_different_result(ds: Path) -> None:
    analysis = _analyze(ds)
    one = plan_resplit(analysis, ratios={"train": 1, "val": 1}, seed=1)
    two = plan_resplit(analysis, ratios={"train": 1, "val": 1}, seed=2)
    assert one.final != two.final


# ── the locked test split ─────────────────────────────────────────────────────


def test_test_is_locked_by_default_and_identical_in_the_result(
    ds: Path, tmp_path: Path
) -> None:
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    built = build_resplit_dataset(analysis, plan, "out")
    assert built is not None
    dest = tmp_path / "out"
    test_files = {f for f in _files(ds) if f.startswith("test/")}
    assert {f for f in _files(dest) if f.startswith("test/")} == test_files
    for rel in test_files:
        assert (dest / rel).read_bytes() == (ds / rel).read_bytes()
    assert plan.locked_splits == ["test"]


def test_lock_off_pools_every_split(ds: Path) -> None:
    plan = plan_resplit(
        _analyze(ds), ratios={"train": 1, "val": 1, "test": 1}, lock_test=False
    )
    assert plan.locked_splits == []
    # 12 images per class, 4 each.
    assert _counts(plan, "test") == {"cat": 4, "dog": 4}
    assert _counts(plan, "train") == {"cat": 4, "dog": 4}


def test_a_positive_test_ratio_is_refused_while_the_test_is_locked(ds: Path) -> None:
    analysis = _analyze(ds)
    with pytest.raises(ResplitError) as caught:
        plan_resplit(analysis, ratios={"train": 1, "test": 1})
    assert caught.value.code == "resplit.ratio_test_locked"


def test_lock_is_ignored_when_the_dataset_has_no_test(ds_no_test: Path) -> None:
    plan = plan_resplit(_analyze(ds_no_test))
    assert plan.locked_splits == []


# ── removing and creating splits ──────────────────────────────────────────────


def test_a_ratio_of_zero_removes_the_split(ds: Path, tmp_path: Path) -> None:
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 0})
    codes = [(w["code"], w["split"]) for w in plan.warnings]
    assert ("split_removed", "val") in codes
    built = build_resplit_dataset(analysis, plan, "out")
    assert built is not None
    assert _result_counts(tmp_path / "out") == {"train": 20, "val": 0, "test": 4}
    assert not (tmp_path / "out" / "val").exists()


def test_a_new_split_exists_only_when_asked(ds_no_test: Path, tmp_path: Path) -> None:
    analysis = _analyze(ds_no_test)
    default = plan_resplit(analysis)
    assert default.split_totals(after=True)["test"] == 0
    asked = plan_resplit(analysis, ratios={"train": 8, "val": 1, "test": 1})
    assert ("new_split", "test") in [(w["code"], w["split"]) for w in asked.warnings]
    built = build_resplit_dataset(analysis, asked, "out")
    assert built is not None
    assert _result_counts(tmp_path / "out") == {"train": 16, "val": 2, "test": 2}


def test_a_class_missing_from_a_split_is_warned(tmp_path: Path) -> None:
    root = _folder_set(tmp_path / "tiny", {"train": {"cat": 4, "dog": 1}})
    plan = plan_resplit(_analyze(root), ratios={"train": 1, "val": 1})
    missing = [w for w in plan.warnings if w["code"] == "class_missing"]
    assert len(missing) == 1
    assert missing[0]["class"] == "dog"


def test_a_split_that_gets_no_image_is_warned(tmp_path: Path) -> None:
    root = _folder_set(tmp_path / "one", {"train": {"cat": 1}})
    plan = plan_resplit(_analyze(root), ratios={"train": 1, "val": 1})
    empty = [w for w in plan.warnings if w["code"] == "empty_split"]
    assert len(empty) == 1


# ── bad requests ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": -1.0},
        {"train": 0.0, "val": 0.0},
        {"nope": 1.0},
        {"train": float("nan")},
    ],
)
def test_bad_ratios_are_refused(ds: Path, ratios: dict[str, float]) -> None:
    analysis = _analyze(ds)
    with pytest.raises(ResplitError) as caught:
        plan_resplit(analysis, ratios=ratios)
    assert caught.value.code == "resplit.bad_ratios"


def test_by_group_without_a_file_is_refused(ds: Path) -> None:
    analysis = _analyze(ds)
    with pytest.raises(ResplitError) as caught:
        plan_resplit(analysis, by_group=True)
    assert caught.value.code == "resplit.no_group_csv"


def test_nothing_to_split_when_only_the_locked_test_exists(tmp_path: Path) -> None:
    root = _folder_set(tmp_path / "only", {"test": {"cat": 2}})
    analysis = _analyze(root)
    with pytest.raises(ResplitError) as caught:
        plan_resplit(analysis)
    assert caught.value.code == "resplit.nothing_to_split"


def test_group_file_errors(ds: Path, tmp_path: Path) -> None:
    missing = tmp_path / "none.csv"
    with pytest.raises(ResplitError) as caught:
        analyze_resplit(ds, "classification", group_csv=str(missing))
    assert caught.value.code == "resplit.group_csv_missing"
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("file,team\na,b\n", encoding="utf-8")
    with pytest.raises(ResplitError) as caught_cols:
        analyze_resplit(ds, "classification", group_csv=str(wrong))
    assert caught_cols.value.code == "resplit.bad_group_csv"
    repeated = _write_csv(tmp_path / "rep.csv", {"train/cat/cat0.png": "a"})
    repeated.write_text(
        "path,group\ntrain/cat/cat0.png,a\ntrain/cat/cat0.png,b\n", encoding="utf-8"
    )
    with pytest.raises(ResplitError) as caught_rep:
        analyze_resplit(ds, "classification", group_csv=str(repeated))
    assert caught_rep.value.code == "resplit.bad_group_csv"


# ── exact duplicates ──────────────────────────────────────────────────────────


def test_exact_copies_always_land_in_the_same_split(ds: Path) -> None:
    # Two copies of one image in different pooled splits.
    shutil.copyfile(ds / "train/cat/cat0.png", ds / "val/cat/copy_of_cat0.png")
    analysis = _analyze(ds)
    for seed in range(6):
        plan = plan_resplit(analysis, ratios={"train": 1, "val": 1}, seed=seed)
        assert (
            plan.final["train/cat/cat0.png"] == plan.final["val/cat/copy_of_cat0.png"]
        )
    assert plan.duplicate_sets == 1
    assert plan.duplicate_extras == 1
    assert plan.duplicate_sets_split_before == 1


def test_a_copy_of_a_locked_test_image_is_dropped_and_reported(
    ds: Path, tmp_path: Path
) -> None:
    shutil.copyfile(ds / "test/cat/cat0.png", ds / "train/cat/leak.png")
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    dropped = {d.rel: d.reason for d in plan.dropped}
    assert dropped == {"train/cat/leak.png": DUPLICATE_OF_TEST}
    assert "train/cat/leak.png" not in plan.final
    assert plan.final["test/cat/cat0.png"] == "test"
    built = build_resplit_dataset(analysis, plan, "out")
    assert built is not None
    assert "leak.png" not in {Path(f).name for f in _files(tmp_path / "out")}
    assert plan.summary()["dropped_by_reason"] == {DUPLICATE_OF_TEST: 1}


def test_with_the_lock_off_a_copy_of_a_test_image_is_not_dropped(ds: Path) -> None:
    shutil.copyfile(ds / "test/cat/cat0.png", ds / "train/cat/leak.png")
    plan = plan_resplit(
        _analyze(ds), ratios={"train": 1, "val": 1, "test": 1}, lock_test=False
    )
    assert plan.dropped == []
    assert plan.final["train/cat/leak.png"] == plan.final["test/cat/cat0.png"]


# ── groups ────────────────────────────────────────────────────────────────────


def _groups_csv(tmp_path: Path) -> Path:
    rows = {}
    for i in range(0, 8, 2):
        rows[f"train/cat/cat{i}.png"] = f"g{i}"
        rows[f"train/cat/cat{i + 1}.png"] = f"g{i}"
        rows[f"train/dog/dog{i}.png"] = f"g{i}"
    rows["val/cat/cat0.png"] = "g0"
    return _write_csv(tmp_path / "groups.csv", rows)


def test_groups_stay_whole_and_unlisted_images_are_their_own_group(
    ds: Path, tmp_path: Path
) -> None:
    analysis = _analyze(ds, csv=_groups_csv(tmp_path))
    assert analysis.group_count == 4
    # 24 images, 13 listed (3 per group * 4 + 1), 4 of them... the rest are unlisted.
    assert analysis.ungrouped == 24 - 13
    for seed in range(5):
        plan = plan_resplit(analysis, ratios={"train": 1, "val": 1}, seed=seed)
        for g in range(0, 8, 2):
            members = [
                f"train/cat/cat{g}.png",
                f"train/cat/cat{g + 1}.png",
                f"train/dog/dog{g}.png",
            ]
            if g == 0:
                members.append("val/cat/cat0.png")
            assert len({plan.final[m] for m in members}) == 1


def test_a_group_that_touches_the_locked_test_is_dropped_and_reported(
    ds: Path, tmp_path: Path
) -> None:
    csv_path = _write_csv(
        tmp_path / "g.csv",
        {
            "test/cat/cat0.png": "T",
            "train/cat/cat7.png": "T",
            "val/dog/dog1.png": "T",
        },
    )
    plan = plan_resplit(_analyze(ds, csv=csv_path), ratios={"train": 1, "val": 1})
    dropped = {d.rel: d.reason for d in plan.dropped}
    assert dropped == {
        "train/cat/cat7.png": GROUP_IN_TEST,
        "val/dog/dog1.png": GROUP_IN_TEST,
    }
    assert plan.final["test/cat/cat0.png"] == "test"


def test_by_group_can_be_switched_off_with_a_file_loaded(
    ds: Path, tmp_path: Path
) -> None:
    analysis = _analyze(ds, csv=_groups_csv(tmp_path))
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1}, by_group=False)
    assert plan.by_group is False
    assert _counts(plan, "val") == {"cat": 5, "dog": 5}


def test_a_group_file_naming_unknown_images_counts_them(
    ds: Path, tmp_path: Path
) -> None:
    csv_path = _write_csv(
        tmp_path / "g.csv", {"train/cat/cat0.png": "a", "train/cat/ghost.png": "b"}
    )
    analysis = _analyze(ds, csv=csv_path)
    assert analysis.unknown_entries == 1


def test_group_stratification_keeps_the_ratio_close(tmp_path: Path) -> None:
    root = _folder_set(tmp_path / "big", {"train": {"cat": 20, "dog": 20}})
    # Twenty groups of two: one cat and one dog each.
    rows = {f"train/cat/cat{i}.png": f"g{i}" for i in range(20)}
    rows.update({f"train/dog/dog{i}.png": f"g{i}" for i in range(20)})
    analysis = _analyze(root, csv=_write_csv(tmp_path / "g.csv", rows))
    plan = plan_resplit(analysis, ratios={"train": 3, "val": 1})
    assert plan.split_totals(after=True) == {"train": 30, "val": 10, "test": 0}
    assert _counts(plan, "val") == {"cat": 5, "dog": 5}


# ── writing ───────────────────────────────────────────────────────────────────


def test_the_original_tree_is_never_touched(ds: Path, tmp_path: Path) -> None:
    before = _tree_hash(ds)
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    assert build_resplit_dataset(analysis, plan, "out") is not None
    assert _tree_hash(ds) == before
    assert (tmp_path / "out").is_dir()


def test_the_result_has_the_planned_counts_and_every_image_once(
    ds: Path, tmp_path: Path
) -> None:
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    built = build_resplit_dataset(analysis, plan, "out")
    assert built is not None
    dest = tmp_path / "out"
    assert _result_counts(dest) == plan.split_totals(after=True)
    original = sorted(_sha(ds / f) for f in _files(ds))
    rebuilt = sorted(_sha(dest / f) for f in _files(dest) if f != REGISTRY_NAME)
    assert rebuilt == original
    assert built["images"] == 24


def test_a_moved_file_with_a_taken_name_gets_a_free_one(
    ds: Path, tmp_path: Path
) -> None:
    # ``cat0.png`` exists in train, val and test; sending the pool to val makes the
    # train and val ones meet in val/cat.
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"val": 1})
    assert build_resplit_dataset(analysis, plan, "out") is not None
    names = sorted(p.name for p in (tmp_path / "out" / "val" / "cat").iterdir())
    renamed = [n for n in names if n.startswith("cat0-")]
    assert len(names) == 10
    assert len(renamed) == 1
    assert names.count("cat0.png") == 1


def test_hardlinks_are_used_when_possible_and_copies_otherwise(
    ds: Path, tmp_path: Path
) -> None:
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})

    def refuse(_src: Path, _dst: Path) -> None:
        raise PermissionError(1, "no links here")

    built = build_resplit_dataset(analysis, plan, "out", link=refuse)
    assert built is not None
    assert built["registry"]["files_copied"] > 0
    assert built["registry"]["files_linked"] == 0


def test_the_registry_records_the_decisions(ds: Path, tmp_path: Path) -> None:
    csv_path = _groups_csv(tmp_path)
    analysis = _analyze(ds, csv=csv_path)
    plan = plan_resplit(analysis, ratios={"train": 3, "val": 2}, seed=11)
    assert build_resplit_dataset(analysis, plan, "out") is not None
    registry = json.loads((tmp_path / "out" / REGISTRY_NAME).read_text("utf-8"))
    assert registry["schema_version"] == 1
    assert registry["kind"] == "resplit"
    assert registry["seed"] == 11
    assert registry["derived_from"]["path"] == str(ds.resolve())
    assert registry["derived_from"]["fingerprint"]["digest"]
    assert registry["requested_ratios"] == {"train": 0.6, "val": 0.4, "test": 0.0}
    assert registry["lock_test"] is True
    assert registry["group_manifest"]["path"] == str(csv_path)
    assert registry["group_manifest"]["sha256"] == _sha(csv_path)
    assert registry["counts"]["before"] == {"train": 16, "val": 4, "test": 4}
    assert registry["counts"]["after"] == plan.split_totals(after=True)
    assert registry["counts"]["moved"] == plan.moved
    assert registry["classes"] == ["cat", "dog"]
    assert registry["achieved_ratios"] == plan.achieved()


def test_cancelling_leaves_nothing_behind(ds: Path, tmp_path: Path) -> None:
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    token = CancellationToken()
    seen: list[int] = []

    def progress(done: int, _total: int) -> None:
        seen.append(done)
        if done >= 3:
            token.cancel()

    result = build_resplit_dataset(
        analysis, plan, "out", token=token, on_progress=progress
    )
    assert result is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds"]


def test_an_error_while_writing_leaves_nothing_behind(ds: Path, tmp_path: Path) -> None:
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    calls = 0

    def boom(_src: Path, _dst: Path) -> None:
        nonlocal calls
        calls += 1
        if calls > 4:
            raise RuntimeError("disk on fire")

    with pytest.raises(RuntimeError):
        build_resplit_dataset(analysis, plan, "out", link=boom)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds"]


def test_cancelling_the_analysis_returns_none(ds: Path) -> None:
    token = CancellationToken()
    token.cancel()
    assert analyze_resplit(ds, "classification", token=token) is None


def test_a_changed_dataset_is_refused(ds: Path, tmp_path: Path) -> None:
    analysis = _analyze(ds)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    _png(ds / "train/cat/new.png", 5000)
    with pytest.raises(DatasetChangedError) as caught:
        build_resplit_dataset(analysis, plan, "out")
    assert caught.value.code == "resplit.base_changed"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ds"]


def test_a_changed_group_file_is_refused(ds: Path, tmp_path: Path) -> None:
    csv_path = _groups_csv(tmp_path)
    analysis = _analyze(ds, csv=csv_path)
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    csv_path.write_text(
        csv_path.read_text("utf-8") + "train/dog/dog1.png,zzz\n", "utf-8"
    )
    with pytest.raises(DatasetChangedError) as caught:
        build_resplit_dataset(analysis, plan, "out")
    assert caught.value.code == "resplit.group_csv_changed"


def test_destination_name_rules(ds: Path, tmp_path: Path) -> None:
    analysis = _analyze(ds)
    plan = plan_resplit(analysis)
    with pytest.raises(ResplitError) as bad:
        build_resplit_dataset(analysis, plan, "../escape")
    assert bad.value.code == "resplit.bad_dest_name"
    (tmp_path / "taken").mkdir()
    with pytest.raises(ResplitError) as exists:
        build_resplit_dataset(analysis, plan, "taken")
    assert exists.value.code == "resplit.dest_exists"


def test_suggested_name_counts_up(ds: Path, tmp_path: Path) -> None:
    assert suggest_name(ds) == "ds-resplit"
    (tmp_path / "ds-resplit").mkdir()
    assert suggest_name(ds) == "ds-resplit-2"


# ── detection ─────────────────────────────────────────────────────────────────

Row = tuple[str, str, int, list[int] | None]


def _yolo(root: Path, rows: list[Row], names: list[str]) -> Path:
    """``(split, file name, seed, class ids)``; ``None`` ids means no label file."""
    for split, name, seed, ids in rows:
        _png(root / "images" / split / name, seed)
        if ids is not None:
            lines = [f"{i} 0.5 0.5 0.2 0.2" for i in ids]
            label = root / "labels" / split / f"{Path(name).stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + ("\n" if lines else ""), "utf-8")
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": names,
            }
        ),
        "utf-8",
    )
    return root


def _detection(tmp_path: Path) -> Path:
    rows: list[Row] = []
    seed = 0
    for i in range(6):
        seed += 1
        rows.append(("train", f"a{i}.png", seed, [0, 0, 1]))
    for i in range(6):
        seed += 1
        rows.append(("train", f"b{i}.png", seed, [1]))
    for i in range(2):
        seed += 1
        rows.append(("train", f"bg{i}.png", seed, []))
    rows.append(("train", "bg2.png", 910, []))
    rows.append(("val", "a0.png", 900, [0]))
    rows.append(("val", "a1.png", 904, [0, 1, 0]))
    rows.append(("val", "b0.png", 901, [1]))
    rows.append(("val", "b1.png", 905, [1]))
    rows.append(("val", "nolabel.png", 902, None))
    rows.append(("test", "a0.png", 903, [0]))
    return _yolo(tmp_path / "det", rows, ["alpha", "beta"])


def test_detection_strata_are_the_dominant_class_and_background(tmp_path: Path) -> None:
    root = _detection(tmp_path)
    analysis = _analyze(root, "detection")
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    # Pool: alpha 8, beta 8, background 4 (3 empty labels + 1 without a file).
    assert plan.counts(after=True)["train"] == {
        "(background)": 2,
        "alpha": 4,
        "beta": 4,
    }
    assert plan.counts(after=True)["val"] == {"(background)": 2, "alpha": 4, "beta": 4}


def test_detection_labels_follow_their_images(tmp_path: Path) -> None:
    root = _detection(tmp_path)
    before = _tree_hash(root)
    total = len(build_index(root, "detection").images)
    analysis = _analyze(root, "detection")
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    assert build_resplit_dataset(analysis, plan, "out") is not None
    dest = tmp_path / "out"
    assert _tree_hash(root) == before
    index = build_index(dest, "detection")
    assert len(index.images) == total
    for image in index.images:
        if image.label_rel is not None:
            assert Path(image.label_rel).stem == Path(image.rel).stem
            assert (dest / image.label_rel).is_file()
    test_image = (dest / "images/test/a0.png").read_bytes()
    assert test_image == (root / "images/test/a0.png").read_bytes()
    test_label = (dest / "labels/test/a0.txt").read_text("utf-8")
    assert test_label == (root / "labels/test/a0.txt").read_text("utf-8")


def test_detection_name_collisions_rename_the_label_too(tmp_path: Path) -> None:
    root = _detection(tmp_path)
    analysis = _analyze(root, "detection")
    # Everything lands in train, where ``a0.png`` of train and of val now meet.
    plan = plan_resplit(analysis, ratios={"train": 1})
    assert build_resplit_dataset(analysis, plan, "out") is not None
    dest = tmp_path / "out"
    renamed = list((dest / "images" / "train").glob("a0-*.png"))
    assert len(renamed) == 1
    assert (dest / "labels" / "train" / f"{renamed[0].stem}.txt").is_file()
    assert (dest / "labels" / "train" / "a0.txt").is_file()


def test_detection_labels_are_linked_unchanged(tmp_path: Path) -> None:
    root = _detection(tmp_path)
    analysis = _analyze(root, "detection")
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 1})
    assert build_resplit_dataset(analysis, plan, "out") is not None
    dest = tmp_path / "out"
    originals = sorted((root / "labels").rglob("*.txt"), key=lambda p: p.name)
    originals_text = sorted(p.read_text("utf-8") for p in originals)
    rebuilt_text = sorted(
        p.read_text("utf-8") for p in (dest / "labels").rglob("*.txt")
    )
    assert rebuilt_text == originals_text


def test_detection_yaml_follows_the_new_splits(tmp_path: Path) -> None:
    root = _detection(tmp_path)
    analysis = _analyze(root, "detection")
    # test is locked, so a new split cannot appear; drop val instead.
    plan = plan_resplit(analysis, ratios={"train": 1, "val": 0})
    assert build_resplit_dataset(analysis, plan, "out") is not None
    data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text("utf-8"))
    assert data["path"] == str((tmp_path / "out").resolve()) or data["path"] == str(
        tmp_path / "out"
    )
    assert "val" not in data
    assert data["names"] == ["alpha", "beta"]
    assert not (tmp_path / "out" / "images" / "val").exists()


def test_detection_creates_a_new_split_in_the_standard_layout(tmp_path: Path) -> None:
    rows: list[Row] = []
    rows.extend(("train", f"a{i}.png", i + 1, [0]) for i in range(8))
    rows.extend(("train", f"b{i}.png", i + 100, [1]) for i in range(8))
    root = _yolo(tmp_path / "det2", rows, ["alpha", "beta"])
    analysis = _analyze(root, "detection")
    plan = plan_resplit(analysis, ratios={"train": 3, "val": 1})
    assert build_resplit_dataset(analysis, plan, "out") is not None
    dest = tmp_path / "out"
    assert len(list((dest / "images" / "val").glob("*.png"))) == 4
    assert len(list((dest / "labels" / "val").glob("*.txt"))) == 4
    data = yaml.safe_load((dest / "data.yaml").read_text("utf-8"))
    assert data["val"] == "images/val"


def test_summary_is_json_serialisable(ds: Path) -> None:
    summary: dict[str, Any] = plan_resplit(_analyze(ds)).summary()
    assert json.loads(json.dumps(summary)) == summary
