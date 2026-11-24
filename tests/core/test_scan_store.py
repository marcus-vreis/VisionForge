"""The last scan of each dataset is kept on disk (ADR-231)."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from visionforge.core.dataset_explorer import build_index
from visionforge.core.scan_store import load_scan, save_scan, scan_key, scan_signature

_SPLITS = {"train": "train", "val": "val", "test": "test"}


def _dataset(root: Path) -> Path:
    for rel, color in (("train/cat/a.png", "red"), ("val/dog/b.png", "blue")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color).save(root / rel)
    return root


def _signature(root: Path) -> str:
    return scan_signature(build_index(root, "classification", _SPLITS))


def test_a_saved_report_comes_back_fresh(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "data")
    index = build_index(root, "classification", _SPLITS)
    key = scan_key(index, _SPLITS)

    save_scan(tmp_path / "store", key, scan_signature(index), {"scanned": 2})
    kept = load_scan(tmp_path / "store", key, _signature(root))

    assert kept is not None
    assert kept["report"] == {"scanned": 2}
    assert kept["stale"] is False
    assert list((tmp_path / "store").iterdir()) == [tmp_path / "store" / f"{key}.json"]


def test_the_signature_sees_a_new_a_renamed_a_relabeled_and_an_edited_image(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "data")
    before = _signature(root)

    Image.new("RGB", (8, 8), "green").save(root / "train/cat/c.png")
    added = _signature(root)
    (root / "train/cat/c.png").rename(root / "train/cat/d.png")
    renamed = _signature(root)
    (root / "train/dog").mkdir()
    (root / "train/cat/d.png").rename(root / "train/dog/d.png")
    relabeled = _signature(root)
    Image.new("RGB", (16, 16), "green").save(root / "train/dog/d.png")
    edited = _signature(root)

    assert len({before, added, renamed, relabeled, edited}) == 5


def test_an_untouched_dataset_keeps_its_signature(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "data")
    first = _signature(root)
    stat = os.stat(root / "train/cat/a.png")
    os.utime(root / "train/cat/a.png", ns=(stat.st_atime_ns + 10**9, stat.st_mtime_ns))

    assert _signature(root) == first


def test_the_key_tells_split_layouts_and_tasks_apart(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "data")
    index = build_index(root, "classification", _SPLITS)

    keys = {
        scan_key(index, _SPLITS),
        scan_key(index, {**_SPLITS, "val": "train"}),
        scan_key(build_index(root, "detection", _SPLITS), _SPLITS),
    }

    assert len(keys) == 3


def test_nothing_kept_or_an_unreadable_file_is_no_report(tmp_path: Path) -> None:
    store = tmp_path / "store"
    assert load_scan(store, "absent", "sig") is None
    store.mkdir()
    (store / "broken.json").write_text("{not json", encoding="utf-8")
    (store / "old.json").write_text('{"format": 0}', encoding="utf-8")

    assert load_scan(store, "broken", "sig") is None
    assert load_scan(store, "old", "sig") is None
