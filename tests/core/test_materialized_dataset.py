"""A filtered copy that outlives its run, or loses a label, is worse than none."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.materialized_dataset import (
    MaterializedDataset,
    materialize_dataset,
    sweep_orphans,
)

GRAYSCALE = [{"kind": "grayscale"}]


def _dataset(root: Path) -> Path:
    """A YOLO-shaped dataset: images plus the label files that must survive."""
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (16, 16), (200, 40, 10)).save(
                root / split / "images" / f"img{i}.jpg"
            )
            (root / split / "labels" / f"img{i}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    return root


class TestMaterialize:
    def test_mirrors_the_tree_and_keeps_labels_byte_for_byte(
        self, tmp_path: Path
    ) -> None:
        src = _dataset(tmp_path / "src")

        with materialize_dataset(src, GRAYSCALE, cache_root=tmp_path / "cache") as out:
            for split in ("train", "val"):
                stems = sorted(p.stem for p in (out.path / split / "images").iterdir())
                assert stems == ["img0", "img1"]
                original = (src / split / "labels" / "img0.txt").read_bytes()
                assert (
                    out.path / split / "labels" / "img0.txt"
                ).read_bytes() == original

    def test_the_images_are_actually_filtered(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")

        with materialize_dataset(src, GRAYSCALE, cache_root=tmp_path / "cache") as out:
            with Image.open(out.path / "train" / "images" / "img0.png") as img:
                r, g, b = img.convert("RGB").getpixel((0, 0))  # type: ignore[misc]

        assert r == g == b

    def test_the_stem_survives_the_format_change(self, tmp_path: Path) -> None:
        """YOLO matches a label to its image by stem, not by extension."""
        src = _dataset(tmp_path / "src")

        with materialize_dataset(src, GRAYSCALE, cache_root=tmp_path / "cache") as out:
            images = {p.stem for p in (out.path / "train" / "images").iterdir()}
            labels = {p.stem for p in (out.path / "train" / "labels").iterdir()}

        assert images == labels

    def test_empty_pipeline_returns_the_original_untouched(
        self, tmp_path: Path
    ) -> None:
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, [], cache_root=cache) as out:
            assert out.path == src
            assert out.filtered is False
        assert not cache.exists()

    def test_the_same_dataset_and_pipeline_reuse_one_folder(
        self, tmp_path: Path
    ) -> None:
        """A 20-trial sweep must filter once, not twenty times."""
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, GRAYSCALE, cache_root=cache) as first:
            with materialize_dataset(src, GRAYSCALE, cache_root=cache) as second:
                assert first.path == second.path
                assert len(list(cache.iterdir())) == 1

    def test_a_different_pipeline_gets_its_own_folder(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, GRAYSCALE, cache_root=cache) as a:
            with materialize_dataset(
                src, [{"kind": "equalize"}], cache_root=cache
            ) as b:
                assert a.path != b.path

    def test_the_folder_is_removed_when_the_last_user_leaves(
        self, tmp_path: Path
    ) -> None:
        src = _dataset(tmp_path / "src")

        with materialize_dataset(src, GRAYSCALE, cache_root=tmp_path / "cache") as out:
            path = out.path

        assert not path.exists()

    def test_a_raising_run_does_not_leave_the_copy_behind(self, tmp_path: Path) -> None:
        """Runs die for real — see ADR-081."""
        src = _dataset(tmp_path / "src")
        leaked: Path | None = None

        with pytest.raises(RuntimeError, match="boom"):
            with materialize_dataset(
                src, GRAYSCALE, cache_root=tmp_path / "cache"
            ) as out:
                leaked = out.path
                raise RuntimeError("boom")

        assert leaked is not None
        assert not leaked.exists()

    def test_an_inner_user_does_not_delete_it_under_the_outer_one(
        self, tmp_path: Path
    ) -> None:
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, GRAYSCALE, cache_root=cache) as outer:
            with materialize_dataset(src, GRAYSCALE, cache_root=cache):
                pass
            assert outer.path.exists()

    def test_a_failed_build_leaves_nothing_to_be_reused_as_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A half-written copy would look finished to the next run."""
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        def explode(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(
            "visionforge.core.materialized_dataset._write_filtered", explode
        )

        with pytest.raises(OSError, match="disk full"):
            with materialize_dataset(src, GRAYSCALE, cache_root=cache):
                pass

        assert list(cache.iterdir()) == []


class TestSweepOrphans:
    def test_removes_a_folder_left_by_a_killed_process(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        (cache / "abc123").mkdir(parents=True)

        assert sweep_orphans(cache) == 1
        assert not (cache / "abc123").exists()

    def test_leaves_a_folder_that_is_in_use(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, GRAYSCALE, cache_root=cache) as out:
            assert sweep_orphans(cache) == 0
            assert out.path.exists()

    def test_a_missing_cache_root_is_not_an_error(self, tmp_path: Path) -> None:
        assert sweep_orphans(tmp_path / "nope") == 0


class TestEstimate:
    def test_reports_the_source_size_so_the_caller_can_warn(
        self, tmp_path: Path
    ) -> None:
        src = _dataset(tmp_path / "src")

        assert MaterializedDataset.estimate_bytes(src) > 0
