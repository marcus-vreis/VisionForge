from __future__ import annotations

import json
from pathlib import Path

import pytest
from make_subset import make_subset


def _dataset(root: Path, per_class: int = 20) -> Path:
    for split, n in (("train", per_class), ("val", 3), ("test", 4)):
        for cls in ("a", "b"):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{cls}{i:02d}.png").write_bytes(b"x")
    return root


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.rglob("*.png"))


class TestMakeSubset:
    def test_takes_the_fraction_of_every_class(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")

        counts = make_subset(src, tmp_path / "dst", 0.1)

        assert counts == {"a": 2, "b": 2}
        assert len(list((tmp_path / "dst" / "train" / "a").iterdir())) == 2

    def test_val_and_test_are_copied_whole(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")

        make_subset(src, tmp_path / "dst", 0.1)

        for split in ("val", "test"):
            assert _names(tmp_path / "dst" / split) == _names(src / split)

    def test_the_same_seed_draws_the_same_files(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")

        make_subset(src, tmp_path / "one", 0.1, seed=7)
        make_subset(src, tmp_path / "two", 0.1, seed=7)

        assert _names(tmp_path / "one" / "train") == _names(tmp_path / "two" / "train")

    def test_a_tiny_class_keeps_one_file(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src", per_class=3)

        assert make_subset(src, tmp_path / "dst", 0.1) == {"a": 1, "b": 1}

    def test_records_its_provenance(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")

        make_subset(src, tmp_path / "dst", 0.1, seed=7)

        meta = json.loads(
            (tmp_path / "dst" / "subset.json").read_text(encoding="utf-8")
        )
        assert meta["seed"] == 7
        assert meta["fraction"] == 0.1
        assert meta["train_counts"] == {"a": 2, "b": 2}

    def test_refuses_to_overwrite(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")
        (tmp_path / "dst").mkdir()

        with pytest.raises(FileExistsError):
            make_subset(src, tmp_path / "dst", 0.1)

    @pytest.mark.parametrize("fraction", [0.0, 1.5])
    def test_rejects_a_fraction_outside_zero_one(
        self, tmp_path: Path, fraction: float
    ) -> None:
        src = _dataset(tmp_path / "src")

        with pytest.raises(ValueError, match="fraction"):
            make_subset(src, tmp_path / "dst", fraction)
