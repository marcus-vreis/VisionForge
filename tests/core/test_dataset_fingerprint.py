"""A fingerprint that over-promises is worse than none: these pin what each
method does and does not detect, and that it can never break a training run.
"""

from __future__ import annotations

from pathlib import Path

from visionforge.core.dataset_fingerprint import (
    fingerprint_dataset,
    fingerprint_from_config,
    same_dataset,
)


def _dataset(root: Path, contents: dict[str, str]) -> Path:
    for relative, text in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class TestManifestMethod:
    def test_same_data_same_digest(self, tmp_path: Path) -> None:
        files = {"train/a/1.txt": "aa", "train/b/2.txt": "bbb"}
        first = fingerprint_dataset(_dataset(tmp_path / "one", files))
        second = fingerprint_dataset(_dataset(tmp_path / "two", files))
        # The root path differs; the data does not.
        assert first.digest == second.digest
        assert first.root != second.root

    def test_counts_files_and_bytes(self, tmp_path: Path) -> None:
        fp = fingerprint_dataset(_dataset(tmp_path, {"a.txt": "12345", "b.txt": "67"}))
        assert fp.n_files == 2
        assert fp.total_bytes == 7

    def test_added_file_changes_the_digest(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, {"a.txt": "x"})
        before = fingerprint_dataset(root).digest
        (root / "b.txt").write_text("y", encoding="utf-8")
        assert fingerprint_dataset(root).digest != before

    def test_rename_changes_the_digest(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, {"a.txt": "x"})
        before = fingerprint_dataset(root).digest
        (root / "a.txt").rename(root / "renamed.txt")
        assert fingerprint_dataset(root).digest != before

    def test_size_change_changes_the_digest(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, {"a.txt": "x"})
        before = fingerprint_dataset(root).digest
        (root / "a.txt").write_text("xx", encoding="utf-8")
        assert fingerprint_dataset(root).digest != before

    def test_same_size_edit_is_NOT_detected_and_the_note_says_so(
        self, tmp_path: Path
    ) -> None:
        """The honest limitation of the fast method — asserted, not just
        documented, so nobody quietly 'optimizes' the note away."""
        root = _dataset(tmp_path, {"a.txt": "xx"})
        fp = fingerprint_dataset(root)
        (root / "a.txt").write_text("yy", encoding="utf-8")  # same byte count
        assert fingerprint_dataset(root).digest == fp.digest
        assert "preserves file size is not detected" in fp.note

    def test_order_of_creation_does_not_matter(self, tmp_path: Path) -> None:
        a = _dataset(tmp_path / "a", {"z.txt": "1", "a.txt": "2"})
        b = tmp_path / "b"
        (b / "a.txt").parent.mkdir(parents=True)
        (b / "a.txt").write_text("2", encoding="utf-8")
        (b / "z.txt").write_text("1", encoding="utf-8")
        assert fingerprint_dataset(a).digest == fingerprint_dataset(b).digest


class TestContentMethod:
    def test_detects_a_same_size_edit(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, {"a.txt": "xx"})
        before = fingerprint_dataset(root, method="content").digest
        (root / "a.txt").write_text("yy", encoding="utf-8")
        assert fingerprint_dataset(root, method="content").digest != before

    def test_differs_from_the_manifest_digest_of_the_same_data(
        self, tmp_path: Path
    ) -> None:
        root = _dataset(tmp_path, {"a.txt": "xx"})
        assert (
            fingerprint_dataset(root, method="content").digest
            != fingerprint_dataset(root, method="manifest").digest
        )


class TestFailureModes:
    def test_missing_directory_is_reported_not_raised(self, tmp_path: Path) -> None:
        fp = fingerprint_dataset(tmp_path / "nope")
        assert fp.method == "unavailable"
        assert fp.digest == ""
        assert "not a directory" in fp.note

    def test_too_many_files_bails_out_instead_of_stalling(self, tmp_path: Path) -> None:
        _dataset(tmp_path, {f"f{i}.txt": "x" for i in range(5)})
        fp = fingerprint_dataset(tmp_path, max_files=2)
        assert fp.method == "unavailable"
        assert "skipped for speed" in fp.note

    def test_empty_directory_still_produces_a_digest(self, tmp_path: Path) -> None:
        fp = fingerprint_dataset(tmp_path)
        assert fp.method == "manifest"
        assert fp.n_files == 0
        assert fp.digest  # the digest of "no files" is still a claim


class TestFromConfig:
    class _Data:
        def __init__(self, base_dir: Path) -> None:
            self.base_dir = base_dir

    class _Config:
        def __init__(self, base_dir: Path) -> None:
            self.data = TestFromConfig._Data(base_dir)

    def test_reads_data_base_dir(self, tmp_path: Path) -> None:
        _dataset(tmp_path, {"a.txt": "x"})
        entry = fingerprint_from_config(self._Config(tmp_path))
        assert entry["method"] == "manifest"
        assert entry["n_files"] == 1

    def test_a_config_without_data_never_raises(self) -> None:
        entry = fingerprint_from_config(object())
        assert entry["method"] == "unavailable"
        assert "no data.base_dir" in entry["note"]


class TestSameDataset:
    def _fp(self, digest: str, method: str = "manifest") -> dict:
        return {"digest": digest, "method": method}

    def test_equal_digests_are_the_same_data(self) -> None:
        assert same_dataset(self._fp("abc"), self._fp("abc")) is True

    def test_different_digests_are_not(self) -> None:
        assert same_dataset(self._fp("abc"), self._fp("def")) is False

    def test_unanswerable_when_a_digest_is_missing(self) -> None:
        assert same_dataset(self._fp(""), self._fp("abc")) is None

    def test_unanswerable_across_methods(self) -> None:
        # A manifest digest and a content digest of the same data differ;
        # returning False would wrongly claim the datasets differ.
        assert same_dataset(self._fp("abc"), self._fp("abc", "content")) is None
