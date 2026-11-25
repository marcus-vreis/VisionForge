"""The checks a .zip passes before a byte of it is unpacked on a server (ADR-246)."""

from __future__ import annotations

import io
import stat
import struct
import zipfile
from pathlib import Path

import pytest

from visionforge.gui import upload_archive
from visionforge.gui.upload_archive import ArchiveError, count_entries, extract, inspect

GIB = 1024**3


def _write(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def _refusal(path: Path, max_unpacked: int = GIB) -> str:
    with pytest.raises(ArchiveError) as refused:
        inspect(path, max_unpacked)
    return str(refused.value.key)


def test_a_plain_archive_is_planned_without_writing(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.zip", {"d/a/1.jpg": b"12345", "d/b/2.jpg": b"678"})
    plan = inspect(path, GIB)
    assert (plan.files, plan.unpacked_bytes) == (2, 8)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d.zip"]


def test_the_index_is_counted_before_zipfile_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "d.zip", {f"d/{i}.jpg": b"x" for i in range(5)})
    assert count_entries(path) == 5
    monkeypatch.setattr(upload_archive, "MAX_ENTRIES", 4)
    assert _refusal(path) == "upload.too_many_entries"


def test_an_end_record_that_lies_about_the_count_is_still_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "d.zip", {f"d/{i}.jpg": b"x" for i in range(5)})
    data = bytearray(path.read_bytes())
    end = data.rfind(b"PK\x05\x06")
    struct.pack_into("<2H", data, end + 8, 1, 1)
    path.write_bytes(bytes(data))
    monkeypatch.setattr(upload_archive, "MAX_ENTRIES", 4)
    assert _refusal(path) == "upload.too_many_entries"


def test_an_index_too_large_to_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "d.zip", {f"d/{i}.jpg": b"x" for i in range(5)})
    monkeypatch.setattr(upload_archive, "MAX_INDEX_BYTES", 100)
    assert _refusal(path) == "upload.too_many_entries"


def test_a_zip64_index_is_found(tmp_path: Path) -> None:
    # Past 65,535 entries the counts live in the ZIP64 end record.
    path = tmp_path / "many.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(0x10000):
            archive.writestr(f"d/{index}", b"")
    assert b"PK\x06\x06" in path.read_bytes()[-200:]
    assert count_entries(path) == 0x10000


@pytest.mark.parametrize("junk", [b"", b"not a zip at all", b"PK\x05\x06"])
def test_what_is_not_a_zip_is_refused(tmp_path: Path, junk: bytes) -> None:
    path = tmp_path / "d.zip"
    path.write_bytes(junk)
    assert _refusal(path) == "upload.not_zip"


def test_a_name_used_as_a_file_and_a_folder_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.zip", {"d/a": b"file", "d/a/b.jpg": b"x"})
    assert _refusal(path) == "upload.duplicate"
    path = _write(tmp_path / "e.zip", {"d/a.jpg": b"1", "d/./a.jpg": b"2"})
    assert _refusal(path) == "upload.unsafe_path"


def test_devices_passwords_and_empty_archives_are_refused(tmp_path: Path) -> None:
    fifo = zipfile.ZipInfo("d/pipe")
    fifo.external_attr = (stat.S_IFIFO | 0o644) << 16
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(fifo, b"")
    (tmp_path / "fifo.zip").write_bytes(buffer.getvalue())
    assert _refusal(tmp_path / "fifo.zip") == "upload.special_file"

    # ``writestr`` clears the flags; set "encrypted" in the index afterwards.
    secret = bytearray(_write(tmp_path / "secret.zip", {"d/s.jpg": b"x"}).read_bytes())
    central = secret.find(b"PK\x01\x02")
    struct.pack_into("<H", secret, central + 8, 0x1)
    (tmp_path / "secret.zip").write_bytes(bytes(secret))
    assert _refusal(tmp_path / "secret.zip") == "upload.encrypted"

    folders = _write(tmp_path / "folders.zip", {"d/": b"", "__MACOSX/d/._x": b"x"})
    assert _refusal(folders) == "upload.empty"


def test_the_unpacked_total_has_a_ceiling(tmp_path: Path) -> None:
    path = _write(tmp_path / "d.zip", {"d/a.bin": b"x" * 600, "d/b.bin": b"y" * 600})
    assert inspect(path, 1200).unpacked_bytes == 1200
    assert _refusal(path, 1199) == "upload.too_big_unpacked"


def test_unpacking_keeps_everything_under_the_target(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "d.zip",
        {"train/a/1.jpg": b"1", "val/a/2.jpg": b"22", "train/b/": b""},
    )
    root = extract(path, tmp_path / "out", GIB)
    # More than one top entry: the target itself is the dataset.
    assert root == (tmp_path / "out").resolve()
    assert (root / "val" / "a" / "2.jpg").read_bytes() == b"22"
    assert (root / "train" / "b").is_dir()
