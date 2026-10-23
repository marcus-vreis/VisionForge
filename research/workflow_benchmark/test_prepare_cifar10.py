"""Check public-data preparation using small, explicitly synthetic records."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.workflow_benchmark.prepare_cifar10 import (
    _decode_batch,
    _materialize_subset,
    prepare_cifar10,
)


def _records(per_class: int, offset: int = 0) -> np.ndarray:
    rows = np.empty((10 * per_class, 3073), dtype=np.uint8)
    for index in range(len(rows)):
        rows[index, 0] = index % 10
        rows[index, 1:] = np.random.default_rng(offset + index).integers(
            0, 256, 3072, dtype=np.uint8
        )
    return rows


def test_binary_layout_preserves_channel_order() -> None:
    rows = _records(1)
    rows[0, 1:1025] = 11
    rows[0, 1025:2049] = 22
    rows[0, 2049:] = 33
    decoded = _decode_batch(rows.tobytes(), expected_rows=10)
    assert decoded.shape == (10, 3073)
    assert decoded[0, 0] == 0
    assert decoded[0, 1:].reshape(3, 32, 32)[:, 0, 0].tolist() == [11, 22, 33]


@pytest.mark.parametrize("payload", [b"", b"\x00" * 3072, b"\x0a" * 3073])
def test_rejects_truncated_or_invalid_records(payload: bytes) -> None:
    with pytest.raises(ValueError):
        _decode_batch(payload, expected_rows=1)


def test_subset_is_repeatable_disjoint_and_matches_saved_pixels(tmp_path: Path) -> None:
    train, test = _records(6), _records(3, offset=1000)
    first, second = tmp_path / "first", tmp_path / "second"
    options = {
        "seed": 37,
        "train_per_class": 3,
        "val_per_class": 2,
        "test_per_class": 2,
    }
    provenance = {"fixture": "synthetic; not official CIFAR-10"}
    a = _materialize_subset(train, test, first, provenance, **options)
    b = _materialize_subset(train, test, second, provenance, **options)
    assert a == b == json.loads((first / "manifest.json").read_text())
    assert a["counts"] == {"train": 30, "val": 20, "test": 20}
    selected_train = set()
    selected_val = set()
    for sample in a["samples"]:
        path = first / sample["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sample["png_sha256"]
        source = train if sample["source_split"] == "train" else test
        row = source[sample["source_index"]]
        assert sample["label"] == int(row[0])
        np.testing.assert_array_equal(
            np.asarray(Image.open(path)), row[1:].reshape(3, 32, 32).transpose(1, 2, 0)
        )
        if sample["split"] == "train":
            selected_train.add(sample["source_index"])
        elif sample["split"] == "val":
            assert sample["source_split"] == "train"
            selected_val.add(sample["source_index"])
        else:
            assert sample["source_split"] == "test"
    assert selected_train.isdisjoint(selected_val)
    assert len(selected_train) == 30 and len(selected_val) == 20
    before = (first / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        _materialize_subset(train, test, first, provenance, **options)
    assert (first / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": -1},
        {"train_per_class": 0},
        {"val_per_class": -1},
        {"test_per_class": 0},
        {"train_per_class": 5, "val_per_class": 2},
        {"test_per_class": 4},
    ],
)
def test_invalid_request_writes_nothing(tmp_path: Path, changes: dict) -> None:
    options = {
        "seed": 37,
        "train_per_class": 3,
        "val_per_class": 2,
        "test_per_class": 2,
    }
    options.update(changes)
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        _materialize_subset(_records(6), _records(3), output, {}, **options)
    assert not output.exists()


def test_unverified_archive_cannot_create_official_provenance(tmp_path: Path) -> None:
    archive = tmp_path / "fake.tar.gz"
    archive.write_bytes(b"not an official dataset")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="checksum"):
        prepare_cifar10(archive, output)
    assert not output.exists()
