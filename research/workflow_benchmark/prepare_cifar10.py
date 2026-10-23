"""Prepare an auditable CPU pilot from a locally supplied official CIFAR-10 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SOURCE_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz"
SOURCE_PAGE = "https://www.cs.toronto.edu/~kriz/cifar.html"
SOURCE_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _decode_batch(payload: bytes, *, expected_rows: int = 10_000) -> np.ndarray:
    if len(payload) != expected_rows * 3073:
        raise ValueError("CIFAR-10 binary batch has an invalid record count")
    rows = np.frombuffer(payload, dtype=np.uint8).reshape(expected_rows, 3073)
    if np.any(rows[:, 0] >= len(CLASSES)):
        raise ValueError("CIFAR-10 binary batch contains an invalid class label")
    return rows


def _materialize_subset(
    training: np.ndarray,
    testing: np.ndarray,
    output: Path,
    provenance: dict[str, Any],
    *,
    seed: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if seed < 0 or min(train_per_class, val_per_class, test_per_class) < 1:
        raise ValueError(
            "Seed must be nonnegative and each split needs positive counts"
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    selected: list[tuple[str, str, int]] = []
    for label in range(len(CLASSES)):
        train_indices = np.flatnonzero(training[:, 0] == label)
        test_indices = np.flatnonzero(testing[:, 0] == label)
        if len(train_indices) < train_per_class + val_per_class:
            raise ValueError(
                f"Insufficient official training records for class {label}"
            )
        if len(test_indices) < test_per_class:
            raise ValueError(f"Insufficient official test records for class {label}")
        train_indices = generator.permutation(train_indices)
        test_indices = generator.permutation(test_indices)
        for split, indices, origin in (
            ("train", train_indices[:train_per_class], "train"),
            (
                "val",
                train_indices[train_per_class : train_per_class + val_per_class],
                "train",
            ),
            ("test", test_indices[:test_per_class], "test"),
        ):
            selected.extend((split, origin, int(index)) for index in indices)

    # Only create output after every class has enough records for all three splits.
    output.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    for split, origin, index in selected:
        row = (training if origin == "train" else testing)[index]
        label = int(row[0])
        relative = Path(split) / CLASSES[label] / f"{origin}_{index:05d}.png"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pixels = row[1:].reshape(3, 32, 32).transpose(1, 2, 0)
        Image.fromarray(pixels).save(destination)
        samples.append(
            {
                "split": split,
                "source_split": origin,
                "source_index": index,
                "label": label,
                "path": relative.as_posix(),
                "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "png_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    report = {
        "schema_version": 1,
        "purpose": "engineering_parity_only",
        "source": provenance,
        "selection": {"seed": seed, "generator": "PCG64", "numpy": np.__version__},
        "class_to_idx": {name: index for index, name in enumerate(CLASSES)},
        "counts": {
            "train": 10 * train_per_class,
            "val": 10 * val_per_class,
            "test": 10 * test_per_class,
        },
        "samples": samples,
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def prepare_cifar10(
    archive: Path,
    output: Path,
    *,
    seed: int = 37,
    train_per_class: int = 32,
    val_per_class: int = 8,
    test_per_class: int = 8,
) -> dict[str, Any]:
    """Verify the official archive and preserve source indices in ImageFolder splits."""
    archive, output = archive.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if archive.is_relative_to(output):
        raise ValueError("Archive must be outside the output directory")
    md5, sha256 = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    if md5.hexdigest() != SOURCE_MD5:
        raise ValueError("Official CIFAR-10 binary archive checksum mismatch")
    names = [f"data_batch_{index}.bin" for index in range(1, 6)] + ["test_batch.bin"]
    batches = []
    with tarfile.open(archive, "r:gz") as source:
        for name in names:
            member = source.getmember(f"cifar-10-batches-bin/{name}")
            if not member.isfile() or member.size != 10_000 * 3073:
                raise ValueError(f"Invalid official batch: {name}")
            batch_stream = source.extractfile(member)
            if batch_stream is None:
                raise ValueError(f"Unreadable official batch: {name}")
            with batch_stream:
                batches.append(_decode_batch(batch_stream.read()))
    training, testing = np.concatenate(batches[:5]), batches[5]
    return _materialize_subset(
        training,
        testing,
        output,
        {
            "dataset": "CIFAR-10 binary",
            "page": SOURCE_PAGE,
            "url": SOURCE_URL,
            "archive_md5": md5.hexdigest(),
            "archive_sha256": sha256.hexdigest(),
            "train_index": "data_batch_1..5 concatenated, zero-based",
            "test_index": "test_batch, zero-based",
        },
        seed=seed,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        test_per_class=test_per_class,
    )


def main() -> None:
    """Prepare local records without downloading data, weights or importing torch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--train-per-class", type=int, default=32)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--test-per-class", type=int, default=8)
    args = parser.parse_args()
    report = prepare_cifar10(**vars(args))
    print(
        json.dumps(
            {"manifest": str(args.output / "manifest.json"), "counts": report["counts"]}
        )
    )


if __name__ == "__main__":
    main()
