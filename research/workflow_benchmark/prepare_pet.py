"""Prepare Oxford-IIIT Pet binary segmentation splits from locally supplied archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .trimap_labels import trimap_to_mask

SOURCE_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
LICENSE = "CC BY-SA 4.0; image copyright remains with the original owners"


def _digest(path: Path) -> tuple[str, str]:
    md5, sha256 = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _read(archive: tarfile.TarFile, member: str) -> bytes:
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"{member} is missing from the archive")
    with stream:
        return stream.read()


def _members(
    archive: tarfile.TarFile, wanted: dict[str, str]
) -> Iterator[tuple[tarfile.TarInfo, bytes]]:
    """Stream members once, in archive order; random access on .tar.gz is quadratic."""
    for member in archive:
        if member.name in wanted and member.isfile():
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"{member.name} is unreadable")
            with stream:
                yield member, stream.read()


def _stems(archive: tarfile.TarFile, name: str) -> list[str]:
    text = _read(archive, f"annotations/{name}").decode()
    stems = [line.split()[0] for line in text.splitlines() if line.strip()]
    if len(set(stems)) != len(stems):
        raise ValueError(f"annotations/{name} repeats an image name")
    return stems


def prepare_pet(
    images_archive: Path,
    annotations_archive: Path,
    output: Path,
    *,
    seed: int = 37,
    val_fraction: float = 0.15,
) -> dict[str, Any]:
    """Keep the distributed test split; draw validation only from trainval."""
    from PIL import Image

    images_archive = images_archive.resolve()
    annotations_archive = annotations_archive.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if images_archive.is_relative_to(output) or annotations_archive.is_relative_to(
        output
    ):
        raise ValueError("Archives must be outside the output directory")
    sources = {"images": images_archive, "annotations": annotations_archive}
    digests = {name: _digest(path) for name, path in sources.items()}
    with tarfile.open(annotations_archive, "r:gz") as notes:
        trainval, test = _stems(notes, "trainval.txt"), _stems(notes, "test.txt")
        if set(trainval) & set(test):
            raise ValueError("trainval and test lists overlap")
        wanted = {f"annotations/trimaps/{stem}.png": stem for stem in trainval + test}
        trimaps = {
            wanted[member.name]: data for member, data in _members(notes, wanted)
        }
        if len(trimaps) != len(wanted):
            raise ValueError("trimaps are missing from the annotations archive")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(trainval))
    val_count = max(1, round(len(trainval) * val_fraction))
    val_set = {trainval[i] for i in order[:val_count]}
    split_of = dict.fromkeys(test, "test")
    split_of.update({stem: "val" if stem in val_set else "train" for stem in trainval})
    for split in ("train", "val", "test"):
        (output / split / "images").mkdir(parents=True)
        (output / split / "masks").mkdir(parents=True)
    counts = {"train": 0, "val": 0, "test": 0}
    with tarfile.open(images_archive, "r:gz") as pictures:
        wanted_images = {f"images/{stem}.jpg": stem for stem in split_of}
        found = 0
        for member, data in _members(pictures, wanted_images):
            stem = wanted_images[member.name]
            split = split_of[stem]
            found += 1
            with Image.open(io.BytesIO(trimaps[stem])) as trimap:
                if trimap.mode not in ("L", "P"):
                    raise ValueError(f"trimap {stem} is {trimap.mode}")
                mask = trimap_to_mask(np.array(trimap))
            target = output / split
            (target / "images" / f"{stem}.jpg").write_bytes(data)
            Image.fromarray(mask).save(target / "masks" / f"{stem}.png")
            counts[split] += 1
    if found != len(split_of):
        raise ValueError("images are missing from the images archive")
    report: dict[str, Any] = {
        "dataset": "Oxford-IIIT Pet binary segmentation",
        "page": SOURCE_PAGE,
        "license": LICENSE,
        "archives": {
            name: {
                "file": path.name,
                "md5": digests[name][0],
                "sha256": digests[name][1],
            }
            for name, path in sources.items()
        },
        "mask_map": "trimap 1 pet -> 1, 2 background -> 0, 3 border -> 255",
        "split": "test = distributed test.txt; val drawn from trainval.txt",
        "seed": seed,
        "val_fraction": val_fraction,
        "counts": counts,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Prepare local records without downloading data or importing torch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_archive", type=Path)
    parser.add_argument("annotations_archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    report = prepare_pet(**vars(args))
    print(
        json.dumps(
            {"manifest": str(args.output / "manifest.json"), "counts": report["counts"]}
        )
    )


if __name__ == "__main__":
    main()
