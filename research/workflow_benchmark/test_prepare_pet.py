"""Check Pet preparation with small, explicitly synthetic archives."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.workflow_benchmark.prepare_pet import prepare_pet


def _add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _encoded(array: np.ndarray, image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=image_format)
    return buffer.getvalue()


def _archives(root: Path, trainval: int = 10, test: int = 4) -> tuple[Path, Path]:
    train_names = [f"Cat_{i}" for i in range(trainval)]
    test_names = [f"Dog_{i}" for i in range(test)]
    images, notes = root / "images.tar.gz", root / "annotations.tar.gz"
    picture = _encoded(np.zeros((8, 8, 3), dtype=np.uint8), "JPEG")
    trimap = _encoded(np.array([[1, 2], [3, 2]], dtype=np.uint8), "PNG")
    with tarfile.open(images, "w:gz") as archive:
        for stem in train_names + test_names:
            _add(archive, f"images/{stem}.jpg", picture)
    with tarfile.open(notes, "w:gz") as archive:
        listing = "\n".join(f"{name} 1 1 1" for name in train_names)
        _add(archive, "annotations/trainval.txt", listing.encode())
        listing = "\n".join(f"{name} 2 2 1" for name in test_names)
        _add(archive, "annotations/test.txt", listing.encode())
        for stem in train_names + test_names:
            _add(archive, f"annotations/trimaps/{stem}.png", trimap)
    return images, notes


def _stems(directory: Path) -> list[str]:
    return sorted(path.stem for path in directory.iterdir())


def test_splits_keep_distributed_test_and_map_trimaps(tmp_path: Path) -> None:
    images, notes = _archives(tmp_path)
    output = tmp_path / "pet"

    report = prepare_pet(images, notes, output, seed=37, val_fraction=0.3)

    assert report["counts"] == {"train": 7, "val": 3, "test": 4}
    assert _stems(output / "test" / "images") == [f"Dog_{i}" for i in range(4)]
    stem = _stems(output / "train" / "images")[0]
    mask = np.array(Image.open(output / "train" / "masks" / f"{stem}.png"))
    assert mask.tolist() == [[1, 0], [255, 0]]
    assert json.loads((output / "manifest.json").read_text()) == report
    assert len(report["archives"]["images"]["md5"]) == 32


def test_split_is_deterministic_and_output_is_never_overwritten(
    tmp_path: Path,
) -> None:
    images, notes = _archives(tmp_path)
    prepare_pet(images, notes, tmp_path / "a", seed=5, val_fraction=0.3)
    prepare_pet(images, notes, tmp_path / "b", seed=5, val_fraction=0.3)
    assert _stems(tmp_path / "a" / "val" / "images") == _stems(
        tmp_path / "b" / "val" / "images"
    )
    with pytest.raises(FileExistsError):
        prepare_pet(images, notes, tmp_path / "a")


def test_rejects_invalid_validation_fraction(tmp_path: Path) -> None:
    images, notes = _archives(tmp_path)
    with pytest.raises(ValueError, match="val_fraction"):
        prepare_pet(images, notes, tmp_path / "x", val_fraction=1.0)
    assert not (tmp_path / "x").exists()
