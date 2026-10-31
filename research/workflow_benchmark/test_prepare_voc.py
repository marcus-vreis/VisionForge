"""VOC conversion tests use tiny local uncompressed archives."""

import io
import json
import tarfile

import pytest
from PIL import Image

from .prepare_voc import CLASSES, prepare_voc


def _add(archive, name, data):
    member = tarfile.TarInfo("VOCdevkit/VOC2007/" + name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


def _archives(root, *, mismatch=False):
    paths = root / "trainval.tar", root / "test.tar"
    image = io.BytesIO()
    Image.new("RGB", (10, 8)).save(image, "JPEG")
    obj = "<object><name>cat</name><difficult>{}</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>4</ymax></bndbox></object>"
    for path, splits in zip(
        paths,
        ({"train": "a", "val": "b", "trainval": "a\nb"}, {"test": "c"}),
        strict=True,
    ):
        with tarfile.open(path, "w:") as archive:
            # Images precede XML and lists: conversion cannot depend on tar order.
            for stem in ("a", "b") if "train" in splits else ("c",):
                _add(archive, f"JPEGImages/{stem}.jpg", image.getvalue())
                objects = (
                    obj.format(1) if stem == "b" else obj.format(0) + obj.format(1)
                )
                # A reversed box must be discarded after conversion.
                objects += obj.format(0).replace("<xmin>1", "<xmin>9")
                xml = f"<annotation><size><width>{11 if mismatch else 10}</width><height>8</height></size>{objects}</annotation>"
                _add(archive, f"Annotations/{stem}.xml", xml.encode())
            for split, listing in splits.items():
                _add(archive, f"ImageSets/Main/{split}.txt", listing.encode())
    return paths


def test_official_splits_boxes_difficult_and_empty_label(tmp_path, monkeypatch):
    paths = _archives(tmp_path)
    original = tarfile.TarFile.extractfile

    def sequential(self, member):
        assert isinstance(member, tarfile.TarInfo), "Random member-name access"
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", sequential)
    output = tmp_path / "voc"
    report = prepare_voc(*paths, output)
    assert report["classes"] == list(CLASSES)
    assert report["counts"] == {
        "train": {"images": 1, "objects": 1, "difficult_excluded": 1},
        "val": {"images": 1, "objects": 0, "difficult_excluded": 1},
        "test": {"images": 1, "objects": 1, "difficult_excluded": 1},
    }
    assert (
        output / "train/labels/a.txt"
    ).read_text() == "7 0.250000 0.250000 0.500000 0.500000\n"
    assert (output / "val/labels/b.txt").read_bytes() == b""
    assert [p.stem for p in (output / "test/images").iterdir()] == ["c"]
    assert json.loads((output / "manifest.json").read_text()) == report
    assert all(
        len(row["md5"]) == 32 and len(row["sha256"]) == 64
        for row in report["archives"].values()
    )
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_voc(*paths, output)
    assert (output / "manifest.json").read_bytes() == before


def test_archive_inside_output_rejected(tmp_path):
    paths = _archives(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        prepare_voc(*paths, tmp_path)


def test_size_mismatch_rejected(tmp_path):
    paths = _archives(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="size differs"):
        prepare_voc(*paths, tmp_path / "voc")
