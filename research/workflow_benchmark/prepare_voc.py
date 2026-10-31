"""Convert local VOC2007 archives to official YOLO splits without downloads."""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

from PIL import Image

from .prepare_pet import _digest

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
SOURCE_PAGE = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
CONVERSION = "1-based inclusive xyxy -> [xmin-1,ymin-1,xmax,ymax], clamp to size, normalized YOLO cx cy w h (6 decimals); discard degenerate boxes; exclude difficult=1"
PREFIX = "VOCdevkit/VOC2007/"


def _members(path):
    """Read regular members sequentially, never extracting archive paths."""
    with tarfile.open(path, "r|") as archive:
        seen = set()
        for member in archive:
            if not member.isfile() or not member.name.startswith(PREFIX):
                continue
            if member.name in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(member.name)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable member: {member.name}")
            with stream:
                yield member.name[len(PREFIX) :], stream.read()


def _labels(xml, image_size):
    root = ET.fromstring(xml)
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    if (width, height) != image_size or min(width, height) <= 0:
        raise ValueError("Annotation size differs from image size")
    lines, difficult = [], 0
    for obj in root.findall("object"):
        if obj.findtext("difficult", "0") == "1":
            difficult += 1
            continue
        name = obj.findtext("name")
        if name not in CLASSES:
            raise ValueError(f"Unknown VOC class: {name}")
        box = [
            float(obj.findtext(f"bndbox/{key}", "nan"))
            for key in ("xmin", "ymin", "xmax", "ymax")
        ]
        if not all(math.isfinite(value) for value in box):
            raise ValueError("Nonfinite VOC box")
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1 - 1), max(0, y1 - 1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        values = (
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        )
        lines.append(f"{CLASSES.index(name)} " + " ".join(f"{v:.6f}" for v in values))
    return "".join(line + "\n" for line in lines), difficult, len(lines)


def prepare_voc(trainval_archive, test_archive, output):
    """Preserve official train/val/test membership and exclude difficult objects."""
    sources = {
        "trainval": Path(trainval_archive).resolve(),
        "test": Path(test_archive).resolve(),
    }
    output = Path(output).resolve()
    if any(path.is_relative_to(output) for path in sources.values()):
        raise ValueError("Archives must be outside the output directory")
    if output.exists():
        raise FileExistsError(output)
    digests = {key: _digest(path) for key, path in sources.items()}
    lists, annotations = {}, {}
    # Two sequential passes keep only XML and split metadata in memory, not JPEGs.
    for path in sources.values():
        for name, data in _members(path):
            if name in (
                f"ImageSets/Main/{s}.txt" for s in ("train", "val", "trainval", "test")
            ):
                split = PurePosixPath(name).stem
                stems = [
                    line.split()[0]
                    for line in data.decode().splitlines()
                    if line.strip()
                ]
                if len(stems) != len(set(stems)) or any(
                    PurePosixPath(s).name != s or s in (".", "..") or "\\" in s
                    for s in stems
                ):
                    raise ValueError("Invalid or duplicate split image names")
                lists[split] = stems
            elif name.startswith("Annotations/") and name.endswith(".xml"):
                stem = PurePosixPath(name).stem
                if stem in annotations:
                    raise ValueError(f"Duplicate annotation: {stem}")
                annotations[stem] = data
    splits = ("train", "val", "test")
    if any(not lists.get(s) for s in splits):
        raise ValueError("Missing official split list")
    membership = {}
    for split in splits:
        for stem in lists[split]:
            if stem in membership:
                raise ValueError("Official splits overlap")
            membership[stem] = split
    if "trainval" in lists and set(lists["trainval"]) != set(lists["train"]) | set(
        lists["val"]
    ):
        raise ValueError("trainval does not match train and val")
    if not set(membership) <= annotations.keys():
        raise ValueError("Missing annotations")
    counts = {s: {"images": 0, "objects": 0, "difficult_excluded": 0} for s in splits}
    output.mkdir(parents=True, exist_ok=False)
    for split in splits:
        for kind in ("images", "labels"):
            (output / split / kind).mkdir(parents=True)
    found = set()
    for path in sources.values():
        for name, data in _members(path):
            stem = PurePosixPath(name).stem
            if (
                not name.startswith("JPEGImages/")
                or not name.endswith(".jpg")
                or stem not in membership
            ):
                continue
            if stem in found:
                raise ValueError(f"Duplicate image: {stem}")
            found.add(stem)
            with Image.open(io.BytesIO(data)) as image:
                label, excluded, objects = _labels(annotations[stem], image.size)
            split = membership[stem]
            (output / split / "images" / f"{stem}.jpg").write_bytes(data)
            (output / split / "labels" / f"{stem}.txt").write_text(
                label, encoding="utf-8"
            )
            counts[split]["images"] += 1
            counts[split]["objects"] += objects
            counts[split]["difficult_excluded"] += excluded
    if found != membership.keys():
        raise ValueError("Missing images")
    report = {
        "dataset": "Pascal VOC 2007",
        "source_urls": [SOURCE_PAGE],
        "classes": list(CLASSES),
        "conversion": CONVERSION,
        "counts": counts,
        "archives": {
            key: {
                "file": path.name,
                "url": SOURCE_PAGE + path.name,
                "md5": digests[key][0],
                "sha256": digests[key][1],
            }
            for key, path in sources.items()
        },
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    """Convert locally staged uncompressed VOC archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trainval_archive", "test_archive", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_voc(**vars(args))))


if __name__ == "__main__":
    main()
