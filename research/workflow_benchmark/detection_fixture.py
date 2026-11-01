"""Small synthetic YOLO corpus; verifies integration, not detector quality."""

import json

from PIL import Image, ImageDraw

from .native_detection import make_initial_weights


def make_detection_fixture(root, *, split_sizes=None):
    """Create colored rectangles, three classes and shared real SSDLite weights."""
    dataset = root / "dataset"
    for split, count in (split_sizes or {"train": 6, "val": 4, "test": 4}).items():
        for kind in ("images", "labels"):
            (dataset / split / kind).mkdir(parents=True)
        for idx in range(count):
            image = Image.new("RGB", (64, 64), (20 + idx, 30, 40))
            draw = ImageDraw.Draw(image)
            cls = idx % 3
            color = [(230, 40, 40), (40, 230, 40), (40, 40, 230)][cls]
            draw.rectangle((16, 16, 47, 47), fill=color)
            image.save(dataset / split / "images" / f"{idx:03d}.png")
            (dataset / split / "labels" / f"{idx:03d}.txt").write_text(
                f"{cls} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
    (dataset / "manifest.json").write_text(
        json.dumps({"classes": ["red", "green", "blue"], "synthetic": True}),
        encoding="utf-8",
    )
    initial = root / "initial.pth"
    make_initial_weights(initial, 5, num_classes=3)
    return dataset, initial
