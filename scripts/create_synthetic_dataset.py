"""Script to create a tiny synthetic dataset and run a smoke training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def create_synthetic_dataset(root: Path, n_per_class: int = 20) -> None:
    """Create a minimal ImageFolder dataset with colored noise images."""
    classes = ["class_a", "class_b"]

    rng = np.random.default_rng(42)

    for split in ["train", "val", "test"]:
        for cls in classes:
            folder = root / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                # Each class gets a different dominant color channel for learnability
                arr = rng.integers(0, 64, size=(64, 64, 3), dtype=np.uint8)
                if cls == "class_a":
                    arr[:, :, 0] += 128  # red channel dominant
                else:
                    arr[:, :, 2] += 128  # blue channel dominant
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(folder / f"{i:04d}.png")

    print(f"Dataset created at: {root}")
    for split in ["train", "val", "test"]:
        count = sum(1 for _ in root.glob(f"{split}/**/*.png"))
        print(f"  {split}: {count} images")


if __name__ == "__main__":
    dataset_path = Path("datasets/synthetic_binary")
    create_synthetic_dataset(dataset_path)
    print("\nRun training with:")
    print("  uv run visionforge configs/synthetic_test.yaml")
