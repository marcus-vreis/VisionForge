"""Invalid pair inputs must fail before reserving an output directory (ADR-148)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.workflow_benchmark import paired_classification, paired_segmentation


@pytest.mark.parametrize("task", ["classification", "segmentation"])
@pytest.mark.parametrize("fault", ["epochs", "batch_size", "weights"])
def test_pair_preflight_creates_no_output(
    tmp_path: Path, task: str, fault: str
) -> None:
    """Control errors and missing weights do not create a pair or partial report."""
    dataset = tmp_path / "dataset"
    rng = np.random.default_rng(37)
    for split in ("train", "val", "test"):
        for index in range(4):
            if task == "classification":
                folder = dataset / split / f"class{index % 2}"
            else:
                folder = dataset / split / "images"
            folder.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{index}.png")
            if task == "segmentation":
                masks = dataset / split / "masks"
                masks.mkdir(exist_ok=True)
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[:, :16] = 1
                Image.fromarray(mask).save(masks / f"{index}.png")
    initial = tmp_path / "initial.pth"
    if fault != "weights":
        initial.write_bytes(b"preflight must refuse controls before loading weights")
    output = tmp_path / "pair"
    runner = (
        paired_classification.run_pair
        if task == "classification"
        else paired_segmentation.run_pair
    )
    expected_error = FileNotFoundError if fault == "weights" else ValueError
    message = "initial weights" if fault == "weights" else fault
    with pytest.raises(expected_error, match=message):
        runner(
            dataset,
            initial,
            output,
            epochs=0 if fault == "epochs" else 1,
            batch_size=0 if fault == "batch_size" else 2,
        )
    assert not output.exists()
