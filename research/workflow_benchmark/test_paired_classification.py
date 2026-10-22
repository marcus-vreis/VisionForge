"""Synthetic pairs check implementation parity, not scientific performance."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import models

from research.workflow_benchmark.paired_classification import run_pair


@pytest.mark.parametrize("class_count", [2, 3])
def test_native_and_platform_match_with_shared_weights(
    tmp_path: Path, class_count: int
) -> None:
    dataset = tmp_path / "dataset"
    for role, split in enumerate(("train", "val", "test")):
        for label in range(class_count):
            folder = dataset / split / f"class{label}"
            folder.mkdir(parents=True)
            for index in range(2 if class_count == 3 else 3):
                pixels = np.random.default_rng(
                    role * 100 + label * 10 + index
                ).integers(0, 256, (36, 40, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{index}.png")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    initial = tmp_path / "initial.pth"
    torch.save(model.state_dict(), initial)
    output = tmp_path / "pair"

    report = run_pair(dataset, initial, output, seed=37, epochs=2, batch_size=4)

    assert report["status"] == "passed"
    assert report["purpose"] == "engineering_parity_only"
    assert report["mismatches"] == []
    assert report["native"]["history"] == report["visionforge"]["history"]
    assert json.loads((output / "paired_result.json").read_text()) == report
    with pytest.raises(FileExistsError):
        run_pair(dataset, initial, output)
