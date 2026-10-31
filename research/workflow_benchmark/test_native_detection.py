"""Independent dataset, parameter validation and native phase contract tests."""

import json

import pytest
import torch
from PIL import Image

from . import native_detection as native
from . import phase_contract
from .detection_fixture import make_detection_fixture


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("native_detection")
    return make_detection_fixture(root, split_sizes={"train": 2, "val": 2, "test": 2})


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"momentum": -1}, "momentum"),
        ({"weight_decay": -1}, "weight_decay"),
        ({"seed": -1}, "seed"),
    ],
)
def test_invalid_controls_create_nothing(tmp_path, corpus, changes, match):
    options = {"seed": 37, "epochs": 1, "batch_size": 2}
    options.update(changes)
    with pytest.raises(ValueError, match=match):
        native.run_native(*corpus, tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


def test_dataset_targets_match_torchvision_adapter(corpus):
    from visionforge.core.detection_dataset import DetectionDataset

    dataset, _ = corpus
    actual = native.YoloDataset(dataset / "train")
    reference = DetectionDataset(dataset / "train/images", dataset / "train/labels")
    for idx in range(len(actual)):
        a, at = actual[idx]
        b, bt = reference[idx]
        assert torch.equal(a, b)
        assert at.keys() == bt.keys()
        assert all(torch.equal(at[key], bt[key]) for key in at)
        assert at["boxes"].dtype == torch.float32
        assert at["labels"].dtype == torch.int64


def test_native_history_phases_and_no_overwrite(tmp_path, corpus, monkeypatch):
    iterations = []
    original = native.DataLoader.__iter__

    def tracked(self):
        iterations.append(self.dataset.split.name)
        return original(self)

    monkeypatch.setattr(native.DataLoader, "__iter__", tracked)
    output = tmp_path / "out"
    report = native.run_native(*corpus, output, seed=37, epochs=2, batch_size=2)
    assert iterations == ["train", "val", "val", "train", "val", "val", "test"]
    assert report["status"] == "completed"
    assert [row["epoch"] for row in report["history"]] == [1, 2]
    assert all(
        set(row) == {"epoch", "train_loss", "val_loss", "val_map50"}
        for row in report["history"]
    )
    assert list(report["phases"]) == list(phase_contract.PHASES)
    assert all(
        row["status"] == "completed" and row["seconds"] >= 0 and row["includes"]
        for row in report["phases"].values()
    )
    path = output / "native_result.json"
    assert json.loads(path.read_text()) == report
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        native.run_native(*corpus, output, seed=37, epochs=1, batch_size=2)
    assert path.read_bytes() == before


def test_head_matches_factory_without_downloads():
    from visionforge.models.detection_factory import build_torchvision_detector

    torch.random.default_generator.manual_seed(7)
    a = native.build_detector(3)
    torch.random.default_generator.manual_seed(7)
    b = build_torchvision_detector(native.ARCHITECTURE, 3, False)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(
        torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items()
    )


def test_empty_and_clamped_degenerate_boxes(tmp_path):
    for kind in ("images", "labels"):
        (tmp_path / kind).mkdir()
    for stem in ("a", "b"):
        Image.new("RGB", (10, 8)).save(tmp_path / "images" / f"{stem}.png")
    (tmp_path / "labels/a.txt").write_text("")
    (tmp_path / "labels/b.txt").write_text("2 0.5 0.5 2 2\n1 2 2 0.1 0.1\n")
    ds = native.YoloDataset(tmp_path)
    _, empty = ds[0]
    assert empty["boxes"].shape == (0, 4)
    assert empty["labels"].shape == (0,)
    _, target = ds[1]
    assert target["boxes"].tolist() == [[0, 0, 10, 8]]
    assert target["labels"].tolist() == [3]
