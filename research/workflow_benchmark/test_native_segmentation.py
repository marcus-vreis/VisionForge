"""Native segmentation baseline on synthetic CPU fixtures (ADR-149)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from research.workflow_benchmark import native_segmentation as native
from research.workflow_benchmark.paired_segmentation import METRIC_ABS_TOLERANCE
from research.workflow_benchmark.segmentation_fixture import make_segmentation_fixture
from visionforge.core.segmentation_trainer import SegmentationMetricAccumulator


def test_confusion_counts_only_labelled_pixels() -> None:
    targets = np.array([[0, 0], [1, 255]])
    predictions = np.array([[0, 1], [1, 1]])

    confusion = native.confusion_matrix(predictions, targets, 2, 255)

    assert confusion.tolist() == [[1, 1], [0, 1]]


def test_all_ignored_mask_contributes_nothing_to_the_aggregate() -> None:
    labelled = np.array([[0, 1], [1, 0]])
    ignored = np.full((2, 2), 255)
    predictions = np.array([[[0, 0], [1, 1]], [[1, 0], [1, 0]]])

    batch = native.confusion_matrix(predictions, np.stack([labelled, ignored]), 2, 255)
    alone = native.confusion_matrix(predictions[0], labelled, 2, 255)

    assert batch.tolist() == alone.tolist()
    assert native.confusion_matrix(predictions[1], ignored, 2, 255).sum() == 0


def test_metrics_by_hand_average_present_classes() -> None:
    metrics = native.segmentation_metrics(np.array([[1, 1], [0, 1]]))
    # IoU 1/2 per class; Dice 2/3 per class; 2 of 3 labelled pixels correct.
    assert metrics == pytest.approx({"miou": 0.5, "dice": 2 / 3, "pixel_acc": 2 / 3})

    absent = native.segmentation_metrics(np.array([[3, 0, 0], [0, 2, 0], [0, 0, 0]]))
    assert absent == {"miou": 1.0, "dice": 1.0, "pixel_acc": 1.0}


def test_metrics_refuse_a_split_without_labelled_pixels() -> None:
    with pytest.raises(ValueError, match="no labelled pixels"):
        native.segmentation_metrics(np.zeros((2, 2), dtype=np.int64))


def test_metrics_agree_with_platform_accumulator_within_tolerance() -> None:
    """Independent NumPy reductions match the trainer's torch float64 ones."""
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(3, 2, 9, 7, generator=generator)
    targets = torch.randint(0, 2, (3, 9, 7), generator=generator)
    targets[0, :4] = 255
    accumulator = SegmentationMetricAccumulator(2, 255)
    accumulator.update(logits, targets)

    confusion = native.confusion_matrix(
        logits.argmax(dim=1).numpy(), targets.numpy(), 2, 255
    )
    expected = accumulator.compute()
    result = native.segmentation_metrics(confusion)

    actual = (result["miou"], result["dice"], result["pixel_acc"])
    for got, want in zip(actual, expected, strict=True):
        assert abs(got - want) <= METRIC_ABS_TOLERANCE


def test_best_checkpoint_keeps_first_of_tied_scores(tmp_path: Path) -> None:
    model = native.build_tiny_fcn(2)
    best = native.BestCheckpoint(tmp_path / "best.pth")

    assert best.offer(1, 0.5, model)
    first = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    assert not best.offer(2, 0.5, model)
    saved = torch.load(tmp_path / "best.pth", weights_only=True)
    assert all(torch.equal(saved[k], first[k]) for k in first)
    assert (best.epoch, best.miou) == (1, 0.5)

    assert best.offer(3, 0.75, model)
    saved = torch.load(tmp_path / "best.pth", weights_only=True)
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())
    assert (best.epoch, best.miou) == (3, 0.75)


def test_load_pair_resizes_image_bilinear_and_mask_nearest(tmp_path: Path) -> None:
    image_path, mask_path = tmp_path / "a.png", tmp_path / "a_mask.png"
    Image.new("RGB", (40, 36), (255, 255, 255)).save(image_path)
    mask = np.zeros((36, 40), dtype=np.uint8)
    mask[:, 20:] = 1
    mask[0, :] = 255
    Image.fromarray(mask).save(mask_path)

    image, target = native.load_pair(image_path, mask_path, 32)

    assert image.shape == (3, 32, 32) and image.dtype == torch.float32
    assert torch.equal(image, torch.ones_like(image))
    assert target.shape == (32, 32) and target.dtype == torch.long
    assert set(target.unique().tolist()) == {0, 1, 255}
    assert (target[1:, :16] == 0).all() and (target[1:, 16:] == 1).all()
    assert (target[0] == 255).all()


def _expect_refusal(dataset: Path, initial: Path, output: Path, match: str) -> None:
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError), match=match):
        native.run_native(dataset, initial, output, seed=37, epochs=1, batch_size=2)
    assert not output.exists()


def _write_mask(path: Path, array: np.ndarray) -> None:
    path.unlink()
    Image.fromarray(array).save(path)


def test_preflight_refuses_an_all_ignored_mask(tmp_path: Path) -> None:
    dataset, initial = make_segmentation_fixture(tmp_path)
    _write_mask(
        dataset / "val" / "masks" / "val_01.png", np.full((36, 40), 255, np.uint8)
    )
    _expect_refusal(dataset, initial, tmp_path / "out", "all ignored")


def test_preflight_judges_ignored_pixels_after_resize(tmp_path: Path) -> None:
    """One labelled pixel can vanish under nearest resize; that mask is refused."""
    dataset, initial = make_segmentation_fixture(tmp_path)
    mask = np.full((36, 40), 255, np.uint8)
    mask[0, 2] = 1
    _write_mask(dataset / "train" / "masks" / "train_02.png", mask)
    _expect_refusal(dataset, initial, tmp_path / "out", "all ignored")


def test_preflight_refuses_unknown_mask_value(tmp_path: Path) -> None:
    dataset, initial = make_segmentation_fixture(tmp_path)
    mask = np.zeros((36, 40), np.uint8)
    mask[5, 5] = 2
    _write_mask(dataset / "test" / "masks" / "test_00.png", mask)
    _expect_refusal(dataset, initial, tmp_path / "out", "mask values")


def test_preflight_refuses_rgb_mask(tmp_path: Path) -> None:
    dataset, initial = make_segmentation_fixture(tmp_path)
    path = dataset / "train" / "masks" / "train_00.png"
    path.unlink()
    Image.new("RGB", (40, 36)).save(path)
    _expect_refusal(dataset, initial, tmp_path / "out", "single-channel")


def test_preflight_refuses_unpaired_and_unsupported_files(tmp_path: Path) -> None:
    dataset, initial = make_segmentation_fixture(tmp_path)
    (dataset / "train" / "masks" / "train_03.png").unlink()
    _expect_refusal(dataset, initial, tmp_path / "out", "unpaired")

    dataset, initial = make_segmentation_fixture(tmp_path / "second")
    (dataset / "val" / "images" / "notes.txt").write_text("x", encoding="utf-8")
    _expect_refusal(dataset, initial, tmp_path / "out", "unsupported")


def test_preflight_refuses_existing_output(tmp_path: Path) -> None:
    dataset, initial = make_segmentation_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError):
        native.run_native(dataset, initial, output, seed=37, epochs=1, batch_size=2)
    assert list(output.iterdir()) == []


def test_native_repeats_exactly_and_selects_best_val_miou(tmp_path: Path) -> None:
    dataset, initial = make_segmentation_fixture(tmp_path)
    results = [
        native.run_native(
            dataset, initial, tmp_path / name, seed=37, epochs=3, batch_size=2
        )
        for name in ("first", "repeat")
    ]
    first, repeat = results
    varying = {"phases", "best_model", "test_predictions_file"}
    assert {k: v for k, v in first.items() if k not in varying} == {
        k: v for k, v in repeat.items() if k not in varying
    }
    assert first["workflow"] == "native_pytorch"
    assert first["status"] == "completed"
    scores = [row["val_miou"] for row in first["history"]]
    assert first["best_epoch"] == scores.index(max(scores)) + 1
    assert first["best_val_miou"] == max(scores)
    assert all(np.isfinite(row["train_loss"]) for row in first["history"])
    assert (
        first["initial_weights_sha256"]
        == hashlib.sha256(initial.read_bytes()).hexdigest()
    )

    predictions_path = Path(first["test_predictions_file"])
    assert (
        first["test_predictions_sha256"]
        == hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    )
    predictions = np.load(predictions_path, allow_pickle=False)
    assert predictions.shape == (2, 32, 32) and predictions.dtype == np.uint8
    per_image = np.array(first["test_confusion_per_image"])
    assert per_image.shape == (2, 2, 2)
    summed = native.segmentation_metrics(per_image.sum(axis=0))
    assert first["test_metrics"] == summed
