"""Suspicious labels by confident learning (ADR-237), on a hand-computed case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visionforge.core.label_suspects import (
    class_thresholds,
    confident_labels,
    find_suspects,
    read_folder_predictions,
)

NAMES = ["cat", "dog", "fox"]
LABELS = np.array([0, 0, 0, 1, 1, 2, 2])
PROBS = np.array(
    [
        [0.90, 0.05, 0.05],
        [0.80, 0.10, 0.10],
        [0.10, 0.85, 0.05],  # labeled cat, the model is sure it is a dog
        [0.10, 0.80, 0.10],
        [0.20, 0.70, 0.10],  # passes no threshold
        [0.05, 0.05, 0.90],
        [0.30, 0.10, 0.60],  # passes no threshold
    ]
)


def test_thresholds_are_the_mean_self_confidence() -> None:
    # cat: (0.9 + 0.8 + 0.1) / 3; dog: (0.8 + 0.7) / 2; fox: (0.9 + 0.6) / 2.
    assert class_thresholds(PROBS, LABELS, 3) == pytest.approx([0.6, 0.75, 0.75])
    assert class_thresholds(PROBS[:2], LABELS[:2], 3)[1] == np.inf


def test_a_saturated_model_still_flags_the_disputed_image() -> None:
    # The labeled-cat images give cat 0.9994 on average; the dog image the model
    # calls a cat with 0.9988 must still count as a cat (ceiling 0.95).
    labels = np.array([0, 0, 1, 1, 1])
    p_cat = np.array([0.9993, 0.9995, 0.012, 0.023, 0.9988])
    probs = np.stack([p_cat, 1 - p_cat], axis=1)
    assert class_thresholds(probs, labels, 2)[0] == 0.95
    report = find_suspects(probs, labels, ["cat", "dog"])
    assert [s["index"] for s in report["suspects"]] == [4]


def test_confident_label_is_the_best_class_over_its_threshold() -> None:
    confident = confident_labels(PROBS, np.array([0.6, 0.75, 0.75]))
    assert confident.tolist() == [0, 0, 1, 1, -1, 2, -1]


def test_the_disputed_image_is_the_only_suspect() -> None:
    report = find_suspects(PROBS, LABELS, NAMES)
    assert report["suspect_count"] == 1
    assert report["suspects"] == [
        {
            "index": 2,
            "given": "cat",
            "suspected": "dog",
            "p_given": 0.1,
            "p_suspected": 0.85,
        }
    ]
    assert report["joint"] == [[2, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert report["pairs"] == [{"given": "cat", "taken_for": "dog", "images": 1}]


def test_suspects_are_ranked_by_how_little_the_label_keeps() -> None:
    labels = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.3, 0.7], [0.05, 0.95], [0.2, 0.8]])
    # Thresholds: cat (0.9 + 0.3) / 2 = 0.6, dog (0.95 + 0.8) / 2 = 0.875.
    # Image 1 is cat with p(dog) 0.7 < 0.875: not confident. No suspect.
    assert find_suspects(probs, labels, ["cat", "dog"])["suspect_count"] == 0
    probs[1] = [0.05, 0.95]  # now well over dog's threshold
    report = find_suspects(probs, labels, ["cat", "dog"])
    assert [s["index"] for s in report["suspects"]] == [1]


def _csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_labels_come_from_the_class_folder(tmp_path: Path) -> None:
    split = tmp_path / "val"
    csv_path = _csv(
        tmp_path / "p.csv",
        "filename,predicted_class,prob_class_0,prob_class_1,prob_class_2",
        [
            f"{split / 'dog' / 'a.png'},1,0.1,0.8,0.1",
            f"{split / 'cat' / 'b.png'},0,0.7,0.2,0.1",
            f"{split / 'loose.png'},0,0.7,0.2,0.1",  # no class folder: skipped
            f"{split / 'owl' / 'c.png'},0,0.7,0.2,0.1",  # unknown class: skipped
        ],
    )
    paths, probs, labels = read_folder_predictions(csv_path, split, NAMES)
    assert paths == ["dog/a.png", "cat/b.png"]
    assert labels.tolist() == [1, 0]
    assert probs.shape == (2, 3)


def test_a_binary_csv_becomes_two_columns(tmp_path: Path) -> None:
    split = tmp_path / "val"
    csv_path = _csv(
        tmp_path / "p.csv",
        "filename,predicted_class,prob_positive",
        [f"{split / 'dog' / 'a.png'},1,0.75"],
    )
    _, probs, labels = read_folder_predictions(csv_path, split, ["cat", "dog"])
    assert probs.tolist() == [[0.25, 0.75]]
    assert labels.tolist() == [1]
