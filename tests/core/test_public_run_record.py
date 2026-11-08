"""A run record leaving the computer keeps only the final name of local paths."""

from __future__ import annotations

import pytest

from visionforge.core.public_run_record import without_local_paths


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (r"C:\Users\ana\Documents\VisionForge\datasets\gatos", "gatos"),
        ("C:/Users/ana/datasets/gatos/", "gatos"),
        (r"\\servidor\lab\datasets\gatos", "gatos"),
        ("/home/ana/datasets/gatos", "gatos"),
        ("/content/drive/MyDrive/gatos", "gatos"),
        ("~/datasets/gatos", "gatos"),
    ],
)
def test_a_whole_path_becomes_its_final_name(value: str, expected: str) -> None:
    assert without_local_paths(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            r"Falhou ao ler C:\Users\ana\datasets\gatos\a.png agora",
            "Falhou ao ler a.png agora",
        ),
        (
            "Pasta '/home/ana/datasets/gatos' vazia",
            "Pasta 'gatos' vazia",
        ),
        ("cache em /root/.cache/torch/hub e /content/x.pt", "cache em hub e x.pt"),
    ],
)
def test_a_path_inside_text_becomes_its_final_name(value: str, expected: str) -> None:
    assert without_local_paths(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "resnet18",
        "weights/best.pt",
        "https://zenodo.org/home/records/1",
        "/",
        "val/accuracy",
        "Windows-11-10.0.26300-SP0",
    ],
)
def test_text_that_is_not_a_local_path_is_unchanged(value: str) -> None:
    assert without_local_paths(value) == value


def test_nested_records_are_rebuilt_and_other_values_kept() -> None:
    record = {
        "config": {
            "data": {"base_dir": "/home/ana/gatos"},
            "seed": 42,
            "augment": True,
        },
        "warnings": ["sem GPU", r"D:\dados\gatos\x.jpg ignorada"],
        "metrics": {"accuracy": 0.9, "loss": None},
    }
    original = repr(record)
    assert without_local_paths(record) == {
        "config": {"data": {"base_dir": "gatos"}, "seed": 42, "augment": True},
        "warnings": ["sem GPU", "x.jpg ignorada"],
        "metrics": {"accuracy": 0.9, "loss": None},
    }
    assert repr(record) == original
