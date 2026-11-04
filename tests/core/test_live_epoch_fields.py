"""The live figures the epoch event carries for the "Results" tab (ADR-161):
the learning rate of the epoch and, for classification, the validation
confusion. Neither goes into ``run.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from visionforge.core import training_loop
from visionforge.core.live_confusion import (
    FULL_MATRIX_MAX_CLASSES,
    TOP_CONFUSIONS,
    live_confusion,
)
from visionforge.core.trainer import Trainer
from visionforge.utils.config import ExperimentConfig


class TestLiveConfusion:
    def test_counts_true_rows_against_predicted_columns(self) -> None:
        # cat→cat twice, cat→dog once, dog→dog once, bird→cat once.
        labels = [0, 0, 0, 1, 2]
        preds = [0, 0, 1, 1, 0]
        assert live_confusion(labels, preds, ["cat", "dog", "bird"]) == {
            "classes": ["cat", "dog", "bird"],
            "matrix": [[2, 1, 0], [0, 1, 0], [1, 0, 0]],
        }

    def test_many_classes_send_the_most_frequent_mistakes(self) -> None:
        n = FULL_MATRIX_MAX_CLASSES + 1
        names = [f"c{i}" for i in range(n)]
        # 3×(5→7), 2×(1→2), 2×(0→4), 1×(9→3), and two correct ones.
        labels = [5, 5, 5, 1, 1, 0, 0, 9, 8, 8]
        preds = [7, 7, 7, 2, 2, 4, 4, 3, 8, 8]
        out = live_confusion(labels, preds, names)
        assert out is not None
        assert "matrix" not in out
        # The tie between (0,4) and (1,2) goes to the lower pair.
        assert out["top"] == [[5, 7, 3], [0, 4, 2], [1, 2, 2], [9, 3, 1]]

    def test_the_top_list_is_capped(self) -> None:
        n = FULL_MATRIX_MAX_CLASSES + 1
        labels = list(range(n))
        preds = [(i + 1) % n for i in range(n)]
        out = live_confusion(labels, preds, [str(i) for i in range(n)])
        assert out is not None
        assert len(out["top"]) == TOP_CONFUSIONS

    @pytest.mark.parametrize(
        ("labels", "preds", "names"),
        [
            ([], [], ["a", "b"]),  # nothing measured
            ([0], [0], []),  # no class list
            ([0, 1], [0], ["a", "b"]),  # lengths differ
            ([0, 2], [0, 1], ["a", "b"]),  # a label outside the list
        ],
    )
    def test_nothing_rather_than_a_wrong_picture(
        self, labels: list[int], preds: list[int], names: list[str]
    ) -> None:
        assert live_confusion(labels, preds, names) is None


class TestCurrentLr:
    def test_reads_the_single_group(self) -> None:
        opt = torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.05)
        assert training_loop.current_lr(opt) == pytest.approx(0.05)

    def test_reads_the_head_when_the_backbone_is_scaled(self) -> None:
        backbone, head = nn.Linear(2, 2), nn.Linear(2, 1)
        opt = torch.optim.SGD(
            [
                {"params": backbone.parameters(), "lr": 0.001},
                {"params": head.parameters(), "lr": 0.01},
            ]
        )
        assert training_loop.current_lr(opt) == pytest.approx(0.01)

    def test_none_without_an_optimizer(self) -> None:
        assert training_loop.current_lr(None) is None


def test_the_epoch_rate_is_read_before_the_scheduler_steps() -> None:
    opt = torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    epoch_lr = training_loop.EpochLr(opt)
    seen: list[float | None] = []

    def compute(epoch: int) -> int:
        opt.step()
        scheduler.step()  # as the trainers do at the end of an epoch
        return epoch

    run = epoch_lr.around(compute)
    for epoch in (1, 2, 3):
        run(epoch)
        seen.append(epoch_lr.fields()["lr"])
    assert seen == pytest.approx([1.0, 0.1, 0.01])


def test_no_field_before_the_first_epoch() -> None:
    assert training_loop.EpochLr(None).fields() == {}


# ── classification, end to end ────────────────────────────────────────────────


class _Binary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class _NamedDataModule:
    """Two batches of four with known labels: three of class 0, five of class 1."""

    class_names = ["ok", "defect"]
    _labels = ([0, 0, 0, 1], [1, 1, 1, 1])

    def _batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        gen = torch.Generator().manual_seed(0)
        return [
            (torch.randn(4, 3, 8, 8, generator=gen), torch.tensor(labels))
            for labels in self._labels
        ]

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()

    def val_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._batches()


def _classification_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "live_fields",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.1,
                "epochs": 2,
                "batch_size": 4,
                "seed": 0,
                "scheduler": {"kind": "step", "step_size": 1, "gamma": 0.5},
            },
            "data": {"base_dir": str(tmp_path)},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


def test_classification_streams_rate_and_confusion_but_does_not_keep_them(
    tmp_path: Path,
) -> None:
    events: list[dict[str, Any]] = []
    cfg = _classification_config(tmp_path)
    Trainer(cfg).fit(_Binary(), _NamedDataModule(), progress_callback=events.append)

    epochs = [e for e in events if e["event"] == "epoch_end"]
    # StepLR halves the rate after each epoch: the epoch's own rate is reported.
    assert [e["lr"] for e in epochs] == pytest.approx([0.1, 0.05])
    for e in epochs:
        confusion = e["confusion"]
        assert confusion["classes"] == ["ok", "defect"]
        # Each row adds up to how many validation images the class has.
        assert [sum(row) for row in confusion["matrix"]] == [3, 5]

    run_json = next((tmp_path / "models" / "live_fields").glob("*/run.json"))
    history = json.loads(run_json.read_text(encoding="utf-8"))["history"]
    assert all("lr" not in h and "confusion" not in h for h in history)


def test_without_class_names_there_is_no_confusion(tmp_path: Path) -> None:
    class Unnamed(_NamedDataModule):
        class_names: list[str] = []

    events: list[dict[str, Any]] = []
    Trainer(_classification_config(tmp_path)).fit(
        _Binary(), Unnamed(), progress_callback=events.append
    )
    epoch = next(e for e in events if e["event"] == "epoch_end")
    assert "confusion" not in epoch
    assert epoch["lr"] == pytest.approx(0.1)
