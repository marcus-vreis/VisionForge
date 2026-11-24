"""`cross_validation.select_on` picks the epoch that scores each fold (ADR-242).

"best" (the default) reloads the checkpoint of the epoch with the lowest
validation loss on the fold's own validation split, which then scores it; "last"
scores the weights of the fold's last epoch, with no selection on that split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.blocks.test_cross_validation_epochs import _config, _run_json
from visionforge.blocks.cross_validation import CrossValidationBlock
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.selftest_data import build_classification_dataset

_EPOCHS = 3


def _with_rule(config: ExperimentConfig, rule: str) -> ExperimentConfig:
    cv = config.cross_validation
    assert cv is not None
    return config.model_copy(
        update={"cross_validation": cv.model_copy(update={"select_on": rule})}
    )


def _fold_best_epoch(fold: dict[str, Any]) -> int:
    """The best epoch the fold's own run.json recorded."""
    run_json = Path(fold["run_dir"]) / "run.json"
    data: dict[str, Any] = json.loads(run_json.read_text(encoding="utf-8"))
    best: int = data["metrics"]["best_epoch"]
    return best


def _run(config: ExperimentConfig) -> CrossValidationBlock:
    block = CrossValidationBlock()
    block.setup(config)
    block.run()
    return block


class TestARealKFold:
    """A tiny CPU K-fold, trained for real, under each rule."""

    @pytest.fixture
    def base(self, tmp_path: Path) -> Path:
        return build_classification_dataset(tmp_path / "ds")

    def test_last_scores_each_fold_at_its_last_epoch(
        self, base: Path, tmp_path: Path
    ) -> None:
        block = _run(_with_rule(_config(base, tmp_path, epochs=_EPOCHS), "last"))

        assert [r["selected_epoch"] for r in block._fold_results] == [3, 3]

    def test_best_scores_each_fold_at_its_best_epoch(
        self, base: Path, tmp_path: Path
    ) -> None:
        block = _run(_config(base, tmp_path, epochs=_EPOCHS))
        folds = block._fold_results

        assert [r["selected_epoch"] for r in folds] == [
            _fold_best_epoch(r) for r in folds
        ]

    @pytest.mark.parametrize("rule", ["best", "last"])
    def test_the_run_json_names_the_rule_of_every_fold(
        self, base: Path, tmp_path: Path, rule: str
    ) -> None:
        config = _with_rule(_config(base, tmp_path, epochs=1), rule)
        _run(config)

        folds = _run_json(config)["metrics"]["fold_results"]

        assert [f["select_on"] for f in folds] == [rule, rule]


class TestWhichWeightsAreScored:
    """Only "best" reloads the saved checkpoint before scoring a fold."""

    def _run_mocked(
        self, base: Path, tmp_path: Path, rule: str
    ) -> tuple[CrossValidationBlock, MagicMock]:
        config = _with_rule(_config(base, tmp_path, epochs=_EPOCHS), rule)
        result = MagicMock(
            stopped=False, best_epoch=1, total_epochs=_EPOCHS, best_val_loss=0.4
        )
        result.model_path = config.output.models_dir / "fold" / "best_model.pth"
        block = CrossValidationBlock()
        block.setup(config)
        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as trainer_cls,
            patch("visionforge.blocks.cross_validation.Evaluator") as evaluator_cls,
            patch(
                "visionforge.blocks.cross_validation.torch.load", return_value={}
            ) as load,
        ):
            trainer_cls.return_value.fit.return_value = result
            evaluator_cls.return_value.evaluate.return_value = MagicMock(
                accuracy=0.5, f1=0.5
            )
            block.run()
        return block, load

    @pytest.mark.parametrize(
        ("rule", "epoch", "loads"), [("best", 1, 2), ("last", _EPOCHS, 0)]
    )
    def test_the_rule_decides_the_epoch_and_the_reload(
        self, tmp_path: Path, rule: str, epoch: int, loads: int
    ) -> None:
        base = build_classification_dataset(tmp_path / "ds")

        block, load = self._run_mocked(base, tmp_path, rule)

        assert [r["selected_epoch"] for r in block._fold_results] == [epoch, epoch]
        assert load.call_count == loads

    def test_a_fold_that_scored_nothing_has_no_selected_epoch(
        self, tmp_path: Path
    ) -> None:
        base = build_classification_dataset(tmp_path / "ds")
        config = _with_rule(_config(base, tmp_path, epochs=1), "last")
        block = CrossValidationBlock()
        block.setup(config)

        with patch("visionforge.blocks.cross_validation.Trainer") as trainer_cls:
            trainer_cls.return_value.fit.side_effect = RuntimeError("boom")
            block.run()

        assert [r["selected_epoch"] for r in block._fold_results] == [None, None]
