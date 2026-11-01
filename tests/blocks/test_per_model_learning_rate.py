"""Per-model learning rates in comparisons and grids (ADR-158).

The fakes stand in for training only; everything up to the config a trial
receives, and everything written after it, is the real code.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from visionforge.blocks.grid_search import GridSearchBlock
from visionforge.blocks.model_comparison import ModelComparisonBlock
from visionforge.utils.config import (
    CURRENT_SCHEMA_VERSION,
    ExperimentConfig,
    load_config,
)
from visionforge.utils.messages import using_lang

LR_AXIS = "training.learning_rate"


def _raw(tmp_path: Path, **sections: Any) -> dict[str, Any]:
    return {
        "name": "lr_test",
        "task": "multiclass",
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "training": {"learning_rate": 0.005, "epochs": 1, "batch_size": 2},
        "data": {
            "base_dir": str(tmp_path),
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {"image_size": 64},
        },
        "output": {
            "models_dir": str(tmp_path / "models"),
            "graphics_dir": str(tmp_path / "graphics"),
            "logs_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
        },
        **sections,
    }


def _comparison(tmp_path: Path, **mc: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        _raw(
            tmp_path,
            block="model_comparison",
            model_comparison={"model_names": ["vgg16", "resnet50"], **mc},
        )
    )


def _grid(tmp_path: Path, axes: dict[str, list[Any]], **grid: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        _raw(
            tmp_path,
            block="grid_search",
            grid_search={"hyperparameters": axes, **grid},
        )
    )


@dataclass
class _FinishedRun:
    status: str = "success"
    stopped: bool = False
    error: str = ""
    run_dir: Path | None = None
    training_time_s: float = 1.0
    metrics: dict[str, float] = field(
        default_factory=lambda: {"accuracy": 0.8, "f1": 0.7, "auc_roc": 0.9}
    )


class _FakeRunner:
    """Records the config of each architecture instead of training it."""

    seen: list[ExperimentConfig]

    def __init__(self) -> None:
        self._cancel_token = None

    def run(self, config: ExperimentConfig) -> _FinishedRun:
        type(self).seen.append(config)
        return _FinishedRun()


def _run_comparison(config: ExperimentConfig) -> dict[str, float]:
    """Rate each architecture actually received, by architecture."""
    _FakeRunner.seen = []
    block = ModelComparisonBlock()
    block.setup(config)
    with (
        patch("visionforge.blocks.model_comparison.ClassificationRunner", _FakeRunner),
        patch("visionforge.blocks._search_utils.write_run_group"),
    ):
        block.run()
    return {c.model.name: c.training.learning_rate for c in _FakeRunner.seen}


class _FakeClassificationBlock:
    """Records each grid trial's config; lowest rate gets the best val loss."""

    seen: list[ExperimentConfig]

    def setup(self, config: ExperimentConfig) -> None:
        self.config = config
        type(self).seen.append(config)

    def run(self) -> None:
        pass

    def report(self) -> dict[str, Any]:
        rate = self.config.training.learning_rate
        return {
            "train": {"best_val_loss": rate},
            "eval": {"accuracy": 0.8, "f1": 0.7},
        }


def _run_grid(config: ExperimentConfig) -> GridSearchBlock:
    _FakeClassificationBlock.seen = []
    block = GridSearchBlock()
    block.setup(config)
    with (
        patch(
            "visionforge.blocks._search_utils.ClassificationBlock",
            _FakeClassificationBlock,
        ),
        patch("visionforge.blocks._search_utils.write_run_group"),
    ):
        block.run()
    return block


class TestComparisonPolicy:
    def test_suggested_gives_each_model_its_own_rate(self, tmp_path: Path) -> None:
        rates = _run_comparison(_comparison(tmp_path, lr_policy="suggested"))
        assert rates == {"vgg16": 1e-4, "resnet50": 1e-3}

    def test_fixed_gives_every_model_the_configured_rate(self, tmp_path: Path) -> None:
        rates = _run_comparison(_comparison(tmp_path, lr_policy="fixed"))
        assert rates == {"vgg16": 0.005, "resnet50": 0.005}

    def test_policy_defaults_to_fixed(self, tmp_path: Path) -> None:
        config = _comparison(tmp_path)
        assert config.model_comparison is not None
        assert config.model_comparison.lr_policy == "fixed"

    def test_suggestion_follows_the_configured_optimizer(self, tmp_path: Path) -> None:
        raw = _comparison(tmp_path, lr_policy="suggested").model_dump(mode="json")
        raw["training"]["optimizer"] = "sgd"
        rates = _run_comparison(ExperimentConfig.model_validate(raw))
        assert rates == {"vgg16": 1e-2, "resnet50": 1e-2}

    def test_rate_reaches_every_report_artifact(self, tmp_path: Path) -> None:
        config = _comparison(tmp_path, lr_policy="suggested")
        _FakeRunner.seen = []
        block = ModelComparisonBlock()
        block.setup(config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationRunner", _FakeRunner
            ),
            patch("visionforge.blocks._search_utils.write_run_group") as group,
        ):
            block.run()

        reports = tmp_path / "reports" / "lr_test"
        summary = json.loads((reports / "comparison_summary.json").read_text("utf-8"))
        assert {t["model_arch"]: t["effective_learning_rate"] for t in summary} == {
            "vgg16": 1e-4,
            "resnet50": 1e-3,
        }
        with (reports / "ranking.csv").open(encoding="utf-8") as f:
            ranking = {r["model_arch"]: r for r in csv.DictReader(f)}
        assert float(ranking["vgg16"]["effective_learning_rate"]) == 1e-4
        top = block.report()["top_3"]
        assert {t["effective_learning_rate"] for t in top} == {1e-4, 1e-3}

        rows = group.call_args.kwargs["report"]["trials"]
        vgg = next(r for r in rows if r["overrides"]["model.name"] == "vgg16")
        assert vgg["effective_learning_rate"] == 1e-4
        assert vgg["overrides"][LR_AXIS] == 1e-4

    def test_fixed_history_row_keeps_its_overrides(self, tmp_path: Path) -> None:
        block = ModelComparisonBlock()
        block.setup(_comparison(tmp_path))
        _FakeRunner.seen = []
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationRunner", _FakeRunner
            ),
            patch("visionforge.blocks._search_utils.write_run_group") as group,
        ):
            block.run()
        rows = group.call_args.kwargs["report"]["trials"]
        assert rows[0]["overrides"] == {"model.name": "vgg16"}
        assert rows[0]["effective_learning_rate"] == 0.005


class TestGridMultiplier:
    AXES: dict[str, list[Any]] = {
        "model.name": ["vgg16", "resnet50"],
        LR_AXIS: [0.3, 1, 3],
    }

    def test_axis_is_read_as_a_factor_of_each_models_rate(self, tmp_path: Path) -> None:
        _run_grid(_grid(tmp_path, self.AXES, lr_mode="suggested_multiplier"))
        trained = [
            (c.model.name, c.training.learning_rate)
            for c in _FakeClassificationBlock.seen
        ]
        # Computed by hand: vgg16 suggests 1e-4, resnet50 1e-3 under Adam.
        assert trained == [
            ("vgg16", 3e-5),
            ("vgg16", 1e-4),
            ("vgg16", 3e-4),
            ("resnet50", 3e-4),
            ("resnet50", 1e-3),
            ("resnet50", 3e-3),
        ]

    def test_absolute_axis_stays_the_same_for_every_model(self, tmp_path: Path) -> None:
        axes: dict[str, list[Any]] = {
            "model.name": ["vgg16", "resnet50"],
            LR_AXIS: [0.01],
        }
        _run_grid(_grid(tmp_path, axes))
        assert [c.training.learning_rate for c in _FakeClassificationBlock.seen] == [
            0.01,
            0.01,
        ]

    def test_grid_without_rate_axis_records_the_base_rate(self, tmp_path: Path) -> None:
        block = _run_grid(_grid(tmp_path, {"model.name": ["resnet18", "resnet34"]}))
        assert [t["effective_learning_rate"] for t in block._trials] == [0.005, 0.005]

    def test_optimizer_axis_changes_the_suggestion(self, tmp_path: Path) -> None:
        axes: dict[str, list[Any]] = {
            "model.name": ["vgg16"],
            "training.optimizer": ["adam", "sgd"],
            LR_AXIS: [1],
        }
        _run_grid(_grid(tmp_path, axes, lr_mode="suggested_multiplier"))
        assert [c.training.learning_rate for c in _FakeClassificationBlock.seen] == [
            1e-4,
            1e-2,
        ]

    def test_trial_record_keeps_factor_and_effective_rate(self, tmp_path: Path) -> None:
        block = _run_grid(_grid(tmp_path, self.AXES, lr_mode="suggested_multiplier"))
        first = block._trials[0]
        assert first["lr_multiplier"] == 0.3
        assert first["effective_learning_rate"] == 3e-5
        assert first[LR_AXIS] == 3e-5
        assert "lr3e-05" in _FakeClassificationBlock.seen[0].name.lower()

    def test_best_config_holds_the_effective_rate(self, tmp_path: Path) -> None:
        _run_grid(_grid(tmp_path, self.AXES, lr_mode="suggested_multiplier"))
        best = yaml.safe_load(
            (tmp_path / "reports" / "lr_test" / "best_config.yaml").read_text("utf-8")
        )
        assert best["model"]["name"] == "vgg16"
        assert best["training"]["learning_rate"] == 3e-5

    def test_summary_csv_has_both_columns(self, tmp_path: Path) -> None:
        _run_grid(_grid(tmp_path, self.AXES, lr_mode="suggested_multiplier"))
        path = tmp_path / "reports" / "lr_test" / "grid_search_summary.csv"
        with path.open(encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        assert float(row["effective_learning_rate"]) == 3e-5
        assert float(row["lr_multiplier"]) == 0.3


class TestGridMultiplierValidation:
    @pytest.fixture(autouse=True)
    def _english(self):
        with using_lang("en"):
            yield

    @pytest.mark.parametrize(
        "axes",
        [{LR_AXIS: [1, 3]}, {"model.name": ["vgg16", "resnet50"]}, {}],
    )
    def test_needs_both_axes(self, tmp_path: Path, axes: dict[str, list[Any]]) -> None:
        with pytest.raises(ValidationError, match="requires both"):
            _grid(tmp_path, axes, lr_mode="suggested_multiplier")

    @pytest.mark.parametrize("factor", [0, -1, "3x", True])
    def test_refuses_a_factor_that_is_not_positive(
        self, tmp_path: Path, factor: Any
    ) -> None:
        axes = {"model.name": ["vgg16", "resnet50"], LR_AXIS: [factor]}
        with pytest.raises(ValidationError, match="must be positive numbers"):
            _grid(tmp_path, axes, lr_mode="suggested_multiplier")


class TestOldConfigs:
    """A v1 file trains exactly as it did: only the version number moves."""

    def _write(self, tmp_path: Path, raw: dict[str, Any]) -> Path:
        path = tmp_path / "old.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    def test_old_comparison_loads_with_same_values(self, tmp_path: Path) -> None:
        raw = _raw(
            tmp_path,
            block="model_comparison",
            schema_version=1,
            model_comparison={"model_names": ["vgg16", "resnet50"]},
        )
        config = load_config(self._write(tmp_path, raw))
        dumped = config.model_dump(mode="json")
        assert dumped.pop("schema_version") == CURRENT_SCHEMA_VERSION == 2
        assert dumped["model_comparison"] == {
            "model_names": ["vgg16", "resnet50"],
            "metric": "f1",
            "lr_policy": "fixed",
        }
        assert dumped["training"]["learning_rate"] == 0.005
        assert _run_comparison(config) == {"vgg16": 0.005, "resnet50": 0.005}

    def test_old_grid_without_a_version_reads_rates_as_rates(
        self, tmp_path: Path
    ) -> None:
        raw = _raw(
            tmp_path,
            block="grid_search",
            grid_search={"hyperparameters": {LR_AXIS: [0.01, 0.1]}},
        )
        config = load_config(self._write(tmp_path, raw))
        assert config.grid_search is not None
        assert config.grid_search.lr_mode == "absolute"
        _run_grid(config)
        assert [c.training.learning_rate for c in _FakeClassificationBlock.seen] == [
            0.01,
            0.1,
        ]


def _write_tiny_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for name in ("class_a", "class_b"):
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(2):
                pixels = np.full((64, 64, 3), 40 + 100 * i, dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{i}.png")


@pytest.mark.slow
def test_each_run_json_records_the_rate_it_trained_at(tmp_path: Path) -> None:
    """End to end on the CPU, with no fake: run.json is written by the trainer."""
    _write_tiny_dataset(tmp_path)
    raw = _comparison(tmp_path, lr_policy="suggested").model_dump(mode="json")
    raw["model_comparison"]["model_names"] = ["alexnet", "resnet18"]
    raw["model_comparison"]["metric"] = "accuracy"
    raw["device"] = {"preferred": "cpu"}
    block = ModelComparisonBlock()
    block.setup(ExperimentConfig.model_validate(raw))
    block.run()

    trials = {t["model_arch"]: t for t in block._trials}
    assert {t["status"] for t in trials.values()} == {"success"}, trials
    recorded = {
        arch: json.loads((Path(t["run_dir"]) / "run.json").read_text("utf-8"))
        for arch, t in trials.items()
    }
    assert recorded["alexnet"]["config"]["training"]["learning_rate"] == 1e-4
    assert recorded["resnet18"]["config"]["training"]["learning_rate"] == 1e-3
