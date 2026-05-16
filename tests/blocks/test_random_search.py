from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from visionforge.blocks.random_search import (
    RandomSearchBlock,
    _ChoiceParam,
    _LogUniformParam,
    _parse_param,
    _UniformParam,
)
from visionforge.blocks.registry import BlockRegistry
from visionforge.utils.config import ExperimentConfig

# ── fixtures ──────────────────────────────────────────────────────────────────


class _TinyBinaryModel(nn.Module):
    """Minimal model that avoids loading a full ResNet."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Minimal ImageFolder layout for two classes."""
    from PIL import Image

    for split in ["train", "val", "test"]:
        for cls in ["class_a", "class_b"]:
            folder = tmp_path / split / cls
            folder.mkdir(parents=True)
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(folder / "image.png")
    return tmp_path


@pytest.fixture
def base_config(dataset_root: Path, tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "rs_test",
            "task": "binary",
            "block": "random_search",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.01,
                "epochs": 1,
                "batch_size": 2,
                "early_stopping_patience": 1,
                "seed": 10,
            },
            "data": {
                "base_dir": str(dataset_root),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32},
            },
            "output": {
                "models_dir": str(tmp_path / "models"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "random_search": {
                "n_trials": 3,
                "seed": 7,
                "search_space": {},
            },
        }
    )


def _config_with_search_space(
    base_config: ExperimentConfig, search_space: dict[str, Any]
) -> ExperimentConfig:
    raw = base_config.model_dump(mode="json")
    raw["random_search"]["search_space"] = search_space
    return ExperimentConfig.model_validate(raw)


def _make_mock_block(val_loss: float = 0.5, accuracy: float = 0.8) -> MagicMock:
    mock = MagicMock()
    mock.report.return_value = {
        "train": {"best_val_loss": val_loss, "best_epoch": 1, "total_epochs": 1},
        "eval": {"accuracy": accuracy, "f1": 0.75, "precision": 0.8, "recall": 0.7},
    }
    return mock


# ── param unit tests ──────────────────────────────────────────────────────────


class TestUniformParam:
    def test_sampling_is_bounded(self) -> None:
        param = _UniformParam(type="uniform", low=1.0, high=5.0)
        rng = random.Random(0)
        for _ in range(100):
            v = param.sample(rng)
            assert 1.0 <= v <= 5.0

    def test_parse_param_returns_uniform(self) -> None:
        p = _parse_param("lr", {"type": "uniform", "low": 0.0, "high": 1.0})
        assert isinstance(p, _UniformParam)


class TestLogUniformParam:
    def test_sampling_is_bounded(self) -> None:
        param = _LogUniformParam(type="log_uniform", low=1e-5, high=1e-2)
        rng = random.Random(0)
        for _ in range(100):
            v = param.sample(rng)
            assert 1e-5 <= v <= 1e-2

    def test_parse_param_returns_log_uniform(self) -> None:
        p = _parse_param("lr", {"type": "log_uniform", "low": 1e-5, "high": 1e-2})
        assert isinstance(p, _LogUniformParam)


class TestChoiceParam:
    def test_sampling_is_valid(self) -> None:
        options = ["adam", "sgd", "adamw"]
        param = _ChoiceParam(type="choice", options=options)
        rng = random.Random(0)
        for _ in range(50):
            assert param.sample(rng) in options

    def test_parse_param_returns_choice(self) -> None:
        p = _parse_param("opt", {"type": "choice", "options": ["adam", "sgd"]})
        assert isinstance(p, _ChoiceParam)


class TestParseParam:
    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValueError, match="'type' key"):
            _parse_param("lr", {"low": 0.0, "high": 1.0})

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown param type"):
            _parse_param("lr", {"type": "bayesian", "low": 0.0})

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="'type' key"):
            _parse_param("lr", "log_uniform")


# ── seed reproducibility ──────────────────────────────────────────────────────


class TestSeedReproducibility:
    def test_same_seed_produces_same_sequence(self) -> None:
        param = _UniformParam(type="uniform", low=0.0, high=1.0)
        seq_a = [param.sample(random.Random(42)) for _ in range(10)]
        seq_b = [param.sample(random.Random(42)) for _ in range(10)]
        assert seq_a == seq_b

    def test_different_seeds_differ(self) -> None:
        param = _UniformParam(type="uniform", low=0.0, high=1.0)
        seq_a = [param.sample(random.Random(1)) for _ in range(20)]
        seq_b = [param.sample(random.Random(2)) for _ in range(20)]
        assert seq_a != seq_b


# ── RandomSearchBlock setup ───────────────────────────────────────────────────


class TestRandomSearchBlockSetup:
    def test_setup_requires_random_search_config(
        self, base_config: ExperimentConfig
    ) -> None:
        raw = base_config.model_dump(mode="json")
        raw["random_search"] = None
        config = ExperimentConfig.model_validate(raw)
        block = RandomSearchBlock()
        with pytest.raises(ValueError, match="random_search"):
            block.setup(config)

    def test_unknown_search_space_key_raises(
        self, base_config: ExperimentConfig
    ) -> None:
        config = _config_with_search_space(
            base_config,
            {"training.nonexistent_key": {"type": "uniform", "low": 0.0, "high": 1.0}},
        )
        block = RandomSearchBlock()
        with pytest.raises(ValueError, match="Unknown hyperparameter path"):
            block.setup(config)

    def test_invalid_param_type_raises_at_setup(
        self, base_config: ExperimentConfig
    ) -> None:
        config = _config_with_search_space(
            base_config,
            {"training.learning_rate": {"type": "bad_type", "low": 0.0, "high": 1.0}},
        )
        block = RandomSearchBlock()
        with pytest.raises(ValueError, match="Unknown param type"):
            block.setup(config)


# ── run and trial count ───────────────────────────────────────────────────────


class TestRandomSearchBlockRun:
    def test_runs_n_trials(self, base_config: ExperimentConfig) -> None:
        """Inner block run() must be called exactly n_trials times."""
        mock_block = _make_mock_block()

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        assert mock_block.run.call_count == base_config.random_search.n_trials  # type: ignore[union-attr]

    def test_training_seed_per_trial_is_master_plus_index(
        self, base_config: ExperimentConfig
    ) -> None:
        """Each trial's training seed must equal base_seed + trial_index."""
        seeds_seen: list[int] = []

        def capture_setup(config: ExperimentConfig) -> None:
            seeds_seen.append(config.training.seed)

        mock_block = _make_mock_block()
        mock_block.setup.side_effect = capture_setup

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        base_seed = base_config.training.seed
        expected = [base_seed + i for i in range(base_config.random_search.n_trials)]  # type: ignore[union-attr]
        assert seeds_seen == expected

    def test_empty_search_space_runs_n_trials(
        self, base_config: ExperimentConfig
    ) -> None:
        """Empty search space runs n_trials identical (base-config) trials."""
        mock_block = _make_mock_block()

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        assert mock_block.run.call_count == base_config.random_search.n_trials  # type: ignore[union-attr]

    @patch("visionforge.blocks.random_search.torch.cuda.empty_cache")
    def test_vram_empty_cache_called_per_trial(
        self, mock_cache: MagicMock, base_config: ExperimentConfig
    ) -> None:
        """torch.cuda.empty_cache() must be called once per trial (in finally)."""
        mock_block = _make_mock_block()

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        n = base_config.random_search.n_trials  # type: ignore[union-attr]
        assert mock_cache.call_count == n


# ── artifacts ─────────────────────────────────────────────────────────────────


class TestRandomSearchArtifacts:
    def test_summary_csv_written(self, base_config: ExperimentConfig) -> None:
        """summary CSV must exist after run()."""
        mock_block = _make_mock_block()

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        csv_path = (
            base_config.output.reports_dir
            / base_config.name
            / "random_search_summary.csv"
        )
        assert csv_path.exists()

    def test_summary_csv_has_correct_row_count(
        self, base_config: ExperimentConfig
    ) -> None:
        """CSV must have exactly n_trials data rows (excluding header)."""
        mock_block = _make_mock_block()

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        csv_path = (
            base_config.output.reports_dir
            / base_config.name
            / "random_search_summary.csv"
        )
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        n = base_config.random_search.n_trials  # type: ignore[union-attr]
        assert len(rows) == n

    def test_summary_csv_has_trial_seed_column(
        self, base_config: ExperimentConfig
    ) -> None:
        """CSV must include a trial_seed column."""
        mock_block = _make_mock_block()

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        csv_path = (
            base_config.output.reports_dir
            / base_config.name
            / "random_search_summary.csv"
        )
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

        assert "trial_seed" in fieldnames

    def test_best_config_yaml_written(self, base_config: ExperimentConfig) -> None:
        """best_config.yaml must exist after a successful run."""
        mock_block = _make_mock_block()

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        yaml_path = (
            base_config.output.reports_dir / base_config.name / "best_config.yaml"
        )
        assert yaml_path.exists()

    def test_best_config_yaml_not_written_when_all_fail(
        self, base_config: ExperimentConfig
    ) -> None:
        """best_config.yaml must NOT be written if all trials fail."""
        mock_block = MagicMock()
        mock_block.run.side_effect = RuntimeError("simulated failure")

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        yaml_path = (
            base_config.output.reports_dir / base_config.name / "best_config.yaml"
        )
        assert not yaml_path.exists()


# ── failure handling ──────────────────────────────────────────────────────────


class TestRandomSearchFailureHandling:
    def test_one_failed_one_ok_trial(self, base_config: ExperimentConfig) -> None:
        """Sweep must complete; CSV has one success and one failed row."""
        raw = base_config.model_dump(mode="json")
        raw["random_search"]["n_trials"] = 2
        config = ExperimentConfig.model_validate(raw)

        call_count = 0

        def run_side_effect() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("trial 0 fails on purpose")

        good_mock = _make_mock_block(val_loss=0.3)
        good_mock.run.side_effect = run_side_effect

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=good_mock,
        ):
            block = RandomSearchBlock()
            block.setup(config)
            block.run()

        csv_path = config.output.reports_dir / config.name / "random_search_summary.csv"
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        statuses = [r["status"] for r in rows]
        assert statuses.count("failed") == 1
        assert statuses.count("success") == 1

    def test_csv_error_column_roundtrips_with_commas_and_newlines(
        self, base_config: ExperimentConfig
    ) -> None:
        """Error messages with commas and newlines must survive CSV round-trip intact."""
        raw = base_config.model_dump(mode="json")
        raw["random_search"]["n_trials"] = 1
        config = ExperimentConfig.model_validate(raw)

        nasty_message = "failed: expected value, got None\ndetail: line 2"
        mock_block = MagicMock()
        mock_block.run.side_effect = RuntimeError(nasty_message)

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(config)
            block.run()

        csv_path = config.output.reports_dir / config.name / "random_search_summary.csv"
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["error"] == nasty_message

    def test_all_trials_fail_report_raises(self, base_config: ExperimentConfig) -> None:
        """report() must raise RuntimeError when no trials succeeded."""
        mock_block = MagicMock()
        mock_block.run.side_effect = RuntimeError("always fails")

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        with pytest.raises(RuntimeError, match="all trials failed"):
            block.report()


# ── report ────────────────────────────────────────────────────────────────────


class TestRandomSearchReport:
    def test_report_returns_best_metrics(self, base_config: ExperimentConfig) -> None:
        """report() must return best_trial, total_trials, successful_trials."""
        mock_block = _make_mock_block(val_loss=0.4, accuracy=0.9)

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(base_config)
            block.run()

        result = block.report()
        assert "best_trial" in result
        assert result["total_trials"] == base_config.random_search.n_trials  # type: ignore[union-attr]
        assert result["successful_trials"] > 0
        assert result["best_trial"]["status"] == "success"

    def test_report_selects_lowest_val_loss(
        self, base_config: ExperimentConfig
    ) -> None:
        """Best trial must be the one with the lowest best_val_loss."""
        raw = base_config.model_dump(mode="json")
        raw["random_search"]["n_trials"] = 3
        config = ExperimentConfig.model_validate(raw)

        losses = [0.8, 0.2, 0.5]
        call_idx = 0

        def make_mock() -> MagicMock:
            nonlocal call_idx
            m = _make_mock_block(val_loss=losses[call_idx])
            call_idx += 1
            return m

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            side_effect=lambda: make_mock(),
        ):
            block = RandomSearchBlock()
            block.setup(config)
            block.run()

        result = block.report()
        assert result["best_trial"]["trial_index"] == 1
        assert result["best_trial"]["best_val_loss"] == pytest.approx(0.2)


# ── registry ──────────────────────────────────────────────────────────────────


class TestBlockRegistry:
    def test_random_search_block_is_registered(self) -> None:
        """BlockRegistry.discover() must include RandomSearchBlock."""
        import visionforge.blocks  # noqa: F401 — ensure __init__ imports trigger registration

        discovered = BlockRegistry.discover()
        assert "RandomSearchBlock" in discovered
        assert discovered["RandomSearchBlock"] is RandomSearchBlock


# ── config validation ─────────────────────────────────────────────────────────


class TestRandomSearchConfig:
    def test_zero_trials_raises_validation_error(
        self, dataset_root: Path, tmp_path: Path
    ) -> None:
        """n_trials=0 must raise a Pydantic ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(
                {
                    "name": "zero_trial_test",
                    "task": "binary",
                    "model": {
                        "name": "resnet18",
                        "num_classes": 1,
                        "pretrained": False,
                    },
                    "training": {
                        "learning_rate": 0.01,
                        "epochs": 1,
                        "batch_size": 2,
                        "early_stopping_patience": 1,
                    },
                    "data": {
                        "base_dir": str(dataset_root),
                        "num_workers": 0,
                        "pin_memory": False,
                    },
                    "random_search": {"n_trials": 0, "seed": 42, "search_space": {}},
                }
            )

    def test_single_trial_runs_once(self, base_config: ExperimentConfig) -> None:
        """n_trials=1 must execute exactly one trial."""
        raw = base_config.model_dump(mode="json")
        raw["random_search"]["n_trials"] = 1
        config = ExperimentConfig.model_validate(raw)

        mock_block = _make_mock_block()

        with patch(
            "visionforge.blocks.random_search.ClassificationBlock",
            return_value=mock_block,
        ):
            block = RandomSearchBlock()
            block.setup(config)
            block.run()

        assert mock_block.run.call_count == 1
