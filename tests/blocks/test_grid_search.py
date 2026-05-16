from __future__ import annotations

import csv as csv_mod
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from visionforge.blocks.registry import BlockRegistry
from visionforge.utils.config import ExperimentConfig

# ── fixtures ──────────────────────────────────────────────────────────────────


def _base_config(
    tmp_path: Path, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "gs_test",
        "task": "binary",
        "block": "grid_search",
        "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
        "training": {
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 2,
            "seed": 10,
        },
        "data": {
            "base_dir": str(tmp_path),
            "num_workers": 0,
            "pin_memory": False,
        },
        "output": {
            "models_dir": str(tmp_path / "models"),
            "graphics_dir": str(tmp_path / "graphics"),
            "logs_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "grid_search": {"hyperparameters": {}},
    }
    if overrides:
        for k, v in overrides.items():
            raw[k] = v
    return raw


@pytest.fixture
def base_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_base_config(tmp_path))


@pytest.fixture
def grid_2x2_config(tmp_path: Path) -> ExperimentConfig:
    raw = _base_config(
        tmp_path,
        {
            "grid_search": {
                "hyperparameters": {
                    "training.learning_rate": [0.001, 0.01],
                    "training.optimizer": ["adam", "sgd"],
                }
            }
        },
    )
    return ExperimentConfig.model_validate(raw)


def _mock_report(trial_idx: int = 0) -> dict[str, Any]:
    return {
        "train": {
            "best_epoch": 1,
            "best_val_loss": 0.5 - trial_idx * 0.05,
            "total_epochs": 1,
        },
        "eval": {
            "accuracy": 0.8 + trial_idx * 0.01,
            "f1": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "auc_roc": 0.85,
        },
    }


# ── setup validation ──────────────────────────────────────────────────────────


class TestGridSearchSetup:
    def test_setup_raises_without_grid_search_config(self, tmp_path: Path) -> None:
        """setup() must raise ValueError when grid_search is None."""
        from visionforge.blocks.grid_search import GridSearchBlock

        raw = _base_config(tmp_path)
        raw["grid_search"] = None
        config = ExperimentConfig.model_validate(raw)
        block = GridSearchBlock()
        with pytest.raises(ValueError, match="grid_search"):
            block.setup(config)

    def test_unknown_dot_key_raises_at_setup(self, tmp_path: Path) -> None:
        """Unknown hyperparameter path must raise ValueError during setup."""
        from visionforge.blocks.grid_search import GridSearchBlock

        raw = _base_config(
            tmp_path,
            {"grid_search": {"hyperparameters": {"training.bogus": [1, 2]}}},
        )
        config = ExperimentConfig.model_validate(raw)
        block = GridSearchBlock()
        with pytest.raises(ValueError, match="Unknown hyperparameter path"):
            block.setup(config)

    def test_valid_setup_does_not_raise(
        self, grid_2x2_config: ExperimentConfig
    ) -> None:
        """setup() with a valid config must not raise."""
        from visionforge.blocks.grid_search import GridSearchBlock

        block = GridSearchBlock()
        block.setup(grid_2x2_config)


# ── trial count ───────────────────────────────────────────────────────────────


class TestTrialGeneration:
    def test_cartesian_product_trial_count(
        self, grid_2x2_config: ExperimentConfig
    ) -> None:
        """2x2 search space must produce 4 trials."""
        from visionforge.blocks.grid_search import GridSearchBlock

        call_count: list[int] = [0]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            return _mock_report(idx)

        block = GridSearchBlock()
        block.setup(grid_2x2_config)
        with (
            patch("visionforge.blocks._search_utils.ClassificationBlock.run", mock_run),
            patch(
                "visionforge.blocks._search_utils.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()

        assert len(block._trials) == 4

    def test_empty_search_space_runs_single_trial(
        self, base_config: ExperimentConfig
    ) -> None:
        """Empty hyperparameters dict must run exactly 1 trial."""
        from visionforge.blocks.grid_search import GridSearchBlock

        call_count: list[int] = [0]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            return _mock_report(idx)

        block = GridSearchBlock()
        block.setup(base_config)
        with (
            patch("visionforge.blocks._search_utils.ClassificationBlock.run", mock_run),
            patch(
                "visionforge.blocks._search_utils.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()

        assert len(block._trials) == 1


# ── seed propagation ──────────────────────────────────────────────────────────


class TestSeedPropagation:
    def test_trial_seeds_increment_by_index(
        self, grid_2x2_config: ExperimentConfig
    ) -> None:
        """Each trial's seed must equal base_seed + trial_index."""
        from visionforge.blocks.grid_search import GridSearchBlock

        captured_seeds: list[int] = []

        def mock_run(self: Any) -> None:  # noqa: ANN001
            captured_seeds.append(self._config.training.seed)

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            return _mock_report(0)

        block = GridSearchBlock()
        block.setup(grid_2x2_config)
        with (
            patch("visionforge.blocks._search_utils.ClassificationBlock.run", mock_run),
            patch(
                "visionforge.blocks._search_utils.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()

        base_seed = grid_2x2_config.training.seed
        expected = [base_seed + i for i in range(4)]
        assert captured_seeds == expected


# ── artifacts ─────────────────────────────────────────────────────────────────


class TestArtifacts:
    def _run_block(self, config: ExperimentConfig) -> Any:
        from visionforge.blocks.grid_search import GridSearchBlock

        call_count: list[int] = [0]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            return _mock_report(idx)

        block = GridSearchBlock()
        block.setup(config)
        with (
            patch("visionforge.blocks._search_utils.ClassificationBlock.run", mock_run),
            patch(
                "visionforge.blocks._search_utils.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()
        return block

    def test_writes_summary_csv(self, grid_2x2_config: ExperimentConfig) -> None:
        """run() must write grid_search_summary.csv with correct columns and row count."""
        self._run_block(grid_2x2_config)
        csv_path = (
            grid_2x2_config.output.reports_dir
            / grid_2x2_config.name
            / "grid_search_summary.csv"
        )
        assert csv_path.exists()

        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))

        assert len(rows) == 4
        assert "trial_index" in rows[0]
        assert "seed" in rows[0]
        assert "status" in rows[0]
        assert "test_accuracy" in rows[0]

    def test_writes_best_config_yaml(self, grid_2x2_config: ExperimentConfig) -> None:
        """run() must write best_config.yaml that is loadable as ExperimentConfig."""
        self._run_block(grid_2x2_config)
        yaml_path = (
            grid_2x2_config.output.reports_dir
            / grid_2x2_config.name
            / "best_config.yaml"
        )
        assert yaml_path.exists()

        with yaml_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        assert isinstance(raw, dict)
        assert "name" in raw
        assert "training" in raw

    def test_report_returns_best_trial(self, grid_2x2_config: ExperimentConfig) -> None:
        """report() must return dict with best_trial, total_trials, successful_trials."""
        block = self._run_block(grid_2x2_config)
        result = block.report()

        assert "best_trial" in result
        assert result["total_trials"] == 4
        assert result["successful_trials"] == 4
        assert result["best_trial"]["status"] == "success"


# ── failure handling ──────────────────────────────────────────────────────────


class TestFailureHandling:
    def test_failed_trial_recorded_and_sweep_continues(
        self, grid_2x2_config: ExperimentConfig
    ) -> None:
        """A failing trial must be recorded as failed; remaining trials must complete."""
        from visionforge.blocks.grid_search import GridSearchBlock

        run_count: list[int] = [0]
        report_count: list[int] = [0]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            idx = run_count[0]
            run_count[0] += 1
            if idx == 1:
                raise RuntimeError("injected failure")

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = report_count[0]
            report_count[0] += 1
            return _mock_report(idx)

        block = GridSearchBlock()
        block.setup(grid_2x2_config)
        with (
            patch("visionforge.blocks._search_utils.ClassificationBlock.run", mock_run),
            patch(
                "visionforge.blocks._search_utils.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()

        statuses = [t["status"] for t in block._trials]
        assert statuses.count("failed") == 1
        assert statuses.count("success") == 3
        assert block._trials[1]["error"] == "injected failure"

    def test_zero_successful_trials_raises_in_report(
        self, base_config: ExperimentConfig
    ) -> None:
        """report() must raise RuntimeError when all trials failed."""
        from visionforge.blocks.grid_search import GridSearchBlock

        def mock_run(self: Any) -> None:  # noqa: ANN001
            raise RuntimeError("always fails")

        block = GridSearchBlock()
        block.setup(base_config)
        with patch(
            "visionforge.blocks._search_utils.ClassificationBlock.run", mock_run
        ):
            block.run()

        with pytest.raises(RuntimeError, match="all trials failed"):
            block.report()

    def test_wrong_type_at_dot_key_is_recorded_as_trial_failure(
        self, tmp_path: Path
    ) -> None:
        """Trials with type-invalid hyperparameter values must be recorded as failed."""
        from visionforge.blocks.grid_search import GridSearchBlock

        raw = _base_config(
            tmp_path,
            {"grid_search": {"hyperparameters": {"training.epochs": ["a", "b"]}}},
        )
        config = ExperimentConfig.model_validate(raw)
        block = GridSearchBlock()
        block.setup(config)
        block.run()

        assert all(t["status"] == "failed" for t in block._trials)
        assert len(block._trials) == 2


# ── registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_registry_discovers_grid_search_block(self) -> None:
        """BlockRegistry must include GridSearchBlock after import."""
        import visionforge.blocks.grid_search  # noqa: F401

        registry = BlockRegistry.discover()
        assert "GridSearchBlock" in registry
