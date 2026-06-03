from __future__ import annotations

import csv as csv_mod
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from visionforge.blocks._search_utils import (
    best_trial,
    make_trial_progress_wrapper,
    run_trial,
    set_nested,
    validate_dot_keys,
    write_best_config_yaml,
    write_trials_csv,
)
from visionforge.utils.config import ExperimentConfig

# ── set_nested ────────────────────────────────────────────────────────────────


class TestSetNested:
    def test_sets_top_level_key(self) -> None:
        d: dict[str, Any] = {"a": 1}
        set_nested(d, "a", 99)
        assert d["a"] == 99

    def test_sets_nested_key(self) -> None:
        d: dict[str, Any] = {"training": {"seed": 42}}
        set_nested(d, "training.seed", 7)
        assert d["training"]["seed"] == 7

    def test_unknown_intermediate_key_raises(self) -> None:
        d: dict[str, Any] = {"training": {"seed": 42}}
        with pytest.raises(ValueError, match="Unknown hyperparameter path"):
            set_nested(d, "bogus.seed", 1)

    def test_unknown_leaf_key_raises(self) -> None:
        d: dict[str, Any] = {"training": {"seed": 42}}
        with pytest.raises(ValueError, match="Unknown hyperparameter path"):
            set_nested(d, "training.bogus", 1)


# ── validate_dot_keys ─────────────────────────────────────────────────────────


class TestValidateDotKeys:
    def test_valid_keys_do_not_raise(self) -> None:
        base: dict[str, Any] = {"training": {"seed": 42, "epochs": 10}}
        validate_dot_keys(base, {"training.seed": [1, 2]})

    def test_unknown_key_raises(self) -> None:
        base: dict[str, Any] = {"training": {"seed": 42}}
        with pytest.raises(ValueError, match="Unknown hyperparameter path"):
            validate_dot_keys(base, {"training.bogus": [1, 2]})


# ── best_trial ────────────────────────────────────────────────────────────────


class TestBestTrial:
    def test_returns_none_when_no_successful_trials(self) -> None:
        trials: list[dict[str, Any]] = [{"status": "failed", "best_val_loss": None}]
        assert best_trial(trials) is None

    def test_returns_trial_with_lowest_val_loss(self) -> None:
        a: dict[str, Any] = {"status": "success", "best_val_loss": 0.5}
        b: dict[str, Any] = {"status": "success", "best_val_loss": 0.2}
        c: dict[str, Any] = {"status": "failed", "best_val_loss": None}
        trials: list[dict[str, Any]] = [a, b, c]
        result = best_trial(trials)
        assert result is not None
        assert result["best_val_loss"] == 0.2

    def test_ignores_failed_trials(self) -> None:
        trials: list[dict[str, Any]] = [
            {"status": "failed", "best_val_loss": 0.1},
            {"status": "success", "best_val_loss": 0.8},
        ]
        result = best_trial(trials)
        assert result is not None
        assert result["best_val_loss"] == 0.8


# ── write_trials_csv ──────────────────────────────────────────────────────────


class TestWriteTrialsCsv:
    def test_writes_csv_with_correct_rows(self, tmp_path: Path) -> None:
        trials = [
            {"trial_index": 0, "status": "success", "error": "", "best_val_loss": 0.5},
            {
                "trial_index": 1,
                "status": "failed",
                "error": "boom",
                "best_val_loss": None,
            },
        ]
        csv_path = tmp_path / "out.csv"
        write_trials_csv(csv_path, trials)

        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))

        assert len(rows) == 2
        assert rows[1]["status"] == "failed"
        assert rows[1]["error"] == "boom"

    def test_handles_error_message_with_commas(self, tmp_path: Path) -> None:
        """Error messages containing commas must round-trip cleanly through CSV."""
        trials = [
            {
                "trial_index": 0,
                "status": "failed",
                "error": "value is not a valid integer, got: abc",
                "best_val_loss": None,
            }
        ]
        csv_path = tmp_path / "out.csv"
        write_trials_csv(csv_path, trials)

        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))

        assert rows[0]["error"] == "value is not a valid integer, got: abc"

    def test_no_op_on_empty_trials(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "out.csv"
        write_trials_csv(csv_path, [])
        assert not csv_path.exists()


# ── write_best_config_yaml ────────────────────────────────────────────────────


class TestWriteBestConfigYaml:
    def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        import yaml

        config = ExperimentConfig.model_validate(
            {
                "name": "t",
                "task": "binary",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.01,
                    "epochs": 1,
                    "batch_size": 2,
                    "seed": 5,
                },
                "data": {
                    "base_dir": str(tmp_path),
                    "num_workers": 0,
                    "pin_memory": False,
                },
            }
        )
        yaml_path = tmp_path / "best.yaml"
        write_best_config_yaml(
            yaml_path,
            config,
            best_idx=2,
            best_overrides={"training.learning_rate": 0.001},
        )

        with yaml_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        assert raw["training"]["seed"] == 7  # 5 + 2
        assert raw["training"]["learning_rate"] == 0.001


# ── run_trial ─────────────────────────────────────────────────────────────────


class TestRunTrial:
    def _make_config(self, tmp_path: Path) -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            {
                "name": "t",
                "task": "binary",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.01,
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
            }
        )

    def test_successful_trial_sets_status(self, tmp_path: Path) -> None:
        config = self._make_config(tmp_path)
        record: dict[str, Any] = {
            "status": "failed",
            "error": "",
            "best_val_loss": None,
            "test_accuracy": None,
            "test_f1": None,
        }

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            return {
                "train": {"best_val_loss": 0.3},
                "eval": {"accuracy": 0.9, "f1": 0.85},
            }

        with (
            patch("visionforge.blocks._search_utils.ClassificationBlock.run", mock_run),
            patch(
                "visionforge.blocks._search_utils.ClassificationBlock.report",
                mock_report,
            ),
        ):
            run_trial(config, 0, 1, 10, {}, record)

        assert record["status"] == "success"
        assert record["best_val_loss"] == 0.3
        assert record["test_accuracy"] == 0.9

    def test_failed_trial_records_error(self, tmp_path: Path) -> None:
        config = self._make_config(tmp_path)
        record: dict[str, Any] = {
            "status": "failed",
            "error": "",
            "best_val_loss": None,
            "test_accuracy": None,
            "test_f1": None,
        }

        def mock_run(self: Any) -> None:  # noqa: ANN001
            raise RuntimeError("injected")

        with patch(
            "visionforge.blocks._search_utils.ClassificationBlock.run", mock_run
        ):
            run_trial(config, 0, 1, 10, {}, record)

        assert record["status"] == "failed"
        assert "injected" in record["error"]

    def test_vram_cache_cleared_on_success(self, tmp_path: Path) -> None:
        config = self._make_config(tmp_path)
        record: dict[str, Any] = {
            "status": "failed",
            "error": "",
            "best_val_loss": None,
            "test_accuracy": None,
            "test_f1": None,
        }

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            return {
                "train": {"best_val_loss": 0.3},
                "eval": {"accuracy": 0.9, "f1": 0.85},
            }

        with (
            patch("visionforge.blocks._search_utils.ClassificationBlock.run", mock_run),
            patch(
                "visionforge.blocks._search_utils.ClassificationBlock.report",
                mock_report,
            ),
            patch(
                "visionforge.blocks._search_utils.torch.cuda.empty_cache"
            ) as mock_cache,
        ):
            run_trial(config, 0, 1, 10, {}, record)

        mock_cache.assert_called_once()

    def test_vram_cache_cleared_on_failure(self, tmp_path: Path) -> None:
        config = self._make_config(tmp_path)
        record: dict[str, Any] = {
            "status": "failed",
            "error": "",
            "best_val_loss": None,
            "test_accuracy": None,
            "test_f1": None,
        }

        def mock_run(self: Any) -> None:  # noqa: ANN001
            raise RuntimeError("injected")

        with (
            patch("visionforge.blocks._search_utils.ClassificationBlock.run", mock_run),
            patch(
                "visionforge.blocks._search_utils.torch.cuda.empty_cache"
            ) as mock_cache,
        ):
            run_trial(config, 0, 1, 10, {}, record)

        mock_cache.assert_called_once()

    def test_forwards_progress_callback_to_inner_block(self, tmp_path: Path) -> None:
        """run_trial must inject its progress_callback onto the inner block so the
        trial's Trainer streams epoch progress to the GUI."""
        config = self._make_config(tmp_path)
        record: dict[str, Any] = {
            "status": "failed",
            "error": "",
            "best_val_loss": None,
            "test_accuracy": None,
            "test_f1": None,
        }
        seen: dict[str, Any] = {}

        def mock_run(self: Any) -> None:  # noqa: ANN001
            seen["callback"] = self._progress_callback

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            return {"train": {"best_val_loss": 0.3}, "eval": {"accuracy": 0.9}}

        def cb(_event: dict[str, Any]) -> None:
            pass

        with (
            patch("visionforge.blocks._search_utils.ClassificationBlock.run", mock_run),
            patch(
                "visionforge.blocks._search_utils.ClassificationBlock.report",
                mock_report,
            ),
        ):
            run_trial(config, 0, 1, 10, {}, record, progress_callback=cb)

        assert seen["callback"] is cb


# ── make_trial_progress_wrapper ───────────────────────────────────────────────


class TestTrialProgressWrapper:
    def test_returns_none_for_none_callback(self) -> None:
        assert make_trial_progress_wrapper(None, 0, 1) is None

    def test_drops_inner_start_event(self) -> None:
        """The sweep emits its own trial_start; the inner Trainer 'start' is noise."""
        seen: list[dict[str, Any]] = []
        wrapped = make_trial_progress_wrapper(seen.append, 2, 5)
        assert wrapped is not None
        wrapped({"event": "start", "total_epochs": 3, "device": "cpu"})
        assert seen == []

    def test_annotates_epoch_with_trial_context(self) -> None:
        seen: list[dict[str, Any]] = []
        wrapped = make_trial_progress_wrapper(seen.append, 2, 5)
        assert wrapped is not None
        wrapped({"event": "epoch_end", "epoch": 1, "total_epochs": 3})
        assert seen[0]["event"] == "epoch_end"
        assert seen[0]["trial_index"] == 2
        assert seen[0]["total_trials"] == 5
        assert seen[0]["epoch"] == 1

    def test_rewrites_inner_end_to_trial_end(self) -> None:
        """A bare 'end' closes the SSE stream client-side; rewriting it to
        'trial_end' keeps the stream open for the next trial."""
        seen: list[dict[str, Any]] = []
        wrapped = make_trial_progress_wrapper(seen.append, 3, 5)
        assert wrapped is not None
        wrapped({"event": "end", "total_epochs": 2})
        assert seen[0]["event"] == "trial_end"
        assert seen[0]["trial_index"] == 3
