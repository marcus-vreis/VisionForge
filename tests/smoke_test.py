import importlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestPackageImport:
    def test_package_is_importable(self) -> None:
        """The top-level visionforge package must be importable."""
        import visionforge  # noqa: F401

    def test_utils_config_is_importable(self) -> None:
        """utils.config must expose ExperimentConfig and load_config."""
        mod = importlib.import_module("visionforge.utils.config")
        assert hasattr(mod, "ExperimentConfig")
        assert hasattr(mod, "load_config")
        assert hasattr(mod, "TransformConfig")

    def test_utils_logger_is_importable(self) -> None:
        """utils.logger must expose logger and setup_logger."""
        mod = importlib.import_module("visionforge.utils.logger")
        assert hasattr(mod, "logger")
        assert hasattr(mod, "setup_logger")

    def test_utils_cuda_is_importable(self) -> None:
        """utils.cuda must expose CUDAInfo, check_cuda and log_cuda_status."""
        mod = importlib.import_module("visionforge.utils.cuda")
        assert hasattr(mod, "CUDAInfo")
        assert hasattr(mod, "check_cuda")
        assert hasattr(mod, "log_cuda_status")

    def test_models_factory_is_importable(self) -> None:
        """models must expose ModelFactory."""
        mod = importlib.import_module("visionforge.models")
        assert hasattr(mod, "ModelFactory")

    def test_blocks_are_importable(self) -> None:
        """blocks must expose ExperimentBlock, ClassificationBlock and BlockRegistry."""
        mod = importlib.import_module("visionforge.blocks")
        assert hasattr(mod, "ExperimentBlock")
        assert hasattr(mod, "ClassificationBlock")
        assert hasattr(mod, "BlockRegistry")

    def test_core_is_importable(self) -> None:
        """core must expose DataModule, Trainer, Evaluator and MetricsPlotter."""
        mod = importlib.import_module("visionforge.core")
        assert hasattr(mod, "DataModule")
        assert hasattr(mod, "Trainer")
        assert hasattr(mod, "Evaluator")
        assert hasattr(mod, "MetricsPlotter")

    def test_config_exposes_classification_config(self) -> None:
        """utils.config must expose ClassificationConfig."""
        mod = importlib.import_module("visionforge.utils.config")
        assert hasattr(mod, "ClassificationConfig")

    def test_main_entrypoint_is_importable(self) -> None:
        """__main__ must expose a callable main()."""
        mod = importlib.import_module("visionforge.__main__")
        assert callable(getattr(mod, "main", None))

    def test_main_runs_without_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() must complete without raising any exception."""
        import visionforge.__main__ as main_module

        monkeypatch.setattr(sys, "argv", ["visionforge"])
        try:
            main_module.main()
        except Exception as exc:
            pytest.fail(f"main() raised an unexpected exception: {exc}")


class TestMainWithConfig:
    def test_main_loads_config_and_runs_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() with a config path must load config and run the block."""
        dataset_dir = tmp_path / "data"
        for split in ["train", "val", "test"]:
            for cls in ["a", "b"]:
                (dataset_dir / split / cls).mkdir(parents=True)

        config_data: dict[str, Any] = {
            "name": "cli_test",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.01,
                "epochs": 1,
                "batch_size": 1,
                "seed": 0,
            },
            "data": {"base_dir": str(dataset_dir)},
            "output": {
                "models_dir": str(tmp_path / "models"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
            },
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        mock_block = MagicMock()
        mock_block.report.return_value = {"status": "ok"}

        monkeypatch.setattr(sys, "argv", ["visionforge", str(config_file)])

        with patch(
            "visionforge.blocks.classification.ClassificationBlock",
            return_value=mock_block,
        ):
            from visionforge.__main__ import main

            main()

        mock_block.setup.assert_called_once()
        mock_block.run.assert_called_once()
        mock_block.report.assert_called_once()
