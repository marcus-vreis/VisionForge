import importlib

import pytest


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
        """blocks must expose ExperimentBlock and BlockRegistry."""
        mod = importlib.import_module("visionforge.blocks")
        assert hasattr(mod, "ExperimentBlock")
        assert hasattr(mod, "BlockRegistry")

    def test_core_is_importable(self) -> None:
        """core must expose DataModule and Trainer."""
        mod = importlib.import_module("visionforge.core")
        assert hasattr(mod, "DataModule")
        assert hasattr(mod, "Trainer")

    def test_main_entrypoint_is_importable(self) -> None:
        """__main__ must expose a callable main()."""
        mod = importlib.import_module("visionforge.__main__")
        assert callable(getattr(mod, "main", None))

    def test_main_runs_without_error(self) -> None:
        """main() must complete without raising any exception."""
        import visionforge.__main__ as main_module

        try:
            main_module.main()
        except Exception as exc:
            pytest.fail(f"main() raised an unexpected exception: {exc}")
