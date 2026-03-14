"""
Smoke test — verifies that the VisionForge package can be imported
and that its public surface is minimally functional.

These tests run first in CI to catch packaging / import errors early,
before the full test suite is executed.
"""

import importlib


class TestPackageImport:
    def test_package_is_importable(self) -> None:
        """The top-level visionforge package must be importable."""
        import visionforge  # noqa: F401

    def test_utils_config_is_importable(self) -> None:
        """The config utilities must be importable."""
        mod = importlib.import_module("visionforge.utils.config")
        assert hasattr(mod, "ExperimentConfig")
        assert hasattr(mod, "load_config")

    def test_utils_logger_is_importable(self) -> None:
        """The logger utilities must be importable."""
        mod = importlib.import_module("visionforge.utils.logger")
        assert hasattr(mod, "logger")
        assert hasattr(mod, "setup_logger")

    def test_main_entrypoint_is_importable(self) -> None:
        """The __main__ entrypoint must be importable and expose main()."""
        mod = importlib.import_module("visionforge.__main__")
        assert callable(getattr(mod, "main", None))
