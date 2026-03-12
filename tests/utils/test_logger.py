import pytest
from pathlib import Path
from loguru import logger

from visionforge.utils.logger import setup_logger


@pytest.fixture(autouse=True)
def reset_logger():
    """Remove all sinks before and after each test to prevent state leaking."""
    logger.remove()
    yield
    logger.remove()


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


class TestSetupLogger:

    def test_setup_does_not_raise(self, tmp_log_dir: Path) -> None:
        """setup_logger() should run without raising any exception."""
        try:
            setup_logger(log_dir=tmp_log_dir)
        except Exception as exc:
            pytest.fail(f"setup_logger() raised an unexpected exception: {exc}")

    def test_log_file_is_created(self, tmp_log_dir: Path) -> None:
        """Log file should exist on disk after setup_logger() is called."""
        setup_logger(log_dir=tmp_log_dir)
        logger.info("triggering file creation")

        assert (tmp_log_dir / "visionforge.log").exists()

    def test_log_message_written_to_file(self, tmp_log_dir: Path) -> None:
        """Messages written with logger.info() should appear in the log file."""
        setup_logger(log_dir=tmp_log_dir)

        message = "hello from test"
        logger.info(message)
        logger.complete()

        content = (tmp_log_dir / "visionforge.log").read_text(encoding="utf-8")
        assert message in content

    def test_all_levels_written_to_file(self, tmp_log_dir: Path) -> None:
        """All log levels should be written to file regardless of terminal level."""
        # Terminal set to INFO — file must still capture DEBUG
        setup_logger(log_dir=tmp_log_dir, level="INFO")

        logger.debug("msg-debug")
        logger.info("msg-info")
        logger.warning("msg-warning")
        logger.error("msg-error")
        logger.complete()

        content = (tmp_log_dir / "visionforge.log").read_text(encoding="utf-8")

        for msg in ["msg-debug", "msg-info", "msg-warning", "msg-error"]:
            assert msg in content, f"Expected '{msg}' in log file."

    def test_double_setup_does_not_duplicate_messages(
        self, tmp_log_dir: Path, capsys
    ) -> None:
        """Calling setup_logger() twice must not duplicate terminal output."""
        setup_logger(log_dir=tmp_log_dir, level="INFO")
        setup_logger(log_dir=tmp_log_dir, level="INFO")

        logger.info("unique-message")
        logger.complete()

        captured = capsys.readouterr()
        assert captured.out.count("unique-message") == 1

    def test_log_dir_created_if_missing(self, tmp_path: Path) -> None:
        """setup_logger() should create the log directory if it does not exist."""
        missing_dir = tmp_path / "new" / "logs"
        assert not missing_dir.exists()

        setup_logger(log_dir=missing_dir)
        logger.info("trigger")
        logger.complete()

        assert missing_dir.exists()
    
    def test_log_is_none(self, tmp_log_dir: Path) -> None:
        """setup_logger() should not return any value."""
        assert setup_logger(log_dir=tmp_log_dir) is None

    
    def test_default_log_dir_used_when_log_dir_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When log_dir=None, setup_logger() must fall back to _DEFAULT_LOG_DIR."""
        import visionforge.utils.logger as logger_module

        default_dir = tmp_path / "default_logs"
        monkeypatch.setattr(logger_module, "_DEFAULT_LOG_DIR", default_dir)

        setup_logger()

        assert default_dir.exists()


class TestLoggerExports:

    def test_logger_is_importable(self) -> None:
        """logger object must be importable from the module."""
        from visionforge.utils.logger import logger as vf_logger
        assert vf_logger is not None

    def test_setup_logger_is_callable(self) -> None:
        """setup_logger must be importable and callable."""
        assert callable(setup_logger)