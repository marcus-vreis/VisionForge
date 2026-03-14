import sys
from pathlib import Path

from loguru import logger

_DEFAULT_LOG_DIR = Path.cwd() / "outputs" / "logs"

LOG_FORMAT_TERMINAL = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

LOG_FORMAT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)


def setup_logger(level: str = "DEBUG", log_dir: Path | None = None) -> None:
    """
    Configure logger sinks: colored terminal output and rotating file sink.

    Args:
        level:   Minimum level for terminal output. Use "INFO" in production.
        log_dir: Directory for log files. Defaults to outputs/logs/.
    """
    if log_dir is None:
        log_dir = _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sink=sys.stdout,
        colorize=True,
        level=level,
        format=LOG_FORMAT_TERMINAL,
    )

    logger.add(
        sink=log_dir / "visionforge.log",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        level="DEBUG",
        format=LOG_FORMAT_FILE,
        encoding="utf-8",
    )


__all__ = ["logger", "setup_logger"]
