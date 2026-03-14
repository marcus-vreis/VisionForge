"""
Entry point for ``python -m visionforge``.

Allows running the package directly from the command line:
    python -m visionforge
or, after installation:
    visionforge
"""

from visionforge.utils.logger import logger, setup_logger


def main() -> None:
    """Bootstrap the VisionForge application."""
    setup_logger()
    logger.info("VisionForge initialized — ready to forge vision models.")


if __name__ == "__main__":
    main()
