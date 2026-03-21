from visionforge.utils.logger import logger, setup_logger


def main() -> None:
    """Bootstrap the VisionForge application."""
    setup_logger()
    logger.info("VisionForge initialized — ready to forge vision models.")


if __name__ == "__main__":
    main()
