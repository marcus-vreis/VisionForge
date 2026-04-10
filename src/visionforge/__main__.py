"""VisionForge CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Run a VisionForge experiment from a YAML config file."""
    from visionforge.utils.config import load_config
    from visionforge.utils.logger import setup_logger

    parser = argparse.ArgumentParser(
        prog="visionforge",
        description="Run a VisionForge experiment.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=None,
        help="path to a .yaml experiment config (omit to run smoke check)",
    )
    args = parser.parse_args()

    setup_logger()

    if args.config is None:
        from visionforge.utils.logger import logger

        logger.info("VisionForge initialized — pass a config path to start training.")
        return

    from visionforge.blocks.classification import ClassificationBlock
    from visionforge.utils.logger import logger

    logger.info("Loading config: {}", args.config)
    config = load_config(args.config)

    logger.info(
        "Experiment: {} | task: {} | mode: {}",
        config.name,
        config.task,
        config.classification.mode,
    )

    block = ClassificationBlock()
    block.setup(config)

    logger.info("Starting {}...", config.classification.mode)
    block.run()

    report = block.report()
    logger.success("Done. Report: {}", report)


if __name__ == "__main__":
    main()
