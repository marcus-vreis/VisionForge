"""VisionForge CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Run a VisionForge experiment from a YAML config file."""
    from visionforge.utils.logger import setup_logger

    parser = argparse.ArgumentParser(
        prog="visionforge",
        description="VisionForge — Computer Vision experimentation platform.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment from YAML.")
    run_parser.add_argument("config", type=Path, help="path to .yaml config")

    gui_parser = subparsers.add_parser("gui", help="Start the VisionForge web GUI.")
    gui_parser.add_argument(
        "--host", default="127.0.0.1", help="bind host (default 127.0.0.1)"
    )
    gui_parser.add_argument(
        "--port", type=int, default=8000, help="bind port (default 8000)"
    )

    args = parser.parse_args()

    setup_logger()

    if args.command == "gui":
        from visionforge.gui.server import start_server

        start_server(host=args.host, port=args.port)
        return

    if args.command == "run":
        import visionforge.blocks.classification  # noqa: F401
        import visionforge.blocks.cross_validation  # noqa: F401
        import visionforge.blocks.grid_search  # noqa: F401
        import visionforge.blocks.random_search  # noqa: F401
        from visionforge.blocks.registry import BlockRegistry
        from visionforge.utils.config import load_config
        from visionforge.utils.logger import logger

        # Map config.block snake_case literals to registry class names.
        _BLOCK_ALIASES: dict[str, str] = {
            "classification": "ClassificationBlock",
            "cross_validation": "CrossValidationBlock",
            "grid_search": "GridSearchBlock",
            "random_search": "RandomSearchBlock",
        }

        logger.info("Loading config: {}", args.config)
        config = load_config(args.config)

        logger.info(
            "Experiment: {} | task: {} | block: {}",
            config.name,
            config.task,
            config.block,
        )

        registry = BlockRegistry.discover()
        class_name = _BLOCK_ALIASES[config.block]
        block_cls = registry[class_name]
        block = block_cls()
        block.setup(config)

        logger.info("Running {}...", class_name)
        block.run()

        report = block.report()
        logger.success("Done. Report: {}", report)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
