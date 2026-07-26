"""VisionForge CLI entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Map ExperimentConfig.block snake_case literals to BlockRegistry class names.
_BLOCK_ALIASES: dict[str, str] = {
    "classification": "ClassificationBlock",
    "cross_validation": "CrossValidationBlock",
    "grid_search": "GridSearchBlock",
    "random_search": "RandomSearchBlock",
    "transfer_learning": "TransferLearningBlock",
    "model_comparison": "ModelComparisonBlock",
    "batch_prediction": "BatchPredictionBlock",
    "export_onnx": "ExportONNXBlock",
}


def _peek_task(config_path: Path) -> str:
    """Read just the ``task`` field from a YAML config (defaults to multiclass)."""
    import yaml

    with config_path.open(encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, dict):
        return "multiclass"
    return str(raw.get("task", "multiclass"))


def build_task_block(config_path: Path) -> tuple[Any, Any]:
    """Load a config and return ``(config, block)`` ready to run, dispatched by task.

    The standalone tasks (detection/regression/segmentation/anomaly) go to their
    own config loader + block; the classification family goes to
    ``ExperimentConfig`` + the ``BlockRegistry`` block named by ``config.block``.
    The block is ``setup()``-d but not run.

    Raises:
        ValueError: if the config's ``task`` is not recognized.
    """
    task = _peek_task(config_path)

    if task == "detection":
        from visionforge.blocks.detection import DetectionBlock
        from visionforge.utils.detection_config import load_detection_config

        config: Any = load_detection_config(config_path)
        block: Any = DetectionBlock()
    elif task == "regression":
        from visionforge.blocks.regression import RegressionBlock
        from visionforge.utils.regression_config import load_regression_config

        config = load_regression_config(config_path)
        block = RegressionBlock()
    elif task == "segmentation":
        from visionforge.blocks.segmentation import SegmentationBlock
        from visionforge.utils.segmentation_config import load_segmentation_config

        config = load_segmentation_config(config_path)
        block = SegmentationBlock()
    elif task == "anomaly":
        from visionforge.blocks.anomaly import AnomalyBlock
        from visionforge.utils.anomaly_config import load_anomaly_config

        config = load_anomaly_config(config_path)
        block = AnomalyBlock()
    elif task in ("binary", "multiclass"):
        import visionforge.blocks.batch_prediction  # noqa: F401
        import visionforge.blocks.classification  # noqa: F401
        import visionforge.blocks.cross_validation  # noqa: F401
        import visionforge.blocks.export_onnx  # noqa: F401
        import visionforge.blocks.grid_search  # noqa: F401
        import visionforge.blocks.model_comparison  # noqa: F401
        import visionforge.blocks.random_search  # noqa: F401
        import visionforge.blocks.transfer_learning  # noqa: F401
        from visionforge.blocks.registry import BlockRegistry
        from visionforge.utils.config import load_config

        config = load_config(config_path)
        registry = BlockRegistry.discover()
        block = registry[_BLOCK_ALIASES[config.block]]()
    else:
        raise ValueError(
            f"Unknown task '{task}'. Expected one of: binary, multiclass, "
            f"detection, regression, segmentation, anomaly."
        )

    block.setup(config)
    return config, block


def main() -> None:
    """Run a VisionForge experiment or start the GUI from the command line."""
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

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check the environment and recommend the correct torch wheel."
    )
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="Run the recommended install after prompting for confirmation.",
    )

    selftest_parser = subparsers.add_parser(
        "selftest",
        help="Train every task through the real GUI API on synthetic data (ADR-060).",
    )
    selftest_parser.add_argument(
        "--tasks",
        default="all",
        help="comma-separated task keys, or 'all' (default): "
        "classification,regression,segmentation,anomaly,detection,custom",
    )
    selftest_parser.add_argument(
        "--strategies",
        default="all",
        help="comma-separated strategies, or 'all' (default): simple,cv,sweep,replicates",
    )
    selftest_parser.add_argument(
        "--quick",
        action="store_true",
        help="only the 'simple' strategy — a fast is-my-install-sane check",
    )
    selftest_parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="where datasets and outputs land (default: a temp dir, kept on failure)",
    )
    selftest_parser.add_argument(
        "--json", type=Path, default=None, help="also write the outcomes as JSON here"
    )

    newtask_parser = subparsers.add_parser(
        "new-task",
        help="Scaffold a custom task under user_tasks/ (ADR-058).",
    )
    newtask_parser.add_argument(
        "key", help="task key: lowercase letters/digits/underscores"
    )
    newtask_parser.add_argument(
        "--package",
        action="store_true",
        help="create user_tasks/<key>/task.py (room for assets) instead of a flat file",
    )
    newtask_parser.add_argument(
        "--force", action="store_true", help="overwrite an existing file"
    )

    args = parser.parse_args()

    setup_logger()

    if args.command == "selftest":
        import sys as _sys
        import tempfile

        from visionforge.utils.logger import logger
        from visionforge.utils.selftest import (
            STRATEGIES,
            TASKS,
            format_report,
            run_selftest,
        )

        tasks = TASKS if args.tasks == "all" else tuple(args.tasks.split(","))
        if args.quick:
            strategies: tuple[str, ...] = ("simple",)
        else:
            strategies = (
                STRATEGIES
                if args.strategies == "all"
                else tuple(args.strategies.split(","))
            )
        unknown = [t for t in tasks if t not in TASKS] + [
            s for s in strategies if s not in STRATEGIES
        ]
        if unknown:
            logger.error("Unknown task/strategy: {}", ", ".join(unknown))
            _sys.exit(2)

        workdir = args.workdir or Path(tempfile.mkdtemp(prefix="visionforge_selftest_"))
        logger.info("Self-test workdir: {}", workdir)
        outcomes = run_selftest(workdir, tasks=tasks, strategies=strategies)
        report = format_report(outcomes)
        print(f"\n{report}\n")  # noqa: T201 — the table IS the command's output

        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps([vars(o) for o in outcomes], indent=2), encoding="utf-8"
            )
            logger.info("Wrote {}", args.json)

        _sys.exit(0 if all(o.status == "passed" for o in outcomes) else 1)

    if args.command == "new-task":
        import sys as _sys

        from visionforge.tasks.scaffold import scaffold_task
        from visionforge.utils.logger import logger

        try:
            target = scaffold_task(args.key, package=args.package, force=args.force)
        except (ValueError, FileExistsError) as exc:
            logger.error("{}", exc)
            _sys.exit(1)
        logger.success("Task template created: {}", target)
        logger.info(
            "Next: edit the TODOs in {}, then `visionforge gui` — the "
            "'{}' tab appears automatically. Guide: user_tasks/README.md",
            target,
            args.key,
        )
        return

    if args.command == "doctor":
        import sys as _sys

        from visionforge.utils.doctor import _default_confirm, run_doctor

        _sys.exit(run_doctor(fix=args.fix, confirm_fn=_default_confirm))

    if args.command == "gui":
        from visionforge.gui.server import start_server

        start_server(host=args.host, port=args.port)
        return

    if args.command == "run":
        from visionforge.utils.logger import logger

        logger.info("Loading config: {}", args.config)
        config, block = build_task_block(args.config)

        logger.info(
            "Experiment: {} | task: {} | block: {}",
            config.name,
            config.task,
            type(block).__name__,
        )
        logger.info("Running {}...", type(block).__name__)
        block.run()

        report = block.report()
        logger.success("Done. Report: {}", report)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
