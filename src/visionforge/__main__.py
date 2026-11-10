"""VisionForge CLI entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from visionforge import __version__

# Map ExperimentConfig.block snake_case literals to BlockRegistry class names.
# A URL, not `user_tasks/README.md`: that file ships only in the source tree,
# and whoever runs `new-task` most likely installed from PyPI (ADR-058).
_TASK_GUIDE_URL = (
    "https://github.com/marcus-vreis/VisionForge/blob/main/docs/custom/TASKS.md"
)

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


_RUN_EPILOG = """\
strategies (strategy.kind):
  replicates             train config once per seed; mean, 95% CI per metric
                         options: seeds [..] | n_replicates, metric
  replicated_comparison  train each variant on the same seeds; paired tests
                         options: variants {label: {dot.path: value}},
                         seeds | n_replicates, metric, alpha
  sweep                  grid/random/optuna search ranked on validation
                         (detection, regression, segmentation, anomaly)
                         options: search_space, mode, n_trials, seed, metric
  cv                     K-fold cross-validation (regression, segmentation)
                         options: n_folds, shuffle, fold_seed

example:
  strategy:
    kind: replicates
    seeds: [0, 1, 2]
  config:
    task: regression
    name: reg_resnet18
    ...
"""


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


def _bind_language_from_env() -> None:
    """Bind ``VISIONFORGE_LANG`` for the whole run, when it names a language (ADR-116).

    Without it the command line has no language: messages are Portuguese and the
    figures it saves are English. Under ``gui`` the page's own header rebinds a
    language on every request, so this changes nothing there.
    """
    from visionforge.utils.messages import lang_from_env, set_lang

    lang = lang_from_env(os.environ)
    if lang is not None:
        set_lang(lang)


def _parse_audit_splits(values: list[str] | None) -> dict[str, str] | None:
    """Parse unique NAME=PATH pairs for a dataset audit."""
    if not values:
        return None
    splits: dict[str, str] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in splits:
            raise ValueError("Each split must be a unique NAME=PATH pair.")
        splits[name] = path
    return splits


def _run_audit_data_command(args: argparse.Namespace) -> None:
    """Audit a dataset and optionally save a new JSON report."""
    from visionforge.core.dataset_audit import audit_dataset

    splits = _parse_audit_splits(args.split)
    report = audit_dataset(
        args.root,
        splits,
        group_manifest=args.groups,
        strict_groups=args.strict_groups,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    print(payload)
    if not report["ok"]:
        raise SystemExit(1)


def _run_bundle_command(args: argparse.Namespace) -> None:
    """Export an experiment bundle and print its path as JSON."""
    from visionforge.core.experiment_bundle import export_experiment_bundle

    path = export_experiment_bundle(args.run_dir, args.destination)
    print(json.dumps({"bundle": str(path)}, ensure_ascii=False))


def _run_verify_bundle_command(args: argparse.Namespace) -> None:
    """Verify a bundle and print the resulting JSON report."""
    from visionforge.core.experiment_bundle import verify_experiment_bundle

    print(
        json.dumps(verify_experiment_bundle(args.bundle), indent=2, ensure_ascii=False)
    )


def _run_evidence_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Run read-only audits or explicit bundle exports with JSON output."""
    commands = {
        "audit-data": _run_audit_data_command,
        "bundle": _run_bundle_command,
    }
    command = commands.get(args.command, _run_verify_bundle_command)
    try:
        command(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"visionforge: {exc}\n")


def _run_selftest_command(args: argparse.Namespace) -> None:
    """Validate selections and run the self-test command."""
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


def _run_strategy_command(config_path: Path, parser: argparse.ArgumentParser) -> bool:
    """Run a strategy envelope (ADR-184); False when the YAML is a plain config.

    An invalid envelope, task or option exits with status 2 before any
    training; a job none of whose units produced a result exits with 1. A file
    that cannot be read is left to the plain path, which reports it as it
    always has.
    """
    from visionforge.cli_strategies import read_envelope, run_strategy
    from visionforge.utils.logger import logger

    try:
        envelope = read_envelope(config_path)
    except OSError:
        return False
    except ValueError as exc:
        parser.exit(2, f"visionforge: {exc}\n")
    if envelope is None:
        return False
    strategy, config = envelope
    try:
        logger.info(
            "Strategy: {} | task: {}",
            strategy["kind"],
            config.get("task", "multiclass"),
        )
        report = run_strategy(strategy, config)
    except ValueError as exc:
        parser.exit(2, f"visionforge: {exc}\n")
    except RuntimeError as exc:
        parser.exit(1, f"visionforge: {exc}\n")
    logger.success("Done. Report: {}", report["report_dir"])
    return True


def _use_utf8_console() -> None:
    """Make stdout and stderr UTF-8, before anything prints (``--help`` included).

    Windows consoles default to a legacy codepage (cp1252), which cannot encode
    the arrows and box characters the reports use. Without this,
    `visionforge selftest` crashed with a UnicodeEncodeError while printing a
    failure — the diagnostic command dying exactly when it found something. It
    runs before the parser is built: argparse prints ``--help`` and exits inside
    ``parse_args``, so a later call never reached it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    """Run a VisionForge experiment or start the GUI from the command line."""
    _use_utf8_console()
    from visionforge.utils.logger import setup_logger
    from visionforge.utils.third_party_privacy import (
        apply_env_opt_outs,
        apply_opt_outs,
    )

    # Before any SDK is imported: they read their switches at import time.
    apply_env_opt_outs()

    parser = argparse.ArgumentParser(
        prog="visionforge",
        description="VisionForge — Computer Vision experimentation platform.",
    )
    # A bug report without a version is a bug report nobody can act on.
    parser.add_argument(
        "--version", action="version", version=f"visionforge {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment from YAML.",
        description=(
            "Run an experiment from YAML.\n\n"
            "A plain task config trains once (or runs its classification block).\n"
            "A YAML whose top level is 'strategy:' + 'config:' runs the same\n"
            "multi-run jobs as the GUI and writes their report (JSON, CSV,\n"
            "LaTeX table) and History entry. 'config' is the task config as is."
        ),
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument("config", type=Path, help="path to .yaml config")

    gui_parser = subparsers.add_parser("gui", help="Start the VisionForge web GUI.")
    gui_parser.add_argument(
        "--host", default="127.0.0.1", help="bind host (default 127.0.0.1)"
    )
    gui_parser.add_argument(
        "--port", type=int, default=8000, help="bind port (default 8000)"
    )
    gui_parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "extra host name the server answers to, besides localhost (and, with "
            "a non-loopback --host, this machine's names and addresses); repeatable"
        ),
    )
    gui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser tab (used when an update restarts the GUI)",
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
        help="Check the install by training every task on generated data.",
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
        help="Create a new task of your own under user_tasks/.",
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

    audit_parser = subparsers.add_parser(
        "audit-data", help="Audit dataset splits without training."
    )
    audit_parser.add_argument("root", type=Path, help="dataset root")
    audit_parser.add_argument(
        "--split", action="append", help="split NAME=PATH; repeat for each split"
    )
    audit_parser.add_argument(
        "--groups", type=Path, help="path,group CSV relative to the dataset root"
    )
    audit_parser.add_argument(
        "--strict-groups", action="store_true", help="report groups crossing splits"
    )
    audit_parser.add_argument(
        "--json", type=Path, help="write a new JSON report (never overwrite)"
    )
    bundle_parser = subparsers.add_parser(
        "bundle", help="Export an experiment record with integrity hashes."
    )
    bundle_parser.add_argument("run_dir", type=Path)
    bundle_parser.add_argument("destination", type=Path)
    verify_parser = subparsers.add_parser(
        "verify-bundle", help="Verify an experiment record without extracting it."
    )
    verify_parser.add_argument("bundle", type=Path)

    args = parser.parse_args()
    _bind_language_from_env()
    if args.command != "gui":
        # Imports ultralytics (~1.7 s) to switch its events off. The GUI imports
        # it only when a run needs it, and each of those imports does this first
        # (ADR-180, #258).
        apply_opt_outs()

    if args.command in {"audit-data", "bundle", "verify-bundle"}:
        _run_evidence_command(args, parser)
        return

    setup_logger()

    if args.command == "selftest":
        _run_selftest_command(args)

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
            "'{}' tab appears automatically. Guide: {}",
            target,
            args.key,
            _TASK_GUIDE_URL,
        )
        return

    if args.command == "doctor":
        import sys as _sys

        from visionforge.utils.doctor import _default_confirm, run_doctor

        _sys.exit(run_doctor(fix=args.fix, confirm_fn=_default_confirm))

    if args.command == "gui":
        from visionforge.gui.server import start_server

        # The updater relaunches with VISIONFORGE_NO_BROWSER rather than the flag:
        # the relaunch runs the *new* version, and an environment variable it does
        # not know is ignored where an unknown flag would stop it (ADR-120).
        quiet = args.no_browser or bool(os.environ.get("VISIONFORGE_NO_BROWSER"))
        start_server(
            host=args.host,
            port=args.port,
            open_browser=not quiet,
            allowed_hosts=args.allowed_host,
        )
        return

    if args.command == "run":
        from visionforge.utils.logger import logger

        logger.info("Loading config: {}", args.config)
        if _run_strategy_command(args.config, parser):
            return
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
