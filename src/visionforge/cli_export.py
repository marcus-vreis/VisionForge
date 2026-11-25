"""The ``visionforge export`` command: list or run third-party exporters (ADR-244)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from visionforge.core.exporter_plugins import discover_exporters, run_exporter
from visionforge.utils.messages import tr


def add_export_parser(subparsers: Any) -> None:
    """Register ``export`` on the CLI's subparsers."""
    parser = subparsers.add_parser(
        "export",
        help="Send a run to a third-party exporter (visionforge.exporters).",
    )
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--to", metavar="NAME", help="exporter name (see --list)")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="option passed to the exporter; repeatable",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the installed exporters and exit"
    )


def _parse_options(pairs: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(tr("exporters.bad_option", option=pair))
        options[key] = value
    return options


def run_export_command(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Run ``export``; exit 2 on a usage or input error."""
    exporters = discover_exporters()
    if args.list:
        if not exporters:
            print(tr("exporters.none_installed"))
        for name, exporter in sorted(exporters.items()):
            print(f"{name}\t{exporter.description}")
        return
    if args.run_dir is None or not args.to:
        parser.exit(2, f"visionforge: {tr('exporters.usage')}\n")
    chosen = exporters.get(args.to)
    if chosen is None:
        available = ", ".join(sorted(exporters)) or tr("exporters.none_available")
        parser.exit(
            2,
            "visionforge: "
            + tr("exporters.unknown", name=args.to, available=available)
            + "\n",
        )
    try:
        result = run_exporter(chosen, args.run_dir, _parse_options(args.option))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"visionforge: {exc}\n")
    print(result.destination)
    for note in result.notes:
        print(note, file=sys.stderr)
