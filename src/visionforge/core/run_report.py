"""A run as a folder of text: README report, public run.json and charts.

Shared by the exporters that write a run into a repository of experiments:
GitHub (ADR-175) and DVC (ADR-176). The README is generated from run.json
alone (final metrics, a table per epoch, the charts); run.json is the public
record of ADR-174, without this computer's paths.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from visionforge.core.experiment_bundle import contained_run_file
from visionforge.core.public_run_record import without_local_paths

README = "README.md"
RECORD = "run.json"
_FIGURES = {".png", ".jpg", ".jpeg"}
_MAX_FIGURE = 5 * 1024 * 1024
_MAX_FIGURES = 20
_MAX_EPOCH_ROWS = 200
_MAX_COLUMNS = 8
_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]+")
_NAME = re.compile(r"[A-Za-z0-9_./-]{1,60}")
_NO_WEIGHTS = "- The weights are not in this repository."

__all__ = [
    "README",
    "RECORD",
    "numeric_values",
    "public_record",
    "report_files",
    "run_folder",
]


def report_files(
    root: Path,
    data: dict[str, Any],
    *,
    model: str | None = None,
    weights: str = _NO_WEIGHTS,
) -> dict[str, bytes]:
    """README first, then the public run.json and ``figures/*``.

    ``model`` is the weights' name inside the folder when the repository
    holds them; ``weights`` is the README line that says where they are.
    """
    folder = run_folder(root, data)
    figures = _figures(root, data.get("artifacts", {}).get("graphics"))
    record = public_record(data)
    record["artifacts"] = {"model": model, "graphics": list(figures), "report": README}
    return {
        README: _readme(data, folder, list(figures), weights).encode("utf-8"),
        RECORD: json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8"),
        **figures,
    }


def run_folder(root: Path, data: dict[str, Any]) -> str:
    """``runs/<experiment>/<run>`` with safe segments only."""
    experiment = data.get("experiment")
    return "runs/" + "/".join(
        _segment(name) for name in (experiment, root.name) if _segment(name)
    )


def public_record(data: dict[str, Any]) -> dict[str, Any]:
    """run.json without artifacts and with local paths reduced (ADR-174)."""
    record = without_local_paths(
        {key: value for key, value in data.items() if key != "artifacts"}
    )
    record["run_dir"] = "."
    return dict(record)


def _segment(name: Any) -> str:
    """A path segment made of safe characters, or '' for nothing usable."""
    if not isinstance(name, str):
        return ""
    return _SEGMENT.sub("-", name).strip("-.")[:100]


def _figures(root: Path, graphics: Any) -> dict[str, bytes]:
    """The run's charts as ``figures/<name>``: images only, bounded in size."""
    figures: dict[str, bytes] = {}
    values = graphics if isinstance(graphics, list) else []
    for value in values[:_MAX_FIGURES]:
        if not isinstance(value, str):
            continue
        try:
            path = contained_run_file(root, value)
        except (OSError, ValueError):
            continue
        if path.suffix.lower() not in _FIGURES or path.stat().st_size > _MAX_FIGURE:
            continue
        figures[f"figures/{_segment(path.name)}"] = path.read_bytes()
    return figures


def _readme(data: dict[str, Any], folder: str, figures: list[str], weights: str) -> str:
    """A Markdown report of the run, from run.json alone."""
    lines = [f"# {PurePosixPath(folder).name}", "", *_summary(data), ""]
    lines += ["## Final metrics", "", *_table([numeric_values(data.get("metrics"))])]
    lines += _epochs(data.get("history"))
    if figures:
        lines += ["## Charts", ""]
        lines += [f"![{PurePosixPath(path).stem}]({path})" for path in figures]
        lines.append("")
    lines += [
        "## Files",
        "",
        "- `run.json`: the run record, with this computer's paths reduced to their "
        "final name.",
        weights,
        "",
        "Exported by VisionForge.",
        "",
    ]
    return "\n".join(lines)


def _summary(data: dict[str, Any]) -> list[str]:
    """``- **Label:** value`` lines for the run's plain descriptive fields."""
    config = _mapping(data.get("config"))
    fields = (
        ("Experiment", data.get("experiment")),
        ("Task", config.get("task")),
        ("Model", _mapping(config.get("model")).get("name")),
        ("Status", data.get("status")),
        ("Date", data.get("timestamp")),
    )
    return [
        f"- **{label}:** {_cell(value)}"
        for label, value in fields
        if isinstance(value, str) and value
    ]


def _epochs(history: Any) -> list[str]:
    """The per-epoch table, bounded; nothing when no epoch has numbers."""
    epochs = history if isinstance(history, list) else []
    rows = [numeric_values(row) for row in epochs[:_MAX_EPOCH_ROWS]]
    if not any(rows):
        return []
    lines = ["## Per epoch", "", *_table(rows)]
    if len(epochs) > _MAX_EPOCH_ROWS:
        lines += [f"First {_MAX_EPOCH_ROWS} of {len(epochs)} epochs.", ""]
    return lines


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def numeric_values(row: Any) -> dict[str, float]:
    """The finite numeric values of a mapping, keyed by plain names."""
    if not isinstance(row, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in row.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and _NAME.fullmatch(str(key))
    }


def _table(rows: list[dict[str, float]]) -> list[str]:
    """A Markdown table of the first ``_MAX_COLUMNS`` names; nothing if empty."""
    columns = list(dict.fromkeys(key for row in rows for key in row))[:_MAX_COLUMNS]
    if not columns:
        return ["No numeric values were recorded.", ""]
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "---|" * len(columns)
    body = [
        "| " + " | ".join(_number(row.get(key)) for key in columns) + " |"
        for row in rows
    ]
    return [header, rule, *body, ""]


def _number(value: float | None) -> str:
    if value is None:
        return ""
    return (
        str(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.4g}"
    )


def _cell(text: str) -> str:
    """Text safe inside one Markdown line: no newline, pipe or HTML."""
    return re.sub(r"[\r\n|<>`]+", " ", text)[:120]
