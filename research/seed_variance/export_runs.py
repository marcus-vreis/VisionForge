"""Copy each trial's ``run.json`` into ``runs/``, without local paths.

The per-cell reports in ``results/`` name each trial's run folder under the
gitignored ``outputs/``; this script writes the record of every one of the 270
trials to ``runs/<experiment>.json`` so ``summarize.py`` can re-derive every
number from a clean clone. Absolute paths are reduced to their final name
(ADR-174), the run folder becomes ``.`` and each artifact keeps only its file
name: checkpoints and plots are not published. ``--root`` is the checkout whose
``outputs/`` holds the runs (default: this one).

    .venv\\Scripts\\python.exe research\\seed_variance\\export_runs.py [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

from visionforge.core.public_run_record import without_local_paths

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = HERE / "runs"
REPO = HERE.parents[1]


def run_name(run_dir: str) -> str:
    """The experiment folder a trial's ``run_dir`` sits in (``<experiment>/<stamp>``)."""
    return PureWindowsPath(run_dir).parent.name


def public_record(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` with no local path: final names only, run folder ``.``."""
    record: dict[str, Any] = without_local_paths(
        {key: value for key, value in data.items() if key != "artifacts"}
    )
    record["run_dir"] = "."
    artifacts: dict[str, Any] = {}
    for category, values in data.get("artifacts", {}).items():
        if isinstance(values, list):
            artifacts[category] = [_file_name(v) for v in values]
        else:
            artifacts[category] = _file_name(values)
    record["artifacts"] = artifacts
    return record


def _file_name(value: Any) -> Any:
    return PureWindowsPath(value).name if isinstance(value, str) else value


def trial_run_dirs() -> list[str]:
    """Every trial's ``run_dir`` across the grid's reports, in a stable order."""
    dirs: list[str] = []
    for path in sorted(RESULTS.glob("*__*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for variant in report["variants"].values():
            dirs.extend(t["run_dir"] for t in variant["trials"] if t.get("run_dir"))
    return dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the study's run.json records.")
    parser.add_argument("--root", type=Path, default=REPO)
    root: Path = parser.parse_args(argv).root
    RUNS.mkdir(exist_ok=True)
    missing = []
    for run_dir in trial_run_dirs():
        source = root / PureWindowsPath(run_dir).as_posix() / "run.json"
        if not source.is_file():
            missing.append(run_dir)
            continue
        data = json.loads(source.read_text(encoding="utf-8"))
        (RUNS / f"{run_name(run_dir)}.json").write_text(
            json.dumps(public_record(data), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if missing:
        print(f"missing run.json: {', '.join(missing)}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
