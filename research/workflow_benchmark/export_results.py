"""Copy the workflow-benchmark series the paper cites into ``results/``, scrubbed.

The series behind the Research impact statement of ``paper/paper.md`` were
measured in the maintainer's research checkout. This script copies the four
cited series (``real-v1``, ``gui-v1``, ``diag-v1``, ``gui-v2``) with their
recipes, protocols, per-run records, logs and CPU monitors, plus the original
aggregation scripts and their summaries. Screenshots are left out. Every text
file has the home folder of any local path replaced by ``~`` (so the account
name never leaves the machine); nothing else is changed.

    .venv\\Scripts\\python.exe research\\workflow_benchmark\\export_results.py --source DIR
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SERIES = ("real-v1", "gui-v1", "diag-v1", "gui-v2")
TOP_FILES = (
    "aggregate_real.py",
    "aggregate_gui.py",
    "aggregate_gui_v2.py",
    "aggregate_diag.py",
    "real_summary.json",
    "real_summary.md",
    "gui_summary.json",
    "gui_summary.md",
    "gui_v2_summary.json",
    "gui_v2_summary.md",
    "diag_summary.json",
    "diag_summary.md",
)
SKIPPED_SUFFIXES = {".png", ".jpg", ".jpeg", ".pt", ".pth"}
# One path separator as it appears in a file: "/", "\" or JSON's escaped "\\".
_SEP = r"(?:\\\\|\\|/)"
# The home folder of a local path: C:\Users\<name>, /c/Users/<name>, /home/<name>.
HOME = re.compile(
    rf"(?:\b[A-Za-z]:{_SEP}+|(?<![\w.~])/[A-Za-z]/)Users{_SEP}+[^\\/\"'\s]+"
    r"|(?<![\w.~])/home/[^\\/\"'\s]+",
    re.IGNORECASE,
)


def scrub(text: str) -> str:
    """``text`` with the home folder of every local path replaced by ``~``."""
    return HOME.sub("~", text)


def files_to_publish(source: Path) -> list[Path]:
    """The files of the cited series and the top-level analysis, relative to ``source``."""
    found = [Path(name) for name in TOP_FILES if (source / name).is_file()]
    for series in SERIES:
        for path in sorted((source / series).rglob("*")):
            if path.is_file() and path.suffix.lower() not in SKIPPED_SUFFIXES:
                found.append(path.relative_to(source))
    return found


def export(source: Path, target: Path) -> int:
    """Copy and scrub every file; the number of files written."""
    files = files_to_publish(source)
    for rel in files:
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = (source / rel).read_text(encoding="utf-8")
        out.write_text(scrub(text), encoding="utf-8", newline="\n")
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the cited benchmark series.")
    parser.add_argument("--source", type=Path, required=True)
    source: Path = parser.parse_args(argv).source
    if not all((source / s).is_dir() for s in SERIES):
        print(f"{source} lacks one of {', '.join(SERIES)}", file=sys.stderr)
        return 1
    print(f"{export(source, RESULTS)} files written to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
