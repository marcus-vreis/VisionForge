"""Aggregate diag-v1: epoch time per client state, relative to native.

Reads results/diag-v1 and writes results/diag_summary.json and results/diag_summary.md.
Outcome from diag-v1/PROTOCOL.md: mean epoch time from the end of epoch 1 to the end of epoch 4.

Usage: python results/aggregate_diag.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "diag-v1"
SEEDS = (37, 38, 39)
CLIENTS = ("server", "hidden", "visible")
STAMP = "%Y-%m-%d %H:%M:%S"


def epoch_s(task: str, seed: int, side: str) -> float:
    d = DIR / task / f"seed{seed}" / side
    if side == "native":
        h = json.loads((d / "native_result.json").read_text(encoding="utf-8"))["history"]
        return (h[3]["seconds"] - h[0]["seconds"]) / 3
    t = [datetime.strptime(x, STAMP) for x in (d / "epoch_times.txt").read_text(encoding="utf-8").split("\n") if x]
    return (t[3] - t[0]).total_seconds() / 3


def ratio(values: list[float]) -> tuple[float, float, float]:
    logs = [math.log(x) for x in values]
    half = float(stats.t.ppf(0.95, len(logs) - 1)) * statistics.stdev(logs) / len(logs) ** 0.5
    mean = statistics.fmean(logs)
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def load() -> dict[tuple[str, int, str], dict[str, float]]:
    rows = [(datetime.strptime(r[0], STAMP), [float(x) for x in r[2:6]]) for r in csv.reader((DIR / "monitor.csv").open()) if len(r) >= 6]
    pat = re.compile(r"^(\w+) seed=(\d+) side=(\w+) exit=0 wall=(\d+)s (\S+ \S+)")
    out = {}
    for line in (DIR / "summary.log").read_text(encoding="utf-8").splitlines():
        if m := pat.match(line):
            end = datetime.strptime(m[5], STAMP)
            x = [r[1] for r in rows if end - timedelta(seconds=int(m[4])) <= r[0] <= end]
            out[m[1], int(m[2]), m[3]] = {k: statistics.median(v[i] for v in x) for i, k in enumerate(("python", "chrome", "node", "other"))}
    return out


def main() -> None:
    cpu = load()
    out = {}
    for task in ("cls", "seg"):
        native = [epoch_s(task, s, "native") for s in SEEDS]
        out[task] = {"native_epoch_s": native}
        for c in CLIENTS:
            e = [epoch_s(task, s, c) for s in SEEDS]
            r, low, high = ratio([a / b for a, b in zip(e, native)])
            out[task][c] = {"epoch_s": e, "ratio": r, "ci90": [low, high]}
        for c in ("hidden", "visible"):
            r, low, high = ratio([a / b for a, b in zip(out[task][c]["epoch_s"], out[task]["server"]["epoch_s"])])
            out[task][f"{c}_over_server"] = {"ratio": r, "ci90": [low, high]}
        out[task]["cpu_median_pct_of_core"] = {c: {k: statistics.fmean(cpu[task, s, c][k] for s in SEEDS) for k in ("python", "chrome", "node", "other")} for c in ("native", *CLIENTS)}
    (ROOT / "diag_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# diag-v1, epoch time by client state", "", "n = 3 blocks, 4 epochs, exploratory. Ratios with 90% CI.", "",
             "| Task | Native epoch (s) | Server/native | Hidden/native | Visible/native | Hidden/server | Visible/server |", "|---|---|---|---|---|---|---|"]
    fmt = lambda v: f"{v['ratio']:.3f} [{v['ci90'][0]:.3f}, {v['ci90'][1]:.3f}]"
    for task, v in out.items():
        lines.append(f"| {task} | {statistics.fmean(v['native_epoch_s']):.1f} | {fmt(v['server'])} | {fmt(v['hidden'])} | {fmt(v['visible'])} | {fmt(v['hidden_over_server'])} | {fmt(v['visible_over_server'])} |")
    (ROOT / "diag_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
