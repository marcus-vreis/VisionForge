"""Aggregate the gui-v1 series (VisionForge through the browser vs native, same recipe).

Reads results/gui-v1 and results/real-v1 and writes results/gui_summary.json and results/gui_summary.md.
Outcomes and margins come from results/gui-v1/PROTOCOL.md, frozen before the runs. No run is dropped.

Usage: python results/aggregate_gui.py
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "gui-v1"
SEEDS = (37, 38, 39)
TASKS = {
    "cls": ("ResNet-18, CIFAR-10, 30 ep", "test accuracy", "test_accuracy"),
    "seg": ("DeepLabV3-ResNet50, Pet, 40 ep", "test mIoU", "test_miou"),
}
TIME_MARGIN = 0.05
QUALITY_MARGIN = 0.005


def ci(values: list[float], level: float) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    half = float(stats.t.ppf(0.5 + level / 2, len(values) - 1)) * statistics.stdev(values) / len(values) ** 0.5
    return mean, mean - half, mean + half


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def native_metric(base: Path, task: str, seed: int) -> float:
    n = load(base / task / f"seed{seed}" / "native" / "native_result.json")
    return n["test_accuracy"] if task == "cls" else n["test_metrics"]["miou"]


def main() -> None:
    pat = re.compile(r"^(\w+) seed=(\d+) side=(\w+) exit=0 wall=(\d+)s")
    wall = {}
    for line in (DIR / "summary.log").read_text(encoding="utf-8").splitlines():
        if m := pat.match(line):
            wall[m[1], int(m[2]), m[3]] = int(m[4])
    out = {}
    for task, (recipe, name, key) in TASKS.items():
        gui = [load(DIR / task / f"seed{s}" / "gui" / "run.json")["metrics"][key] for s in SEEDS]
        nat = [native_metric(DIR, task, s) for s in SEEDS]
        cli = [load(ROOT / "real-v1" / task / f"seed{s}" / "vf" / "run.json")["metrics"][key] for s in SEEDS]
        diff, low, high = ci([a - b for a, b in zip(gui, nat)], 0.95)
        ratios = [wall[task, s, "gui"] / wall[task, s, "native"] for s in SEEDS]
        r, r_low, r_high = (math.exp(x) for x in ci([math.log(x) for x in ratios], 0.90))
        click = [load(DIR / task / f"seed{s}" / "gui" / "timing.json")["click_to_done_s"] for s in SEEDS]
        c_ratios = [c / wall[task, s, "native"] for c, s in zip(click, SEEDS)]
        c, c_low, c_high = (math.exp(x) for x in ci([math.log(x) for x in c_ratios], 0.90))
        out[task] = {
            "recipe": recipe, "metric": name, "seeds": SEEDS, "gui": gui, "native": nat, "cli_real_v1": cli,
            "diff": diff, "diff_ci95": [low, high],
            "quality_within": -QUALITY_MARGIN <= low and high <= QUALITY_MARGIN,
            "gui_minus_cli": [a - b for a, b in zip(gui, cli)],
            "wall_gui": [wall[task, s, "gui"] for s in SEEDS], "wall_native": [wall[task, s, "native"] for s in SEEDS],
            "ratio": r, "ratio_ci90": [r_low, r_high], "ratio_per_seed": ratios,
            "time_within": 1 - TIME_MARGIN <= r_low and r_high <= 1 + TIME_MARGIN,
            "click_ratio": c, "click_ratio_ci90": [c_low, c_high],
        }
    (ROOT / "gui_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# gui-v1, VisionForge through the browser vs native", "",
        "n = 3 seeds (37, 38, 39). Margins frozen in gui-v1/PROTOCOL.md: time ±5% (90% CI), quality ±0.005 (95% CI).",
        "Failing the quality margin at n = 3 means inconclusive, not worse.", "",
        "| Recipe | Metric | Paired diff (95% CI) | Quality within | Time GUI/native (90% CI) | Per seed | Time within | Click to end/native (90% CI) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in out.values():
        lines.append(
            f"| {v['recipe']} | {v['metric']} | {v['diff']:+.4f} [{v['diff_ci95'][0]:+.4f}, {v['diff_ci95'][1]:+.4f}] "
            f"| {'yes' if v['quality_within'] else 'inconclusive'} "
            f"| {v['ratio']:.3f} [{v['ratio_ci90'][0]:.3f}, {v['ratio_ci90'][1]:.3f}] "
            f"| {', '.join(f'{x:.3f}' for x in v['ratio_per_seed'])} "
            f"| {'yes' if v['time_within'] else 'no'} "
            f"| {v['click_ratio']:.3f} [{v['click_ratio_ci90'][0]:.3f}, {v['click_ratio_ci90'][1]:.3f}] |"
        )
    (ROOT / "gui_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
