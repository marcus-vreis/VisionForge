"""Aggregate the real-training series (VisionForge vs native, same recipe).

Reads results/real-v1 and writes results/real_summary.json and results/real_summary.md.
Metrics use the first run of every seed. Time is run wall clock from the summary logs, so it
includes startup and final test evaluation on both sides (the det test scoring is outside it).
The main time table swaps in the reruns of the three runs inside the 2026-10-30 slowdown;
the sensitivity table keeps the original runs. No run is dropped.

Usage: python results/aggregate_real.py
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "real-v1"
TASKS = {
    "cls": ("ResNet-18, CIFAR-10, 30 ep", "test accuracy", (37, 38, 39, 40, 41, 42)),
    "seg": ("DeepLabV3-ResNet50, Pet, 40 ep", "test mIoU", (37, 38, 39, 40, 41, 42)),
    "det": ("YOLO11s, VOC2007, 100 ep", "test mAP50", (37, 38, 39)),
}
RERUNS = (("cls", 39, "vf"), ("cls", 39, "native"), ("seg", 37, "vf"))
MARGIN = 0.05


def metric(base: Path, task: str, seed: int, side: str) -> float:
    d = base / task / f"seed{seed}" / side
    if task == "det":
        return json.loads((d / "det_test.json").read_text(encoding="utf-8"))["map50"]
    if side == "vf":
        m = json.loads((d / "run.json").read_text(encoding="utf-8"))["metrics"]
        return m["test_accuracy"] if task == "cls" else m["test_miou"]
    n = json.loads((d / "native_result.json").read_text(encoding="utf-8"))
    return n["test_accuracy"] if task == "cls" else n["test_metrics"]["miou"]


def walls(*names: str) -> dict[tuple[str, int, str], int]:
    pat = re.compile(r"^(\w+) seed=(\d+) side=(\w+) exit=0 wall=(\d+)s")
    out = {}
    for name in names:
        for line in (DIR / name).read_text(encoding="utf-8").splitlines():
            if m := pat.match(line):
                out[m[1], int(m[2]), m[3]] = int(m[4])
    return out


def ci(values: list[float], level: float) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    half = float(stats.t.ppf(0.5 + level / 2, len(values) - 1)) * statistics.stdev(values) / len(values) ** 0.5
    return mean, mean - half, mean + half


def time_ratio(wall: dict, task: str, seeds: tuple[int, ...]) -> dict:
    logs = [math.log(wall[task, s, "vf"] / wall[task, s, "native"]) for s in seeds]
    mean, low, high = (math.exp(x) for x in ci(logs, 0.90))
    return {
        "mean": mean, "ci90_low": low, "ci90_high": high, "within_5pct": 1 - MARGIN <= low and high <= 1 + MARGIN,
        "per_seed": [wall[task, s, "vf"] / wall[task, s, "native"] for s in seeds],
    }


def main() -> None:
    original = walls("summary.log", "ext-summary.log")
    clean = {**original, **walls("rerun-summary.log")}
    out = {}
    for task, (recipe, name, seeds) in TASKS.items():
        vf = [metric(DIR, task, s, "vf") for s in seeds]
        nat = [metric(DIR, task, s, "native") for s in seeds]
        diff = [a - b for a, b in zip(vf, nat)]
        mean, low, high = ci(diff, 0.95)
        out[task] = {
            "recipe": recipe, "metric": name, "seeds": seeds, "vf": vf, "native": nat,
            "vf_mean": statistics.fmean(vf), "vf_sd": statistics.stdev(vf),
            "native_mean": statistics.fmean(nat), "native_sd": statistics.stdev(nat),
            "diff_mean": mean, "diff_ci95": [low, high],
            "time_with_reruns": time_ratio(clean, task, seeds), "time_original": time_ratio(original, task, seeds),
        }
    out["rerun_metrics"] = [
        {"task": t, "seed": s, "side": w, "first": metric(DIR, t, s, w), "rerun": metric(DIR / "rerun", t, s, w)} for t, s, w in RERUNS
    ]
    (ROOT / "real_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    fmt = lambda t: f"{t['mean']:.3f} [{t['ci90_low']:.3f}, {t['ci90_high']:.3f}]"
    lines = [
        "# Real training, VisionForge vs native", "",
        "cls and seg: n = 6 seeds (37-42), VF first on 37, 39, 41 and native first on 38, 40, 42. det: n = 3 seeds (37-39), VF first on 37, 39.",
        "Time is run wall clock. The margin for time is ±5%, checked with the 90% CI of the paired ratio.", "",
        "| Recipe | Metric | VF mean ± SD | Native mean ± SD | Paired diff (95% CI) | Time VF/native, with reruns (90% CI) | Within ±5% | Time, original runs (90% CI) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for task in TASKS:
        r = out[task]
        t = r["time_with_reruns"]
        lines.append(
            f"| {r['recipe']} | {r['metric']} | {r['vf_mean']:.4f} ± {r['vf_sd']:.4f} | {r['native_mean']:.4f} ± {r['native_sd']:.4f} "
            f"| {r['diff_mean']:+.4f} [{r['diff_ci95'][0]:+.4f}, {r['diff_ci95'][1]:+.4f}] | {fmt(t)} | {'yes' if t['within_5pct'] else 'no'} | {fmt(r['time_original'])} |"
        )
    lines += [
        "",
        "Reruns: both cls seed 39 runs and seg seed 37 VF fell inside a slowdown of unknown cause, about 13:11 to 15:15 on 2026-10-30.",
        "They were repeated with the machine idle. Metrics above use the first runs. No run was dropped.",
        "Detection metrics match to full precision, but best.pt hashes differ (det_best_pt_sha256.txt), and weight equality was not tested.",
        "Pet mIoU ignores label 255.",
    ]
    (ROOT / "real_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
