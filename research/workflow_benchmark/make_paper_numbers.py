"""Recompute every number of the paper's workflow benchmark from ``results/``.

The Research impact statement of ``paper/paper.md`` reports four series:
``real-v1`` (command line vs plain PyTorch/Ultralytics), ``gui-v1`` (browser
vs native), ``diag-v1`` (epoch time by browser state) and ``gui-v2`` (browser
vs native after the animation fix). This script reads the published per-run
records and logs, recomputes each difference, ratio and interval with the
methods the series' protocols fixed, and writes each as the phrase the paper
prints. It exits 1 if a phrase is missing from the paper.

Methods: quality is the paired difference (VisionForge - native) with a 95%
t interval; time is the geometric mean of the paired ratios with a 90% t
interval on the log scale; diag-v1 times the mean epoch from the end of epoch 1
to the end of epoch 4.

    .venv\\Scripts\\python.exe research\\workflow_benchmark\\make_paper_numbers.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PAPER = HERE.parents[1] / "paper" / "paper.md"
STAMP = "%Y-%m-%d %H:%M:%S"
WALL = re.compile(r"^(\w+) seed=(\d+) side=(\w+) exit=0 wall=(\d+)s(?: (\S+ \S+))?")
WORDS = {3: "three", 4: "four", 6: "six"}
TASK_KEY = {"cls": "test_accuracy", "seg": "test_miou"}
GUI_SEEDS = (37, 38, 39)
Wall = dict[tuple[str, int, str], tuple[int, datetime | None]]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def t_interval(values: list[float], level: float) -> tuple[float, float, float]:
    """Mean and two-sided ``level`` t interval of ``values``."""
    mean = statistics.fmean(values)
    t = float(stats.t.ppf(0.5 + level / 2, len(values) - 1))
    half = t * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def ratio_interval(ratios: list[float]) -> tuple[float, float, float]:
    """Geometric mean of ``ratios`` and its 90% interval (t on the log scale)."""
    mean, low, high = t_interval([math.log(r) for r in ratios], 0.90)
    return math.exp(mean), math.exp(low), math.exp(high)


def walls(*logs: Path) -> Wall:
    """Wall seconds and end time of each successful run in the queue logs."""
    out: Wall = {}
    for log in logs:
        for line in log.read_text(encoding="utf-8").splitlines():
            if m := WALL.match(line):
                end = datetime.strptime(m[5], STAMP) if m[5] else None
                out[m[1], int(m[2]), m[3]] = (int(m[4]), end)
    return out


def native_metric(run: Path, task: str) -> float:
    n = _load(run / "native_result.json")
    return float(n["test_accuracy"] if task == "cls" else n["test_metrics"]["miou"])


def vf_metric(run: Path, task: str) -> float:
    return float(_load(run / "run.json")["metrics"][TASK_KEY[task]])


def seeds_of(series: Path, task: str) -> list[int]:
    return sorted(int(p.name[4:]) for p in (series / task).glob("seed*"))


def _quality(diffs: list[float], first: str = "95% CI ") -> str:
    mean, low, high = t_interval(diffs, 0.95)
    return f"{mean:+.4f} ({first}{low:+.4f} to {high:+.4f})"


def _ratio(ratios: list[float], first: str = "") -> str:
    mean, low, high = ratio_interval(ratios)
    return f"{mean:.3f} ({first}{low:.3f} to {high:.3f})"


def _time_ratios(wall: Wall, task: str, seeds: list[int], side: str) -> list[float]:
    return [wall[task, s, side][0] / wall[task, s, "native"][0] for s in seeds]


def real_phrases(base: Path) -> list[str]:
    """Command line vs native, with the reruns of the slowdown swapped in for time."""
    real = base / "real-v1"
    original = walls(real / "summary.log", real / "ext-summary.log")
    reruns = walls(real / "rerun-summary.log")
    clean = {**original, **reruns}
    seeds = {task: seeds_of(real, task) for task in ("cls", "seg", "det")}
    epochs = {
        task: yaml.safe_load(
            (real / task / f"seed{seeds[task][0]}" / "vf" / "config.yaml").read_text(
                encoding="utf-8"
            )
        )["training"]["epochs"]
        for task in seeds
    }
    diffs = {
        task: [
            vf_metric(real / task / f"seed{s}" / "vf", task)
            - native_metric(real / task / f"seed{s}" / "native", task)
            for s in seeds[task]
        ]
        for task in ("cls", "seg")
    }
    det_equal = all(
        _load(real / "det" / f"seed{s}" / "vf" / "det_test.json")["map50"]
        == _load(real / "det" / f"seed{s}" / "native" / "det_test.json")["map50"]
        for s in seeds["det"]
    )
    times = {t: _time_ratios(clean, t, seeds[t], "vf") for t in seeds}
    margin = int(real_time_margin(base) * 100)
    within = all(
        1 - margin / 100 <= ratio_interval(r)[1]
        and ratio_interval(r)[2] <= 1 + margin / 100
        for r in times.values()
    )
    _, seg_low, seg_high = ratio_interval(
        _time_ratios(original, "seg", seeds["seg"], "vf")
    )
    return [
        f"ResNet-18 on CIFAR-10 ({epochs['cls']} epochs, "
        f"{WORDS[len(seeds['cls'])]} seeds)",
        f"DeepLabV3-ResNet50 on Oxford-IIIT Pet ({epochs['seg']} epochs, "
        f"{WORDS[len(seeds['seg'])]} seeds)",
        f"YOLO11s on VOC2007 ({epochs['det']} epochs, "
        f"{WORDS[len(seeds['det'])]} seeds)",
        f"accuracy was {_quality(diffs['cls'])}",
        f"in test mIoU {_quality(diffs['seg'], '')}",
        "test mAP50 matched to full precision" if det_equal else "mAP50 differs",
        f"was {_ratio(times['cls'], '90% CI ')}, {_ratio(times['seg'])} and "
        f"{_ratio(times['det'])}",
        f"inside a margin of {margin}%" if within else "outside the margin",
        f"{WORDS.get(len(reruns), str(len(reruns))).capitalize()} runs that fell "
        "inside a slowdown",
        f"interval, {seg_low:.3f} to {seg_high:.3f}, leaves the margin",
    ]


def real_time_margin(base: Path) -> float:
    """The time margin real-v1's summary states ("The margin for time is ±5%")."""
    text = (base / "real_summary.md").read_text(encoding="utf-8")
    found = re.search(r"margin for time is ±(\d+)%", text)
    if found is None:
        raise ValueError("no time margin in real_summary.md")
    return int(found[1]) / 100


def _gui_series(series: Path) -> dict[str, Any]:
    wall = walls(series / "summary.log")
    out: dict[str, Any] = {}
    for task in ("cls", "seg"):
        runs = [series / task / f"seed{s}" for s in GUI_SEEDS]
        out[task] = {
            "diffs": [
                vf_metric(r / "gui", task) - native_metric(r / "native", task)
                for r in runs
            ],
            "ratios": _time_ratios(wall, task, list(GUI_SEEDS), "gui"),
        }
    out["wall"] = wall
    return out


def protocol_margins(series: Path) -> tuple[str, str]:
    """The quality and time margins the series' PROTOCOL.md froze."""
    text = (series / "PROTOCOL.md").read_text(encoding="utf-8")
    quality = re.search(r"margin ±(0\.\d+) absolute", text)
    time = re.search(r"margin ±(\d+)%", text)
    if quality is None or time is None:
        raise ValueError(f"no margins in {series.name}/PROTOCOL.md")
    return quality[1], time[1]


def gui_v1_phrases(base: Path) -> list[str]:
    series = base / "gui-v1"
    g = _gui_series(series)
    quality, time = protocol_margins(series)
    return [
        f"with {WORDS[len(GUI_SEEDS)]} seeds",
        f"margin of {quality}: the paired difference was "
        f"{_quality(g['cls']['diffs'])} in test accuracy and "
        f"{_quality(g['seg']['diffs'], '')} in test mIoU",
        f"inside its margin of {time}%: the ratio, browser over native, was "
        f"{_ratio(g['cls']['ratios'], '90% CI ')} and {_ratio(g['seg']['ratios'])}",
    ]


def diag_epoch_s(run: Path) -> float:
    """Mean epoch time from the end of epoch 1 to the end of epoch 4."""
    if run.name == "native":
        h = _load(run / "native_result.json")["history"]
        return float(h[3]["seconds"] - h[0]["seconds"]) / 3
    text = (run / "epoch_times.txt").read_text(encoding="utf-8")
    t = [datetime.strptime(x, STAMP) for x in text.split("\n") if x]
    return (t[3] - t[0]).total_seconds() / 3


def diag_phrases(base: Path) -> list[str]:
    series = base / "diag-v1"
    seeds = seeds_of(series, "cls")
    epochs = len(
        _load(series / "cls" / f"seed{seeds[0]}" / "native" / "native_result.json")[
            "history"
        ]
    )
    ratio: dict[tuple[str, str], list[float]] = {}
    for task in ("cls", "seg"):
        for client in ("server", "hidden", "visible"):
            ratio[task, client] = [
                diag_epoch_s(series / task / f"seed{s}" / client)
                / diag_epoch_s(series / task / f"seed{s}" / "native")
                for s in seeds
            ]
    return [
        f"({WORDS[len(seeds)]} blocks, {WORDS[epochs]} epochs",
        f"the ratio was {_ratio(ratio['cls', 'server'], '90% CI ')} for "
        f"classification and {_ratio(ratio['seg', 'server'])} for segmentation",
        f"minimized, {_ratio(ratio['cls', 'hidden'])} and "
        f"{_ratio(ratio['seg', 'hidden'])}",
        f"visible, {_ratio(ratio['cls', 'visible'])} and "
        f"{_ratio(ratio['seg', 'visible'])}",
    ]


def other_cpu_mean(series: Path, wall: Wall, key: tuple[str, int, str]) -> float:
    """Mean CPU of the "other" process group (% of one core) during one run."""
    seconds, end = wall[key]
    if end is None:
        raise ValueError(f"no end time for {key}")
    start = end - timedelta(seconds=seconds)
    with (series / "monitor.csv").open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.reader(handle) if len(r) >= 6]
    values = [
        float(r[5]) for r in rows if start <= datetime.strptime(r[0], STAMP) <= end
    ]
    return statistics.fmean(values)


def gui_v2_phrases(base: Path) -> list[str]:
    series = base / "gui-v2"
    g = _gui_series(series)
    cls_ratios = g["cls"]["ratios"]
    slowest = max(range(len(GUI_SEEDS)), key=lambda i: cls_ratios[i])
    cpu = {s: other_cpu_mean(series, g["wall"], ("cls", s, "gui")) for s in GUI_SEEDS}
    others = [cpu[s] for s in GUI_SEEDS if s != GUI_SEEDS[slowest]]
    excess = cpu[GUI_SEEDS[slowest]] / statistics.fmean(others)
    return [
        f"Quality stayed inside its margin: {_quality(g['cls']['diffs'])} in test "
        f"accuracy and {_quality(g['seg']['diffs'], '')} in test mIoU",
        f"The time ratio fell to {_ratio(g['seg']['ratios'], '90% CI ')} for "
        "segmentation",
        f"and to {_ratio(cls_ratios)} for classification",
        f"(ratio {cls_ratios[slowest]:.3f})",
        f"about {excess:.1f} times the CPU",
    ]


def paper_phrases(base: Path = RESULTS) -> list[str]:
    """Every number of the workflow benchmark, as the paper prints it."""
    return [
        *real_phrases(base),
        *gui_v1_phrases(base),
        *diag_phrases(base),
        *gui_v2_phrases(base),
    ]


def normalized(text: str) -> str:
    """``text`` with every run of whitespace (line breaks too) as one space."""
    return " ".join(text.split())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # cp1252 console
    paper = normalized(PAPER.read_text(encoding="utf-8"))
    missing = 0
    for phrase in paper_phrases():
        found = phrase in paper
        missing += not found
        print(f"{'ok ' if found else 'MISSING'} {phrase}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
