"""Apply PROTOCOL.md's per-cell analysis to the grid and judge H1-H3.

Reads every ``results/<dataset>__<architecture>.json`` that run_study.py saved,
plus the ``run.json`` of each cell's baseline at seed 0 (for the evaluation-noise
item), and writes ``results/summary.json``. Prints the tables as Markdown.

    .venv\\Scripts\\python.exe research\\seed_variance\\summarize.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from analysis import (
    noise_to_effect,
    paired_differences,
    per_seed,
    reversal_rate,
    seeds_for_power,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parents[1]
DATASETS = ["usk_full", "usk_10pct", "beans"]
ARCHITECTURES = ["resnet18", "convnext_tiny", "swin_t"]
ABLATIONS = ["no_aug", "res160"]
REVERSAL_KS = [1, 3, 5]
SMALLEST_K_CANDIDATES = [1, 3, 5, 10]
NOISE_EFFECT_THRESHOLD = 0.5  # H1
REVERSAL_THRESHOLD = 0.10  # H2 and item 5a


def _comparison(report: dict[str, Any], ablation: str) -> dict[str, Any]:
    for row in report["comparisons"]:
        if {row["label_a"], row["label_b"]} == {"baseline", ablation}:
            return dict(row)
    raise KeyError(f"no baseline-vs-{ablation} comparison in the report")


def smallest_stable_k(diffs: list[float]) -> int | None:
    """The smallest k of 1, 3, 5, 10 whose reversal rate is at most 10% (item 5a)."""
    for k in SMALLEST_K_CANDIDATES:
        if k <= len(diffs) and reversal_rate(diffs, k) <= REVERSAL_THRESHOLD:
            return k
    return None


def evaluation_half_width(report: dict[str, Any]) -> float | None:
    """Half-width of the 95% bootstrap CI of the baseline's test accuracy at seed 0."""
    trial = next(
        (t for t in report["variants"]["baseline"]["trials"] if t["seed"] == 0), None
    )
    if trial is None or not trial.get("run_dir"):
        return None
    run_json = REPO / Path(trial["run_dir"].replace("\\", "/")) / "run.json"
    if not run_json.is_file():
        return None
    ci = json.loads(run_json.read_text(encoding="utf-8"))["metric_cis"]["accuracy"]
    return (float(ci["ci_high"]) - float(ci["ci_low"])) / 2


def cell_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per ablation: items 1-5a of the protocol, plus the secondary F1."""
    baseline = list(per_seed(report, "baseline").values())
    f1 = report["variants"]["baseline"]["aggregates"]["f1"]
    rows = []
    for ablation in ABLATIONS:
        diffs = paired_differences(report, ablation)
        test = _comparison(report, ablation)
        delta = statistics.fmean(diffs)
        rows.append(
            {
                "ablation": ablation,
                "n_pairs": len(diffs),
                "baseline_mean": statistics.fmean(baseline),
                "baseline_sd": statistics.stdev(baseline),
                "baseline_f1_mean": f1["mean"],
                "baseline_f1_sd": f1["std"],
                "delta": delta,
                # The report's CI is for baseline - variant; flip it to variant - baseline.
                "delta_ci95": [-test["ci_high"], -test["ci_low"]],
                "test": test["test"],
                "p_value": test["p_value"],
                "significant_holm": bool(test["significant"]),
                "noise_to_effect": noise_to_effect(baseline, diffs),
                "reversal": {
                    str(k): reversal_rate(diffs, k) if delta != 0 else None
                    for k in REVERSAL_KS
                },
                "smallest_stable_k": smallest_stable_k(diffs) if delta != 0 else None,
                "seeds_for_power": seeds_for_power(diffs),
            }
        )
    return rows


def judge_h1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """H1 holds when noise/effect >= 0.5 in at least half of the cells."""
    high = sum(1 for r in rows if r["noise_to_effect"] >= NOISE_EFFECT_THRESHOLD)
    return {
        "cells": len(rows),
        "at_or_above": high,
        "verdict": "supported" if 2 * high >= len(rows) else "rejected",
    }


def judge_h2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """H2 holds when every dataset has a cell whose k=1 reversal rate exceeds 10%.

    The protocol rejects H2 only when no cell anywhere exceeds 10%; a grid where
    some datasets have such a cell and others do not is reported as neither.
    """
    by_dataset: dict[str, bool] = {}
    for r in rows:
        k1 = r["reversal"]["1"]
        hit = k1 is not None and k1 > REVERSAL_THRESHOLD
        by_dataset[r["dataset"]] = by_dataset.get(r["dataset"], False) or hit
    if all(by_dataset.values()):
        verdict = "supported"
    elif not any(by_dataset.values()):
        verdict = "rejected"
    else:
        verdict = "neither"
    return {"dataset_has_cell_above": by_dataset, "verdict": verdict}


def judge_h3(sd: dict[tuple[str, str], float]) -> dict[str, Any]:
    """H3 holds when SD(10%) > SD(full) for every architecture."""
    per_arch = {
        arch: {"sd_10pct": sd[("usk_10pct", arch)], "sd_full": sd[("usk_full", arch)]}
        for arch in ARCHITECTURES
        if ("usk_10pct", arch) in sd and ("usk_full", arch) in sd
    }
    larger = all(v["sd_10pct"] > v["sd_full"] for v in per_arch.values())
    return {
        "per_architecture": per_arch,
        "verdict": "supported" if per_arch and larger else "rejected",
    }


def _fmt(x: float | None, digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def print_tables(rows: list[dict[str, Any]], verdicts: dict[str, Any]) -> None:
    print(
        "| Dataset | Arch | Baseline acc (SD) | Ablation | Δ [95% CI] | p (Holm sig.)"
        " | noise/effect | rev k=1/3/5 | k≤10% | n80 | eval ±|"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        rev = "/".join(_fmt(r["reversal"][str(k)], 2) for k in REVERSAL_KS)
        lo, hi = r["delta_ci95"]
        sig = "*" if r["significant_holm"] else ""
        print(
            f"| {r['dataset']} | {r['architecture']} "
            f"| {_fmt(r['baseline_mean'])} ({_fmt(r['baseline_sd'])}) "
            f"| {r['ablation']} | {r['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
            f"| {r['p_value']:.3f}{sig} | {r['noise_to_effect']:.2f} | {rev} "
            f"| {r['smallest_stable_k'] or '>10'} | {r['seeds_for_power'] or '—'} "
            f"| {_fmt(r['evaluation_half_width'])} |"
        )
    print()
    for name, v in verdicts.items():
        print(
            f"- {name}: {v['verdict']} — {json.dumps({k: x for k, x in v.items() if k != 'verdict'})}"
        )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # cp1252 console
    rows: list[dict[str, Any]] = []
    sd: dict[tuple[str, str], float] = {}
    missing = []
    for dataset in DATASETS:
        for arch in ARCHITECTURES:
            path = RESULTS / f"{dataset}__{arch}.json"
            if not path.is_file():
                missing.append(path.name)
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            half_width = evaluation_half_width(report)
            for row in cell_rows(report):
                rows.append(
                    {
                        "dataset": dataset,
                        "architecture": arch,
                        **row,
                        "evaluation_half_width": half_width,
                    }
                )
            sd[(dataset, arch)] = statistics.stdev(
                per_seed(report, "baseline").values()
            )
    if missing:
        print(f"missing reports: {', '.join(missing)}", file=sys.stderr)
    verdicts = {"H1": judge_h1(rows), "H2": judge_h2(rows), "H3": judge_h3(sd)}
    summary = {"cells": rows, "hypotheses": verdicts, "missing": missing}
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_tables(rows, verdicts)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
