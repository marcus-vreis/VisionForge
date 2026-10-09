"""Replicate groups as History runs (ADR-113).

A replicate set and a replicated comparison (ADR-056, ADR-061) are one job made
of several trainings. Each seed is an ordinary run, named ``<name>_s<seed>``,
and History listed them as loose runs: the mean, the confidence interval and
the paired tests lived only in a report under ``outputs/reports`` that nothing
in the interface pointed to. This module gives the job a ``run.json`` of its
own, next to the runs it is made of, the way a K-fold's top-level ``run.json``
does (``blocks/cross_validation.py``), so History can list it as one entry and
show those numbers.

Nothing is computed here. Every statistic in the group is copied from the
report the job already built: a second computation could drift from the report
a paper's table is made from, and the number on screen must be the one on
disk. The module only reads the report, tags the children, and writes JSON.
"""

from __future__ import annotations

import contextlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger

GroupKind = Literal["replicates", "replicated_comparison"]

# The folder under the models directory holds one sub-folder per group, named
# by the group id, so running the same set twice leaves both in History rather
# than the second overwriting the first (which would orphan its children).
_DIR_SUFFIX: dict[str, str] = {
    "replicates": "replicates",
    "replicated_comparison": "comparison",
}


def group_run_dir(models_dir: Path, name: str, kind: GroupKind, group_id: str) -> Path:
    """Where a group's ``run.json`` lives: ``<models>/<name>_<suffix>/<group_id>``."""
    return Path(models_dir) / f"{name}_{_DIR_SUFFIX[kind]}" / group_id


def _json_safe(value: Any) -> Any:
    """``value`` with every NaN / infinity replaced by ``None``.

    A bootstrap interval or a test statistic can come out non-finite, and
    Python writes that as a bare ``NaN`` that no browser parses and the API
    refuses to serve. The report keeps the token it always had; the group, which
    the interface reads, says "no value" instead.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _child(row: dict[str, Any], variant: str | None) -> dict[str, Any]:
    """One seed of the group, as History lists it.

    ``run_id`` is the folder name of the seed's own run (what ``/api/runs``
    calls ``run_id``), or ``None`` for a seed that never got a directory.
    """
    run_dir = row.get("run_dir") or ""
    return {
        "seed": row.get("seed"),
        "variant": variant,
        "status": row.get("status"),
        "run_id": Path(run_dir).name if run_dir else None,
        "run_dir": run_dir or None,
        "metrics": row.get("metrics") or {},
        "error": row.get("error") or "",
    }


def _count(children: list[dict[str, Any]]) -> int:
    return sum(1 for c in children if c["status"] == "success")


def _seeds_finished(children: list[dict[str, Any]]) -> list[int]:
    return [c["seed"] for c in children if c["status"] == "success"]


def build_group(
    kind: GroupKind, report: dict[str, Any], report_json: Path | None
) -> dict[str, Any]:
    """The ``group`` section of the group's ``run.json``, copied from ``report``."""
    group: dict[str, Any] = _json_safe(_copy_group(kind, report, report_json))
    return group


def _copy_group(
    kind: GroupKind, report: dict[str, Any], report_json: Path | None
) -> dict[str, Any]:
    """The section as the report states it, before non-finite numbers are cleared."""
    group: dict[str, Any] = {
        "kind": kind,
        "metric": report.get("metric"),
        "seeds": list(report.get("seeds") or []),
        "stopped": bool(report.get("stopped")),
        # The convention behind every ``std`` below (statistics.stdev).
        "std_ddof": 1,
        "report_path": str(report_json) if report_json is not None else None,
        "report_dir": report.get("report_dir"),
    }

    if kind == "replicates":
        children = [_child(row, None) for row in report.get("trials") or []]
        group.update(
            {
                "n_requested": len(group["seeds"]),
                "n_finished": _count(children),
                "seeds_finished": _seeds_finished(children),
                "children": children,
                "aggregates": report.get("aggregates") or {},
            }
        )
        return group

    variants: dict[str, Any] = {}
    all_children: list[dict[str, Any]] = []
    for label, variant in (report.get("variants") or {}).items():
        children = [_child(row, label) for row in variant.get("trials") or []]
        all_children.extend(children)
        variants[label] = {
            "overrides": variant.get("overrides") or {},
            "aggregates": variant.get("aggregates") or {},
            "successful": variant.get("successful"),
            "seeds_finished": _seeds_finished(children),
            "children": children,
        }
    # Trainings, not seeds: every variant trains every seed, the ones a stop
    # kept from starting included (``not_run``).
    n_variants = len(variants) + len(report.get("not_run") or [])
    group.update(
        {
            "n_requested": n_variants * len(group["seeds"]),
            "n_finished": _count(all_children),
            "metric_direction": report.get("metric_direction"),
            "alpha": report.get("alpha"),
            "variants": variants,
            "comparisons": report.get("comparisons") or [],
            "best_by_mean": report.get("best_by_mean"),
            "ranked_by_mean": report.get("ranked_by_mean") or [],
            "ranking_seeds": report.get("ranking_seeds") or [],
            "significant_pairs": report.get("significant_pairs"),
            "skipped_variants": report.get("skipped_variants") or [],
            "not_run": report.get("not_run") or [],
            "underpowered": bool(report.get("underpowered")),
            "power_note": report.get("power_note") or "",
        }
    )
    return group


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace ``path`` whole, so a reader never sees a half-written run.json."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        # Do not leave the half step behind: the next reader of this folder
        # would find a stray file next to a run.json that never changed.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _writes_bare_names(config: dict[str, Any], donor: dict[str, Any] | None) -> bool:
    """Whether this task's run.json names its metrics without a ``test_`` prefix.

    Detection (its score is the best validation mAP) and a researcher-defined
    task (the metrics are whatever it declared) write the bare name; the other
    built-ins write the held-out split's score as ``test_<name>``.
    """
    task = (donor or {}).get("task")
    if isinstance(task, str) and task.startswith("custom:"):
        return True
    return config.get("task") == "detection"


def _children_of(group: dict[str, Any]) -> list[dict[str, Any]]:
    if group["kind"] == "replicates":
        return list(group["children"])
    return [c for v in group["variants"].values() for c in v["children"]]


def write_run_group(
    *,
    kind: GroupKind,
    group_id: str,
    name: str,
    config: dict[str, Any],
    report: dict[str, Any],
    report_json: Path | None,
    models_dir: Path,
) -> Path:
    """Write the group's ``run.json`` and tag every child run with its id.

    Returns the directory written. The top-level keys are the ones
    ``_parse_run_summary`` reads from any run (``experiment``, ``status``,
    ``timestamp``, ``config``, ``metrics.total_epochs``), so the group is listed
    by the code that lists every other run; ``group`` carries what only a group
    has. Children get an additive ``group_id`` in their own ``run.json`` so the
    list can fold them under the group.

    A child whose ``run.json`` cannot be read or rewritten is left alone and
    logged: the group still stands, and that seed stays a loose run rather than
    failing a job whose training already finished.
    """
    group = build_group(kind, report, report_json)
    children = _children_of(group)

    total_epochs = 0
    donor: dict[str, Any] | None = None
    for child in children:
        if not child["run_dir"]:
            continue
        path = Path(child["run_dir"]) / "run.json"
        data = _read_json(path)
        if data is None:
            logger.warning("Replicate group {}: no readable {}", group_id, path)
            continue
        donor = donor or data
        total_epochs += int((data.get("metrics") or {}).get("total_epochs") or 0)
        data["group_id"] = group_id
        try:
            _write_json(path, data)
        except OSError as exc:
            # Windows: an antivirus or the indexer can hold a run.json for a
            # moment. That seed stays a loose run; the others are still tagged
            # and the group is still written.
            logger.warning(
                "Replicate group {}: could not tag {} ({}); that seed stays loose",
                group_id,
                path,
                exc,
            )

    # Task identity and provenance come from a child: a researcher-defined task
    # stamps ``task: custom:<key>`` at the top level, which the group's own
    # config cannot say.
    inherited = {
        key: donor[key]
        for key in (
            "task",
            "task_label",
            "device_used",
            "environment",
            "dataset_fingerprint",
        )
        if donor is not None and key in donor
    }

    metrics: dict[str, Any] = {"total_epochs": total_epochs}
    if kind == "replicates":
        # The means under the keys a single run of this task writes, so a
        # reader of ``metrics`` that knows nothing about groups (the history
        # list, the comparison table) still finds the number the job is about
        # in the row where it belongs. ``metric_keys`` says which aggregate
        # filled which key, so nobody has to guess the correspondence.
        bare = _writes_bare_names(config, donor)
        metric_keys: dict[str, str] = {}
        for metric_name, agg in group["aggregates"].items():
            if isinstance(agg, dict) and agg.get("mean") is not None:
                key = metric_name if bare else f"test_{metric_name}"
                metrics[key] = agg["mean"]
                metric_keys[key] = metric_name
        group["metric_keys"] = metric_keys

    run_dir = group_run_dir(models_dir, name, kind, group_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json: dict[str, Any] = {
        "id": group_id,
        "experiment": name,
        **inherited,
        # Completed, as for a K-fold: a stop is the researcher's request, and
        # ``stopped`` says that the job was cut (ADR-111).
        "status": "completed",
        "stopped": group["stopped"],
        # Naive local time like every other run.json writer.
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir.resolve()),
        "config": config,
        "metrics": metrics,
        "history": [],
        "artifacts": {"model": None, "graphics": [], "report": None},
        "tests": [],
        "block": kind,
        "group": group,
    }
    _write_json(run_dir / "run.json", run_json)
    return run_dir


__all__ = ["GroupKind", "build_group", "group_run_dir", "write_run_group"]
