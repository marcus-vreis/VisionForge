"""Replicate groups and K-folds as History runs (ADR-113, ADR-123).

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

A K-fold is the same shape with folds in place of seeds (ADR-123): its children
are the folds' own runs and its aggregates the K-fold's mean ± std per metric.
A sweep or a model comparison (ADR-124) has trials for children, each with the
hyperparameters it tried, and names the trial its ranking put first.

The ``finish_*`` functions at the end are the step every caller of an
orchestrator shares once its units have run (ADR-184): decide whether the job
stopped or failed, shape the orchestrator's output into the report, write the
report (JSON, ranking CSV, ``.tex``) and the group. The GUI routes and
``visionforge run`` both call them, so the command line cannot drift from the
numbers the interface shows. Every statistic still comes from the
orchestrators; these functions only arrange and write it.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from visionforge.core.cancellation import STOPPED, CancellationToken, job_was_stopped
from visionforge.core.latex_export import report_to_latex
from visionforge.utils.messages import tr
from visionforge.utils.paths import validate_path_component

if TYPE_CHECKING:
    # The orchestrators import torch (through task_runner) and the server
    # imports this module, so they are imported where they are used.
    from visionforge.core.replicates import ReplicateTrial
    from visionforge.core.sweep import SweepTrial
    from visionforge.core.task_runner import TaskRunner

GroupKind = Literal["replicates", "replicated_comparison", "kfold", "sweep"]

# The folder under the models directory holds one sub-folder per group, named
# by the group id, so running the same set twice leaves both in History rather
# than the second overwriting the first (which would orphan its children).
_DIR_SUFFIX: dict[str, str] = {
    "replicates": "replicates",
    "replicated_comparison": "comparison",
    "kfold": "cv",
    "sweep": "sweep",
}


def group_run_dir(models_dir: Path, name: str, kind: GroupKind, group_id: str) -> Path:
    """Where a group's ``run.json`` lives: ``<models>/<name>_<suffix>/<group_id>``."""
    validate_path_component(name)
    validate_path_component(group_id)
    path = Path(models_dir) / f"{name}_{_DIR_SUFFIX[kind]}" / group_id
    _contained_path(models_dir, path)
    return path


def _contained_path(root: Path, path: Path) -> Path:
    lexical_root = Path(os.path.abspath(root))
    canonical_root = lexical_root.resolve()
    lexical_path = Path(os.path.abspath(path))
    if not (
        lexical_path.is_relative_to(lexical_root)
        or lexical_path.is_relative_to(canonical_root)
    ):
        raise ValueError(tr("group.path_outside_root", path=path, root=root))
    resolved = lexical_path.resolve()
    if not resolved.is_relative_to(canonical_root):
        raise ValueError(tr("group.path_outside_root", path=path, root=root))
    return resolved


def _metadata_path(root: Path, run_dir: Path) -> Path:
    path = _contained_path(root, run_dir / "run.json")
    staging = _contained_path(root, path.with_name(path.name + ".tmp"))
    for candidate in (path, staging):
        if candidate.is_file() and candidate.stat().st_nlink > 1:
            raise ValueError(tr("group.linked_metadata", path=candidate))
    return path


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


def _fold_child(row: dict[str, Any]) -> dict[str, Any]:
    """One fold of a K-fold, as History lists it; ``seed`` stays empty."""
    child = _child({**row, "seed": None}, None)
    child["fold"] = row.get("fold")
    return child


def _trial_child(row: dict[str, Any]) -> dict[str, Any]:
    """One trial of a sweep or a model comparison, as History lists it."""
    child = _child({**row, "seed": None}, None)
    child["trial_index"] = row.get("trial_index")
    child["overrides"] = row.get("overrides") or {}
    return child


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

    copiers = {
        "sweep": _copy_sweep_group,
        "kfold": _copy_kfold_group,
        "replicates": _copy_replicate_group,
    }
    copiers.get(kind, _copy_comparison_group)(group, report)
    return group


def _copy_sweep_group(group: dict[str, Any], report: dict[str, Any]) -> None:
    """Copy trial children and the report's sweep ranking."""
    children = [_trial_child(row) for row in report.get("trials") or []]
    group.update(
        {
            "mode": report.get("mode"),
            "metric_direction": report.get("metric_direction"),
            "n_requested": report.get("n_planned") or len(children),
            "n_finished": _count(children),
            "children": children,
            # The ranking is the report's; the group only says who led it.
            "best_index": report.get("best_index"),
        }
    )


def _copy_kfold_group(group: dict[str, Any], report: dict[str, Any]) -> None:
    """Copy fold children and their reported aggregates."""
    children = [_fold_child(row) for row in report.get("fold_results") or []]
    group.update(
        {
            "n_folds": report.get("n_folds") or len(children),
            "n_requested": report.get("n_folds") or len(children),
            "n_finished": _count(children),
            "children": children,
            "aggregates": report.get("aggregate") or {},
        }
    )


def _copy_replicate_group(group: dict[str, Any], report: dict[str, Any]) -> None:
    """Copy seed children and their reported aggregates."""
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


def _copy_comparison_group(group: dict[str, Any], report: dict[str, Any]) -> None:
    """Copy variant children and the comparison statistics already reported."""
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
    if group["kind"] in ("replicates", "kfold", "sweep"):
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
    status: str = "completed",
    extra_metrics: dict[str, Any] | None = None,
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

    ``status`` is "failed" only for a job none of whose units finished and that
    nobody stopped; ``extra_metrics`` carries keys a writer kept for readers of
    its older ``run.json`` (the classification K-fold's ``cv_aggregate``).
    """
    group = build_group(kind, report, report_json)
    children = _children_of(group)
    run_dir = _contained_path(
        models_dir, group_run_dir(models_dir, name, kind, group_id)
    )
    group_json = _metadata_path(models_dir, run_dir)
    # Validate the complete write set before the first otherwise valid child is tagged.
    child_paths = [
        _metadata_path(models_dir, Path(child["run_dir"]))
        for child in children
        if child["run_dir"]
    ]

    total_epochs, donor = _tag_group_children(child_paths, group_id)

    # Task identity and provenance come from a child: a researcher-defined task
    # stamps ``task: custom:<key>`` at the top level, which the group's own
    # config cannot say.
    # Annotated: newer mypy infers Literal keys from the tuple and then rejects
    # unpacking the dict into a `dict[str, Any]`.
    inherited: dict[str, Any] = {
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
    _copy_group_metrics(group, config, donor, metrics)

    metrics.update(extra_metrics or {})
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json: dict[str, Any] = {
        "id": group_id,
        "experiment": name,
        **inherited,
        # Completed even when stopped: a stop is the researcher's request, and
        # ``stopped`` says that the job was cut (ADR-111).
        "status": status,
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
    _write_json(group_json, run_json)
    return run_dir


def _tag_group_children(
    child_paths: list[Path], group_id: str
) -> tuple[int, dict[str, Any] | None]:
    """Tag readable children and retain their epoch total and first donor."""
    total_epochs = 0
    donor: dict[str, Any] | None = None
    for path in child_paths:
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

    return total_epochs, donor


def _copy_group_metrics(
    group: dict[str, Any],
    config: dict[str, Any],
    donor: dict[str, Any] | None,
    metrics: dict[str, Any],
) -> None:
    """Copy aggregate means under the metric names used by History readers."""
    kind = group["kind"]
    if kind not in ("replicates", "kfold"):
        return
    # Fold means describe validation; replicate means follow the task's run keys.
    prefix = "val_"
    if kind == "replicates":
        prefix = "" if _writes_bare_names(config, donor) else "test_"
    metric_keys: dict[str, str] = {}
    for metric_name, agg in group["aggregates"].items():
        if isinstance(agg, dict) and agg.get("mean") is not None:
            key = metric_name if not prefix else f"{prefix}{metric_name}"
            metrics[key] = agg["mean"]
            metric_keys[key] = metric_name
    group["metric_keys"] = metric_keys


# ---------------------------------------------------------------------------
# Finishing a group job (ADR-184)
# ---------------------------------------------------------------------------

_SUCCESS = "success"
_REPORTS_DIR = "outputs/reports"


def refuse_output_paths(paths: Iterable[str], where: str) -> None:
    """Reject a dot-path that would move a trial's output folders.

    ``Profile.scope_dict`` points ``output`` at the profile before a sweep or a
    comparison starts, but both apply their dot-paths to that dict once per
    trial. A path under ``output`` would therefore undo the scoping (ADR-114),
    and no experiment legitimately varies where its files go. The GUI routes and
    ``visionforge run`` (ADR-184) share this check.

    ``where`` is the catalog key of the phrase that ends the sentence ("in a
    sweep"), because that phrase differs between languages.

    Raises:
        ValueError: naming the first offending path.
    """
    for path in paths:
        if str(path).split(".", 1)[0] == "output":
            raise ValueError(tr("profile.output_locked", where=tr(where), path=path))


def resolve_seeds(
    complete: dict[str, Any], seeds: list[int] | None, n_replicates: int
) -> list[int]:
    """The seeds a replicate job trains: the explicit list, else consecutive ones.

    Without an explicit list, ``n_replicates`` consecutive seeds start at the
    validated config's own ``training.seed``.

    Raises:
        ValueError: if the explicit list repeats a seed.
    """
    if seeds is not None:
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate seeds")
        return list(seeds)
    base_seed = int(complete.get("training", {}).get("seed", 42))
    return [base_seed + i for i in range(n_replicates)]


def require_reported_metric(
    metric: str, per_trial_metrics: list[dict[str, float]], kind: str
) -> None:
    """Fail loudly when no trial reported the metric the run is ranked/aggregated by.

    Without this the failure is silent and misleading: a sweep ranks every
    trial as 0.0 and calls an arbitrary one "best", and a replicate set returns
    ``headline: None`` that renders as an empty card. The metric is chosen
    before any training, so the earliest honest place to catch a typo is here,
    once results exist.

    Raises:
        RuntimeError: naming the metric and what the trials actually reported.
    """
    if any(metric in m for m in per_trial_metrics):
        return
    available = sorted({key for m in per_trial_metrics for key in m})
    raise RuntimeError(
        f"No {kind} reported the metric '{metric}' — "
        f"available: {available or 'none'}. Pick one of those."
    )


def replicates_report(
    trials: list[ReplicateTrial], seeds: list[int], metric: str
) -> dict[str, Any]:
    """Shape replicate trials + per-metric aggregates into the replicates report."""
    from visionforge.core.replicates import aggregate_replicates

    rows = [asdict(t) for t in trials]
    aggregates = aggregate_replicates(trials)
    return {
        "metric": metric,
        "seeds": seeds,
        "trials": rows,
        "aggregates": aggregates,
        "headline": aggregates.get(metric),
        "total_replicates": len(rows),
        "successful_replicates": sum(1 for r in rows if r["status"] == _SUCCESS),
    }


def sweep_report(
    trials: list[SweepTrial],
    mode: str,
    metric: str,
    direction: Literal["higher", "lower"],
) -> dict[str, Any]:
    """Shape ranked sweep trials into the sweep report."""
    from visionforge.core.task_runner import SELECTION_SPLIT

    rows = [asdict(t) for t in trials]
    successful = [t for t in rows if t["status"] == _SUCCESS]
    return {
        "mode": mode,
        "metric": metric,
        # Which way the trials were ranked; best_trial is the lowest for "lower".
        "metric_direction": direction,
        # Each trial's `metrics` are measured on this split and rank it; its
        # `evaluation_metrics` are the final (test) numbers, not used (ADR-131).
        "selection_split": SELECTION_SPLIT,
        "trials": rows,
        "best_trial": successful[0] if successful else None,
        "total_trials": len(rows),
        "successful_trials": len(successful),
        "failed_count": sum(1 for t in rows if t["status"] == "failed"),
        "stopped_count": sum(1 for t in rows if t["status"] == STOPPED),
    }


def _flatten_trial(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten a comparison/sweep trial row (nested metrics/overrides) for CSV."""
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if key in ("metrics", "overrides") and isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def write_group_summary(
    config_dict: dict[str, Any], kind: str, report: dict[str, Any]
) -> str:
    """Persist a group job's report (JSON + ranking CSV + LaTeX) under outputs/reports.

    Mirrors the classification ModelComparisonBlock artifact layout so a
    comparison, sweep, replicate set or K-fold leaves the same durable record
    on disk instead of vanishing when the result view closes. The ``.tex``
    table (ADR-061) is the paper-ready form: retyping numbers into a
    manuscript is where transcription errors enter, and nothing downstream
    catches them.
    """
    name = str(config_dict.get("name", kind))
    reports_dir = config_dict.get("output", {}).get("reports_dir") or _REPORTS_DIR
    out_dir = Path(reports_dir) / name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"{kind}_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    rows = report.get("trials") or report.get("fold_results") or []
    # Like the classification comparison's ranking.csv: only units that ran to
    # the end, so a cut unit's half-trained numbers never sit among finished
    # ones (ADR-111). The JSON summary above keeps every unit with its status.
    finished = [r for r in rows if r.get("status", _SUCCESS) == _SUCCESS]
    flat = [_flatten_trial(r) for r in finished]
    if kind in ("comparison", "sweep"):
        # Already sorted by the metric, best first.
        flat = [{"rank": i, **row} for i, row in enumerate(flat, start=1)]
    if flat:
        fieldnames = list(dict.fromkeys(k for row in flat for k in row))
        with (out_dir / f"{kind}_ranking.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat)

    latex = report_to_latex(report, experiment=name)
    if latex:
        (out_dir / f"{kind}_table.tex").write_text(latex, encoding="utf-8")

    # The strip sits beside the table, so the figure and the numbers of a
    # paper come from the same finished seeds (GUI-9).
    if kind == "replicates" and report.get("metric"):
        # Imported here: replicates pulls the plotter (matplotlib) lazily, and
        # this module stays light for the GUI's startup and the CLI.
        from visionforge.core.replicates import write_replicates_strip

        write_replicates_strip(rows, str(report["metric"]), out_dir, config_dict)

    return str(out_dir)


def record_run_group(
    kind: GroupKind,
    run_id: str,
    config: dict[str, Any],
    report: dict[str, Any],
    report_name: str,
    *,
    models_dir: Path,
) -> None:
    """Give a finished group job its own entry in History (ADR-113, ADR-123).

    The job's trainings are ordinary runs and its numbers sit in the report
    just written; this writes the ``run.json`` that ties them together, copied
    from that report, under the config's ``output.models_dir`` (``models_dir``
    when the config names none). It is the last step of a job whose training
    and report are already done, so a failure here is logged rather than
    turned into a failed job: the researcher keeps a report and loses only the
    grouped entry.
    """
    target = Path((config.get("output") or {}).get("models_dir") or models_dir)
    try:
        write_run_group(
            kind=kind,
            group_id=run_id,
            name=str(config.get("name", report_name)),
            config=config,
            report=report,
            report_json=Path(report["report_dir"]) / f"{report_name}_summary.json",
            models_dir=target,
        )
    # the report is on disk; do not fail the job
    except Exception:  # noqa: BLE001
        logger.exception("Could not write the History entry of group {}", run_id)


def record_sweep_group(
    run_id: str,
    runner: TaskRunner,
    config_dict: dict[str, Any],
    report: dict[str, Any],
    *,
    mode: str,
    planned: int,
    trials: list[dict[str, Any]],
    best_index: int | None,
    report_name: str,
    models_dir: Path,
) -> None:
    """Give a finished sweep or model comparison its History entry (ADR-124)."""
    # The validated config names the task (config.task); the raw dict may not.
    try:
        config = runner.config_type.model_validate(config_dict).model_dump(mode="json")
    # the entry still stands on the raw config
    except Exception:  # noqa: BLE001
        config = config_dict
    group_report = {
        "metric": report.get("metric"),
        "metric_direction": report.get("metric_direction"),
        "mode": mode,
        "n_planned": planned,
        "trials": trials,
        "best_index": best_index,
        "stopped": report.get("stopped", False),
        "report_dir": report.get("report_dir"),
    }
    record_run_group(
        "sweep", run_id, config, group_report, report_name, models_dir=models_dir
    )


def finish_replicates(
    trials: list[ReplicateTrial],
    *,
    seeds: list[int],
    metric: str,
    config: dict[str, Any],
    run_id: str,
    token: CancellationToken | None,
    models_dir: Path,
) -> dict[str, Any]:
    """Report, summary files and History group of a replicate set that has run.

    Raises:
        RuntimeError: when every replicate failed and nobody stopped the set,
            or no replicate reported ``metric``.
    """
    stopped = job_was_stopped([t.status for t in trials], len(seeds), token)
    finished = any(t.status == _SUCCESS for t in trials)
    if not finished and not stopped:
        raise RuntimeError("All replicates failed — no aggregate available.")
    if finished:
        require_reported_metric(metric, [t.metrics for t in trials], "replicate")
    report = replicates_report(trials, seeds, metric)
    report["stopped"] = stopped
    report["report_dir"] = write_group_summary(config, "replicates", report)
    record_run_group(
        "replicates", run_id, config, report, "replicates", models_dir=models_dir
    )
    return report


def finish_replicated_comparison(
    report: dict[str, Any],
    *,
    n_planned: int,
    config: dict[str, Any],
    run_id: str,
    token: CancellationToken | None,
    models_dir: Path,
) -> dict[str, Any]:
    """Summary files and History group of a replicated comparison that has run.

    A stopped comparison reports the variants that did run, even with no pair
    left to test: the researcher asked for the stop (ADR-111).

    Raises:
        RuntimeError: when no variant has enough successful replicates to
            compare and nobody stopped the job.
    """
    stopped = job_was_stopped(
        [
            t["status"]
            for v in report.get("variants", {}).values()
            for t in v.get("trials", [])
        ],
        n_planned,
        token,
    )
    report["stopped"] = stopped
    if not report["comparisons"] and not stopped:
        raise RuntimeError(
            "No variant produced enough successful replicates to compare "
            f"(skipped: {report['skipped_variants']})."
        )
    report["report_dir"] = write_group_summary(config, "comparison", report)
    record_run_group(
        "replicated_comparison",
        run_id,
        config,
        report,
        "comparison",
        models_dir=models_dir,
    )
    return report


def finish_sweep(
    trials: list[SweepTrial],
    *,
    runner: TaskRunner,
    config: dict[str, Any],
    search_space: dict[str, Any],
    mode: str,
    n_trials: int,
    metric: str,
    run_id: str,
    token: CancellationToken | None,
    models_dir: Path,
) -> dict[str, Any]:
    """Ranked report, summary files and History group of a sweep that has run.

    Raises:
        RuntimeError: when every trial failed and nobody stopped the sweep, or
            no trial reported ``metric``.
    """
    from visionforge.core.sweep import planned_trials
    from visionforge.core.task_runner import runner_metric_direction

    planned = planned_trials(search_space, mode=mode, n_trials=n_trials)
    stopped = job_was_stopped([t.status for t in trials], planned, token)
    finished = any(t.status == _SUCCESS for t in trials)
    if not finished and not stopped:
        raise RuntimeError("All sweep trials failed — no ranking available.")
    if finished:
        require_reported_metric(metric, [t.metrics for t in trials], "sweep")
    report = sweep_report(trials, mode, metric, runner_metric_direction(runner, metric))
    report["stopped"] = stopped
    # total_trials counts the trials that ran; the table and the GUI need to
    # say of how many (ADR-111).
    report["planned_trials"] = planned
    report["report_dir"] = write_group_summary(config, "sweep", report)
    best = report["best_trial"]
    record_sweep_group(
        run_id,
        runner,
        config,
        report,
        mode=mode,
        planned=planned,
        trials=report["trials"],
        best_index=best["trial_index"] if best else None,
        report_name="sweep",
        models_dir=models_dir,
    )
    return report


def finish_kfold(
    cv: Any,
    *,
    n_folds: int,
    config: dict[str, Any],
    validated_config: dict[str, Any],
    run_id: str,
    token: CancellationToken | None,
    models_dir: Path,
) -> dict[str, Any]:
    """Fold report, summary files and History group of a task K-fold that has run.

    ``cv`` is the ``CrossValidationReport`` the task's CV runner returned;
    the report is written where ``config`` says, and the group records
    ``validated_config``, which names the task (``config.task``) where the raw
    dict may leave it to the default.

    Raises:
        RuntimeError: when every fold failed and nobody stopped the K-fold.
    """
    stopped = job_was_stopped([f.status for f in cv.folds], n_folds, token)
    if not any(f.status == _SUCCESS for f in cv.folds) and not stopped:
        raise RuntimeError("All folds failed — no aggregate available.")
    report: dict[str, Any] = {
        "n_folds": cv.n_folds,
        "metric": cv.metric,
        "fold_results": [asdict(f) for f in cv.folds],
        "aggregate": cv.aggregate,
        "successful_folds": sum(1 for f in cv.folds if f.status == _SUCCESS),
        "stopped": stopped,
    }
    report["report_dir"] = write_group_summary(config, "cv", report)
    # One History entry for the job, its folds under it (ADR-123).
    record_run_group(
        "kfold", run_id, validated_config, report, "cv", models_dir=models_dir
    )
    return report


__all__ = [
    "GroupKind",
    "build_group",
    "finish_kfold",
    "finish_replicated_comparison",
    "finish_replicates",
    "finish_sweep",
    "group_run_dir",
    "record_run_group",
    "record_sweep_group",
    "replicates_report",
    "require_reported_metric",
    "resolve_seeds",
    "sweep_report",
    "write_group_summary",
    "write_run_group",
]
