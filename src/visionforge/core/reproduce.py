"""Re-train one seed of a verified experiment bundle and compare it with the record.

``visionforge reproduce BUNDLE.zip`` is the third party's check of a published
result: the bundle (ADR-135) says what was trained and what came out; this
module plans a new training of one of its seeds, on a dataset the reproducer
has locally, and compares what that training reports with what the bundle
recorded. Nothing here trains or picks a task: the caller hands in the same
``runner_for`` dispatcher ``visionforge run`` uses for its strategies, so the
reproduction trains through the code that produced the record.

The recorded values come from the bundle's ``run.json`` (a replicate group's
``group.aggregates``, written by ``core.run_groups``) and the reproduced values
from the runner's own result. The group's ``ci95`` is shown as recorded, but it
does not judge the seed: it is the Student-t interval of the *mean* of the
seeds (half-width t·s/√n), and one new seed legitimately falls outside it far
more often than 5% of the time. The verdict uses the 95% *prediction* interval
for one new observation, ``mean ± t(0.975, n-1) · std · sqrt(1 + 1/n)``,
computed here from the recorded ``n``, ``mean`` and ``std`` with the same t
quantile and the same clipping to the metric's domain as the recorded ``ci95``
(``core.replicates``). It is the only number derived here; it is not written
back into any record. The bundle is read member by member from the ZIP;
nothing is extracted, and the original run is never written to: the new run
gets its own name (``<name>_repro``) and lands in the outputs folder.
"""

from __future__ import annotations

import copy
import json
import math
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from visionforge.core.experiment_bundle import (
    RUN_FILENAME,
    read_run_record,
    verify_experiment_bundle,
)
from visionforge.core.replicates import (
    _clip_to_domain,
    _t_critical_95,
    metric_domain,
)
from visionforge.core.run_groups import _writes_bare_names
from visionforge.utils.environment import capture_environment

if TYPE_CHECKING:
    from visionforge.core.task_runner import RunResult, TaskRunner

Kind = Literal["single", "replicates"]
Split = Literal["test", "validation"]
Verdict = Literal["inside_interval", "outside_interval", "no_interval"]

REPRO_SUFFIX = "_repro"
# The versions that can move a metric between two otherwise identical runs.
ENVIRONMENT_KEYS = ("python", "torch", "torchvision", "visionforge")
# Where a config names its dataset; ``data_yaml`` exists only for detection.
DATASET_FIELDS = ("base_dir", "data_yaml")
_OUTPUT_DIRS = ("models_dir", "graphics_dir", "logs_dir", "reports_dir")
_CONFIG_MEMBERS = ("config.json", "config.yaml", "config.yml")
_CLASSIFICATION_TASKS = ("binary", "multiclass", "multilabel")
_MAX_RECORD_BYTES = 16 * 1024 * 1024
_SUCCESS = "success"


class ReproduceError(ValueError):
    """The bundle, its kind, the seed or the dataset cannot be reproduced."""


@dataclass(frozen=True)
class BundleRecord:
    """A verified bundle's ``run.json`` and the task config it trained."""

    path: Path
    record: dict[str, Any]
    config: dict[str, Any]


@dataclass(frozen=True)
class ReproductionPlan:
    """Everything decided before training: what to train and what to compare with."""

    bundle: Path
    kind: Kind
    seed: int
    task: str
    record: dict[str, Any]
    # The seed's own entry in a replicate group (None for a single run).
    child: dict[str, Any] | None
    # The config the new run trains: recorded one, new seed, name, data, outputs.
    config: dict[str, Any]


def _read_member(bundle: zipfile.ZipFile, name: str) -> bytes:
    if bundle.getinfo(name).file_size > _MAX_RECORD_BYTES:
        raise ReproduceError(f"{name} in the bundle exceeds the size limit")
    return bundle.read(name)


def _parse_config(name: str, raw: bytes) -> Any:
    try:
        if name.endswith(".json"):
            return json.loads(raw)
        return yaml.safe_load(raw)
    except (ValueError, yaml.YAMLError) as exc:
        raise ReproduceError(f"{name} in the bundle does not parse") from exc


def read_bundle_record(bundle_path: Path | str) -> BundleRecord:
    """Verify the bundle and read its ``run.json`` and config, without extracting.

    The config is the one ``run.json`` recorded; a run.json without one falls
    back to a ``config.json``/``config.yaml`` member of the bundle.

    Raises:
        ValueError: the bundle fails verification, or its run.json or config
            is not a usable record (``ReproduceError``).
        OSError: the file cannot be read.
    """
    path = Path(bundle_path)
    manifest = verify_experiment_bundle(path)
    members = set(manifest["files"])
    with zipfile.ZipFile(path) as bundle:
        try:
            record = json.loads(_read_member(bundle, RUN_FILENAME))
        except ValueError as exc:
            raise ReproduceError("run.json in the bundle does not parse") from exc
        if not isinstance(record, dict):
            raise ReproduceError("run.json in the bundle is not a run record")
        config = record.get("config")
        if not isinstance(config, dict):
            name = next((n for n in _CONFIG_MEMBERS if n in members), None)
            if name is not None:
                config = _parse_config(name, _read_member(bundle, name))
    if not isinstance(config, dict):
        raise ReproduceError("the bundle records no task config")
    return BundleRecord(path=path, record=record, config=config)


def bundle_kind(record: Mapping[str, Any]) -> Kind:
    """``single`` for one training, ``replicates`` for a replicate group.

    Raises:
        ReproduceError: for any other group kind (K-fold, sweep, comparison).
    """
    block = record.get("block")
    group = record.get("group")
    if block is None and group is None:
        return "single"
    if block is not None:
        kind = block
    else:
        kind = group.get("kind") if isinstance(group, dict) else None
    if kind == "replicates" and isinstance(group, dict):
        return "replicates"
    raise ReproduceError(
        f"reproducing a '{kind}' bundle is not supported yet; "
        "supported: a single run and a replicates group"
    )


def task_of(record: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """The task the runner dispatcher knows the config by.

    Raises:
        ReproduceError: for a researcher-defined task or a classification
            config whose block is not a plain training.
    """
    top = record.get("task")
    if isinstance(top, str) and top.startswith("custom:"):
        raise ReproduceError(
            f"reproducing the custom task '{top}' is not supported yet"
        )
    task = str(config.get("task", "multiclass"))
    block = config.get("block", "classification")
    if task in _CLASSIFICATION_TASKS and block != "classification":
        raise ReproduceError(
            f"reproducing a classification '{block}' run is not supported yet"
        )
    return task


def pick_seed(
    kind: Kind,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    requested: int | None,
) -> tuple[int, dict[str, Any] | None]:
    """The seed to train and, for a replicate group, that seed's recorded entry.

    A single run trains ``requested`` or its recorded ``training.seed``. A
    replicate group trains ``requested`` when it is one of the group's seeds,
    else the first seed that finished (the first child when none did).

    Raises:
        ReproduceError: the seed is not one of the group's, or none is recorded.
    """
    if kind == "single":
        if requested is not None:
            return requested, None
        seed = (config.get("training") or {}).get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ReproduceError("the bundle records no training seed; pass --seed N")
        return seed, None
    group: Mapping[str, Any] = record["group"]
    children: list[dict[str, Any]] = [
        c for c in group.get("children") or [] if isinstance(c, dict)
    ]
    seeds = [s for s in group.get("seeds") or [] if isinstance(s, int)]
    by_seed = {c.get("seed"): c for c in children}
    if requested is not None:
        if requested not in seeds:
            listed = ", ".join(str(s) for s in seeds) or "none"
            raise ReproduceError(
                f"seed {requested} is not one of the group's seeds: {listed}"
            )
        return requested, by_seed.get(requested)
    finished = [c for c in children if c.get("status") == _SUCCESS]
    for child in finished or children:
        if isinstance(child.get("seed"), int):
            return int(child["seed"]), child
    if seeds:
        return seeds[0], None
    raise ReproduceError("the replicate group records no seed")


def dataset_section(
    config: Mapping[str, Any], task: str, data: Path | None
) -> dict[str, Any]:
    """The config's ``data`` section pointing at a dataset that exists here.

    ``data`` replaces the recorded path(s): a folder becomes ``base_dir``, and
    for detection a file becomes ``data_yaml`` (the other source is cleared).
    Without it, the recorded paths are kept when they all exist on this machine.

    Raises:
        ReproduceError: ``data`` is missing or of the wrong kind, or a recorded
            path is not here and ``data`` was not given.
    """
    section = dict(config.get("data") or {})
    if data is not None:
        if task == "detection" and data.is_file():
            section.update({"data_yaml": str(data), "base_dir": None})
        elif data.is_dir():
            section["base_dir"] = str(data)
            if task == "detection":
                section["data_yaml"] = None
        else:
            raise ReproduceError(f"--data {data} is not an existing dataset folder")
        return section
    recorded = [str(section[f]) for f in DATASET_FIELDS if section.get(f)]
    if not recorded:
        raise ReproduceError(
            "the bundle records no dataset path; pass --data DATASET_DIR"
        )
    for value in recorded:
        if not Path(value).exists():
            raise ReproduceError(
                f"the recorded dataset '{value}' is not on this machine; "
                "pass --data DATASET_DIR"
            )
    return section


def reproduction_config(
    config: Mapping[str, Any],
    *,
    name: str,
    seed: int,
    data: dict[str, Any],
    output: Path | None,
) -> dict[str, Any]:
    """The recorded config with the new run's name, seed, dataset and outputs.

    The recorded output folders belong to the machine that trained; they are
    dropped, so the run lands in the default outputs folder, or under
    ``output`` (``output/models``, ``output/graphics``...) when one is given.
    """
    new = copy.deepcopy(dict(config))
    new["name"] = name
    new["training"] = {**(new.get("training") or {}), "seed": seed}
    new["data"] = data
    section = {
        k: v for k, v in (new.get("output") or {}).items() if k not in _OUTPUT_DIRS
    }
    if output is not None:
        section.update(
            {key: str(output / key.removesuffix("_dir")) for key in _OUTPUT_DIRS}
        )
    new["output"] = section
    return new


def plan_reproduction(
    bundle_path: Path | str,
    *,
    data: Path | None = None,
    seed: int | None = None,
    output: Path | None = None,
) -> ReproductionPlan:
    """Verify the bundle and decide the seed, dataset and config to train.

    Raises:
        ValueError: anything that makes the bundle not reproducible as asked
            (exit code 2 on the command line).
        OSError: the bundle cannot be read.
    """
    bundle = read_bundle_record(bundle_path)
    kind = bundle_kind(bundle.record)
    task = task_of(bundle.record, bundle.config)
    chosen, child = pick_seed(kind, bundle.record, bundle.config, seed)
    base = str(bundle.config.get("name") or bundle.record.get("experiment") or "run")
    name = f"{base}_s{chosen}" if kind == "replicates" else base
    config = reproduction_config(
        bundle.config,
        name=f"{name}{REPRO_SUFFIX}",
        seed=chosen,
        data=dataset_section(bundle.config, task, data),
        output=output,
    )
    return ReproductionPlan(
        bundle=bundle.path,
        kind=kind,
        seed=chosen,
        task=task,
        record=bundle.record,
        child=child,
        config=config,
    )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def recorded_value(plan: ReproductionPlan, name: str, split: Split) -> float | None:
    """What the bundle recorded for metric ``name`` of the reproduced seed.

    A replicate group's children carry the runner's own metric names. A single
    run's run.json writes the test score as ``test_<name>`` (the bare name for
    detection and custom tasks, ``core.run_groups``) and a validation score as
    ``val_<name>``, or the bare name in older records.
    """
    if plan.kind == "replicates":
        return _finite(((plan.child or {}).get("metrics") or {}).get(name))
    metrics: Mapping[str, Any] = plan.record.get("metrics") or {}
    if split == "test":
        bare = _writes_bare_names(plan.config, plan.record)
        return _finite(metrics.get(name if bare else f"test_{name}"))
    value = _finite(metrics.get(f"val_{name}"))
    return value if value is not None else _finite(metrics.get(name))


def recorded_intervals(record: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """The group's Student-t 95% interval per metric, where it has both ends."""
    aggregates = (record.get("group") or {}).get("aggregates") or {}
    intervals: dict[str, tuple[float, float]] = {}
    for name, agg in aggregates.items():
        if not isinstance(agg, Mapping):
            continue
        low, high = _finite(agg.get("ci95_low")), _finite(agg.get("ci95_high"))
        if low is not None and high is not None:
            intervals[str(name)] = (low, high)
    return intervals


def prediction_intervals(
    record: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    """The 95% prediction interval of one new seed per metric of the group.

    ``mean ± t(0.975, n-1) · std · sqrt(1 + 1/n)`` from the recorded ``n``,
    ``mean`` and ``std`` (sample, ddof=1, as ``aggregate_replicates`` writes
    it), clipped to the metric's domain as the recorded ``ci95`` is. A metric
    without ``n >= 2`` and a finite mean and std gets none.
    """
    aggregates = (record.get("group") or {}).get("aggregates") or {}
    intervals: dict[str, tuple[float, float]] = {}
    for name, agg in aggregates.items():
        if not isinstance(agg, Mapping):
            continue
        n = agg.get("n")
        mean, std = _finite(agg.get("mean")), _finite(agg.get("std"))
        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            continue
        if mean is None or std is None:
            continue
        half = _t_critical_95(n - 1) * std * math.sqrt(1 + 1 / n)
        low, high, _ = _clip_to_domain(
            mean - half, mean + half, metric_domain(str(name))
        )
        intervals[str(name)] = (low, high)
    return intervals


def compare_metrics(
    names: list[str],
    recorded: Mapping[str, float | None],
    reproduced: Mapping[str, float | None],
    intervals: Mapping[str, tuple[float, float]],
    predictions: Mapping[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """One row per metric: both values, their difference and the interval check.

    ``ci95`` (the recorded interval of the mean) is reported for context;
    ``inside`` is judged against ``pi95``, the prediction interval of one seed.
    """
    rows: list[dict[str, Any]] = []
    for name in names:
        old, new = recorded.get(name), reproduced.get(name)
        interval, prediction = intervals.get(name), predictions.get(name)
        inside = (
            None
            if prediction is None or new is None
            else prediction[0] <= new <= prediction[1]
        )
        rows.append(
            {
                "name": name,
                "recorded": old,
                "reproduced": new,
                "delta": None if old is None or new is None else new - old,
                "ci95": list(interval) if interval is not None else None,
                "pi95": list(prediction) if prediction is not None else None,
                "inside": inside,
            }
        )
    return rows


def overall_verdict(rows: list[dict[str, Any]]) -> Verdict:
    """Outside if any metric left its pi95, inside if any was checked, else none."""
    checked = [row["inside"] for row in rows if row["inside"] is not None]
    if not checked:
        return "no_interval"
    return "inside_interval" if all(checked) else "outside_interval"


def environment_diff(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Recorded against current library versions and device, from two run.json.

    ``same`` is None when either side did not record the value.
    """
    old_env: Mapping[str, Any] = recorded.get("environment") or {}
    new_env: Mapping[str, Any] = current.get("environment") or {}
    pairs = {key: (old_env.get(key), new_env.get(key)) for key in ENVIRONMENT_KEYS}
    pairs["device"] = (recorded.get("device_used"), current.get("device_used"))
    return {
        key: {
            "recorded": old,
            "current": new,
            "same": None if old is None or new is None else old == new,
        }
        for key, (old, new) in pairs.items()
    }


def _current_record(run_dir: Path | None) -> dict[str, Any]:
    """The new run's run.json, or this process's environment when it has none."""
    if run_dir is not None:
        try:
            return read_run_record(run_dir)[1]
        except (OSError, ValueError):
            pass
    return {"environment": capture_environment()}


def run_reproduction(
    plan: ReproductionPlan, runner_for: Callable[[str], TaskRunner]
) -> dict[str, Any]:
    """Train the planned seed and return the comparison report.

    The compared metrics are the runner's final ones (test split), or its
    validation ones when the task has no test split, plus every metric the
    group's aggregates name. ``runner_for`` is the dispatcher of ``visionforge
    run`` (``cli_strategies.runner_for``).

    Raises:
        ValueError: the config does not validate for its task (nothing trained).
        RuntimeError: the training failed.
    """
    runner = runner_for(plan.task)
    cfg = runner.config_type.model_validate(plan.config)
    result: RunResult = runner.run(cfg)
    if result.status != _SUCCESS:
        raise RuntimeError(f"the reproduction training failed: {result.error}")
    reproduced: dict[str, float | None] = {
        k: _finite(v) for k, v in runner.metrics(result).items()
    }
    split: Split = "test"
    if not reproduced:
        reproduced = {k: _finite(v) for k, v in result.selection_metrics.items()}
        split = "validation"
    intervals = recorded_intervals(plan.record)
    aggregates = (plan.record.get("group") or {}).get("aggregates") or {}
    names = list(dict.fromkeys([*reproduced, *map(str, aggregates)]))
    recorded = {name: recorded_value(plan, name, split) for name in names}
    rows = compare_metrics(
        names, recorded, reproduced, intervals, prediction_intervals(plan.record)
    )
    return {
        "bundle": str(plan.bundle),
        "kind": plan.kind,
        "seed": plan.seed,
        "run_dir": str(result.run_dir) if result.run_dir is not None else None,
        "environment_diff": environment_diff(
            plan.record, _current_record(result.run_dir)
        ),
        "metrics": rows,
        "verdict": overall_verdict(rows),
    }


def exit_code(verdict: str) -> int:
    """1 when a metric left its 95% prediction interval, else 0."""
    return 1 if verdict == "outside_interval" else 0


__all__ = [
    "BundleRecord",
    "ReproduceError",
    "ReproductionPlan",
    "bundle_kind",
    "compare_metrics",
    "dataset_section",
    "environment_diff",
    "exit_code",
    "overall_verdict",
    "pick_seed",
    "plan_reproduction",
    "prediction_intervals",
    "read_bundle_record",
    "recorded_intervals",
    "recorded_value",
    "reproduction_config",
    "run_reproduction",
    "task_of",
]
