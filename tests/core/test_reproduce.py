"""Bundle reproduction: reading, seed, dataset, config and the comparison."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.experiment_bundle import export_experiment_bundle
from visionforge.core.replicates import ReplicateTrial
from visionforge.core.reproduce import (
    ReproduceError,
    ReproductionPlan,
    bundle_kind,
    compare_metrics,
    dataset_section,
    environment_diff,
    exit_code,
    overall_verdict,
    pick_seed,
    plan_reproduction,
    prediction_intervals,
    read_bundle_record,
    recorded_intervals,
    recorded_value,
    reproduction_config,
    run_reproduction,
    task_of,
)
from visionforge.core.run_groups import replicates_report, write_run_group
from visionforge.core.task_runner import RunResult
from visionforge.utils.regression_config import RegressionConfig

R2 = "r2"
MAE = "mae"
ENVIRONMENT = {
    "python": "3.12.0",
    "torch": "2.9.0",
    "torchvision": "0.24.0",
    "visionforge": "0.22.1",
}


def _config(tmp_path: Path, name: str = "reg", seed: int = 7) -> dict[str, Any]:
    (tmp_path / "data").mkdir(exist_ok=True)
    return {
        "name": name,
        "task": "regression",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(tmp_path / "data"), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 4, "seed": seed},
        "output": {
            "models_dir": str(tmp_path / "old" / "models"),
            "graphics_dir": str(tmp_path / "old" / "graphics"),
            "figure_dpi": 150,
        },
        "device": {"kind": "cpu"},
    }


def _single_run(tmp_path: Path, metrics: dict[str, Any] | None = None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = {
        "experiment": "reg",
        "status": "completed",
        "environment": ENVIRONMENT,
        "device_used": "cpu",
        "config": _config(tmp_path),
        "metrics": metrics or {"test_r2": 0.8, "test_mae": 0.1, "total_epochs": 1},
        "artifacts": {},
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def _bundle(tmp_path: Path, run_dir: Path, name: str = "bundle.zip") -> Path:
    return export_experiment_bundle(run_dir, tmp_path / name)


def _replicate_group(
    tmp_path: Path,
    values: dict[int, float],
    *,
    failed: tuple[int, ...] = (),
) -> Path:
    """A real replicate group (``write_run_group``) over child runs on disk."""
    models = tmp_path / "models"
    trials: list[ReplicateTrial] = []
    for seed in [*failed, *values]:
        if seed in failed:
            trials.append(ReplicateTrial(seed=seed, error="boom"))
            continue
        run_dir = models / f"reg_s{seed}" / "20260101_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "experiment": f"reg_s{seed}",
                    "environment": ENVIRONMENT,
                    "device_used": "cuda",
                    "metrics": {"test_r2": values[seed], "total_epochs": 1},
                }
            ),
            encoding="utf-8",
        )
        trials.append(
            ReplicateTrial(
                seed=seed,
                status="success",
                metrics={R2: values[seed], MAE: 0.1 + seed / 100},
                run_dir=str(run_dir),
            )
        )
    seeds = [t.seed for t in trials]
    return write_run_group(
        kind="replicates",
        group_id="g1",
        name="reg",
        config=_config(tmp_path, seed=seeds[0]),
        report=replicates_report(trials, seeds, R2),
        report_json=None,
        models_dir=models,
    )


class FakeRunner:
    """Trains nothing; returns the metrics it was built with."""

    config_type = RegressionConfig

    def __init__(
        self,
        metrics: dict[str, float],
        *,
        status: str = "success",
        selection: dict[str, float] | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self._metrics = metrics
        self._status = status
        self._selection = selection or {}
        self._run_dir = run_dir
        self.configs: list[Any] = []

    def run(self, cfg: Any) -> RunResult:
        self.configs.append(cfg)
        return RunResult(
            metrics=dict(self._metrics),
            selection_metrics=dict(self._selection),
            status=self._status,
            error="" if self._status == "success" else "out of memory",
            run_dir=self._run_dir,
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return dict(result.metrics)

    def primary_metric(self) -> str:
        return R2


# ---------------------------------------------------------------- reading


def test_reads_run_json_and_its_config_from_a_verified_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, _single_run(tmp_path))
    record = read_bundle_record(bundle)
    assert record.path == bundle
    assert record.record["metrics"]["test_r2"] == pytest.approx(0.8)
    assert record.config["name"] == "reg"
    assert not (tmp_path / "run.json").exists()


def test_falls_back_to_a_config_member_when_run_json_has_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text('{"artifacts": {}}', encoding="utf-8")
    (run_dir / "config.yaml").write_text("name: from_yaml\ntask: regression\n")
    record = read_bundle_record(_bundle(tmp_path, run_dir))
    assert record.config == {"name": "from_yaml", "task": "regression"}


def test_a_bundle_without_any_config_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text('{"artifacts": {}}', encoding="utf-8")
    bundle = _bundle(tmp_path, run_dir)
    with pytest.raises(ReproduceError, match="no task config"):
        read_bundle_record(bundle)


def test_an_unparsable_config_member_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text('{"artifacts": {}}', encoding="utf-8")
    (run_dir / "config.json").write_text("{not json", encoding="utf-8")
    bundle = _bundle(tmp_path, run_dir)
    with pytest.raises(ReproduceError, match=r"config\.json"):
        read_bundle_record(bundle)


def test_a_tampered_bundle_fails_verification(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, _single_run(tmp_path))
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(tampered, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            dst.writestr(info, data.replace(b"0.8", b"0.9"))
    with pytest.raises(ValueError, match="integrity"):
        read_bundle_record(tampered)


def test_a_file_that_is_not_a_zip_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="invalid experiment bundle"):
        read_bundle_record(path)


# ---------------------------------------------------------------- kind, task


def test_a_record_without_block_or_group_is_a_single_run() -> None:
    assert bundle_kind({"metrics": {}}) == "single"


def test_a_replicates_group_is_recognised() -> None:
    assert bundle_kind({"block": "replicates", "group": {"kind": "replicates"}}) == (
        "replicates"
    )
    assert bundle_kind({"group": {"kind": "replicates"}}) == "replicates"


@pytest.mark.parametrize("kind", ["kfold", "sweep", "replicated_comparison"])
def test_other_group_kinds_are_not_supported_yet(kind: str) -> None:
    with pytest.raises(ReproduceError, match=f"'{kind}'.*not supported yet"):
        bundle_kind({"block": kind, "group": {"kind": kind}})


def test_a_group_that_is_not_an_object_is_not_supported() -> None:
    with pytest.raises(ReproduceError, match="'None'"):
        bundle_kind({"group": ["x"]})


def test_task_of_names_the_runner_task() -> None:
    assert task_of({}, {"task": "regression"}) == "regression"
    assert task_of({}, {"block": "classification"}) == "multiclass"


def test_custom_tasks_are_not_supported_yet() -> None:
    with pytest.raises(ReproduceError, match="custom:mine"):
        task_of({"task": "custom:mine"}, {"task": "custom:mine"})


def test_a_custom_task_group_stopped_before_any_run_json_is_refused(
    tmp_path: Path,
) -> None:
    # Issue 343: no child wrote run.json, so the group inherits no
    # ``task: custom:<key>``, and a custom task's config names no task or block.
    # It must not fall back to classification.
    config = {
        "name": "mine",
        "data": {"base_dir": str(tmp_path)},
        "training": {"epochs": 1, "seed": 3},
    }
    run_dir = write_run_group(
        kind="replicates",
        group_id="g-custom",
        name="mine",
        config=config,
        report=replicates_report([ReplicateTrial(seed=3, error="stopped")], [3], R2),
        report_json=None,
        models_dir=tmp_path / "models",
    )
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert "task" not in record
    with pytest.raises(ReproduceError, match="does not say which task"):
        task_of(record, record["config"])


def test_a_classification_search_block_is_not_supported_yet() -> None:
    with pytest.raises(ReproduceError, match="cross_validation"):
        task_of({}, {"task": "binary", "block": "cross_validation"})


# ---------------------------------------------------------------- seed


def test_a_single_run_trains_its_recorded_seed_or_the_requested_one() -> None:
    config = {"training": {"seed": 7}}
    assert pick_seed("single", {}, config, None) == (7, None)
    assert pick_seed("single", {}, config, 3) == (3, None)


def test_a_single_run_without_a_seed_asks_for_one() -> None:
    with pytest.raises(ReproduceError, match="--seed"):
        pick_seed("single", {}, {"training": {}}, None)


def _group_record(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    group_dir = _replicate_group(tmp_path, **kwargs)
    data: dict[str, Any] = json.loads(
        (group_dir / "run.json").read_text(encoding="utf-8")
    )
    return data


def test_a_replicate_group_trains_the_requested_seed_with_its_entry(
    tmp_path: Path,
) -> None:
    record = _group_record(tmp_path, values={0: 0.8, 1: 0.84})
    seed, child = pick_seed("replicates", record, record["config"], 1)
    assert seed == 1
    assert child is not None
    assert child["metrics"][R2] == pytest.approx(0.84)


def test_a_seed_outside_the_group_lists_the_group_seeds(tmp_path: Path) -> None:
    record = _group_record(tmp_path, values={0: 0.8, 1: 0.84})
    with pytest.raises(ReproduceError, match="seed 5 .* 0, 1"):
        pick_seed("replicates", record, record["config"], 5)


def test_the_default_is_the_first_seed_that_finished(tmp_path: Path) -> None:
    record = _group_record(tmp_path, values={3: 0.8, 4: 0.84}, failed=(9,))
    assert record["group"]["children"][0]["seed"] == 9
    seed, child = pick_seed("replicates", record, record["config"], None)
    assert seed == 3
    assert child is not None
    assert child["status"] == "success"


def test_a_group_with_no_children_falls_back_to_its_first_seed() -> None:
    record = {"group": {"seeds": [4, 5], "children": []}}
    assert pick_seed("replicates", record, {}, None) == (4, None)


def test_a_group_with_no_seed_at_all_is_refused() -> None:
    with pytest.raises(ReproduceError, match="no seed"):
        pick_seed("replicates", {"group": {}}, {}, None)


# ---------------------------------------------------------------- dataset


def test_data_replaces_the_recorded_dataset(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    section = dataset_section(
        {"data": {"base_dir": "/elsewhere/ds", "images_dir": "img"}},
        "regression",
        local,
    )
    assert section == {"base_dir": str(local), "images_dir": "img"}


def test_detection_takes_a_data_yaml_file_and_clears_base_dir(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [a]\n", encoding="utf-8")
    config = {"data": {"base_dir": "/elsewhere", "data_yaml": None}}
    section = dataset_section(config, "detection", data_yaml)
    assert section == {"base_dir": None, "data_yaml": str(data_yaml)}


def test_detection_takes_a_folder_and_clears_data_yaml(tmp_path: Path) -> None:
    config = {"data": {"data_yaml": "/elsewhere/data.yaml"}}
    section = dataset_section(config, "detection", tmp_path)
    assert section == {"data_yaml": None, "base_dir": str(tmp_path)}


def test_data_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReproduceError, match="not an existing dataset folder"):
        dataset_section({"data": {}}, "regression", tmp_path / "missing")


def test_a_file_is_not_a_dataset_outside_detection(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text("x: 1\n", encoding="utf-8")
    with pytest.raises(ReproduceError, match="not an existing dataset folder"):
        dataset_section({"data": {}}, "segmentation", path)


def test_a_recorded_dataset_that_exists_here_is_kept(tmp_path: Path) -> None:
    config = {"data": {"base_dir": str(tmp_path)}}
    assert dataset_section(config, "anomaly", None) == {"base_dir": str(tmp_path)}


def test_a_recorded_dataset_missing_here_asks_for_data(tmp_path: Path) -> None:
    config = {"data": {"base_dir": str(tmp_path / "gone")}}
    with pytest.raises(ReproduceError, match="pass --data DATASET_DIR"):
        dataset_section(config, "regression", None)


def test_a_config_without_a_dataset_path_asks_for_data() -> None:
    with pytest.raises(ReproduceError, match="no dataset path"):
        dataset_section({"data": {"base_dir": None}}, "regression", None)


# ---------------------------------------------------------------- config


def test_reproduction_config_renames_reseeds_and_moves_outputs(
    tmp_path: Path,
) -> None:
    original = _config(tmp_path)
    new = reproduction_config(
        original,
        name="reg_repro",
        seed=3,
        data={"base_dir": "/d"},
        output=tmp_path / "out",
    )
    assert new["name"] == "reg_repro"
    assert new["training"] == {"epochs": 1, "batch_size": 4, "seed": 3}
    assert new["data"] == {"base_dir": "/d"}
    assert new["output"] == {
        "figure_dpi": 150,
        "models_dir": str(tmp_path / "out" / "models"),
        "graphics_dir": str(tmp_path / "out" / "graphics"),
        "logs_dir": str(tmp_path / "out" / "logs"),
        "reports_dir": str(tmp_path / "out" / "reports"),
    }
    assert original["name"] == "reg"
    assert original["training"]["seed"] == 7


def test_without_output_the_recorded_folders_give_way_to_the_defaults(
    tmp_path: Path,
) -> None:
    new = reproduction_config(_config(tmp_path), name="n", seed=1, data={}, output=None)
    assert new["output"] == {"figure_dpi": 150}


def test_plan_names_a_replicate_seed_run_after_the_seed(tmp_path: Path) -> None:
    group_dir = _replicate_group(tmp_path, values={0: 0.8, 1: 0.84})
    bundle = _bundle(tmp_path, group_dir)
    plan = plan_reproduction(bundle, seed=1, output=tmp_path / "out")
    assert plan.kind == "replicates"
    assert plan.seed == 1
    assert plan.task == "regression"
    assert plan.config["name"] == "reg_s1_repro"
    assert plan.config["training"]["seed"] == 1
    assert plan.config["data"]["base_dir"] == str(tmp_path / "data")


def test_plan_of_a_single_run_keeps_its_name_with_the_suffix(tmp_path: Path) -> None:
    plan = plan_reproduction(_bundle(tmp_path, _single_run(tmp_path)))
    assert (plan.kind, plan.seed, plan.child) == ("single", 7, None)
    assert plan.config["name"] == "reg_repro"


# ---------------------------------------------------------------- comparison


def _plan(kind: str, record: dict[str, Any], child: dict[str, Any] | None = None):
    return ReproductionPlan(
        bundle=Path("b.zip"),
        kind=kind,  # type: ignore[arg-type]
        seed=0,
        task=str(record.get("config", {}).get("task", "regression")),
        record=record,
        child=child,
        config=record.get("config", {}),
    )


def test_a_single_run_records_test_scores_under_the_test_prefix() -> None:
    plan = _plan(
        "single", {"config": {"task": "regression"}, "metrics": {"test_r2": 0.7}}
    )
    assert recorded_value(plan, R2, "test") == pytest.approx(0.7)


def test_detection_records_its_scores_under_the_bare_name() -> None:
    record = {"config": {"task": "detection"}, "metrics": {"map50": 0.4}}
    assert recorded_value(_plan("single", record), "map50", "test") == 0.4


def test_validation_scores_are_read_as_val_or_the_legacy_bare_name() -> None:
    record = {"config": {"task": "segmentation"}, "metrics": {"val_miou": 0.5}}
    assert recorded_value(_plan("single", record), "miou", "validation") == 0.5
    legacy = {"config": {"task": "segmentation"}, "metrics": {"miou": 0.6}}
    assert recorded_value(_plan("single", legacy), "miou", "validation") == 0.6


def test_a_non_finite_or_missing_record_is_none() -> None:
    record = {"config": {"task": "regression"}, "metrics": {"test_r2": float("nan")}}
    assert recorded_value(_plan("single", record), R2, "test") is None
    assert recorded_value(_plan("single", record), MAE, "test") is None


def test_a_replicate_seed_records_its_own_child_value() -> None:
    child = {"seed": 0, "status": "success", "metrics": {R2: 0.81}}
    plan = _plan("replicates", {"group": {}}, child=child)
    assert recorded_value(plan, R2, "test") == pytest.approx(0.81)
    assert recorded_value(_plan("replicates", {"group": {}}), R2, "test") is None


def test_recorded_intervals_keep_only_metrics_with_both_ends() -> None:
    record = {
        "group": {
            "aggregates": {
                R2: {"ci95_low": 0.7, "ci95_high": 0.9},
                MAE: {"ci95_low": None, "ci95_high": None},
                "bad": "x",
            }
        }
    }
    assert recorded_intervals(record) == {R2: (0.7, 0.9)}


def _aggregate(n: int, mean: float, std: float | None) -> dict[str, Any]:
    return {"n": n, "mean": mean, "std": std, "std_ddof": 1}


def test_prediction_interval_of_one_new_seed_matches_a_hand_computation() -> None:
    # t(0.975, 4) = 2.776445; half = 2.776445 * 0.05 * sqrt(1 + 1/5) = 0.152072
    record = {"group": {"aggregates": {"accuracy": _aggregate(5, 0.80, 0.05)}}}
    (low, high) = prediction_intervals(record)["accuracy"]
    assert low == pytest.approx(0.647928, abs=1e-4)
    assert high == pytest.approx(0.952072, abs=1e-4)


def test_prediction_interval_is_clipped_to_the_metric_domain() -> None:
    record = {"group": {"aggregates": {R2: _aggregate(5, 0.97, 0.05)}}}
    (low, high) = prediction_intervals(record)[R2]
    assert low == pytest.approx(0.97 - 0.152072, abs=1e-4)
    assert high == 1.0


def test_prediction_interval_of_a_percent_scale_metric_is_not_inverted() -> None:
    # Accuracy recorded on a 0-100 scale: the name says [0, 1], the values do not.
    aggregate = {**_aggregate(3, 85.0, 1.0), "min": 84.0, "max": 86.0}
    record = {"group": {"aggregates": {"accuracy": aggregate}}}
    (low, high) = prediction_intervals(record)["accuracy"]
    assert low <= high
    assert low <= 84.0
    assert high >= 86.0
    assert low < 85.0 < high


def test_prediction_interval_is_clipped_when_every_observed_value_is_in_domain() -> (
    None
):
    aggregate = {**_aggregate(5, 0.97, 0.05), "min": 0.9, "max": 0.99}
    (low, high) = prediction_intervals({"group": {"aggregates": {R2: aggregate}}})[R2]
    assert high == 1.0
    assert low == pytest.approx(0.97 - 0.152072, abs=1e-4)


@pytest.mark.parametrize(
    "aggregate",
    [
        _aggregate(1, 0.8, None),
        _aggregate(5, 0.8, None),
        _aggregate(5, 0.8, float("nan")),
        {"n": 5, "std": 0.05},
        {"n": True, "mean": 0.8, "std": 0.05},
        "not an aggregate",
    ],
)
def test_no_prediction_interval_without_n_of_two_a_mean_and_a_std(
    aggregate: Any,
) -> None:
    assert prediction_intervals({"group": {"aggregates": {R2: aggregate}}}) == {}


def test_a_seed_outside_the_mean_ci_but_inside_the_prediction_interval_passes() -> None:
    # ci95 of the mean: 0.80 ± 2.776445 * 0.05 / sqrt(5) = (0.737916, 0.862084).
    record = {
        "group": {
            "aggregates": {
                "accuracy": {
                    **_aggregate(5, 0.80, 0.05),
                    "ci95_low": 0.737916,
                    "ci95_high": 0.862084,
                }
            }
        }
    }
    rows = compare_metrics(
        ["accuracy"],
        {"accuracy": 0.81},
        {"accuracy": 0.70},
        recorded_intervals(record),
        prediction_intervals(record),
    )
    (row,) = rows
    assert not row["ci95"][0] <= 0.70 <= row["ci95"][1]
    assert row["pi95"] == [
        pytest.approx(0.647928, abs=1e-4),
        pytest.approx(0.952072, abs=1e-4),
    ]
    assert row["inside"] is True
    assert overall_verdict(rows) == "inside_interval"


def test_compare_metrics_checks_each_value_against_its_prediction_interval() -> None:
    rows = compare_metrics(
        [R2, MAE, "rmse"],
        {R2: 0.8, MAE: 0.1, "rmse": None},
        {R2: 0.95, MAE: 0.3, "rmse": 0.2},
        {R2: (0.7, 0.9), MAE: (0.05, 0.2)},
        {R2: (0.6, 1.0), MAE: (0.0, 0.25)},
    )
    assert rows[0] == {
        "name": R2,
        "recorded": 0.8,
        "reproduced": 0.95,
        "delta": pytest.approx(0.15),
        "ci95": [0.7, 0.9],
        "pi95": [0.6, 1.0],
        "inside": True,
    }
    assert rows[1]["inside"] is False
    assert rows[2] == {
        "name": "rmse",
        "recorded": None,
        "reproduced": 0.2,
        "delta": None,
        "ci95": None,
        "pi95": None,
        "inside": None,
    }


def test_a_ci95_alone_does_not_judge_a_seed() -> None:
    rows = compare_metrics([R2], {}, {R2: 0.95}, {R2: (0.7, 0.9)}, {})
    assert rows[0]["ci95"] == [0.7, 0.9]
    assert rows[0]["pi95"] is None
    assert rows[0]["inside"] is None


def test_without_a_reproduced_value_nothing_is_judged() -> None:
    rows = compare_metrics([R2], {R2: 0.8}, {}, {}, {R2: (0.6, 1.0)})
    assert rows[0]["inside"] is None


def test_the_interval_ends_count_as_inside() -> None:
    rows = compare_metrics([R2], {}, {R2: 0.9}, {}, {R2: (0.7, 0.9)})
    assert rows[0]["inside"] is True


@pytest.mark.parametrize(
    ("insides", "verdict"),
    [
        ([True, None], "inside_interval"),
        ([True, False], "outside_interval"),
        ([None, None], "no_interval"),
        ([], "no_interval"),
    ],
)
def test_overall_verdict(insides: list[bool | None], verdict: str) -> None:
    assert overall_verdict([{"inside": i} for i in insides]) == verdict


def test_exit_code_is_1_only_outside_the_interval() -> None:
    assert exit_code("outside_interval") == 1
    assert exit_code("inside_interval") == 0
    assert exit_code("no_interval") == 0


def test_environment_diff_compares_versions_and_device() -> None:
    diff = environment_diff(
        {"environment": ENVIRONMENT, "device_used": "cuda"},
        {"environment": {**ENVIRONMENT, "torch": "2.10.0"}, "device_used": "cpu"},
    )
    assert set(diff) == {"python", "torch", "torchvision", "visionforge", "device"}
    assert diff["python"] == {"recorded": "3.12.0", "current": "3.12.0", "same": True}
    assert diff["torch"]["same"] is False
    assert diff["device"] == {"recorded": "cuda", "current": "cpu", "same": False}


def test_environment_diff_says_nothing_about_an_unrecorded_value() -> None:
    diff = environment_diff({}, {"environment": ENVIRONMENT})
    assert diff["torch"] == {"recorded": None, "current": "2.9.0", "same": None}


# ---------------------------------------------------------------- training


def _new_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "new_run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {"environment": {**ENVIRONMENT, "torch": "2.10.0"}, "device_used": "cpu"}
        ),
        encoding="utf-8",
    )
    return run_dir


def test_a_single_run_reproduction_has_no_interval(tmp_path: Path) -> None:
    plan = plan_reproduction(_bundle(tmp_path, _single_run(tmp_path)))
    runner = FakeRunner({R2: 0.75, MAE: 0.1}, run_dir=_new_run(tmp_path))
    report = run_reproduction(plan, lambda task: runner)
    assert runner.configs[0].name == "reg_repro"
    assert runner.configs[0].training.seed == 7
    assert report["kind"] == "single"
    assert report["seed"] == 7
    assert report["run_dir"] == str(tmp_path / "new_run")
    assert report["verdict"] == "no_interval"
    assert [m["name"] for m in report["metrics"]] == [R2, MAE]
    assert report["metrics"][0]["recorded"] == pytest.approx(0.8)
    assert report["metrics"][0]["delta"] == pytest.approx(-0.05)
    assert report["environment_diff"]["torch"]["same"] is False
    assert report["environment_diff"]["device"]["same"] is True


def test_a_replicate_seed_outside_the_interval_is_reported(tmp_path: Path) -> None:
    group_dir = _replicate_group(tmp_path, values={0: 0.80, 1: 0.82, 2: 0.81})
    plan = plan_reproduction(_bundle(tmp_path, group_dir), seed=2)
    runner = FakeRunner({R2: 0.10, MAE: 0.12})
    report = run_reproduction(plan, lambda task: runner)
    rows = {m["name"]: m for m in report["metrics"]}
    assert rows[R2]["recorded"] == pytest.approx(0.81)
    # 0.81 ± t(2)=4.302653 * 0.01 * sqrt(1 + 1/3) = 0.81 ± 0.049683
    assert rows[R2]["pi95"] == [
        pytest.approx(0.760317, abs=1e-4),
        pytest.approx(0.859683, abs=1e-4),
    ]
    assert rows[R2]["inside"] is False
    assert rows[MAE]["inside"] is True
    assert report["verdict"] == "outside_interval"
    # The run recorded its environment through a child of the group.
    assert report["environment_diff"]["device"]["recorded"] == "cuda"


def test_a_replicate_seed_inside_the_interval_passes(tmp_path: Path) -> None:
    group_dir = _replicate_group(tmp_path, values={0: 0.80, 1: 0.82, 2: 0.81})
    plan = plan_reproduction(_bundle(tmp_path, group_dir))
    report = run_reproduction(plan, lambda task: FakeRunner({R2: 0.80, MAE: 0.1}))
    assert report["seed"] == 0
    assert report["verdict"] == "inside_interval"


def test_a_real_group_judges_the_seed_by_prediction_not_mean_interval(
    tmp_path: Path,
) -> None:
    group_dir = _replicate_group(tmp_path, values={0: 0.80, 1: 0.82, 2: 0.81})
    plan = plan_reproduction(_bundle(tmp_path, group_dir))
    report = run_reproduction(plan, lambda task: FakeRunner({R2: 0.845, MAE: 0.1}))
    row = next(m for m in report["metrics"] if m["name"] == R2)
    # ci95 of the mean: 0.81 ± 4.302653 * 0.01 / sqrt(3) = (0.785159, 0.834841)
    assert row["ci95"][1] == pytest.approx(0.834841, abs=1e-4)
    assert row["inside"] is True
    assert report["verdict"] == "inside_interval"


def test_a_group_metric_the_reproduction_did_not_report_is_listed(
    tmp_path: Path,
) -> None:
    group_dir = _replicate_group(tmp_path, values={0: 0.80, 1: 0.82})
    plan = plan_reproduction(_bundle(tmp_path, group_dir))
    report = run_reproduction(plan, lambda task: FakeRunner({R2: 0.80}))
    rows = {m["name"]: m for m in report["metrics"]}
    assert rows[MAE]["reproduced"] is None
    assert rows[MAE]["inside"] is None
    assert rows[MAE]["ci95"] is not None
    assert rows[MAE]["pi95"] is not None


def test_validation_metrics_stand_in_when_there_is_no_test_split(
    tmp_path: Path,
) -> None:
    run_dir = _single_run(tmp_path, metrics={"val_r2": 0.6, "total_epochs": 1})
    plan = plan_reproduction(_bundle(tmp_path, run_dir))
    report = run_reproduction(plan, lambda task: FakeRunner({}, selection={R2: 0.65}))
    assert report["metrics"] == [
        {
            "name": R2,
            "recorded": 0.6,
            "reproduced": 0.65,
            "delta": pytest.approx(0.05),
            "ci95": None,
            "pi95": None,
            "inside": None,
        }
    ]


def test_without_a_new_run_record_the_current_environment_is_used(
    tmp_path: Path,
) -> None:
    plan = plan_reproduction(_bundle(tmp_path, _single_run(tmp_path)))
    report = run_reproduction(plan, lambda task: FakeRunner({R2: 0.8}))
    assert report["run_dir"] is None
    assert report["environment_diff"]["torch"]["current"] is not None
    assert report["environment_diff"]["device"]["current"] is None


def test_a_failed_training_raises(tmp_path: Path) -> None:
    plan = plan_reproduction(_bundle(tmp_path, _single_run(tmp_path)))
    with pytest.raises(RuntimeError, match="out of memory"):
        run_reproduction(plan, lambda task: FakeRunner({}, status="failed"))


def test_a_config_that_does_not_validate_trains_nothing(tmp_path: Path) -> None:
    plan = plan_reproduction(_bundle(tmp_path, _single_run(tmp_path)))
    plan.config["training"]["epochs"] = 0
    runner = FakeRunner({R2: 0.8})
    with pytest.raises(ValueError, match="epochs"):
        run_reproduction(plan, lambda task: runner)
    assert runner.configs == []


# ------------------------------------------------- review findings (2026-11-18)


def test_reading_never_extracts_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path, _single_run(tmp_path))

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the bundle was extracted")

    monkeypatch.setattr(zipfile.ZipFile, "extract", refuse)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", refuse)
    assert read_bundle_record(bundle).config["name"] == "reg"


def test_a_non_finite_reproduced_metric_with_an_interval_is_outside() -> None:
    # The runner reported the metric, but as NaN: the reproduction did not
    # produce the value the group recorded, so it cannot pass as "no interval".
    rows = compare_metrics([R2], {R2: 0.8}, {R2: None}, {}, {R2: (0.6, 1.0)})
    assert rows[0]["inside"] is False
    assert overall_verdict(rows) == "outside_interval"


def test_a_replicate_seed_whose_training_gave_nan_exits_outside(
    tmp_path: Path,
) -> None:
    group_dir = _replicate_group(tmp_path, values={0: 0.80, 1: 0.82, 2: 0.81})
    plan = plan_reproduction(_bundle(tmp_path, group_dir))
    runner = FakeRunner({R2: float("nan"), MAE: 0.1})
    report = run_reproduction(plan, lambda task: runner)
    assert report["verdict"] == "outside_interval"


def test_a_group_reproduced_without_its_test_split_is_refused(
    tmp_path: Path,
) -> None:
    group_dir = _replicate_group(tmp_path, values={0: 0.80, 1: 0.82, 2: 0.81})
    plan = plan_reproduction(_bundle(tmp_path, group_dir))
    # No test split here: the runner only has validation scores, which the
    # group's test aggregates cannot judge.
    runner = FakeRunner({}, selection={R2: 0.81})
    with pytest.raises(ReproduceError, match="test split"):
        run_reproduction(plan, lambda task: runner)


def test_a_single_run_with_another_seed_has_no_recorded_value(
    tmp_path: Path,
) -> None:
    plan = plan_reproduction(_bundle(tmp_path, _single_run(tmp_path)), seed=3)
    report = run_reproduction(plan, lambda task: FakeRunner({R2: 0.75}))
    row = report["metrics"][0]
    assert (row["recorded"], row["delta"]) == (None, None)
    same = plan_reproduction(_bundle(tmp_path, tmp_path / "run", "b2.zip"), seed=7)
    assert recorded_value(same, R2, "test") == pytest.approx(0.8)


def test_a_stopped_replicate_is_not_a_recorded_baseline() -> None:
    child = {"seed": 0, "status": "stopped", "metrics": {R2: 0.4}}
    plan = _plan("replicates", {"group": {}}, child=child)
    assert recorded_value(plan, R2, "test") is None


@pytest.mark.parametrize("section", ["training", "output"])
def test_a_config_section_that_is_not_a_mapping_is_refused(
    tmp_path: Path, section: str
) -> None:
    config = {**_config(tmp_path), section: ["not", "a", "mapping"]}
    with pytest.raises(ReproduceError, match=section):
        reproduction_config(config, name="x", seed=1, data={}, output=None)


def test_a_training_section_that_is_not_a_mapping_has_no_seed_to_pick(
    tmp_path: Path,
) -> None:
    config = {**_config(tmp_path), "training": ["not", "a", "mapping"]}
    with pytest.raises(ReproduceError, match="training"):
        pick_seed("single", {}, config, None)


def test_an_output_that_is_a_file_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    out.write_text("x", encoding="utf-8")
    bundle = _bundle(tmp_path, _single_run(tmp_path))
    with pytest.raises(ReproduceError, match="out.txt"):
        plan_reproduction(bundle, output=out)
