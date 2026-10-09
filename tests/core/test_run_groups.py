"""A replicate group's run.json is the job's report, copied (ADR-113).

The group is written from the report the job already built, so these tests
hand the writer a report and read back what it put on disk: every number must
be the report's, children must carry the group's id, and a group nobody could
rank must say so instead of inventing a winner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.replicated_comparison import VariantResult, build_report
from visionforge.core.replicates import ReplicateTrial, aggregate_replicates
from visionforge.core.run_groups import build_group, group_run_dir, write_run_group


def _child_run(models_dir: Path, name: str, epochs: int = 1) -> Path:
    """A finished single run on disk, as the trainers leave it."""
    run_dir = models_dir / name / "20260101_000000_000001"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "id": f"{name}_20260101_000000_000001",
                "experiment": name,
                "status": "completed",
                "timestamp": "2026-01-01T00:00:00",
                "device_used": "cpu",
                "config": {"task": "regression"},
                "metrics": {"total_epochs": epochs, "r2": 0.5},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _trials(models_dir: Path, name: str, values: dict[int, float]) -> list[Any]:
    return [
        ReplicateTrial(
            seed=seed,
            status="success",
            metrics={"r2": value},
            training_time_s=0.1,
            run_dir=str(_child_run(models_dir, f"{name}_s{seed}")),
        )
        for seed, value in values.items()
    ]


def _replicates_report(
    trials: list[ReplicateTrial], seeds: list[int]
) -> dict[str, Any]:
    from dataclasses import asdict

    aggregates = aggregate_replicates(trials)
    return {
        "metric": "r2",
        "seeds": seeds,
        "trials": [asdict(t) for t in trials],
        "aggregates": aggregates,
        "headline": aggregates.get("r2"),
        "stopped": False,
        "report_dir": "outputs/reports/x/1",
    }


class TestReplicatesGroup:
    def test_the_aggregate_is_the_reports_not_a_recomputation(
        self, tmp_path: Path
    ) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.80, 2: 0.90, 3: 0.85})
        report = _replicates_report(trials, [1, 2, 3])

        run_dir = write_run_group(
            kind="replicates",
            group_id="rep_20260101",
            name="rep",
            config={"name": "rep", "task": "regression"},
            report=report,
            report_json=Path("outputs/reports/x/1/replicates_summary.json"),
            models_dir=tmp_path,
        )

        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        group = data["group"]
        assert group["kind"] == "replicates"
        assert group["aggregates"] == json.loads(json.dumps(report["aggregates"]))
        agg = group["aggregates"]["r2"]
        assert agg["n"] == 3
        assert agg["std_ddof"] == 1
        assert agg["ci95_low"] < agg["mean"] < agg["ci95_high"]
        assert group["seeds"] == [1, 2, 3]
        assert group["seeds_finished"] == [1, 2, 3]
        assert (group["n_requested"], group["n_finished"]) == (3, 3)
        assert group["report_path"].endswith("replicates_summary.json")

    def test_top_level_keys_are_the_ones_the_history_list_reads(
        self, tmp_path: Path
    ) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.8, 2: 0.9})
        run_dir = write_run_group(
            kind="replicates",
            group_id="rep_g1",
            name="rep",
            config={"name": "rep"},
            report=_replicates_report(trials, [1, 2]),
            report_json=None,
            models_dir=tmp_path,
        )

        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run_dir == group_run_dir(tmp_path, "rep", "replicates", "rep_g1")
        assert data["id"] == "rep_g1"
        assert data["experiment"] == "rep"
        assert data["status"] == "completed"
        assert data["stopped"] is False
        assert data["block"] == "replicates"
        assert data["config"] == {"name": "rep"}
        # Epochs are the children's, read off their run.json.
        assert data["metrics"]["total_epochs"] == 2
        # The mean sits where a single run of the task writes its score, and
        # ``metric_keys`` names which aggregate filled it.
        assert data["metrics"]["test_r2"] == data["group"]["aggregates"]["r2"]["mean"]
        assert data["group"]["metric_keys"] == {"test_r2": "r2"}
        # Provenance comes from a child.
        assert data["device_used"] == "cpu"

    def test_children_are_tagged_with_the_group_id(self, tmp_path: Path) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.8, 2: 0.9})
        write_run_group(
            kind="replicates",
            group_id="rep_g1",
            name="rep",
            config={},
            report=_replicates_report(trials, [1, 2]),
            report_json=None,
            models_dir=tmp_path,
        )

        for trial in trials:
            child = json.loads((Path(trial.run_dir) / "run.json").read_text("utf-8"))
            assert child["group_id"] == "rep_g1"
            # Additive: nothing the trainer wrote was lost.
            assert child["metrics"]["r2"] == 0.5

    def test_a_seed_that_never_trained_is_listed_without_a_run(
        self, tmp_path: Path
    ) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.8, 2: 0.9})
        trials.append(ReplicateTrial(seed=3, status="failed", error="boom"))
        group = build_group("replicates", _replicates_report(trials, [1, 2, 3]), None)

        failed = group["children"][2]
        assert failed["run_id"] is None
        assert failed["status"] == "failed"
        assert failed["error"] == "boom"
        assert group["seeds_finished"] == [1, 2]
        assert (group["n_requested"], group["n_finished"]) == (3, 2)

    def test_a_stopped_group_says_so_and_keeps_the_cut_seed_out_of_the_count(
        self, tmp_path: Path
    ) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.8})
        cut = _trials(tmp_path, "rep", {2: 0.1})[0]
        cut.status = "stopped"
        report = _replicates_report([*trials, cut], [1, 2, 3])
        report["stopped"] = True

        run_dir = write_run_group(
            kind="replicates",
            group_id="rep_g1",
            name="rep",
            config={},
            report=report,
            report_json=None,
            models_dir=tmp_path,
        )

        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert data["stopped"] is True
        assert data["status"] == "completed"
        assert data["group"]["stopped"] is True
        assert data["group"]["seeds_finished"] == [1]
        assert data["group"]["n_requested"] == 3
        # The cut seed is still a child of the group, listed as stopped.
        assert [c["status"] for c in data["group"]["children"]] == [
            "success",
            "stopped",
        ]

    def test_a_task_the_group_cannot_name_is_taken_from_a_child(
        self, tmp_path: Path
    ) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.8, 2: 0.9})
        child_json = Path(trials[0].run_dir) / "run.json"
        child = json.loads(child_json.read_text(encoding="utf-8"))
        child["task"] = "custom:toy"
        child["task_label"] = "Toy"
        child_json.write_text(json.dumps(child), encoding="utf-8")

        run_dir = write_run_group(
            kind="replicates",
            group_id="rep_g1",
            name="rep",
            config={},
            report=_replicates_report(trials, [1, 2]),
            report_json=None,
            models_dir=tmp_path,
        )

        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert data["task"] == "custom:toy"
        assert data["task_label"] == "Toy"
        # A researcher's task names its metrics itself: no ``test_`` prefix.
        assert data["metrics"]["r2"] == data["group"]["aggregates"]["r2"]["mean"]
        assert data["group"]["metric_keys"] == {"r2": "r2"}

    def test_detection_keeps_the_bare_metric_name(self, tmp_path: Path) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.8, 2: 0.9})
        run_dir = write_run_group(
            kind="replicates",
            group_id="rep_g1",
            name="rep",
            config={"task": "detection"},
            report=_replicates_report(trials, [1, 2]),
            report_json=None,
            models_dir=tmp_path,
        )

        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert data["group"]["metric_keys"] == {"r2": "r2"}
        assert "test_r2" not in data["metrics"]

    def test_an_unreadable_child_does_not_fail_the_group(self, tmp_path: Path) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.8, 2: 0.9})
        (Path(trials[1].run_dir) / "run.json").write_text("{not json", "utf-8")

        run_dir = write_run_group(
            kind="replicates",
            group_id="rep_g1",
            name="rep",
            config={},
            report=_replicates_report(trials, [1, 2]),
            report_json=None,
            models_dir=tmp_path,
        )

        assert (run_dir / "run.json").is_file()
        first = json.loads((Path(trials[0].run_dir) / "run.json").read_text("utf-8"))
        assert first["group_id"] == "rep_g1"

    def test_a_child_that_cannot_be_rewritten_does_not_fail_the_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Windows: an antivirus or indexer holding a child's run.json makes the
        # replace raise PermissionError. One seed staying loose must not leave
        # the later seeds untagged, the group unwritten and a .tmp behind.
        trials = _trials(tmp_path, "rep", {1: 0.8, 2: 0.9, 3: 0.85})
        held = Path(trials[1].run_dir)
        real_replace = Path.replace

        def replace(self: Path, target: Any) -> Path:
            if self.parent == held:
                raise PermissionError(13, "held by another process")
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", replace)

        run_dir = write_run_group(
            kind="replicates",
            group_id="rep_g1",
            name="rep",
            config={},
            report=_replicates_report(trials, [1, 2, 3]),
            report_json=None,
            models_dir=tmp_path,
        )

        tagged = {
            t.seed: json.loads((Path(t.run_dir) / "run.json").read_text("utf-8")).get(
                "group_id"
            )
            for t in trials
        }
        assert tagged == {1: "rep_g1", 2: None, 3: "rep_g1"}
        assert not list(held.glob("*.tmp"))
        # The held run.json is the trainer's, untouched.
        assert json.loads((held / "run.json").read_text("utf-8"))["metrics"] == {
            "total_epochs": 1,
            "r2": 0.5,
        }
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert data["group"]["seeds_finished"] == [1, 2, 3]
        assert data["group"]["aggregates"]["r2"]["n"] == 3

    def test_two_groups_of_the_same_name_do_not_overwrite_each_other(
        self, tmp_path: Path
    ) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.8, 2: 0.9})
        dirs = {
            write_run_group(
                kind="replicates",
                group_id=gid,
                name="rep",
                config={},
                report=_replicates_report(trials, [1, 2]),
                report_json=None,
                models_dir=tmp_path,
            )
            for gid in ("rep_a", "rep_b")
        }
        assert len(dirs) == 2
        assert all((d / "run.json").is_file() for d in dirs)


def _comparison_report(
    models_dir: Path, values: dict[str, dict[int, float]], *, not_run: list[str]
) -> dict[str, Any]:
    results = [
        VariantResult(
            label=label,
            overrides={"training.learning_rate": 0.01} if label != "base" else {},
            trials=_trials(models_dir, f"cmp_{label}", per_seed),
        )
        for label, per_seed in values.items()
    ]
    seeds = sorted({s for per_seed in values.values() for s in per_seed})
    report = build_report(results, seeds, "r2", not_run=not_run)
    report["stopped"] = bool(not_run)
    report["report_dir"] = "outputs/reports/cmp/1"
    round_trip: dict[str, Any] = json.loads(json.dumps(report))
    return round_trip


class TestReplicatedComparisonGroup:
    def test_variants_paired_tests_and_ranking_are_the_reports(
        self, tmp_path: Path
    ) -> None:
        report = _comparison_report(
            tmp_path,
            {
                "base": {1: 0.70, 2: 0.72, 3: 0.71, 4: 0.69, 5: 0.73, 6: 0.70},
                "lr": {1: 0.80, 2: 0.83, 3: 0.81, 4: 0.79, 5: 0.84, 6: 0.80},
            },
            not_run=[],
        )

        run_dir = write_run_group(
            kind="replicated_comparison",
            group_id="cmp_g1",
            name="cmp",
            config={"name": "cmp"},
            report=report,
            report_json=Path("outputs/reports/cmp/1/comparison_summary.json"),
            models_dir=tmp_path,
        )

        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        group = data["group"]
        assert data["block"] == "replicated_comparison"
        assert group["kind"] == "replicated_comparison"
        assert group["comparisons"] == report["comparisons"]
        pair = group["comparisons"][0]
        assert {"p_value", "significant", "effect_size", "effect_label"} <= set(pair)
        assert group["best_by_mean"] == report["best_by_mean"] == "lr"
        assert group["ranking_seeds"] == report["ranking_seeds"]
        assert group["not_run"] == []
        assert group["metric_direction"] == "higher"
        for label, variant in group["variants"].items():
            assert variant["aggregates"] == report["variants"][label]["aggregates"]
            assert variant["overrides"] == report["variants"][label]["overrides"]
            assert len(variant["children"]) == 6
        assert (group["n_requested"], group["n_finished"]) == (12, 12)
        # A comparison has no single headline number to flatten.
        assert "r2" not in data["metrics"]

    def test_every_seed_of_every_variant_is_tagged(self, tmp_path: Path) -> None:
        report = _comparison_report(
            tmp_path,
            {"base": {1: 0.7, 2: 0.72}, "lr": {1: 0.8, 2: 0.83}},
            not_run=[],
        )
        write_run_group(
            kind="replicated_comparison",
            group_id="cmp_g1",
            name="cmp",
            config={},
            report=report,
            report_json=None,
            models_dir=tmp_path,
        )

        tagged = [
            json.loads(p.read_text(encoding="utf-8")).get("group_id")
            for p in tmp_path.glob("cmp_*_s*/*/run.json")
        ]
        assert tagged == ["cmp_g1"] * 4

    def test_a_stopped_comparison_keeps_not_run_and_a_null_best(
        self, tmp_path: Path
    ) -> None:
        # The two variants that ran finished different seeds, so no pair exists
        # and nothing can be ranked: the group must say that, not pick one.
        report = _comparison_report(
            tmp_path,
            {"base": {1: 0.7, 2: 0.72}, "lr": {3: 0.8, 4: 0.83}},
            not_run=["wd"],
        )

        run_dir = write_run_group(
            kind="replicated_comparison",
            group_id="cmp_g1",
            name="cmp",
            config={},
            report=report,
            report_json=None,
            models_dir=tmp_path,
        )

        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        group = data["group"]
        assert data["stopped"] is True
        assert group["stopped"] is True
        assert group["not_run"] == ["wd"]
        assert group["best_by_mean"] is None
        assert group["comparisons"] == []
        assert group["ranking_seeds"] == []
        # The variant that never started still counts toward what was asked.
        assert (group["n_requested"], group["n_finished"]) == (12, 4)

    def test_a_value_json_cannot_carry_is_stored_as_null(self, tmp_path: Path) -> None:
        trials = _trials(tmp_path, "rep", {1: 0.8, 2: 0.9})
        report = _replicates_report(trials, [1, 2])
        report["aggregates"]["r2"]["ci95_low"] = float("nan")
        report["aggregates"]["r2"]["boot95_high"] = float("inf")

        run_dir = write_run_group(
            kind="replicates",
            group_id="rep_g1",
            name="rep",
            config={},
            report=report,
            report_json=None,
            models_dir=tmp_path,
        )

        text = (run_dir / "run.json").read_text(encoding="utf-8")
        agg = json.loads(text, parse_constant=_reject)["group"]["aggregates"]["r2"]
        assert agg["ci95_low"] is None
        assert agg["boot95_high"] is None


def _reject(constant: str) -> None:
    raise AssertionError(f"run.json carries {constant}, which is not JSON")
