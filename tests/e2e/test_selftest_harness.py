"""The self-test harness, tested (ADR-060).

Two layers:

* fast, always-on: the synthetic dataset builders produce the exact layouts
  the tasks consume, and the case table / report formatter behave — no
  training, milliseconds.
* ``@pytest.mark.slow``: a real end-to-end run through a live server for the
  cheapest task pair. Deselected by default (see pyproject addopts) so the
  pre-commit suite stays fast; run it with ``pytest -m slow``, or exercise
  every task with ``visionforge selftest``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from visionforge.utils.selftest import (
    STRATEGIES,
    TASKS,
    SelfTestOutcome,
    build_cases,
    format_report,
    run_selftest,
)
from visionforge.utils.selftest_data import (
    build_anomaly_dataset,
    build_classification_dataset,
    build_detection_dataset,
    build_regression_dataset,
    build_segmentation_dataset,
)


class TestSyntheticDatasets:
    """Each builder must emit the layout its task's DataModule expects."""

    def test_classification_imagefolder_layout(self, tmp_path: Path) -> None:
        base = build_classification_dataset(tmp_path / "cls", per_class=3)
        for split in ("train", "val", "test"):
            for cls in ("class_a", "class_b"):
                images = list((base / split / cls).glob("*.png"))
                assert len(images) == 3, f"{split}/{cls}"

    def test_regression_manifest_matches_images(self, tmp_path: Path) -> None:
        base = build_regression_dataset(tmp_path / "reg", rows=5)
        with (base / "train.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        assert set(rows[0]) == {"image", "target"}
        for row in rows:
            assert (base / "images" / row["image"]).is_file()
            assert 0.0 <= float(row["target"]) <= 1.0

    def test_segmentation_pairs_are_aligned(self, tmp_path: Path) -> None:
        base = build_segmentation_dataset(tmp_path / "seg", pairs=4)
        images = sorted((base / "train" / "images").glob("*.png"))
        masks = sorted((base / "train" / "masks").glob("*.png"))
        assert len(images) == 4
        assert [p.name for p in images] == [p.name for p in masks]

    def test_anomaly_train_is_normal_only(self, tmp_path: Path) -> None:
        base = build_anomaly_dataset(tmp_path / "anom", normals=5)
        assert [d.name for d in (base / "train").iterdir()] == ["good"]
        assert len(list((base / "train" / "good").glob("*.png"))) == 5
        # test carries both classes so AUROC is computable
        assert {d.name for d in (base / "test").iterdir()} == {"good", "defect"}

    def test_detection_labels_pair_with_images(self, tmp_path: Path) -> None:
        base = build_detection_dataset(tmp_path / "det", per_split=3)
        images = sorted((base / "images" / "train").glob("*.jpg"))
        labels = sorted((base / "labels" / "train").glob("*.txt"))
        assert len(images) == 3
        assert [p.stem for p in images] == [p.stem for p in labels]
        # one YOLO line: class cx cy w h, all normalized
        parts = labels[0].read_text(encoding="utf-8").split()
        assert len(parts) == 5
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


class TestCaseTable:
    def test_every_task_and_strategy_is_reachable(self, tmp_path: Path) -> None:
        cases = build_cases(tmp_path, TASKS, STRATEGIES)
        covered = {(c.task, c.strategy) for c in cases}
        for task in TASKS:
            assert any(t == task for t, _ in covered), f"{task} has no case"
        for strategy in STRATEGIES:
            assert any(s == strategy for _, s in covered), f"{strategy} never used"

    def test_cv_only_where_the_endpoint_exists(self, tmp_path: Path) -> None:
        cases = build_cases(tmp_path, TASKS, ("cv",))
        cv_tasks = {c.task for c in cases}
        assert cv_tasks == {"classification", "regression", "segmentation"}

    def test_every_task_can_be_compared(self, tmp_path: Path) -> None:
        """The replicated comparison (ADR-061) is the newest endpoint and the
        one whose bugs were only visible end to end — every task must be
        covered, or the harness re-opens the gap it exists to close."""
        cases = build_cases(tmp_path, TASKS, ("comparison",))
        assert {c.task for c in cases} == set(TASKS)
        for case in cases:
            assert case.endpoint.endswith("/replicated-comparison")
            assert len(case.payload["variants"]) >= 2
            # Same seed list for every variant is what makes the test paired.
            assert len(case.payload["seeds"]) >= 2
            assert case.multi_trial is True

    def test_strategy_filter_narrows_the_run(self, tmp_path: Path) -> None:
        cases = build_cases(tmp_path, ("custom",), ("simple",))
        assert len(cases) == 1
        assert cases[0].endpoint == "/api/custom/example_counting/run"

    def test_custom_cases_are_skipped_when_no_task_is_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh install has no researcher-defined task, and that is the
        normal state — not a broken install.

        Emitting the cases anyway made `visionforge selftest` report a failure
        on a clean `pip install` and inside the Docker image, where user_tasks/
        is a mount point rather than the repository's example.
        """
        monkeypatch.setattr(
            "visionforge.tasks.registry.load_user_tasks", lambda *a, **k: []
        )
        assert build_cases(tmp_path, ("custom",), STRATEGIES) == []

    def test_multi_trial_flag_marks_the_streaming_contract(
        self, tmp_path: Path
    ) -> None:
        cases = {
            (c.task, c.strategy): c for c in build_cases(tmp_path, TASKS, STRATEGIES)
        }
        assert cases[("regression", "simple")].multi_trial is False
        for strategy in ("cv", "sweep", "replicates"):
            assert cases[("regression", strategy)].multi_trial is True


class TestReportFormatting:
    def test_table_lists_every_case_and_the_verdict(self) -> None:
        text = format_report(
            [
                SelfTestOutcome("classification", "simple", "passed", 2.0, "acc=0.9"),
                SelfTestOutcome("regression", "sweep", "failed", 1.0, "boom"),
            ]
        )
        assert "classification/simple" in text
        assert "PASS" in text and "FAIL" in text
        assert "1/2 cases passed" in text
        assert "failed: regression/sweep" in text

    def test_empty_selection_is_stated_not_crashed(self) -> None:
        assert "No self-test cases" in format_report([])


@pytest.mark.slow
class TestLiveEndToEnd:
    """Real server, real training — the contract the GUI depends on."""

    def test_custom_task_trains_through_the_api(self, tmp_path: Path) -> None:
        outcomes = run_selftest(tmp_path, tasks=("custom",), strategies=("simple",))
        assert len(outcomes) == 1
        assert outcomes[0].status == "passed", outcomes[0].detail
        # the live monitor must have received per-epoch events
        assert outcomes[0].events.get("epoch_end", 0) >= 1

    def test_multi_trial_streams_trial_events(self, tmp_path: Path) -> None:
        outcomes = run_selftest(tmp_path, tasks=("custom",), strategies=("replicates",))
        assert outcomes[0].status == "passed", outcomes[0].detail
        assert outcomes[0].events.get("trial_start", 0) == 2
        assert outcomes[0].events.get("trial_end", 0) == 2
