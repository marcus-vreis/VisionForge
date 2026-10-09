"""A unit the stop cut inside a multi-unit job cannot be continued on its own.

A trainer leaves its `resume.pt` behind when the token cuts it, which is what
lets a stopped single run be continued. In a sweep, a comparison or a set of
replicates the same unit is just one cell of a table the job never finished:
continuing it would train that lone cell to the end and hand History a run
detached from its sweep, whose summary still lists the cell as `stopped`
(ADR-111). The orchestrator therefore drops the cut unit's resume state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.cancellation import CancellationToken
from visionforge.core.comparison import run_model_comparison
from visionforge.core.replicates import run_replicates
from visionforge.core.resume import (
    RESUME_FILENAME,
    ResumeState,
    discard_resume_state,
    save_resume_state,
)
from visionforge.core.sweep import run_sweep
from visionforge.core.task_runner import RunResult, give_cancel_token, run_dir_from
from visionforge.gui.api.routes import _resume_status

EPOCHS = 3


def _state(epoch: int) -> ResumeState:
    return ResumeState(
        epoch=epoch,
        model={},
        optimizer={},
        scheduler=None,
        scaler=None,
        best_metric=0.1,
        best_epoch=1,
        patience_counter=0,
        history=[],
    )


class _EchoConfig:
    @classmethod
    def model_validate(cls, d: dict[str, Any]) -> dict[str, Any]:
        return d


class _UnitRunner:
    """Each run leaves a resume file in its own directory; run ``cut_during`` is cut."""

    config_type = _EchoConfig
    _cancel_token: CancellationToken | None = None

    def __init__(self, root: Path, cut_during: int, *, report_dir: bool = True) -> None:
        self.root = root
        self.cut_during = cut_during
        self.report_dir = report_dir
        self.dirs: list[Path] = []
        self.calls = 0

    def run(self, cfg: dict[str, Any]) -> RunResult:
        index = self.calls
        self.calls += 1
        run_dir = self.root / f"unit{index}"
        run_dir.mkdir()
        save_resume_state(run_dir, _state(1))
        self.dirs.append(run_dir)
        cut = index == self.cut_during
        if cut and self._cancel_token is not None:
            self._cancel_token.cancel()
        return RunResult(
            metrics={"score": 0.5},
            status="success",
            training_time_s=0.0,
            stopped=cut,
            run_dir=run_dir if self.report_dir else None,
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return dict(result.metrics)

    def primary_metric(self) -> str:
        return "score"


def _runner(tmp_path: Path, cut_during: int, **kwargs: Any) -> _UnitRunner:
    runner = _UnitRunner(tmp_path, cut_during, **kwargs)
    give_cancel_token(runner, CancellationToken())
    return runner


def _base() -> dict[str, Any]:
    return {
        "name": "x",
        "model": {"name": "a"},
        "training": {"learning_rate": 0.01, "seed": 1},
    }


def _has_resume(run_dir: Path) -> bool:
    return (run_dir / RESUME_FILENAME).is_file()


class TestOrchestratorsDropTheCutUnitsResumeState:
    def test_sweep(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path, cut_during=1)

        run_sweep(
            runner,
            _base(),
            {"training.learning_rate": [0.1, 0.01, 0.001]},
            mode="grid",
            metric="score",
        )

        assert runner.calls == 2
        assert not _has_resume(runner.dirs[1])
        # Only the unit the stop cut loses it.
        assert _has_resume(runner.dirs[0])

    def test_comparison(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path, cut_during=1)

        run_model_comparison(runner, _base(), ["a", "b", "c"], "score")

        assert runner.calls == 2
        assert not _has_resume(runner.dirs[1])
        assert _has_resume(runner.dirs[0])

    def test_replicates(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path, cut_during=1)

        run_replicates(runner, _base(), [1, 2, 3], "score")

        assert runner.calls == 2
        assert not _has_resume(runner.dirs[1])
        assert _has_resume(runner.dirs[0])

    def test_a_unit_that_reports_no_directory_is_left_alone(
        self, tmp_path: Path
    ) -> None:
        # Custom tasks own their loop and write no resume state to drop.
        runner = _runner(tmp_path, cut_during=0, report_dir=False)

        trials = run_model_comparison(runner, _base(), ["a", "b"], "score")

        assert trials[0].status == "stopped"
        assert _has_resume(runner.dirs[0])

    def test_a_unit_that_finished_despite_the_stop_keeps_what_it_has(
        self, tmp_path: Path
    ) -> None:
        class _FinishedAnyway(_UnitRunner):
            def run(self, cfg: dict[str, Any]) -> RunResult:
                result = super().run(cfg)
                result.stopped = False  # its stop landed in the last epoch
                return result

        runner = _FinishedAnyway(tmp_path, cut_during=0)
        give_cancel_token(runner, CancellationToken())

        trials = run_model_comparison(runner, _base(), ["a", "b"], "score")

        assert trials[0].status == "success"
        assert _has_resume(runner.dirs[0])


class TestWhatHistoryOffers:
    def _run_json(self, run_dir: Path, name: str = "x") -> dict[str, Any]:
        data = {
            "experiment": name,
            "config": {"task": "regression", "training": {"epochs": EPOCHS}},
            "metrics": {"total_epochs": 1},
            "history": [],
            "artifacts": {},
        }
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
        return data

    def test_a_cut_unit_of_a_sweep_is_not_resumable(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path, cut_during=0)
        run_sweep(
            runner,
            _base(),
            {"training.learning_rate": [0.1, 0.01]},
            mode="grid",
            metric="score",
        )
        cut_dir = runner.dirs[0]
        data = self._run_json(cut_dir)

        assert _resume_status(cut_dir, data) == (False, EPOCHS)

    def test_a_stopped_single_run_still_is(self, tmp_path: Path) -> None:
        # No orchestrator touches a lone run: its resume file is what History
        # reads to offer "continue".
        run_dir = tmp_path / "single"
        run_dir.mkdir()
        save_resume_state(run_dir, _state(1))
        data = self._run_json(run_dir)

        assert _resume_status(run_dir, data) == (True, EPOCHS)


class TestRunDirFrom:
    def test_reads_the_directory_a_report_section_names(self, tmp_path: Path) -> None:
        assert run_dir_from({"run_dir": str(tmp_path)}) == tmp_path

    def test_a_section_without_one_has_no_directory(self) -> None:
        assert run_dir_from({}) is None
        assert run_dir_from({"run_dir": None}) is None


class TestDiscardResumeState:
    def test_removes_the_resume_file(self, tmp_path: Path) -> None:
        save_resume_state(tmp_path, _state(1))

        discard_resume_state(tmp_path)

        assert not _has_resume(tmp_path)

    def test_removes_the_ultralytics_state_too(self, tmp_path: Path) -> None:
        # Ultralytics resumes from its own weights/last.pt (ADR-093), which is
        # what _resume_status reads for it; best.pt stays the deliverable.
        weights = tmp_path / "weights"
        weights.mkdir()
        (weights / "last.pt").write_bytes(b"x")
        (weights / "best.pt").write_bytes(b"x")

        discard_resume_state(tmp_path)

        assert not (weights / "last.pt").exists()
        assert (weights / "best.pt").is_file()

    def test_a_directory_with_nothing_to_drop_is_fine(self, tmp_path: Path) -> None:
        discard_resume_state(tmp_path)
        discard_resume_state(tmp_path / "never_created")


class TestRealRegressionSweep:
    """The same thing with a real trainer: the stop lands after epoch 1 of 3."""

    @pytest.fixture
    def cut_after_first_epoch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from visionforge.blocks import regression_runner
        from visionforge.blocks.regression import RegressionBlock

        class _CutBlock(RegressionBlock):
            def run(self) -> None:
                token = self._cancel_token

                def press_stop(event: dict[str, Any]) -> None:
                    if event.get("event") == "epoch_end" and event.get("epoch") == 1:
                        assert token is not None
                        token.cancel()

                self._progress_callback = press_stop
                super().run()

        monkeypatch.setattr(regression_runner, "RegressionBlock", _CutBlock)

    def _config(self, tmp_path: Path) -> dict[str, Any]:
        from visionforge.utils.regression_config import RegressionConfig
        from visionforge.utils.selftest_data import build_regression_dataset

        base = build_regression_dataset(tmp_path / "ds")
        config = RegressionConfig.model_validate(
            {
                "name": "cut_reg",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {
                    "base_dir": str(base),
                    "target_columns": ["target"],
                    "image_size": 32,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": EPOCHS, "batch_size": 4, "seed": 0},
                "output": {
                    "models_dir": str(tmp_path / "models"),
                    "reports_dir": str(tmp_path / "reports"),
                    "graphics_dir": str(tmp_path / "graphics"),
                    "logs_dir": str(tmp_path / "logs"),
                },
                "device": {"kind": "cpu"},
            }
        )
        return config.model_dump(mode="json")

    def _run_dirs(self, tmp_path: Path) -> list[Path]:
        return sorted((tmp_path / "models").glob("cut_reg/*"))

    def test_the_cut_trial_cannot_be_continued_alone(
        self, tmp_path: Path, cut_after_first_epoch: None
    ) -> None:
        from visionforge.blocks.regression_runner import RegressionRunner

        runner = RegressionRunner()
        give_cancel_token(runner, CancellationToken())

        trials = run_sweep(
            runner,
            self._config(tmp_path),
            {"training.learning_rate": [0.001, 0.0001]},
            mode="grid",
            metric="r2",
        )

        assert [t.status for t in trials] == ["stopped"]
        (run_dir,) = self._run_dirs(tmp_path)
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert data["metrics"]["total_epochs"] == 1
        assert not _has_resume(run_dir)
        assert _resume_status(run_dir, data) == (False, EPOCHS)

    def test_the_same_cut_on_a_single_run_can_be_continued(
        self, tmp_path: Path, cut_after_first_epoch: None
    ) -> None:
        from visionforge.blocks.regression_runner import RegressionRunner
        from visionforge.utils.regression_config import RegressionConfig

        runner = RegressionRunner()
        give_cancel_token(runner, CancellationToken())

        result = runner.run(RegressionConfig.model_validate(self._config(tmp_path)))

        assert result.stopped
        (run_dir,) = self._run_dirs(tmp_path)
        assert result.run_dir == run_dir
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert _resume_status(run_dir, data) == (True, EPOCHS)


def _stop_in_second_fold(token: CancellationToken) -> Any:
    """Progress callback that presses stop after the second fold's first epoch."""

    def _callback(event: dict[str, Any]) -> None:
        if (
            event.get("event") == "epoch_end"
            and event.get("trial_index") == 1
            and event.get("epoch") == 1
        ):
            token.cancel()

    return _callback


def _output_dirs(tmp_path: Path) -> dict[str, str]:
    return {
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
        "graphics_dir": str(tmp_path / "graphics"),
        "logs_dir": str(tmp_path / "logs"),
    }


def _fold_dir(tmp_path: Path, name: str, fold: int) -> Path:
    (run_dir,) = (tmp_path / "models").glob(f"{name}_fold{fold}/*")
    return run_dir


class TestRealStandaloneKFold:
    """A fold the stop cut is not offered as a lone run to continue (ADR-111)."""

    def test_regression_cut_fold_is_not_resumable(self, tmp_path: Path) -> None:
        from visionforge.blocks.regression_cv import run_regression_cross_validation
        from visionforge.utils.regression_config import RegressionConfig
        from visionforge.utils.selftest_data import build_regression_dataset

        config = RegressionConfig.model_validate(
            {
                "name": "cut_reg_cv",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {
                    "base_dir": str(build_regression_dataset(tmp_path / "ds")),
                    "target_columns": ["target"],
                    "image_size": 32,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": EPOCHS, "batch_size": 4, "seed": 0},
                "output": _output_dirs(tmp_path),
                "device": {"kind": "cpu"},
            }
        )
        token = CancellationToken()

        report = run_regression_cross_validation(
            config,
            n_folds=3,
            shuffle=False,
            progress_callback=_stop_in_second_fold(token),
            cancel_token=token,
        )

        assert [f.status for f in report.folds] == ["success", "stopped"]
        self._assert_cut_fold_dropped(tmp_path, "cut_reg_cv")

    def test_segmentation_cut_fold_is_not_resumable(self, tmp_path: Path) -> None:
        from visionforge.blocks.segmentation_cv import (
            run_segmentation_cross_validation,
        )
        from visionforge.utils.segmentation_config import SegmentationConfig
        from visionforge.utils.selftest_data import build_segmentation_dataset

        config = SegmentationConfig.model_validate(
            {
                "name": "cut_seg_cv",
                "model": {"name": "unet", "num_classes": 3, "pretrained": False},
                "data": {
                    "base_dir": str(build_segmentation_dataset(tmp_path / "ds")),
                    "image_size": 64,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": EPOCHS, "batch_size": 2, "seed": 0},
                "output": _output_dirs(tmp_path),
                "device": {"kind": "cpu"},
            }
        )
        token = CancellationToken()

        report = run_segmentation_cross_validation(
            config,
            n_folds=3,
            shuffle=False,
            progress_callback=_stop_in_second_fold(token),
            cancel_token=token,
        )

        assert [f.status for f in report.folds] == ["success", "stopped"]
        self._assert_cut_fold_dropped(tmp_path, "cut_seg_cv")

    @staticmethod
    def _assert_cut_fold_dropped(tmp_path: Path, name: str) -> None:
        finished = _fold_dir(tmp_path, name, 0)
        cut = _fold_dir(tmp_path, name, 1)
        cut_data = json.loads((cut / "run.json").read_text(encoding="utf-8"))
        finished_data = json.loads((finished / "run.json").read_text(encoding="utf-8"))

        assert cut_data["metrics"]["total_epochs"] == 1
        assert not _has_resume(cut)
        assert _resume_status(cut, cut_data) == (False, EPOCHS)
        # The cut fold keeps its best checkpoint: only the way back in is shut.
        assert (cut / "best_model.pth").is_file()
        # A fold that ran to the end had no resume state to begin with, and
        # its artifacts are as the trainer left them.
        assert finished_data["metrics"]["total_epochs"] == EPOCHS
        assert (finished / "best_model.pth").is_file()
        assert _resume_status(finished, finished_data) == (False, EPOCHS)
