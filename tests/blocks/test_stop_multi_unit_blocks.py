"""K-fold, comparison and sweep blocks stop where the queue says they do (ADR-111).

Before this, a 3-fold run stopped during its first fold trained all three folds
to the end: the token reached neither the fold's trainer nor the loop over
folds. The K-fold tests here train for real on a tiny dataset and press stop
during the second fold, so what is checked is the stop landing mid-run, the
fold it cut, and an aggregate that counts only the folds that ran to the end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from visionforge.core.cancellation import STOPPED, STOPPED_NOTE, CancellationToken
from visionforge.core.task_runner import RunResult
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.selftest_data import (
    build_classification_dataset,
    build_regression_dataset,
    build_segmentation_dataset,
)


def _output(tmp_path: Path) -> dict[str, str]:
    return {
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
        "graphics_dir": str(tmp_path / "graphics"),
        "logs_dir": str(tmp_path / "logs"),
    }


def _stop_during_fold(token: CancellationToken, fold: int) -> Any:
    """Progress callback that presses stop once ``fold`` finished its first epoch."""
    events: list[dict[str, Any]] = []

    def _callback(event: dict[str, Any]) -> None:
        events.append(event)
        if event.get("event") == "epoch_end" and event.get("trial_index") == fold:
            token.cancel()

    _callback.events = events  # type: ignore[attr-defined]
    return _callback


def _classification_raw(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    base = build_classification_dataset(tmp_path / "ds")
    return {
        "name": "stop_multi",
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "seed": 0,
            "early_stopping_patience": 0,
        },
        "data": {
            "base_dir": str(base),
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {"image_size": 32},
        },
        "output": _output(tmp_path),
        "device": {"kind": "cpu"},
        **extra,
    }


class TestClassificationKFold:
    def _block(self, tmp_path: Path) -> Any:
        from visionforge.blocks.cross_validation import CrossValidationBlock

        config = ExperimentConfig.model_validate(
            _classification_raw(
                tmp_path,
                block="cross_validation",
                cross_validation={"n_folds": 3, "shuffle": False},
            )
        )
        block = CrossValidationBlock()
        block.setup(config)
        return block, config

    def test_stops_inside_the_second_fold(self, tmp_path: Path) -> None:
        block, config = self._block(tmp_path)
        token = CancellationToken()
        callback = _stop_during_fold(token, fold=1)
        block._progress_callback = callback
        block._cancel_token = token

        block.run()

        statuses = [r["status"] for r in block._fold_results]
        assert statuses == ["success", STOPPED]  # the third fold never started
        cut = block._fold_results[1]
        assert cut["error"] == STOPPED_NOTE
        # The fold that ran to the end is the whole aggregate, with its own n,
        # and one fold has no spread to report rather than a spread of 0.0.
        report = block.report()
        assert report["n_folds_ok"] == 1
        assert report["mean_accuracy"] == block._fold_results[0]["accuracy"]
        assert report["std_accuracy"] is None

        run_json = json.loads(
            (config.output.models_dir / "stop_multi_cv" / "run.json").read_text(
                encoding="utf-8"
            )
        )
        assert run_json["status"] == "completed"
        assert run_json["stopped"] is True
        aggregate = run_json["metrics"]["cv_aggregate"]
        assert aggregate["std_ddof"] == 1
        summary = json.loads(
            (config.output.reports_dir / "stop_multi" / "cv_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["aggregate"]["std_ddof"] == 1
        assert aggregate["n_folds"] == 3
        assert aggregate["n_folds_ok"] == 1
        assert aggregate["n_folds_stopped"] == 1
        assert aggregate["n_folds_failed"] == 0

        kinds = [e["event"] for e in callback.events]
        assert kinds[-1] == "end" and kinds.count("end") == 1
        # One trial_end per fold that started, the cut one included, never two.
        assert kinds.count("trial_start") == kinds.count("trial_end") == 2
        # The cut fold stopped after its first epoch rather than running both.
        fold_epochs = [
            e["epoch"]
            for e in callback.events
            if e["event"] == "epoch_end" and e["trial_index"] == 1
        ]
        assert fold_epochs == [1]

    def test_a_stop_in_the_last_epoch_of_a_fold_keeps_the_fold(
        self, tmp_path: Path
    ) -> None:
        """The fold finished: it counts, and the K-fold is not a failure.

        Marking it stopped from the token alone made a 2-fold run stopped here
        write a "failed" run.json with nothing to average.
        """
        block, config = self._block(tmp_path)
        token = CancellationToken()
        events: list[dict[str, Any]] = []

        def press_stop_in_the_last_epoch(event: dict[str, Any]) -> None:
            events.append(event)
            if event.get("event") == "epoch_end" and event.get("epoch") == 2:
                token.cancel()

        block._progress_callback = press_stop_in_the_last_epoch
        block._cancel_token = token

        block.run()

        assert [r["status"] for r in block._fold_results] == ["success"]
        run_json = json.loads(
            (config.output.models_dir / "stop_multi_cv" / "run.json").read_text(
                encoding="utf-8"
            )
        )
        assert run_json["status"] == "completed"
        aggregate = run_json["metrics"]["cv_aggregate"]
        assert (aggregate["n_folds_ok"], aggregate["n_folds_stopped"]) == (1, 0)
        assert block.report()["n_folds_ok"] == 1

    def test_stopped_before_any_fold_finished_is_not_a_failure(
        self, tmp_path: Path
    ) -> None:
        block, config = self._block(tmp_path)
        token = CancellationToken()
        token.cancel()
        block._cancel_token = token

        block.run()

        assert [r["status"] for r in block._fold_results] == [STOPPED]
        report = block.report()  # no exception: the stop was asked for
        assert report["stopped"] is True
        assert report["n_folds_ok"] == 0
        assert report["mean_accuracy"] is None
        run_json = json.loads(
            (config.output.models_dir / "stop_multi_cv" / "run.json").read_text(
                encoding="utf-8"
            )
        )
        assert run_json["status"] == "completed"
        assert run_json["metrics"]["cv_aggregate"]["n_folds_ok"] == 0
        assert run_json["metrics"]["test_accuracy"] is None

    def test_all_folds_failing_without_a_stop_still_raises(
        self, tmp_path: Path
    ) -> None:
        block, _ = self._block(tmp_path)
        block._fold_results = [{"status": "failed", "accuracy": None, "f1": None}]

        with pytest.raises(RuntimeError, match="all folds failed"):
            block.report()


class TestStandaloneKFold:
    def test_regression_stops_inside_the_second_fold(self, tmp_path: Path) -> None:
        from visionforge.blocks.regression_cv import run_regression_cross_validation
        from visionforge.utils.regression_config import RegressionConfig

        base = build_regression_dataset(tmp_path / "ds")
        config = RegressionConfig.model_validate(
            {
                "name": "stop_reg_cv",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {
                    "base_dir": str(base),
                    "target_columns": ["target"],
                    "image_size": 32,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": 2, "batch_size": 4, "seed": 0},
                "output": _output(tmp_path),
                "device": {"kind": "cpu"},
            }
        )
        token = CancellationToken()
        callback = _stop_during_fold(token, fold=1)

        report = run_regression_cross_validation(
            config,
            n_folds=3,
            shuffle=False,
            progress_callback=callback,
            cancel_token=token,
        )

        assert [f.status for f in report.folds] == ["success", STOPPED]
        assert report.aggregate["r2"]["mean"] == report.folds[0].metrics["r2"]
        # One finished fold: its n says so, and it has no spread to report.
        assert report.aggregate["r2"]["n"] == 1
        assert report.aggregate["r2"]["std"] is None
        # Older runs used ddof=0; the readers can tell them apart.
        assert report.aggregate["r2"]["std_ddof"] == 1
        kinds = [e["event"] for e in callback.events]
        assert kinds.count("trial_start") == kinds.count("trial_end") == 2

    def test_segmentation_stopped_up_front_runs_one_empty_fold(
        self, tmp_path: Path
    ) -> None:
        from visionforge.blocks.segmentation_cv import (
            run_segmentation_cross_validation,
        )
        from visionforge.utils.segmentation_config import SegmentationConfig

        base = build_segmentation_dataset(tmp_path / "ds")
        config = SegmentationConfig.model_validate(
            {
                "name": "stop_seg_cv",
                "model": {"name": "unet", "num_classes": 3, "pretrained": False},
                "data": {
                    "base_dir": str(base),
                    "image_size": 64,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": 2, "batch_size": 2, "seed": 0},
                "output": _output(tmp_path),
                "device": {"kind": "cpu"},
            }
        )
        token = CancellationToken()
        token.cancel()

        report = run_segmentation_cross_validation(
            config, n_folds=3, shuffle=False, cancel_token=token
        )

        assert [f.status for f in report.folds] == [STOPPED]
        assert report.aggregate == {}  # an empty fold is not averaged in


class _FakeRunner:
    """Records the token each architecture was handed; presses stop on the second.

    ``cut`` is what that second architecture reports: cut by the stop, or
    trained to its end anyway.
    """

    instances: list[_FakeRunner] = []
    cut = True

    def __init__(self) -> None:
        self._cancel_token: CancellationToken | None = None
        self.tokens_seen: list[CancellationToken | None] = []
        _FakeRunner.instances.append(self)

    def run(self, cfg: Any) -> RunResult:
        self.tokens_seen.append(self._cancel_token)
        stopped = False
        if len(self.tokens_seen) == 2 and self._cancel_token is not None:
            self._cancel_token.cancel()
            stopped = _FakeRunner.cut
        return RunResult(
            metrics={"accuracy": 0.9, "f1": 0.8, "auc_roc": 0.85},
            status="success",
            training_time_s=0.0,
            stopped=stopped,
        )


class TestClassificationModelComparison:
    def _run(self, tmp_path: Path, cut: bool) -> tuple[Any, CancellationToken]:
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        config = ExperimentConfig.model_validate(
            _classification_raw(
                tmp_path,
                block="model_comparison",
                model_comparison={
                    "model_names": ["resnet18", "resnet34", "resnet50"],
                    "metric": "accuracy",
                },
            )
        )
        token = CancellationToken()
        _FakeRunner.instances.clear()
        _FakeRunner.cut = cut
        block = ModelComparisonBlock()
        block.setup(config)
        block._cancel_token = token
        with patch(
            "visionforge.blocks.model_comparison.ClassificationRunner", _FakeRunner
        ):
            block.run()
        return block, token

    def test_an_architecture_that_finished_is_ranked(self, tmp_path: Path) -> None:
        block, _ = self._run(tmp_path, cut=False)

        assert [t["status"] for t in block._trials] == ["success", "success"]
        assert block.report()["stopped_count"] == 0

    def test_stops_between_architectures(self, tmp_path: Path) -> None:
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        _FakeRunner.cut = True
        config = ExperimentConfig.model_validate(
            _classification_raw(
                tmp_path,
                block="model_comparison",
                model_comparison={
                    "model_names": ["resnet18", "resnet34", "resnet50"],
                    "metric": "accuracy",
                },
            )
        )
        token = CancellationToken()
        _FakeRunner.instances.clear()
        block = ModelComparisonBlock()
        block.setup(config)
        block._cancel_token = token

        with patch(
            "visionforge.blocks.model_comparison.ClassificationRunner", _FakeRunner
        ):
            block.run()

        runner = _FakeRunner.instances[0]
        assert runner.tokens_seen == [token, token]  # resnet50 never started
        assert [(t["model_arch"], t["status"]) for t in block._trials] == [
            ("resnet18", "success"),
            ("resnet34", STOPPED),
        ]
        report = block.report()
        assert [t["model_arch"] for t in report["top_3"]] == ["resnet18"]
        assert report["stopped_count"] == 1
        assert report["failed_count"] == 0
        assert report["stopped"] is True


class TestClassificationGridSearch:
    def test_the_cut_trial_is_never_the_best(self, tmp_path: Path) -> None:
        from visionforge.blocks.grid_search import GridSearchBlock

        token = CancellationToken()
        calls: list[int] = []

        class _FakeBlock:
            _cancel_token: CancellationToken | None = None

            def setup(self, config: Any) -> None:
                self._config = config

            def run(self) -> None:
                calls.append(1)
                if len(calls) == 2:
                    token.cancel()

            def report(self) -> dict[str, Any]:
                # The cut trial looks better, which is exactly the trap.
                loss = 0.5 if len(calls) == 1 else 0.1
                return {
                    "train": {"best_val_loss": loss, "stopped": len(calls) == 2},
                    "eval": {"accuracy": 0.8, "f1": 0.7},
                }

        config = ExperimentConfig.model_validate(
            _classification_raw(
                tmp_path,
                block="grid_search",
                grid_search={
                    "hyperparameters": {"training.learning_rate": [0.1, 0.2, 0.3]}
                },
            )
        )
        block = GridSearchBlock()
        block.setup(config)
        block._cancel_token = token

        with patch("visionforge.blocks._search_utils.ClassificationBlock", _FakeBlock):
            block.run()

        assert [t["status"] for t in block._trials] == ["success", STOPPED]
        report = block.report()
        assert report["best_trial"]["trial_index"] == 0
        # A cut trial is told apart from a failed one (ADR-111).
        assert (report["stopped_count"], report["failed_count"]) == (1, 0)
        assert report["stopped"] is True

    def test_a_trial_that_finished_is_ranked(self, tmp_path: Path) -> None:
        from visionforge.blocks.grid_search import GridSearchBlock

        token = CancellationToken()

        class _FinishesAnyway:
            _cancel_token: CancellationToken | None = None

            def setup(self, config: Any) -> None:
                pass

            def run(self) -> None:
                token.cancel()  # pressed during the last epoch

            def report(self) -> dict[str, Any]:
                return {
                    "train": {"best_val_loss": 0.3, "stopped": False},
                    "eval": {"accuracy": 0.8, "f1": 0.7},
                }

        config = ExperimentConfig.model_validate(
            _classification_raw(
                tmp_path,
                block="grid_search",
                grid_search={"hyperparameters": {"training.learning_rate": [0.1, 0.2]}},
            )
        )
        block = GridSearchBlock()
        block.setup(config)
        block._cancel_token = token

        with patch(
            "visionforge.blocks._search_utils.ClassificationBlock", _FinishesAnyway
        ):
            block.run()

        assert [t["status"] for t in block._trials] == ["success"]

    @pytest.mark.parametrize("block_name", ["grid_search", "random_search"])
    def test_stopped_in_the_first_trial_is_reported_not_raised(
        self, tmp_path: Path, block_name: str
    ) -> None:
        from visionforge.blocks.grid_search import GridSearchBlock
        from visionforge.blocks.random_search import RandomSearchBlock

        token = CancellationToken()

        class _CutAtOnce:
            _cancel_token: CancellationToken | None = None

            def setup(self, config: Any) -> None:
                pass

            def run(self) -> None:
                token.cancel()

            def report(self) -> dict[str, Any]:
                return {"train": {"best_val_loss": None, "stopped": True}}

        extra: dict[str, Any] = (
            {"grid_search": {"hyperparameters": {"training.learning_rate": [0.1, 0.2]}}}
            if block_name == "grid_search"
            else {
                "random_search": {
                    "n_trials": 2,
                    "search_space": {
                        "training.learning_rate": {
                            "type": "uniform",
                            "low": 0.001,
                            "high": 0.01,
                        }
                    },
                }
            }
        )
        config = ExperimentConfig.model_validate(
            _classification_raw(tmp_path, block=block_name, **extra)
        )
        block = (
            GridSearchBlock() if block_name == "grid_search" else RandomSearchBlock()
        )
        block.setup(config)
        block._cancel_token = token

        with patch("visionforge.blocks._search_utils.ClassificationBlock", _CutAtOnce):
            block.run()

        report = block.report()
        assert report["best_trial"] is None
        assert (report["total_trials"], report["successful_trials"]) == (1, 0)
        assert (report["stopped_count"], report["failed_count"]) == (1, 0)
        assert report["stopped"] is True


class TestAnErrorDuringTheStopIsAFailure:
    """An OOM that lands after the stop is still an OOM: failed, with its error."""

    def test_grid_search_trial(self, tmp_path: Path) -> None:
        from visionforge.blocks.grid_search import GridSearchBlock

        token = CancellationToken()

        class _OutOfMemory:
            _cancel_token: CancellationToken | None = None

            def setup(self, config: Any) -> None:
                pass

            def run(self) -> None:
                token.cancel()
                raise RuntimeError("CUDA out of memory")

            def report(self) -> dict[str, Any]:  # pragma: no cover - never reached
                return {}

        config = ExperimentConfig.model_validate(
            _classification_raw(
                tmp_path,
                block="grid_search",
                grid_search={"hyperparameters": {"training.learning_rate": [0.1, 0.2]}},
            )
        )
        block = GridSearchBlock()
        block.setup(config)
        block._cancel_token = token

        with patch(
            "visionforge.blocks._search_utils.ClassificationBlock", _OutOfMemory
        ):
            block.run()

        assert [t["status"] for t in block._trials] == ["failed"]
        assert "out of memory" in block._trials[0]["error"]
        assert block.report()["failed_count"] == 1

    def test_model_comparison_architecture(self, tmp_path: Path) -> None:
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        token = CancellationToken()

        class _OutOfMemoryRunner:
            _cancel_token: CancellationToken | None = None

            def run(self, cfg: Any) -> RunResult:
                token.cancel()
                return RunResult(status="failed", error="CUDA out of memory")

        config = ExperimentConfig.model_validate(
            _classification_raw(
                tmp_path,
                block="model_comparison",
                model_comparison={"model_names": ["resnet18", "resnet34"]},
            )
        )
        block = ModelComparisonBlock()
        block.setup(config)
        block._cancel_token = token

        with patch(
            "visionforge.blocks.model_comparison.ClassificationRunner",
            _OutOfMemoryRunner,
        ):
            block.run()

        assert [(t["status"], t["error"]) for t in block._trials] == [
            ("failed", "CUDA out of memory")
        ]
        report = block.report()  # cut nothing, but left resnet34 unrun
        assert (report["failed_count"], report["stopped_count"]) == (1, 0)
        assert report["stopped"] is True

    def test_classification_runner(self) -> None:
        from visionforge.blocks import classification_runner as mod

        token = CancellationToken()

        class _OutOfMemory:
            _cancel_token: CancellationToken | None = None

            def setup(self, cfg: Any) -> None:
                pass

            def run(self) -> None:
                token.cancel()
                raise RuntimeError("CUDA out of memory")

        runner = mod.ClassificationRunner()
        runner._cancel_token = token
        with patch.object(mod, "ClassificationBlock", _OutOfMemory):
            result = runner.run(object())

        assert (result.status, result.stopped) == ("failed", False)
        assert "out of memory" in result.error

    def test_a_failed_run_is_never_relabelled(self) -> None:
        from visionforge.core.replicates import run_replicates

        token = CancellationToken()

        class _Echo:
            @staticmethod
            def model_validate(d: dict[str, Any]) -> dict[str, Any]:
                return d

        class _Runner:
            config_type: Any = _Echo
            _cancel_token: CancellationToken | None = None

            def run(self, cfg: Any) -> RunResult:
                token.cancel()
                # Even a runner that says "stopped" alongside a failure.
                return RunResult(status="failed", error="boom", stopped=True)

            def metrics(self, result: RunResult) -> dict[str, float]:
                return dict(result.metrics)

            def primary_metric(self) -> str:
                return "m"

        runner = _Runner()
        runner._cancel_token = token

        trials = run_replicates(runner, {"name": "x", "training": {}}, [1, 2], "m")

        assert [(t.status, t.error) for t in trials] == [("failed", "boom")]

    def test_kfold_evaluation_error_after_a_cut(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from visionforge.blocks.cross_validation import CrossValidationBlock

        config = ExperimentConfig.model_validate(
            _classification_raw(
                tmp_path,
                block="cross_validation",
                cross_validation={"n_folds": 3, "shuffle": False},
            )
        )
        token = CancellationToken()
        block = CrossValidationBlock()
        block.setup(config)
        block._cancel_token = token

        cut = MagicMock(stopped=True, total_epochs=1, best_val_loss=0.4)

        def fit(*args: Any, **kwargs: Any) -> Any:
            token.cancel()
            return cut

        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as trainer_cls,
            patch("visionforge.blocks.cross_validation.Evaluator") as evaluator_cls,
            patch("visionforge.blocks.cross_validation.torch.load", return_value={}),
        ):
            trainer_cls.return_value.fit.side_effect = fit
            evaluator_cls.return_value.evaluate.side_effect = RuntimeError(
                "CUDA out of memory"
            )
            block.run()

        assert [r["status"] for r in block._fold_results] == ["failed"]
        assert "out of memory" in block._fold_results[0]["error"]


class TestRunnersHandTheTokenToTheirBlock:
    """The runner is how a stop reaches the training inside a sweep or a set."""

    @pytest.mark.parametrize(
        ("module", "runner_name", "block_name"),
        [
            ("classification_runner", "ClassificationRunner", "ClassificationBlock"),
            ("regression_runner", "RegressionRunner", "RegressionBlock"),
            ("segmentation_runner", "SegmentationRunner", "SegmentationBlock"),
            ("detection_runner", "DetectionRunner", "DetectionBlock"),
            ("anomaly_runner", "AnomalyRunner", "AnomalyBlock"),
        ],
    )
    def test_builtin_runner(
        self, module: str, runner_name: str, block_name: str
    ) -> None:
        import importlib

        mod = importlib.import_module(f"visionforge.blocks.{module}")
        seen: list[CancellationToken | None] = []

        class _FakeBlock:
            _cancel_token: CancellationToken | None = None

            def setup(self, cfg: Any) -> None:
                # The standalone blocks reset their slot here, so the runner
                # has to assign the token after setup, not before.
                self._cancel_token = None

            def run(self) -> None:
                seen.append(self._cancel_token)

            def report(self) -> dict[str, Any]:
                return {}

        runner = getattr(mod, runner_name)()
        token = CancellationToken()
        runner._cancel_token = token

        with patch.object(mod, block_name, _FakeBlock):
            result = runner.run(object())

        assert result.status == "success"
        assert result.stopped is False
        assert seen == [token]

    @pytest.mark.parametrize(
        ("module", "runner_name", "block_name", "section"),
        [
            (
                "classification_runner",
                "ClassificationRunner",
                "ClassificationBlock",
                "train",
            ),
            ("regression_runner", "RegressionRunner", "RegressionBlock", "train"),
            ("segmentation_runner", "SegmentationRunner", "SegmentationBlock", "train"),
            ("detection_runner", "DetectionRunner", "DetectionBlock", "detection"),
            ("anomaly_runner", "AnomalyRunner", "AnomalyBlock", "train"),
        ],
    )
    def test_a_cut_training_reaches_the_orchestrator(
        self, module: str, runner_name: str, block_name: str, section: str
    ) -> None:
        import importlib

        mod = importlib.import_module(f"visionforge.blocks.{module}")

        class _CutBlock:
            _cancel_token: CancellationToken | None = None

            def setup(self, cfg: Any) -> None:
                pass

            def run(self) -> None:
                pass

            def report(self) -> dict[str, Any]:
                return {section: {"stopped": True}}

        with patch.object(mod, block_name, _CutBlock):
            result = getattr(mod, runner_name)().run(object())

        assert result.stopped is True

    def test_custom_task_runner(self) -> None:
        from types import SimpleNamespace

        from visionforge.tasks import runner as runner_mod

        captured: dict[str, Any] = {}

        class _FakeEngine:
            def __init__(self, info: Any, cfg: Any) -> None:
                pass

            def run(
                self, progress_callback: Any = None, cancel_token: Any = None
            ) -> Any:
                captured["token"] = cancel_token
                return SimpleNamespace(metrics={"mae": 1.0}, stopped=True)

        info = SimpleNamespace(
            key="toy",
            spec_cls=SimpleNamespace(Config=dict),
            metrics={"mae": "lower"},
            primary_metric="mae",
        )
        runner = runner_mod.CustomTaskRunner(info)  # type: ignore[arg-type]
        token = CancellationToken()
        runner._cancel_token = token

        with patch.object(runner_mod, "GenericTaskEngine", _FakeEngine):
            result = runner.run(object())

        assert result.status == "success"
        assert captured["token"] is token
        assert result.stopped is True  # the engine's verdict, passed through
