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
        # The fold that ran to the end is the whole aggregate, with its own n.
        report = block.report()
        assert report["mean_accuracy"] == block._fold_results[0]["accuracy"]
        assert report["std_accuracy"] == 0.0

        run_json = json.loads(
            (config.output.models_dir / "stop_multi_cv" / "run.json").read_text(
                encoding="utf-8"
            )
        )
        assert run_json["status"] == "completed"
        aggregate = run_json["metrics"]["cv_aggregate"]
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

    def test_stopped_before_any_fold_finished_reports_nothing(
        self, tmp_path: Path
    ) -> None:
        block, config = self._block(tmp_path)
        token = CancellationToken()
        token.cancel()
        block._cancel_token = token

        block.run()

        assert [r["status"] for r in block._fold_results] == [STOPPED]
        with pytest.raises(RuntimeError, match="parada antes de concluir"):
            block.report()
        run_json = json.loads(
            (config.output.models_dir / "stop_multi_cv" / "run.json").read_text(
                encoding="utf-8"
            )
        )
        assert run_json["metrics"]["cv_aggregate"]["n_folds_ok"] == 0
        assert run_json["metrics"]["test_accuracy"] is None


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
    """Records the token each architecture was handed; presses stop on the second."""

    instances: list[_FakeRunner] = []

    def __init__(self) -> None:
        self._cancel_token: CancellationToken | None = None
        self.tokens_seen: list[CancellationToken | None] = []
        _FakeRunner.instances.append(self)

    def run(self, cfg: Any) -> RunResult:
        self.tokens_seen.append(self._cancel_token)
        if len(self.tokens_seen) == 2 and self._cancel_token is not None:
            self._cancel_token.cancel()
        return RunResult(
            metrics={"accuracy": 0.9, "f1": 0.8, "auc_roc": 0.85},
            status="success",
            training_time_s=0.0,
        )


class TestClassificationModelComparison:
    def test_stops_between_architectures(self, tmp_path: Path) -> None:
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
                    "train": {"best_val_loss": loss},
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
        assert block.report()["best_trial"]["trial_index"] == 0


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
        assert seen == [token]

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
                return SimpleNamespace(metrics={"mae": 1.0})

        info = SimpleNamespace(
            key="toy", spec_cls=SimpleNamespace(Config=dict), primary_metric="mae"
        )
        runner = runner_mod.CustomTaskRunner(info)  # type: ignore[arg-type]
        token = CancellationToken()
        runner._cancel_token = token

        with patch.object(runner_mod, "GenericTaskEngine", _FakeEngine):
            result = runner.run(object())

        assert result.status == "success"
        assert captured["token"] is token
