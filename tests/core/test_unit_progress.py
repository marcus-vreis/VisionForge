"""The live stream of a model comparison: one unit per architecture (ADR-118).

``UnitProgress`` announces each unit, forwards its trainer's epochs with the unit's
index, closes it exactly once and ends the job once. These tests drive it directly,
through ``run_model_comparison`` with a fake runner, through every task runner's
``_progress_callback`` slot, and through the classification ``ModelComparisonBlock``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from visionforge.core.comparison import run_model_comparison
from visionforge.core.task_runner import RunResult, UnitProgress


class _EchoConfig:
    @classmethod
    def model_validate(cls, d: dict[str, Any]) -> dict[str, Any]:
        return d


class _StreamingRunner:
    """Each run reports a start, two epochs and an end, then succeeds.

    ``fail`` names the architectures whose run raises before its trainer says
    anything, the way a model that cannot be built dies.
    """

    config_type = _EchoConfig
    _cancel_token = None
    _progress_callback: Any = None

    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        self.fail = fail
        self.callbacks_seen: list[Any] = []

    def run(self, cfg: dict[str, Any]) -> RunResult:
        self.callbacks_seen.append(self._progress_callback)
        # A dict from run_model_comparison, an ExperimentConfig from the block.
        name = cfg["model"]["name"] if isinstance(cfg, dict) else cfg.model.name
        if name in self.fail:
            raise RuntimeError("cannot build")
        emit = self._progress_callback
        if emit is not None:
            emit({"event": "start", "total_epochs": 2})
            emit(
                {"event": "epoch_end", "epoch": 1, "total_epochs": 2, "train_loss": 0.9}
            )
            emit(
                {"event": "epoch_end", "epoch": 2, "total_epochs": 2, "train_loss": 0.5}
            )
            emit({"event": "end", "total_epochs": 2})
        return RunResult(metrics={"score": 0.5}, status="success")

    def metrics(self, result: RunResult) -> dict[str, float]:
        return result.metrics

    def primary_metric(self) -> str:
        return "score"


def _pairs(events: list[dict[str, Any]]) -> list[tuple[str, int | None]]:
    return [(e["event"], e.get("trial_index")) for e in events]


class TestUnitProgress:
    def test_without_a_callback_nothing_is_sent_or_wired(self) -> None:
        runner = _StreamingRunner()
        progress = UnitProgress(None, 2)

        progress.start(runner, 0, {"model.name": "a"})
        progress.finish(runner, 0, "success")
        progress.end(1)

        assert runner._progress_callback is None

    def test_start_announces_the_unit_and_wires_the_runner(self) -> None:
        events: list[dict[str, Any]] = []
        runner = _StreamingRunner()

        UnitProgress(events.append, 3).start(runner, 1, {"model.name": "b"})

        assert events == [
            {
                "event": "trial_start",
                "trial_index": 1,
                "total_trials": 3,
                "overrides": {"model.name": "b"},
            }
        ]
        assert callable(runner._progress_callback)

    def test_the_trainers_epochs_carry_the_unit_and_its_end_is_held(self) -> None:
        events: list[dict[str, Any]] = []
        runner = _StreamingRunner()
        UnitProgress(events.append, 2).start(runner, 0, {"model.name": "a"})

        runner._progress_callback({"event": "start", "total_epochs": 5})
        runner._progress_callback({"event": "epoch_end", "epoch": 1, "total_epochs": 5})
        runner._progress_callback({"event": "end", "total_epochs": 5})

        # The inner start is dropped and the inner end waits for the outcome.
        assert events[1:] == [
            {
                "event": "epoch_end",
                "epoch": 1,
                "total_epochs": 5,
                "trial_index": 0,
                "total_trials": 2,
            },
        ]

    def test_the_unit_closes_once_with_its_outcome_and_epochs(self) -> None:
        events: list[dict[str, Any]] = []
        runner = _StreamingRunner()
        progress = UnitProgress(events.append, 2)
        progress.start(runner, 0, {"model.name": "a"})
        runner._progress_callback({"event": "end", "total_epochs": 2})

        progress.finish(runner, 0, "success")
        progress.finish(runner, 0, "success")  # a second close sends nothing

        ends = [e for e in events if e["event"] == "trial_end"]
        assert ends == [
            {
                "event": "trial_end",
                "trial_index": 0,
                "total_trials": 2,
                "status": "success",
                "total_epochs": 2,
            }
        ]
        assert runner._progress_callback is None

    def test_a_unit_that_died_first_is_closed_with_its_status(self) -> None:
        events: list[dict[str, Any]] = []
        runner = _StreamingRunner()
        progress = UnitProgress(events.append, 2)
        progress.start(runner, 0, {"model.name": "a"})

        progress.finish(runner, 0, "failed")

        assert events[-1] == {
            "event": "trial_end",
            "trial_index": 0,
            "total_trials": 2,
            "status": "failed",
        }

    def test_end_is_the_single_terminal_event(self) -> None:
        events: list[dict[str, Any]] = []

        UnitProgress(events.append, 3).end(2)

        assert events == [{"event": "end", "total_epochs": 0, "total_trials": 2}]


class TestComparisonStream:
    def test_each_architecture_streams_as_one_unit(self) -> None:
        events: list[dict[str, Any]] = []

        run_model_comparison(
            _StreamingRunner(),
            {"model": {}},
            ["a", "b"],
            "score",
            progress_callback=events.append,
        )

        assert _pairs(events) == [
            ("trial_start", 0),
            ("epoch_end", 0),
            ("epoch_end", 0),
            ("trial_end", 0),
            ("trial_start", 1),
            ("epoch_end", 1),
            ("epoch_end", 1),
            ("trial_end", 1),
            ("end", None),
        ]
        assert events[0]["overrides"] == {"model.name": "a"}
        assert events[4]["overrides"] == {"model.name": "b"}
        assert events[-1]["total_trials"] == 2

    def test_an_architecture_that_cannot_be_built_is_still_closed(self) -> None:
        events: list[dict[str, Any]] = []

        trials = run_model_comparison(
            _StreamingRunner(fail=("a",)),
            {"model": {}},
            ["a", "b"],
            "score",
            progress_callback=events.append,
        )

        assert _pairs(events)[:3] == [
            ("trial_start", 0),
            ("trial_end", 0),
            ("trial_start", 1),
        ]
        assert events[1]["status"] == "failed"
        assert [t.status for t in trials] == ["success", "failed"]

    def test_without_a_callback_the_comparison_is_unchanged(self) -> None:
        runner = _StreamingRunner()

        trials = run_model_comparison(runner, {"model": {}}, ["a", "b"], "score")

        assert [t.model_arch for t in trials] == ["a", "b"]
        assert runner.callbacks_seen == [None, None]


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
def test_every_runner_hands_its_callback_to_the_block(
    monkeypatch: pytest.MonkeyPatch, module: str, runner_name: str, block_name: str
) -> None:
    mod = importlib.import_module(f"visionforge.blocks.{module}")
    built: list[Any] = []

    class _FakeBlock:
        _progress_callback: Any = None
        _cancel_token: Any = None

        def __init__(self) -> None:
            built.append(self)

        def setup(self, cfg: Any) -> None:
            pass

        def run(self) -> None:
            pass

        def report(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(mod, block_name, _FakeBlock)
    runner = getattr(mod, runner_name)()

    def sink(event: dict[str, Any]) -> None:
        pass

    runner._progress_callback = sink
    runner.run(MagicMock())

    assert built and built[0]._progress_callback is sink


def test_the_classification_comparison_block_streams_its_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import visionforge.blocks.model_comparison as mc_mod
    from visionforge.blocks.model_comparison import ModelComparisonBlock
    from visionforge.utils.config import ExperimentConfig

    monkeypatch.setattr(mc_mod, "ClassificationRunner", _StreamingRunner)
    (tmp_path / "data").mkdir()
    config = ExperimentConfig.model_validate(
        {
            "name": "mc",
            "task": "binary",
            "block": "model_comparison",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "data": {"base_dir": str(tmp_path / "data")},
            "output": {
                "models_dir": str(tmp_path / "models"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "model_comparison": {
                "model_names": ["resnet18", "resnet34"],
                "metric": "f1",
            },
        }
    )
    events: list[dict[str, Any]] = []
    block = ModelComparisonBlock()
    block.setup(config)
    block._progress_callback = events.append

    block.run()

    assert [e["event"] for e in events] == [
        "trial_start",
        "epoch_end",
        "epoch_end",
        "trial_end",
        "trial_start",
        "epoch_end",
        "epoch_end",
        "trial_end",
        "end",
    ]
    assert events[4]["overrides"] == {"model.name": "resnet34"}


def test_a_sweep_over_architectures_streams_each_trials_epochs() -> None:
    # Comparing architectures in the GUI is a one-axis grid over model.name
    # (ADR-059): its trials now carry their epochs, not only their start and end.
    from visionforge.core.sweep import run_sweep

    events: list[dict[str, Any]] = []

    run_sweep(
        _StreamingRunner(),
        {"model": {"name": "x"}},
        {"model.name": ["a", "b"]},
        mode="grid",
        metric="score",
        progress_callback=events.append,
    )

    assert _pairs(events) == [
        ("trial_start", 0),
        ("epoch_end", 0),
        ("epoch_end", 0),
        ("trial_end", 0),
        ("trial_start", 1),
        ("epoch_end", 1),
        ("epoch_end", 1),
        ("trial_end", 1),
    ]
    assert events[3]["status"] == "success"
    assert events[3]["total_epochs"] == 2
