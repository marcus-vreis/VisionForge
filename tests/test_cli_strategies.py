"""``visionforge run`` with a strategy envelope (ADR-184, issue #199).

The orchestrators run for real over a fake runner that trains nothing, so the
tests check the dispatch and what lands on disk without paying for training;
one test trains a tiny regression model on CPU end to end.
"""

from __future__ import annotations

import contextvars
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import visionforge.__main__ as main_mod
import visionforge.cli_strategies as cli
from visionforge.core.task_runner import RunResult
from visionforge.utils.messages import message, using_lang
from visionforge.utils.regression_config import RegressionConfig

R2 = "r2"
LEARNING_RATE = "training.learning_rate"


class FakeRunner:
    """Trains nothing; its metric depends on the seed and the learning rate."""

    config_type = RegressionConfig

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def run(self, cfg: Any) -> RunResult:
        seed = int(cfg.training.seed)
        lr = float(cfg.training.learning_rate)
        self.calls.append((seed, lr))
        value = 0.5 + 0.01 * seed + lr
        return RunResult(
            metrics={R2: value},
            selection_metrics={R2: value},
            status="success",
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return dict(result.metrics)

    def primary_metric(self) -> str:
        return R2


def _task_config(tmp_path: Path, name: str = "reg") -> dict[str, Any]:
    (tmp_path / "data").mkdir(exist_ok=True)
    return {
        "name": name,
        "task": "regression",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(tmp_path / "data"), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001, "seed": 0},
        "output": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
            "graphics_dir": str(tmp_path / "graphics"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "device": {"kind": "cpu"},
    }


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _run_main(monkeypatch: pytest.MonkeyPatch, config_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["visionforge", "run", str(config_path)])
    # A copy of the context keeps the language the command binds from leaking.
    contextvars.copy_context().run(main_mod.main)


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    runner = FakeRunner()
    monkeypatch.setattr(cli, "runner_for", lambda task: runner)
    return runner


def _group_json(models_dir: Path, folder: str) -> dict[str, Any]:
    paths = list((models_dir / folder).glob("*/run.json"))
    assert len(paths) == 1
    data: dict[str, Any] = json.loads(paths[0].read_text(encoding="utf-8"))
    return data


class TestReplicates:
    def test_seeds_run_through_run_replicates_and_the_group_is_written(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_runner: FakeRunner,
    ) -> None:
        import visionforge.core.replicates as replicates_mod

        real = replicates_mod.run_replicates
        seen: list[list[int]] = []

        def spy(runner: Any, config: dict[str, Any], seeds: list[int], *a: Any) -> Any:
            seen.append(list(seeds))
            return real(runner, config, seeds, *a)

        monkeypatch.setattr(replicates_mod, "run_replicates", spy)
        path = _write(
            tmp_path,
            {
                "strategy": {"kind": "replicates", "seeds": [0, 1]},
                "config": _task_config(tmp_path),
            },
        )

        _run_main(monkeypatch, path)

        assert seen == [[0, 1]]
        assert [seed for seed, _ in fake_runner.calls] == [0, 1]
        group = _group_json(tmp_path / "models", "reg_replicates")
        assert group["group"]["kind"] == "replicates"
        assert group["group"]["seeds"] == [0, 1]
        report_dir = Path(group["group"]["report_path"]).parent
        assert (report_dir / "replicates_table.tex").is_file()

    def test_the_report_carries_the_orchestrator_aggregate(
        self, tmp_path: Path, fake_runner: FakeRunner
    ) -> None:
        report = cli.run_strategy(
            {"kind": "replicates", "seeds": [3, 5], "metric": R2},
            _task_config(tmp_path),
        )
        assert report["seeds"] == [3, 5]
        assert report["headline"]["mean"] == pytest.approx(0.5 + 0.04 + 0.001)

    def test_seeds_default_to_consecutive_from_the_config_seed(
        self, tmp_path: Path, fake_runner: FakeRunner
    ) -> None:
        config = _task_config(tmp_path)
        config["training"]["seed"] = 7
        cli.run_strategy({"kind": "replicates", "n_replicates": 3}, config)
        assert [seed for seed, _ in fake_runner.calls] == [7, 8, 9]

    def test_duplicate_seeds_are_refused_before_training(
        self, tmp_path: Path, fake_runner: FakeRunner
    ) -> None:
        strategy = {"kind": "replicates", "seeds": [1, 1]}
        config = _task_config(tmp_path)
        with pytest.raises(ValueError, match="seeds"):
            cli.run_strategy(strategy, config)
        assert fake_runner.calls == []


class TestReplicatedComparison:
    def test_both_variants_train_on_the_same_seeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_runner: FakeRunner,
    ) -> None:
        path = _write(
            tmp_path,
            {
                "strategy": {
                    "kind": "replicated_comparison",
                    "seeds": [0, 1, 2],
                    "variants": {
                        "baseline": {},
                        "high_lr": {LEARNING_RATE: 0.01},
                    },
                },
                "config": _task_config(tmp_path),
            },
        )

        _run_main(monkeypatch, path)

        by_lr: dict[float, list[int]] = {}
        for seed, lr in fake_runner.calls:
            by_lr.setdefault(lr, []).append(seed)
        assert by_lr == {0.001: [0, 1, 2], 0.01: [0, 1, 2]}
        group = _group_json(tmp_path / "models", "reg_comparison")
        assert group["group"]["kind"] == "replicated_comparison"
        report_dir = Path(group["group"]["report_path"]).parent
        assert (report_dir / "comparison_table.tex").is_file()

    def test_an_unknown_override_path_fails_before_training(
        self, tmp_path: Path, fake_runner: FakeRunner
    ) -> None:
        strategy = {
            "kind": "replicated_comparison",
            "seeds": [0, 1],
            "variants": {"a": {}, "b": {"training.no_such_field": 1}},
        }
        config = _task_config(tmp_path)
        with pytest.raises(ValueError, match="no_such_field"):
            cli.run_strategy(strategy, config)
        assert fake_runner.calls == []

    def test_duplicate_seeds_are_refused(
        self, tmp_path: Path, fake_runner: FakeRunner
    ) -> None:
        strategy = {
            "kind": "replicated_comparison",
            "seeds": [2, 2],
            "variants": {"a": {}, "b": {LEARNING_RATE: 0.01}},
        }
        config = _task_config(tmp_path)
        with pytest.raises(ValueError, match="seeds"):
            cli.run_strategy(strategy, config)


class TestSweep:
    def test_a_standalone_sweep_calls_run_sweep(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_runner: FakeRunner,
    ) -> None:
        import visionforge.core.sweep as sweep_mod

        real = sweep_mod.run_sweep
        spaces: list[dict[str, Any]] = []

        def spy(
            runner: Any, config: dict[str, Any], space: dict[str, Any], **kw: Any
        ) -> Any:
            spaces.append(space)
            return real(runner, config, space, **kw)

        monkeypatch.setattr(sweep_mod, "run_sweep", spy)
        space = {LEARNING_RATE: [0.001, 0.01]}
        report = cli.run_strategy(
            {"kind": "sweep", "search_space": space}, _task_config(tmp_path)
        )

        assert spaces == [space]
        assert report["best_trial"]["overrides"] == {LEARNING_RATE: 0.01}
        assert report["planned_trials"] == 2
        group = _group_json(tmp_path / "models", "reg_sweep")
        assert group["group"]["kind"] == "sweep"

    def test_classification_points_to_the_search_blocks(self, tmp_path: Path) -> None:
        config = {"name": "c", "task": "multiclass"}
        strategy = {"kind": "sweep", "search_space": {LEARNING_RATE: [0.1]}}
        with pytest.raises(ValueError, match="grid_search"):
            cli.run_strategy(strategy, config)

    def test_an_unknown_sweep_path_fails_before_training(
        self, tmp_path: Path, fake_runner: FakeRunner
    ) -> None:
        strategy = {"kind": "sweep", "search_space": {"training.nope": [1, 2]}}
        config = _task_config(tmp_path)
        with pytest.raises(ValueError, match="nope"):
            cli.run_strategy(strategy, config)
        assert fake_runner.calls == []


class TestCrossValidation:
    def test_detection_fails_with_a_clear_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("VISIONFORGE_LANG", "en")
        path = _write(
            tmp_path,
            {
                "strategy": {"kind": "cv", "n_folds": 3},
                "config": {"name": "d", "task": "detection"},
            },
        )
        with pytest.raises(SystemExit) as exit_info:
            _run_main(monkeypatch, path)
        assert exit_info.value.code == 2
        err = capsys.readouterr().err
        assert "K-fold cross-validation is not available for task 'detection'" in err
        assert "regression, segmentation" in err

    def test_regression_calls_the_task_kfold_and_writes_the_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import visionforge.blocks.regression_cv as regression_cv
        from visionforge.blocks._cv_common import CrossValidationReport, FoldResult

        calls: list[dict[str, Any]] = []

        def fake_cv(config: RegressionConfig, **kwargs: Any) -> CrossValidationReport:
            calls.append(kwargs)
            folds = [
                FoldResult(
                    fold=i,
                    status="success",
                    train_size=8,
                    val_size=4,
                    metrics={R2: 0.5 + i},
                )
                for i in range(kwargs["n_folds"])
            ]
            aggregate: dict[str, dict[str, float | int | None]] = {
                R2: {"mean": 1.0, "std": 0.5, "n": 3}
            }
            return CrossValidationReport(
                n_folds=kwargs["n_folds"], metric=R2, folds=folds, aggregate=aggregate
            )

        monkeypatch.setattr(regression_cv, "run_regression_cross_validation", fake_cv)
        report = cli.run_strategy(
            {"kind": "cv", "n_folds": 3, "fold_seed": 11}, _task_config(tmp_path)
        )

        assert calls == [{"n_folds": 3, "shuffle": True, "seed": 11}]
        assert report["successful_folds"] == 3
        group = _group_json(tmp_path / "models", "reg_cv")
        assert group["group"]["kind"] == "kfold"


class TestEnvelope:
    def test_a_yaml_without_strategy_takes_the_current_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[str] = []

        class Block:
            def run(self) -> None:
                ran.append("run")

            def report(self) -> dict[str, Any]:
                return {}

        class Config:
            name = "plain"
            task = "regression"

        def fake_build(path: Path) -> tuple[Any, Any]:
            ran.append("build")
            return Config(), Block()

        def no_strategy(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a plain config must not reach run_strategy")

        monkeypatch.setattr(main_mod, "build_task_block", fake_build)
        monkeypatch.setattr(cli, "run_strategy", no_strategy)
        path = _write(tmp_path, _task_config(tmp_path))

        _run_main(monkeypatch, path)

        assert ran == ["build", "run"]
        assert cli.read_envelope(path) is None

    @pytest.mark.parametrize(
        "raw",
        [
            {"strategy": {"kind": "replicates"}},
            {"strategy": ["replicates"], "config": {}},
            {"strategy": {"kind": "replicates"}, "config": {}, "name": "x"},
        ],
    )
    def test_a_malformed_envelope_is_refused(
        self, tmp_path: Path, raw: dict[str, Any]
    ) -> None:
        path = _write(tmp_path, raw)
        with using_lang("en"), pytest.raises(ValueError) as exc_info:
            cli.read_envelope(path)
        assert str(exc_info.value) == message("en", "cli.strategy_shape")

    def test_an_unknown_kind_names_the_known_ones(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"strategy": {"kind": "bagging"}, "config": {}})
        with using_lang("en"), pytest.raises(ValueError) as exc_info:
            cli.read_envelope(path)
        text = str(exc_info.value)
        assert "'bagging'" in text
        assert "replicates, replicated_comparison, sweep, cv" in text

    def test_an_invalid_option_exits_with_status_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write(
            tmp_path,
            {
                "strategy": {"kind": "replicates", "seeds": [0]},
                "config": _task_config(tmp_path),
            },
        )
        with pytest.raises(SystemExit) as exit_info:
            _run_main(monkeypatch, path)
        assert exit_info.value.code == 2

    def test_a_job_with_no_result_exits_with_status_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_runner: FakeRunner,
    ) -> None:
        def failing(cfg: Any) -> RunResult:
            return RunResult(status="failed", error="boom")

        monkeypatch.setattr(fake_runner, "run", failing)
        path = _write(
            tmp_path,
            {
                "strategy": {"kind": "replicates", "seeds": [0, 1]},
                "config": _task_config(tmp_path),
            },
        )
        with pytest.raises(SystemExit) as exit_info:
            _run_main(monkeypatch, path)
        assert exit_info.value.code == 1


class TestRunnerFor:
    @pytest.mark.parametrize(
        ("task", "runner_name"),
        [
            ("binary", "ClassificationRunner"),
            ("multiclass", "ClassificationRunner"),
            ("detection", "DetectionRunner"),
            ("regression", "RegressionRunner"),
            ("segmentation", "SegmentationRunner"),
            ("anomaly", "AnomalyRunner"),
        ],
    )
    def test_each_task_gets_the_runner_the_gui_uses(
        self, task: str, runner_name: str
    ) -> None:
        assert type(cli.runner_for(task)).__name__ == runner_name

    def test_an_unknown_task_is_refused(self) -> None:
        with using_lang("en"), pytest.raises(ValueError, match="'pose'"):
            cli.runner_for("pose")


def test_help_documents_every_strategy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["visionforge", "run", "--help"])
    with pytest.raises(SystemExit):
        contextvars.copy_context().run(main_mod.main)
    out = capsys.readouterr().out
    for kind in cli.STRATEGY_KINDS:
        assert f"  {kind} " in out


def test_replicates_train_a_real_model_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two seeds of a 1-epoch regression on 32 px images, on CPU, nothing faked."""
    from visionforge.utils.selftest_data import build_regression_dataset

    base = build_regression_dataset(tmp_path / "data", rows=8)
    config = _task_config(tmp_path, name="e2e")
    config["data"].update(
        {
            "base_dir": str(base),
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {"image_size": 32},
        }
    )
    path = _write(
        tmp_path,
        {"strategy": {"kind": "replicates", "seeds": [0, 1]}, "config": config},
    )

    _run_main(monkeypatch, path)

    group = _group_json(tmp_path / "models", "e2e_replicates")
    assert group["group"]["seeds_finished"] == [0, 1]
    children = sorted(c["run_id"] for c in group["group"]["children"])
    assert len(children) == 2
    report_dir = Path(group["group"]["report_path"]).parent
    summary = json.loads(
        (report_dir / "replicates_summary.json").read_text(encoding="utf-8")
    )
    assert summary["successful_replicates"] == 2
    assert summary["headline"]["n"] == 2
    assert (report_dir / "replicates_table.tex").is_file()
