from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import Field
from torch import nn

from visionforge.tasks import (
    BaseTaskConfig,
    TaskSpec,
    clear_task_registry,
    get_task,
    register_task,
)
from visionforge.tasks.registry import TaskInfo
from visionforge.tasks.runner import CustomTaskRunner


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_task_registry()
    yield
    clear_task_registry()


class ToyConfig(BaseTaskConfig):
    scale: float = Field(default=1.0, gt=0)


def _register_toy(key: str = "toyrunner") -> None:
    @register_task(
        key=key,
        label="Toy Runner",
        accent="#2dd4bf",
        metrics={"mae": "lower"},
        primary_metric="mae",
    )
    class ToyRunnerTask(TaskSpec):
        Config = ToyConfig

        def build_model(self, cfg: Any) -> nn.Module:
            return nn.Linear(3, 1)

        def build_loaders(self, cfg: Any):
            gen = torch.Generator().manual_seed(0)
            batches = [
                (torch.randn(4, 3, generator=gen), torch.randn(4, 1, generator=gen))
                for _ in range(2)
            ]
            return batches, batches, None

        def compute_loss(self, model: nn.Module, batch: Any, cfg: Any):
            inputs, targets = batch
            return nn.functional.mse_loss(model(inputs), targets)

        def compute_metrics(self, model: nn.Module, loader: Any, cfg: Any):
            errs = [(model(x) - y).abs().mean().item() for x, y in loader]
            # scale exists so sweeps over the task's own field are observable
            return {"mae": (sum(errs) / len(errs)) * cfg.scale}


def _config_dict(tmp_path: Path) -> dict[str, Any]:
    return {
        "name": "toy_runner_run",
        "data": {"base_dir": str(tmp_path)},
        "training": {"epochs": 1, "batch_size": 4},
        "output": {"models_dir": str(tmp_path / "models")},
        "device": {"kind": "cpu"},
    }


class TestCustomTaskRunner:
    def test_run_trains_and_returns_success(self, tmp_path: Path) -> None:
        _register_toy()
        runner = CustomTaskRunner(get_task("toyrunner"))
        cfg = runner.config_type.model_validate(_config_dict(tmp_path))
        result = runner.run(cfg)

        assert result.status == "success"
        assert result.error == ""
        assert result.training_time_s is not None and result.training_time_s > 0
        assert "mae" in runner.metrics(result)
        assert runner.primary_metric() == "mae"

    def test_failure_is_captured_not_raised(self, tmp_path: Path) -> None:
        @register_task(
            key="broken",
            label="Broken",
            accent="#112233",
            metrics={"mae": "lower"},
            primary_metric="mae",
        )
        class BrokenTask(TaskSpec):
            Config = ToyConfig

            def build_model(self, cfg: Any) -> nn.Module:
                raise RuntimeError("researcher bug in build_model")

            def build_loaders(self, cfg: Any):
                return [], [], None

            def compute_loss(self, model: Any, batch: Any, cfg: Any):
                raise AssertionError("unreachable")

            def compute_metrics(self, model: Any, loader: Any, cfg: Any):
                raise AssertionError("unreachable")

        runner = CustomTaskRunner(get_task("broken"))
        cfg = runner.config_type.model_validate(_config_dict(tmp_path))
        result = runner.run(cfg)

        assert result.status == "failed"
        assert "researcher bug in build_model" in result.error
        assert result.metrics == {}

    def test_info_without_spec_cls_rejected(self) -> None:
        info = TaskInfo(key="ghost", label="Ghost", accent="#000000", description="")
        with pytest.raises(ValueError, match="no spec class"):
            CustomTaskRunner(info)


class TestOrchestratorsOverCustomRunner:
    def test_replicates_run_for_free(self, tmp_path: Path) -> None:
        from visionforge.core.replicates import run_replicates

        _register_toy()
        runner = CustomTaskRunner(get_task("toyrunner"))
        base = runner.config_type.model_validate(_config_dict(tmp_path)).model_dump(
            mode="json"
        )

        trials = run_replicates(runner, base, seeds=[7, 8], metric="mae")

        assert [t.seed for t in trials] == [7, 8]  # seed order, never ranked
        assert all(t.status == "success" for t in trials)
        assert all("mae" in t.metrics for t in trials)

    def test_sweep_over_the_tasks_own_field(self, tmp_path: Path) -> None:
        from visionforge.core.sweep import run_sweep

        _register_toy()
        runner = CustomTaskRunner(get_task("toyrunner"))
        base = runner.config_type.model_validate(_config_dict(tmp_path)).model_dump(
            mode="json"
        )

        # The search space targets `scale` — a field only this task declares.
        trials = run_sweep(
            runner, base, {"scale": [1.0, 3.0]}, mode="grid", metric="mae"
        )

        assert len(trials) == 2
        assert all(t.status == "success" for t in trials)
        by_scale = {t.overrides["scale"]: t.metrics["mae"] for t in trials}
        assert by_scale[3.0] == pytest.approx(by_scale[1.0] * 3.0)
