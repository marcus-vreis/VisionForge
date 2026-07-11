from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from visionforge.tasks import (
    BaseTaskConfig,
    TaskSpec,
    clear_task_registry,
    get_task,
    register_task,
)
from visionforge.tasks.engine import GenericTaskEngine, TaskRunContext


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_task_registry()
    yield
    clear_task_registry()


def _make_batches(n: int = 4) -> list[tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(0)
    return [
        (torch.randn(4, 3, generator=gen), torch.randn(4, 1, generator=gen))
        for _ in range(n)
    ]


def _register_toy(key: str = "toyreg") -> Any:
    @register_task(
        key=key,
        label="Toy",
        accent="#2dd4bf",
        metrics={"mae": "lower"},
        primary_metric="mae",
    )
    class ToyTask(TaskSpec):
        def build_model(self, cfg: Any) -> nn.Module:
            return nn.Linear(3, 1)

        def build_loaders(self, cfg: Any):
            return _make_batches(), _make_batches(2), _make_batches(1)

        def compute_loss(self, model: nn.Module, batch: Any, cfg: Any):
            inputs, targets = batch
            return nn.functional.mse_loss(model(inputs), targets)

        def compute_metrics(self, model: nn.Module, loader: Any, cfg: Any):
            errs = [(model(x) - y).abs().mean().item() for x, y in loader]
            return {"mae": sum(errs) / len(errs)}

    return ToyTask


def _config(tmp_path: Path, **training: Any) -> BaseTaskConfig:
    return BaseTaskConfig.model_validate(
        {
            "name": "toy_run",
            "data": {"base_dir": str(tmp_path)},
            "training": {"epochs": 3, "batch_size": 4, **training},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


class TestGenericTaskEngineLevel1:
    def test_trains_streams_and_writes_run_json(self, tmp_path: Path) -> None:
        _register_toy()
        events: list[dict[str, Any]] = []
        result = GenericTaskEngine(get_task("toyreg"), _config(tmp_path)).run(
            events.append
        )

        kinds = [e["event"] for e in events]
        assert kinds[0] == "start" and kinds[-1] == "end"
        assert kinds.count("epoch_end") == 3
        assert all("val_mae" in e for e in events if e["event"] == "epoch_end")

        assert result.total_epochs == 3
        assert result.best_epoch >= 1
        assert "mae" in result.metrics
        assert "test_mae" in result.metrics  # test loader was provided
        assert result.model_path is not None and result.model_path.is_file()

        run_json = json.loads((result.run_dir / "run.json").read_text("utf-8"))
        assert run_json["task"] == "custom:toyreg"
        assert run_json["status"] == "completed"
        assert run_json["metrics"]["mae"] == pytest.approx(result.metrics["mae"])
        assert len(run_json["history"]) == 3
        assert "cuda" in run_json["environment"]
        # the primary-metric curve is rendered next to the checkpoint
        assert (result.run_dir / "mae_curve.png").is_file()

    def test_early_stopping_respects_patience(self, tmp_path: Path) -> None:
        @register_task(
            key="stuck",
            label="Stuck",
            accent="#112233",
            metrics={"score": "higher"},
            primary_metric="score",
        )
        class StuckTask(TaskSpec):
            def build_model(self, cfg: Any) -> nn.Module:
                return nn.Linear(3, 1)

            def build_loaders(self, cfg: Any):
                return _make_batches(1), _make_batches(1), None

            def compute_loss(self, model: nn.Module, batch: Any, cfg: Any):
                inputs, targets = batch
                return nn.functional.mse_loss(model(inputs), targets)

            def compute_metrics(self, model: nn.Module, loader: Any, cfg: Any):
                return {"score": 0.5}  # never improves after epoch 1

        cfg = _config(tmp_path, epochs=50, early_stopping_patience=2)
        result = GenericTaskEngine(get_task("stuck"), cfg).run()
        assert result.total_epochs == 3  # 1 best + 2 patience
        assert result.best_epoch == 1

    def test_missing_primary_metric_fails_loudly(self, tmp_path: Path) -> None:
        @register_task(
            key="badmetrics",
            label="Bad",
            accent="#112233",
            metrics={"declared": "higher"},
            primary_metric="declared",
        )
        class BadTask(TaskSpec):
            def build_model(self, cfg: Any) -> nn.Module:
                return nn.Linear(3, 1)

            def build_loaders(self, cfg: Any):
                return _make_batches(1), _make_batches(1), None

            def compute_loss(self, model: nn.Module, batch: Any, cfg: Any):
                inputs, targets = batch
                return nn.functional.mse_loss(model(inputs), targets)

            def compute_metrics(self, model: nn.Module, loader: Any, cfg: Any):
                return {"other": 1.0}

        with pytest.raises(ValueError, match="primary metric"):
            GenericTaskEngine(get_task("badmetrics"), _config(tmp_path)).run()

    def test_lower_is_better_selects_min_checkpoint(self, tmp_path: Path) -> None:
        values = iter([3.0, 1.0, 2.0, 5.0])

        @register_task(
            key="valley",
            label="Valley",
            accent="#112233",
            metrics={"err": "lower"},
            primary_metric="err",
        )
        class ValleyTask(TaskSpec):
            def build_model(self, cfg: Any) -> nn.Module:
                return nn.Linear(3, 1)

            def build_loaders(self, cfg: Any):
                return _make_batches(1), _make_batches(1), None

            def compute_loss(self, model: nn.Module, batch: Any, cfg: Any):
                inputs, targets = batch
                return nn.functional.mse_loss(model(inputs), targets)

            def compute_metrics(self, model: nn.Module, loader: Any, cfg: Any):
                try:
                    return {"err": next(values)}
                except StopIteration:
                    return {"err": 9.0}  # final re-evaluation after reload

        cfg = _config(tmp_path, epochs=4, early_stopping_patience=10)
        result = GenericTaskEngine(get_task("valley"), cfg).run()
        assert result.best_epoch == 2  # err=1.0 was the minimum


class TestGenericTaskEngineLevel2:
    def test_custom_run_bypasses_the_loop(self, tmp_path: Path) -> None:
        seen: dict[str, Any] = {}

        @register_task(
            key="freeform",
            label="Freeform",
            accent="#445566",
            metrics={"score": "higher"},
            primary_metric="score",
        )
        class Freeform(TaskSpec):
            def build_model(self, cfg: Any) -> nn.Module:
                return nn.Linear(1, 1)

            def build_loaders(self, cfg: Any):
                return [], [], None

            def compute_loss(self, model: Any, batch: Any, cfg: Any):
                raise AssertionError("Level 1 hooks must not run")

            def compute_metrics(self, model: Any, loader: Any, cfg: Any):
                raise AssertionError("Level 1 hooks must not run")

            def run(self, cfg: Any, ctx: TaskRunContext) -> dict[str, float]:
                seen["run_dir"] = ctx.run_dir
                ctx.save_checkpoint(nn.Linear(1, 1), "final.pth")
                return {"score": 0.9}

        result = GenericTaskEngine(get_task("freeform"), _config(tmp_path)).run()
        assert result.metrics == {"score": 0.9}
        assert (seen["run_dir"] / "final.pth").is_file()
        run_json = json.loads((result.run_dir / "run.json").read_text("utf-8"))
        assert run_json["task"] == "custom:freeform"
        assert run_json["metrics"]["score"] == 0.9
