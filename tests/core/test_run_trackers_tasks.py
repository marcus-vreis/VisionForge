"""Every task's finished run.json reaches MLflow and W&B (#286, ADR-197).

The run records follow the shape each trainer writes (ADR-013); the clients are
the in-memory fakes, so nothing touches the network or a real store.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.mlflow_tracking import MlflowTracker
from visionforge.core.run_trackers import export_run_record
from visionforge.core.tracking_privacy import is_metric_name, tracker_metrics
from visionforge.core.wandb_tracking import WandbTarget, WandbTracker

from ..fake_mlflow import FakeMlflowClient
from ..fake_wandb import FakeGraphQL, FakeWandb, install_fake_wandb

_TOKEN = "example-wandb-key"
_CONFIG = {
    "name": "exp",
    "training": {"epochs": 2},
    "data": {"base_dir": "C:/Users/someone/private-dataset"},
}

# task -> (run.json "task" value, final metrics, history rows)
RECORDS: dict[str, tuple[str | None, dict[str, Any], list[dict[str, Any]]]] = {
    "regression": (
        None,
        {
            "best_val_loss": 0.2,
            "best_epoch": 2,
            "total_epochs": 2,
            "mse": 0.2,
            "rmse": 0.45,
            "mae": 0.3,
            "r2": 0.8,
        },
        [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "val_mse": 0.4,
                "val_rmse": 0.63,
                "val_mae": 0.5,
                "val_r2": 0.6,
            },
            {
                "epoch": 2,
                "train_loss": 0.3,
                "val_loss": 0.2,
                "val_mse": 0.2,
                "val_rmse": 0.45,
                "val_mae": 0.3,
                "val_r2": 0.8,
            },
        ],
    ),
    "segmentation": (
        None,
        {
            "best_val_miou": 0.7,
            "best_epoch": 1,
            "total_epochs": 1,
            "val_miou": 0.7,
            "val_dice": 0.8,
            "val_pixel_acc": 0.9,
        },
        [
            {
                "epoch": 1,
                "train_loss": 0.6,
                "val_loss": 0.5,
                "val_miou": 0.7,
                "val_dice": 0.8,
                "val_pixel_acc": 0.9,
            }
        ],
    ),
    "anomaly": (
        None,
        {
            "best_auroc": 0.95,
            "best_epoch": 1,
            "total_epochs": 1,
            "auroc": 0.95,
            "threshold": 0.4,
            "image_f1": 0.9,
        },
        [
            {
                "epoch": 1,
                "train_loss": 0.1,
                "val_auroc": 0.95,
                "val_image_f1": 0.9,
                "val_threshold": 0.4,
            }
        ],
    ),
    "detection": (
        None,
        {
            "map50_95": 0.5,
            "map50": 0.7,
            "precision": 0.6,
            "recall": 0.55,
            "box_loss": 1.2,
            "best_epoch": 1,
            "total_epochs": 1,
        },
        [
            {
                "epoch": 1,
                "map50": 0.7,
                "map50_95": 0.5,
                "precision": 0.6,
                "recall": 0.55,
                "box_loss": 1.2,
                "train_box_loss": 1.4,
                "val_box_loss": 1.2,
            }
        ],
    ),
    "custom": (
        "custom:denoise",
        {"best_epoch": 1, "total_epochs": 1, "psnr": 31.5, "ssim": 0.9},
        [{"epoch": 1, "train_loss": 0.2, "val_psnr": 31.5, "val_ssim": 0.9}],
    ),
}

# run.json history names the trainer streams under chart keys (ADR-170).
_CHART_KEYS = {"train_loss": "loss/train", "val_loss": "loss/val"}


def _write_run(tmp_path: Path, task: str, **extra: Any) -> tuple[Path, dict]:
    top_task, metrics, history = RECORDS[task]
    record: dict[str, Any] = {
        "status": "completed",
        "stopped": False,
        "config": {**_CONFIG, "task": task},
        "metrics": metrics,
        "history": history,
        **extra,
    }
    if top_task is not None:
        record["task"] = top_task
    run_dir = tmp_path / "models" / task
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir, record


def _expected_history(history: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    return {
        row["epoch"]: {
            _CHART_KEYS.get(key, key): float(value)
            for key, value in row.items()
            if key != "epoch"
        }
        for row in history
    }


def _wandb_factory(tmp_path: Path) -> Any:
    target = WandbTarget("lab", token=_TOKEN)
    return partial(WandbTracker, tmp_path / "wandb", "ana", target=target)


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeWandb, FakeGraphQL]:
    return install_fake_wandb(monkeypatch)


@pytest.mark.parametrize("task", RECORDS)
def test_mlflow_receives_each_task(
    tmp_path: Path, task: str, fake_mlflow: type[FakeMlflowClient]
) -> None:
    run_dir, record = _write_run(tmp_path, task)

    export_run_record([partial(MlflowTracker, tmp_path / "mlflow", "ana")], run_dir)

    (run,) = fake_mlflow.runs.values()
    assert run.status == "FINISHED"
    assert run.artifacts == ["run.json"]
    assert run.params["training.epochs"] == "2"
    assert run.params["task"] == json.dumps(task)
    assert not any("base_dir" in key for key in run.params)
    logged = {
        key: {m.step: m.value for m in values} for key, values in run.metrics.items()
    }
    expected: dict[str, dict[int, float]] = {}
    for epoch, values in _expected_history(record["history"]).items():
        for key, value in values.items():
            expected.setdefault(key, {})[epoch] = value
    assert logged == expected
    saved = json.loads((run_dir / "tracking.json").read_text())
    assert saved["state"] == "finished"


@pytest.mark.parametrize("task", RECORDS)
def test_wandb_receives_each_task(
    tmp_path: Path, task: str, fakes: tuple[FakeWandb, FakeGraphQL]
) -> None:
    sdk, api = fakes
    run_dir, record = _write_run(tmp_path, task)

    export_run_record([_wandb_factory(tmp_path)], run_dir)

    (run,) = sdk.runs.values()
    assert api.projects == {("ana", "lab"): "PRIVATE"}
    assert run.history == _expected_history(record["history"])
    # The final metrics keep their run.json names.
    assert run.summary == record["metrics"]
    assert run.exit_codes == [0]
    assert sdk.inits[0]["config"]["task"] == task
    assert "data.base_dir" not in sdk.inits[0]["config"]
    saved = json.loads((run_dir / "wandb.json").read_text())
    assert saved["state"] == "finished"


def test_both_exporters_get_the_same_run(
    tmp_path: Path,
    fake_mlflow: type[FakeMlflowClient],
    fakes: tuple[FakeWandb, FakeGraphQL],
) -> None:
    sdk, _ = fakes
    run_dir, _ = _write_run(tmp_path, "segmentation")

    export_run_record(
        [
            partial(MlflowTracker, tmp_path / "mlflow", "ana"),
            _wandb_factory(tmp_path),
        ],
        run_dir,
    )

    assert len(fake_mlflow.runs) == 1
    assert len(sdk.runs) == 1


built: list[_Recorder] = []


class _Recorder:
    """A tracker that writes down the calls it gets."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.calls: list[tuple[str, Any]] = []
        built.append(self)

    def start(self, config: dict[str, Any]) -> None:
        self.calls.append(("start", config))

    def epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        self.calls.append(("epoch", (epoch, metrics)))

    def finish(self, status: str) -> None:
        self.calls.append(("finish", status))


@pytest.fixture(autouse=True)
def _clear_built() -> None:
    built.clear()


@pytest.mark.parametrize(
    ("extra", "status"),
    [
        ({}, "FINISHED"),
        ({"stopped": True}, "KILLED"),
        ({"status": "failed"}, "FAILED"),
    ],
)
def test_terminal_status_follows_the_record(
    tmp_path: Path, extra: dict[str, Any], status: str
) -> None:
    run_dir, _ = _write_run(tmp_path, "regression", **extra)
    export_run_record([_Recorder], run_dir)
    (tracker,) = built
    assert tracker.calls[-1] == ("finish", status)


def test_rows_without_an_integer_epoch_are_skipped(tmp_path: Path) -> None:
    run_dir, record = _write_run(tmp_path, "anomaly")
    record["history"] = [{"epoch": True, "x": 1}, "row", {"x": 2}, {"epoch": 3}]
    record["config"] = "not a dict"
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")

    export_run_record([_Recorder], run_dir)

    (tracker,) = built
    assert tracker.calls == [("start", {}), ("epoch", (3, {})), ("finish", "FINISHED")]


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", None])
def test_unreadable_record_exports_nothing(tmp_path: Path, content: str | None) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if content is not None:
        (run_dir / "run.json").write_text(content, encoding="utf-8")

    export_run_record([_Recorder], run_dir)

    assert built == []


def test_a_tracker_that_cannot_be_built_does_not_stop_the_others(
    tmp_path: Path,
) -> None:
    run_dir, _ = _write_run(tmp_path, "detection")

    def broken(_run_dir: Path) -> Any:
        raise RuntimeError("no client")

    export_run_record([broken, _Recorder], run_dir)

    (tracker,) = built
    assert tracker.calls[-1] == ("finish", "FINISHED")


def test_hub_list_is_unchanged() -> None:
    # The Hub card keeps its own list; only the private trackers add these.
    assert not is_metric_name("box_loss")
    assert not is_metric_name("best_val_miou")


def test_custom_task_metrics_keep_their_names() -> None:
    record = {
        "task": "custom:denoise",
        "metrics": {
            "psnr": 31.5,
            "best_epoch": 1,
            "flag": True,
            "nan": float("nan"),
            "label": "x",
            "a.b": 1.0,
        },
    }
    assert tracker_metrics(record) == {"best_epoch": 1, "psnr": 31.5}


def test_built_in_task_metrics_stay_on_the_known_list() -> None:
    record = {"metrics": {"psnr": 31.5, "r2": 0.8, "box_loss": 1.0}}
    assert tracker_metrics(record) == {"box_loss": 1.0, "r2": 0.8}


@pytest.mark.parametrize(
    "record", [None, {"task": "custom:x"}, {"task": "custom:x", "metrics": [1]}]
)
def test_missing_metrics_give_an_empty_summary(record: Any) -> None:
    assert tracker_metrics(record) == {}
