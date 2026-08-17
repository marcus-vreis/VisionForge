from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

from visionforge.core import detection_trainer as dt_mod
from visionforge.core.cancellation import CancellationToken
from visionforge.core.detection_trainer import (
    DetectionTrainer,
    _is_worker_spawn_crash,
)
from visionforge.core.resume import RESUME_FILENAME
from visionforge.utils.detection_config import DetectionConfig


def _layout(base: Path) -> None:
    for split in ("train", "val"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)


def _config(tmp_path: Path, model: dict | None = None) -> DetectionConfig:
    base = tmp_path / "ds"
    _layout(base)
    return DetectionConfig.model_validate(
        {
            "name": "det",
            "model": model or {"name": "yolo11n", "num_classes": 2},
            "data": {"base_dir": str(base), "image_size": 640},
            "training": {
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": 0.01,
                "patience": 5,
                "seed": 0,
                "workers": 0,
            },
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


class _FakeUTrainer:
    def __init__(self, epoch: int, epochs: int) -> None:
        self.epoch = epoch
        self.epochs = epochs
        self.metrics = {
            "metrics/mAP50(B)": 0.5 + 0.1 * epoch,
            "metrics/mAP50-95(B)": 0.3 + 0.1 * epoch,
            "metrics/precision(B)": 0.6 + 0.05 * epoch,
            "metrics/recall(B)": 0.4 + 0.05 * epoch,
            "val/box_loss": 1.0 - 0.1 * epoch,
            "val/cls_loss": 0.8 - 0.1 * epoch,
            "val/dfl_loss": 0.6 - 0.05 * epoch,
            "train/box_loss": 1.2 - 0.1 * epoch,
            "train/cls_loss": 0.9 - 0.1 * epoch,
            "train/dfl_loss": 0.7 - 0.05 * epoch,
        }


def _make_fake_yolo(record: dict[str, Any]) -> type:
    class _FakeYOLO:
        def __init__(self, weights: str) -> None:
            record["weights"] = weights
            self._callbacks: dict[str, list] = {}

        def add_callback(self, event: str, fn: Any) -> None:
            self._callbacks.setdefault(event, []).append(fn)

        def train(self, **kwargs: Any) -> object:
            record["train_kwargs"] = kwargs
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            (run_dir / "weights" / "best.pt").write_bytes(b"fake")
            epochs = int(kwargs["epochs"])
            for e in range(epochs):
                ft = _FakeUTrainer(e, epochs)
                for fn in self._callbacks.get("on_fit_epoch_end", []):
                    fn(ft)
                if getattr(ft, "stop", False):
                    break
            # `final_eval` fires the same callback once more, at epoch + 1, to
            # log the metrics of best.pt. Real Ultralytics does this; the fake
            # has to, or the tests cannot see the phantom epoch it caused.
            final = _FakeUTrainer(epochs, epochs)
            # Better numbers than any real epoch, which is what made the phantom
            # steal `best_epoch` instead of merely padding the history.
            final.metrics = {k: v + 1.0 for k, v in final.metrics.items()}
            for fn in self._callbacks.get("on_fit_epoch_end", []):
                fn(final)
            return object()

    return _FakeYOLO


# ── backend guards ────────────────────────────────────────────────────────────


class TestWorkerSpawnCrashDetection:
    """WinError 1455 (page file too small for N workers to reload torch's
    CUDA DLLs) surfaces under several disguises — all must be recognized so
    the trainer can raise an actionable message instead of the raw crash."""

    @pytest.mark.parametrize(
        "message",
        [
            "[WinError 1455] O arquivo de paginação é muito pequeno",
            'Error loading "...\\torch\\lib\\shm.dll" or one of its dependencies.',
            "The paging file is too small for this operation to complete.",
            "DataLoader worker (pid 1234) exited unexpectedly",
            "Ran out of input",
        ],
    )
    def test_recognizes_page_file_exhaustion_disguises(self, message: str) -> None:
        assert _is_worker_spawn_crash(message)

    def test_unrelated_errors_pass_through(self) -> None:
        assert not _is_worker_spawn_crash("CUDA out of memory")
        assert not _is_worker_spawn_crash("dataset.yaml not found")


class TestBackendGuards:
    def test_missing_ultralytics_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dt_mod, "YOLO", None)
        with pytest.raises(RuntimeError, match="ultralytics"):
            DetectionTrainer(_config(tmp_path)).fit()


# ── ultralytics path (mocked) ─────────────────────────────────────────────────


class TestUltralyticsPath:
    def test_passes_translated_args_to_yolo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record: dict[str, Any] = {}
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo(record))
        DetectionTrainer(_config(tmp_path)).fit()

        kw = record["train_kwargs"]
        assert kw["epochs"] == 2
        assert kw["imgsz"] == 640
        assert kw["lr0"] == 0.01
        assert kw["batch"] == 8
        assert kw["device"] == "cpu"
        assert kw["data"].endswith("data.yaml")
        assert record["weights"] == "yolo11n.pt"

    def test_passes_extended_hyperparameters_to_yolo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Optimizer, schedule, loss-gain, and augmentation knobs all reach train()."""
        record: dict[str, Any] = {}
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo(record))
        cfg = _config(tmp_path)
        # default values flow through unchanged (behaviour-preserving)
        DetectionTrainer(cfg).fit()

        kw = record["train_kwargs"]
        # optimizer
        assert kw["optimizer"] == "auto"
        assert kw["momentum"] == pytest.approx(0.937)
        assert kw["weight_decay"] == pytest.approx(0.0005)
        # schedule
        assert kw["lrf"] == pytest.approx(0.01)
        assert kw["cos_lr"] is False
        assert kw["warmup_epochs"] == pytest.approx(3.0)
        # loss gains
        assert kw["box"] == pytest.approx(7.5)
        assert kw["cls"] == pytest.approx(0.5)
        assert kw["dfl"] == pytest.approx(1.5)
        # regularization / mechanics
        assert kw["close_mosaic"] == 10
        assert kw["amp"] is True
        assert kw["single_cls"] is False
        # augmentation
        assert kw["fliplr"] == pytest.approx(0.5)
        assert kw["mosaic"] == pytest.approx(1.0)
        assert kw["hsv_h"] == pytest.approx(0.015)
        assert kw["auto_augment"] == "randaugment"

    def test_custom_hyperparameters_are_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record: dict[str, Any] = {}
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo(record))
        cfg = _config(tmp_path)
        cfg = cfg.model_copy(
            update={
                "training": cfg.training.model_copy(
                    update={"optimizer": "AdamW", "cos_lr": True, "box": 9.0}
                )
            }
        )
        DetectionTrainer(cfg).fit()
        kw = record["train_kwargs"]
        assert kw["optimizer"] == "AdamW"
        assert kw["cos_lr"] is True
        assert kw["box"] == pytest.approx(9.0)

    def test_streams_progress_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo({}))
        events: list[dict[str, Any]] = []
        DetectionTrainer(_config(tmp_path)).fit(progress_callback=events.append)

        kinds = [e["event"] for e in events]
        assert kinds == ["start", "epoch_end", "epoch_end", "end"]
        epoch_events = [e for e in events if e["event"] == "epoch_end"]
        assert all("map50_95" in e for e in epoch_events)
        assert epoch_events[-1]["epoch"] == 2

    def test_streams_extended_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Precision/recall + train & val loss components reach the SSE event."""
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo({}))
        events: list[dict[str, Any]] = []
        DetectionTrainer(_config(tmp_path)).fit(progress_callback=events.append)

        last = [e for e in events if e["event"] == "epoch_end"][-1]
        for key in (
            "precision",
            "recall",
            "train_box_loss",
            "train_cls_loss",
            "train_dfl_loss",
            "val_box_loss",
            "val_cls_loss",
            "val_dfl_loss",
        ):
            assert last[key] is not None, key

    def test_run_json_history_carries_extended_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo({}))
        result = DetectionTrainer(_config(tmp_path)).fit()

        run_json = json.loads((result.run_dir / "run.json").read_text("utf-8"))
        assert run_json["metrics"]["precision"] is not None
        assert run_json["metrics"]["recall"] is not None
        last_epoch = run_json["history"][-1]
        assert last_epoch["train_box_loss"] is not None
        assert last_epoch["val_cls_loss"] is not None

    def test_writes_compatible_run_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo({}))
        result = DetectionTrainer(_config(tmp_path)).fit()

        run_json = json.loads((result.run_dir / "run.json").read_text("utf-8"))
        assert run_json["status"] == "completed"
        assert run_json["metrics"]["map50_95"] == pytest.approx(0.4)
        assert len(run_json["history"]) == 2
        assert run_json["artifacts"]["model"].endswith("best.pt")
        assert run_json["device_used"] == "cpu"
        assert "torch" in run_json["environment"]

    def test_best_epoch_selected_by_map(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo({}))
        result = DetectionTrainer(_config(tmp_path)).fit()
        assert result.best_epoch == 2
        assert result.best_map50_95 == pytest.approx(0.4)


# ── weight resolution ─────────────────────────────────────────────────────────


class _FakeDetector(nn.Module):
    """Minimal stand-in: returns a real loss dict in train mode so the loop runs."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, images: Any, targets: Any = None) -> Any:
        if self.training:
            loss = self.lin(torch.zeros(1, 1)).sum()
            return {"loss_box_reg": loss}
        return [
            {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,)),
            }
            for _ in images
        ]


def _tv_config(tmp_path: Path) -> DetectionConfig:
    base = tmp_path / "ds"
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (100, 100, 100)).save(img_dir / "a.jpg")
        (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    return DetectionConfig.model_validate(
        {
            "name": "det_tv",
            "model": {
                "backend": "torchvision",
                "name": "fasterrcnn_resnet50_fpn",
                "num_classes": 2,
                "pretrained": False,
            },
            "data": {"base_dir": str(base), "image_size": 320},
            "training": {
                "epochs": 2,
                "batch_size": 1,
                "learning_rate": 0.005,
                "workers": 0,
            },
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


class TestTorchvisionPath:
    def test_requires_base_dir(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("train: t\nval: v\nnames: [a]\n", encoding="utf-8")
        cfg = DetectionConfig.model_validate(
            {
                "name": "det_tv",
                "model": {"backend": "torchvision", "name": "ssd300_vgg16"},
                "data": {"data_yaml": str(data_yaml)},
                "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.01},
                "output": {"models_dir": str(tmp_path / "models")},
            }
        )
        with pytest.raises(ValueError, match="base_dir"):
            DetectionTrainer(cfg).fit()

    def test_trains_streams_and_writes_run_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            dt_mod, "build_torchvision_detector", lambda *a, **k: _FakeDetector()
        )
        events: list[dict[str, Any]] = []
        result = DetectionTrainer(_tv_config(tmp_path)).fit(
            progress_callback=events.append
        )

        kinds = [e["event"] for e in events]
        assert kinds == ["start", "epoch_end", "epoch_end", "end"]
        assert result.total_epochs == 2
        assert result.model_path.exists()  # best.pt saved

        run_json = json.loads((result.run_dir / "run.json").read_text("utf-8"))
        assert run_json["status"] == "completed"
        assert run_json["metrics"]["box_loss"] is not None
        # mAP@50 is computed and recorded (0.0 here: the fake emits no boxes).
        assert run_json["metrics"]["map50"] is not None
        assert run_json["device_used"] == "cpu"
        assert len(run_json["history"]) == 2
        assert all(e.get("map50") is not None for e in run_json["history"])

    def test_writes_results_plot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """torchvision has no Ultralytics results.png, so the trainer renders one."""
        monkeypatch.setattr(
            dt_mod, "build_torchvision_detector", lambda *a, **k: _FakeDetector()
        )
        result = DetectionTrainer(_tv_config(tmp_path)).fit()

        assert (result.run_dir / "results.png").exists()
        run_json = json.loads((result.run_dir / "run.json").read_text("utf-8"))
        graphics = run_json["artifacts"]["graphics"]
        assert any(g.endswith("results.png") for g in graphics)


class TestFinalValidationIsNotAnEpoch:
    """Ultralytics logs best.pt's metrics through the same callback, at epochs+1."""

    def test_the_history_stops_at_the_configured_epochs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo({}))

        result = DetectionTrainer(_config(tmp_path)).fit()

        assert [h.epoch for h in result.history] == [1, 2]
        assert result.total_epochs == 2

    def test_it_cannot_steal_best_epoch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It reports the best epoch's own numbers; crediting them to a step
        past the last epoch pointed the researcher at a run that never ran."""
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo({}))

        result = DetectionTrainer(_config(tmp_path)).fit()

        assert result.best_epoch == 2

    def test_a_cancelled_run_does_not_gain_one_either(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its phantom is within the configured count, so the epoch number alone
        cannot tell it apart -- only the loop having ended can."""
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo({}))
        cfg = _config(tmp_path)
        cfg = cfg.model_copy(
            update={"training": cfg.training.model_copy(update={"epochs": 3})}
        )
        token = CancellationToken()

        def stop_after_first(event: dict[str, Any]) -> None:
            if event.get("event") == "epoch_end" and event.get("epoch") == 1:
                token.cancel()

        result = DetectionTrainer(cfg).fit(
            progress_callback=stop_after_first, cancel_token=token
        )

        assert [h.epoch for h in result.history] == [1]
        assert result.total_epochs == 1


class TestUltralyticsResume:
    """Ultralytics keeps its own resume state; the trainer must use it, not ours."""

    @staticmethod
    def _fake_yolo_resuming(record: dict[str, Any]) -> type:
        """Fake that continues its epoch numbering the way `resume=True` does."""

        class _FakeYOLO:
            def __init__(self, weights: str) -> None:
                record["weights"] = weights

            def add_callback(self, event: str, fn: Any) -> None:
                record.setdefault("callbacks", []).append(fn)

            def train(self, **kwargs: Any) -> object:
                record["train_kwargs"] = kwargs
                run_dir = Path(kwargs["project"]) / kwargs["name"]
                (run_dir / "weights").mkdir(parents=True, exist_ok=True)
                (run_dir / "weights" / "best.pt").write_bytes(b"fake")
                (run_dir / "weights" / "last.pt").write_bytes(b"fake")
                first = int(record.get("resume_from", 0))
                for e in range(first, int(kwargs["epochs"])):
                    for fn in record.get("callbacks", []):
                        fn(_FakeUTrainer(e, int(kwargs["epochs"])))
                return object()

        return _FakeYOLO

    def test_continues_from_last_pt_in_the_same_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record: dict[str, Any] = {}
        monkeypatch.setattr(dt_mod, "YOLO", self._fake_yolo_resuming(record))
        cfg = _config(tmp_path)
        first = DetectionTrainer(cfg).fit()

        # Second attempt: only the last epoch is missing.
        record["callbacks"] = []
        record["resume_from"] = 1
        second = DetectionTrainer(cfg).fit(resume_dir=first.run_dir)

        assert record["train_kwargs"]["resume"] is True
        assert record["weights"].endswith("last.pt")
        assert second.run_dir == first.run_dir
        # The epochs already recorded came back from run.json, so the history is
        # continuous rather than starting where the second attempt did.
        assert [h.epoch for h in second.history] == [1, 2]

    def test_a_fresh_run_never_asks_to_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record: dict[str, Any] = {}
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo(record))
        DetectionTrainer(_config(tmp_path)).fit()

        assert record["train_kwargs"]["resume"] is False


class TestTorchvisionResume:
    def test_continues_the_same_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stopped run continues in its own directory, with one history."""
        monkeypatch.setattr(
            dt_mod, "build_torchvision_detector", lambda *a, **k: _FakeDetector()
        )
        cfg = _tv_config(tmp_path)
        token = CancellationToken()

        def _stop_after_first(event: dict[str, Any]) -> None:
            if event.get("event") == "epoch_end" and event.get("epoch") == 1:
                token.cancel()

        stopped = DetectionTrainer(cfg).fit(
            progress_callback=_stop_after_first, cancel_token=token
        )
        assert stopped.total_epochs == 1
        assert (stopped.run_dir / RESUME_FILENAME).is_file()

        finished = DetectionTrainer(cfg).fit(resume_dir=stopped.run_dir)

        assert finished.run_dir == stopped.run_dir
        assert [h.epoch for h in finished.history] == [1, 2]
        assert not (finished.run_dir / RESUME_FILENAME).exists()
        # The plot needs one train loss per epoch, including the restored ones.
        assert (finished.run_dir / "results.png").exists()

    def test_unreadable_state_starts_a_fresh_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            dt_mod, "build_torchvision_detector", lambda *a, **k: _FakeDetector()
        )
        cfg = _tv_config(tmp_path)
        token = CancellationToken()

        def _stop_after_first(event: dict[str, Any]) -> None:
            if event.get("event") == "epoch_end" and event.get("epoch") == 1:
                token.cancel()

        stopped = DetectionTrainer(cfg).fit(
            progress_callback=_stop_after_first, cancel_token=token
        )
        (stopped.run_dir / RESUME_FILENAME).write_bytes(b"not a checkpoint")

        finished = DetectionTrainer(cfg).fit(resume_dir=stopped.run_dir)

        assert finished.run_dir != stopped.run_dir
        assert [h.epoch for h in finished.history] == [1, 2]


class TestTorchvisionOptimizer:
    def test_auto_maps_to_sgd(self, tmp_path: Path) -> None:
        trainer = DetectionTrainer(_tv_config(tmp_path))
        param = nn.Parameter(torch.zeros(1))
        opt = trainer._build_torchvision_optimizer([param])
        assert isinstance(opt, torch.optim.SGD)
        assert opt.defaults["momentum"] == pytest.approx(0.937)
        assert opt.defaults["weight_decay"] == pytest.approx(0.0005)

    def test_adamw_selected(self, tmp_path: Path) -> None:
        cfg = _tv_config(tmp_path)
        cfg = cfg.model_copy(
            update={"training": cfg.training.model_copy(update={"optimizer": "AdamW"})}
        )
        trainer = DetectionTrainer(cfg)
        opt = trainer._build_torchvision_optimizer([nn.Parameter(torch.zeros(1))])
        assert isinstance(opt, torch.optim.AdamW)


class TestResolveSplitDirs:
    def test_images_split_layout(self, tmp_path: Path) -> None:
        (tmp_path / "images" / "train").mkdir(parents=True)
        trainer = DetectionTrainer(_tv_config(tmp_path / "x"))
        imgs, lbls = trainer._resolve_split_dirs(tmp_path, "train")
        assert imgs == tmp_path / "images" / "train"
        assert lbls == tmp_path / "labels" / "train"

    def test_missing_split_raises(self, tmp_path: Path) -> None:
        trainer = DetectionTrainer(_tv_config(tmp_path / "x"))
        with pytest.raises(ValueError, match="train"):
            trainer._resolve_split_dirs(tmp_path / "empty", "train")


class TestWeightResolution:
    def test_scratch_uses_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record: dict[str, Any] = {}
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo(record))
        cfg = _config(tmp_path, {"name": "yolo11n", "pretrained": False})
        DetectionTrainer(cfg).fit()
        assert record["weights"] == "yolo11n.yaml"

    def test_checkpoint_path_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ckpt = tmp_path / "my.pt"
        ckpt.write_bytes(b"x")
        record: dict[str, Any] = {}
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo(record))
        cfg = _config(tmp_path, {"name": "yolo11n", "weights_path": str(ckpt)})
        DetectionTrainer(cfg).fit()
        assert record["weights"] == str(ckpt)


class TestTensorBoardTracking:
    def test_torchvision_loop_logs_per_epoch_scalars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        calls: list[tuple[str, float, int]] = []

        class FakeWriter:
            def __init__(self, log_dir: str) -> None:
                self.log_dir = log_dir

            def add_scalar(self, tag: str, value: float, step: int) -> None:
                calls.append((tag, value, step))

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        fake_mod = types.ModuleType("torch.utils.tensorboard")
        fake_mod.SummaryWriter = FakeWriter  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_mod)
        monkeypatch.setattr(
            dt_mod, "build_torchvision_detector", lambda *a, **k: _FakeDetector()
        )

        DetectionTrainer(_tv_config(tmp_path)).fit()

        tags = {tag for tag, _, _ in calls}
        assert {"loss/train", "loss/val", "map50/val"} <= tags
        steps = sorted({step for _, _, step in calls})
        assert steps == [1, 2]  # one entry per configured epoch


class TestAugmentationToggle:
    """Ultralytics owns its pipeline; the only way off is neutral values."""

    def test_off_sends_neutral_values_and_leaves_the_config_alone(self) -> None:
        from visionforge.core.detection_trainer import _augmentation_kwargs
        from visionforge.utils.detection_config import DetectionAugmentationConfig

        aug = DetectionAugmentationConfig(
            augment=False, mosaic=1.0, fliplr=0.5, hsv_h=0.015
        )

        kwargs = _augmentation_kwargs(aug)

        assert kwargs["mosaic"] == 0.0
        assert kwargs["fliplr"] == 0.0
        assert kwargs["hsv_h"] == 0.0
        assert kwargs["auto_augment"] is None
        # Turning it back on has to restore the tuning, so nothing was zeroed.
        assert aug.mosaic == 1.0
        assert aug.fliplr == 0.5

    def test_off_still_sends_every_key(self) -> None:
        """An omitted argument gets Ultralytics' own default, which augments."""
        from visionforge.core.detection_trainer import (
            _NEUTRAL_AUGMENTATION,
            _augmentation_kwargs,
        )
        from visionforge.utils.detection_config import DetectionAugmentationConfig

        kwargs = _augmentation_kwargs(DetectionAugmentationConfig(augment=False))

        assert set(kwargs) == set(_NEUTRAL_AUGMENTATION)

    def test_on_passes_the_configured_values(self) -> None:
        from visionforge.core.detection_trainer import _augmentation_kwargs
        from visionforge.utils.detection_config import DetectionAugmentationConfig

        kwargs = _augmentation_kwargs(DetectionAugmentationConfig(mosaic=0.8))

        assert kwargs["mosaic"] == 0.8
        assert kwargs["auto_augment"] == "randaugment"

    def test_defaults_to_on(self) -> None:
        from visionforge.utils.detection_config import DetectionAugmentationConfig

        assert DetectionAugmentationConfig().augment is True
