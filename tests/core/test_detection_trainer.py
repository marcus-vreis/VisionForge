from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from visionforge.core import detection_trainer as dt_mod
from visionforge.core.detection_trainer import DetectionTrainer
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
            "val/box_loss": 1.0 - 0.1 * epoch,
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
            for e in range(int(kwargs["epochs"])):
                ft = _FakeUTrainer(e, int(kwargs["epochs"]))
                for fn in self._callbacks.get("on_fit_epoch_end", []):
                    fn(ft)
            return object()

    return _FakeYOLO


# ── backend guards ────────────────────────────────────────────────────────────


class TestBackendGuards:
    def test_torchvision_backend_not_implemented(self, tmp_path: Path) -> None:
        cfg = _config(
            tmp_path,
            {"backend": "torchvision", "name": "fasterrcnn_resnet50_fpn"},
        )
        with pytest.raises(NotImplementedError, match="torchvision"):
            DetectionTrainer(cfg).fit()

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

    def test_best_epoch_selected_by_map(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dt_mod, "YOLO", _make_fake_yolo({}))
        result = DetectionTrainer(_config(tmp_path)).fit()
        assert result.best_epoch == 2
        assert result.best_map50_95 == pytest.approx(0.4)


# ── weight resolution ─────────────────────────────────────────────────────────


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
