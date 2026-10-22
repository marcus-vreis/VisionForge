"""An explicit continuation must never become a new training run."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from visionforge.core.anomaly_trainer import AnomalyTrainer
from visionforge.core.detection_trainer import DetectionTrainer
from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.core.resume import (
    ResumeStateError,
    load_resume_state,
    require_resume_state,
)
from visionforge.core.segmentation_trainer import SegmentationTrainer
from visionforge.core.trainer import Trainer
from visionforge.models.anomaly_factory import PatchCore
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig


def test_corrupt_requested_resume_never_creates_a_new_run(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    run = tmp_path / "existing"
    run.mkdir()
    checkpoint = run / "resume.pt"
    checkpoint.write_bytes(b"broken checkpoint")
    cfg = ExperimentConfig.model_validate(
        {
            "name": "explicit_resume",
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "data": {"base_dir": str(dataset)},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="resume"):
        Trainer(cfg).fit(model, object(), resume_dir=run)

    assert checkpoint.read_bytes() == b"broken checkpoint"
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("kind", ["missing", "unreadable", "unsupported", "invalid"])
def test_required_state_distinguishes_failures_and_preserves_files(
    tmp_path: Path, kind: str
) -> None:
    checkpoint = tmp_path / "resume.pt"
    if kind == "unreadable":
        checkpoint.write_bytes(b"broken")
    elif kind == "unsupported":
        torch.save({"format_version": 999}, checkpoint)
    elif kind == "invalid":
        torch.save({"format_version": 1}, checkpoint)
    before = checkpoint.read_bytes() if checkpoint.exists() else None

    with pytest.raises(ResumeStateError) as error:
        require_resume_state(tmp_path)

    assert kind in error.value.reason
    assert error.value.path == checkpoint
    assert load_resume_state(tmp_path) is None
    assert (checkpoint.read_bytes() if checkpoint.exists() else None) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        [] if kind == "missing" else ["resume.pt"]
    )


@pytest.mark.parametrize("task", ["regression", "segmentation", "anomaly", "detection"])
@pytest.mark.parametrize("kind", ["missing", "unreadable"])
def test_native_trainers_refuse_unusable_resume_before_output_or_model_setup(
    tmp_path: Path, task: str, kind: str
) -> None:
    dataset = tmp_path / "data"
    dataset.mkdir()
    run = tmp_path / "existing"
    run.mkdir()
    if kind == "unreadable":
        (run / "resume.pt").write_bytes(b"broken")
    payload: dict[str, Any] = {
        "name": "must_not_restart",
        "data": {"base_dir": str(dataset)},
        "output": {"models_dir": str(tmp_path / "models")},
        "device": {"kind": "cpu"},
    }
    trainer: Any
    if task == "regression":
        payload["model"] = {"name": "resnet18", "pretrained": False}
        payload["data"]["target_columns"] = ["target"]
        trainer = RegressionTrainer(RegressionConfig.model_validate(payload))
    elif task == "segmentation":
        payload["model"] = {"name": "unet", "num_classes": 2, "pretrained": False}
        trainer = SegmentationTrainer(SegmentationConfig.model_validate(payload))
    elif task == "anomaly":
        payload["model"] = {"name": "autoencoder", "latent_dim": 8}
        trainer = AnomalyTrainer(AnomalyConfig.model_validate(payload))
    else:
        payload["model"] = {
            "backend": "torchvision",
            "name": "fasterrcnn_resnet50_fpn",
            "num_classes": 2,
            "pretrained": False,
        }
        trainer = DetectionTrainer(DetectionConfig.model_validate(payload))
    model = nn.Linear(4, 2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ResumeStateError, match=kind):
        if task == "detection":
            trainer.fit(resume_dir=run)
        else:
            trainer.fit(model, object(), resume_dir=run)
    assert not (tmp_path / "models").exists()
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )


def test_ultralytics_missing_last_checkpoint_never_starts_fresh(tmp_path: Path) -> None:
    payload = {
        "data": {"base_dir": str(tmp_path)},
        "model": {"backend": "ultralytics", "name": "yolo11n", "pretrained": False},
        "output": {"models_dir": str(tmp_path / "models")},
        "device": {"kind": "cpu"},
    }
    trainer = DetectionTrainer(DetectionConfig.model_validate(payload))
    with pytest.raises(ResumeStateError, match="missing"):
        trainer.fit(resume_dir=tmp_path / "missing")
    assert not (tmp_path / "models").exists()


def test_patchcore_rejects_explicit_continuation(tmp_path: Path) -> None:
    cfg = AnomalyConfig.model_validate(
        {
            "data": {"base_dir": str(tmp_path)},
            "model": {"name": "patchcore", "pretrained": False},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )
    model = PatchCore(pretrained=False)
    with pytest.raises(ValueError, match="Cannot resume PatchCore"):
        AnomalyTrainer(cfg).fit(model, object(), resume_dir=tmp_path)
    assert not (tmp_path / "models").exists()
    assert not model.is_fitted


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", []),
        ("optimizer", None),
        ("history", "epochs"),
        ("scheduler", []),
        ("scaler", 3),
        ("epoch", -1),
        ("epoch", float("inf")),
    ],
)
def test_structurally_invalid_state_is_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    payload = {
        "format_version": 1,
        "epoch": 1,
        "model": {},
        "optimizer": {},
        "scheduler": None,
        "scaler": None,
        "best_metric": 0.1,
        "best_epoch": 1,
        "patience_counter": 0,
        "history": [],
    }
    payload[field] = value
    torch.save(payload, tmp_path / "resume.pt")
    with pytest.raises(ResumeStateError, match="invalid"):
        require_resume_state(tmp_path)
    assert load_resume_state(tmp_path) is None
