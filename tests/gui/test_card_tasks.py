"""The model card reads each task's own config, not the classification shape.

Regression, anomaly, segmentation and custom runs have no ``model.num_classes``
or ``model.pretrained``, and anomaly/segmentation resize to ``data.image_size``
(the ``data.transforms.image_size`` they also carry is never used). The card
used to print ``?`` for the missing fields and the unused image size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from visionforge.gui.api.routes import _render_run_markdown
from visionforge.utils.messages import using_lang

_TRAINING = {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "epochs": 5,
    "batch_size": 4,
    "seed": 0,
}


def _run(config: dict[str, Any], **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "exp_20260101_000000",
        "experiment": "exp",
        "timestamp": "2026-01-01T00:00:00",
        "status": "completed",
        "config": config,
        "metrics": {"best_epoch": 1},
        "history": [],
        "artifacts": {},
    }
    data.update(extra)
    return data


def _card(lang: str, data: dict[str, Any], run_dir: Path = Path(".")) -> str:
    with using_lang(lang):  # type: ignore[arg-type]
        return _render_run_markdown(run_dir, data)


_ANOMALY = {
    "task": "anomaly",
    "model": {"name": "autoencoder", "pretrained": True},
    "training": _TRAINING,
    "data": {"image_size": 128, "transforms": {"image_size": 224, "augment": True}},
}
_SEGMENTATION = {
    "task": "segmentation",
    "model": {"name": "unet", "num_classes": 3, "pretrained": False},
    "training": _TRAINING,
    "data": {"image_size": 64, "transforms": {"image_size": 224, "augment": True}},
}
_REGRESSION = {
    "task": "regression",
    "model": {"name": "resnet18", "num_targets": 2, "pretrained": False},
    "training": _TRAINING,
    "data": {"transforms": {"image_size": 224}},
}
_CUSTOM = {
    "name": "exp",
    "training": {"epochs": 3, "batch_size": 2, "learning_rate": 0.01},
    "data": {"image_size": 96, "transforms": {"image_size": 224}},
}


class TestImageSize:
    @pytest.mark.parametrize("cfg,size", [(_ANOMALY, 128), (_SEGMENTATION, 64)])
    def test_card_shows_the_size_the_training_used(
        self, cfg: dict[str, Any], size: int
    ) -> None:
        card = _card("en", _run(cfg))

        assert f"| image_size | `{size}` |" in card
        assert "| image_size | `224` |" not in card

    def test_classification_keeps_its_transform_size(self) -> None:
        cfg = {
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": _TRAINING,
            "data": {"transforms": {"image_size": 224}},
        }

        assert "| image_size | `224` |" in _card("en", _run(cfg))


class TestNoPlaceholders:
    @pytest.mark.parametrize("cfg", [_ANOMALY, _SEGMENTATION, _REGRESSION, _CUSTOM])
    @pytest.mark.parametrize("lang", ["pt", "en"])
    def test_no_question_mark_value(self, cfg: dict[str, Any], lang: str) -> None:
        data = (
            _run(cfg, task="custom:thing", task_label="Thing")
            if cfg is _CUSTOM
            else _run(cfg)
        )

        card = _card(lang, data)

        assert ":** ?" not in card

    def test_regression_names_its_targets_not_classes(self) -> None:
        card = _card("en", _run(_REGRESSION))

        assert "- **num_targets:** 2" in card
        assert "num_classes" not in card

    def test_anomaly_has_no_class_count_or_missing_fields(self) -> None:
        card = _card("en", _run(_ANOMALY))

        assert "num_classes" not in card
        assert "- **Architecture:** autoencoder" in card

    def test_custom_uses_its_label_as_architecture_and_task(self) -> None:
        data = _run(_CUSTOM, task="custom:thing", task_label="Thing")

        card = _card("en", data)

        assert "- **Architecture:** Thing" in card
        assert "- **Task:** Thing" in card
        assert "Pretrained" not in card
        assert "Seed" not in card


class TestStatusAndNone:
    def test_status_is_translated(self) -> None:
        cfg = _SEGMENTATION
        assert "- **Situação:** concluída" in _card("pt", _run(cfg))
        assert "- **Status:** completed" in _card("en", _run(cfg))

    def test_unknown_status_is_kept_verbatim(self) -> None:
        assert "- **Status:** weird" in _card("en", _run(_SEGMENTATION, status="weird"))

    def test_none_metric_is_a_dash(self) -> None:
        data = _run(_SEGMENTATION, metrics={"best_val_miou": None, "best_epoch": 1})

        card = _card("pt", data)

        assert "| best_val_miou | — |" in card
        assert "None" not in card


class TestRunId:
    def test_card_uses_the_history_id(self) -> None:
        run_dir = Path("outputs/models/exp/20260101_000000")

        card = _card("en", _run(_SEGMENTATION), run_dir)

        assert "- **Run ID:** `20260101_000000`" in card
        assert "exp_20260101_000000" not in card
