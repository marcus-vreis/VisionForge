"""Post-training actions behave the same way across tasks (ADR-077).

Every one of these was found by exercising the actions against real trained runs
rather than by reading the code: the model card 500'd for every task except
classification, and a researcher-defined task crashed all four actions with a raw
ResNet key mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionforge.gui.api.routes import (
    _gradcam_class_names,
    _reject_custom_task_action,
    _render_run_markdown,
)


def _run_json(tmp_path: Path, payload: dict) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


class TestModelCardAcrossTasks:
    """The markdown export used to format every history cell with ':.4f'."""

    def test_segmentation_history_renders_its_own_columns(self) -> None:
        data = {
            "experiment": "seg",
            "metrics": {"test_miou": 0.55},
            "history": [
                {"epoch": 1, "train_loss": 0.65, "val_loss": 0.58, "val_miou": 0.58}
            ],
            "artifacts": {},
            "config": {},
        }

        card = _render_run_markdown(Path("."), data)

        assert "val_miou" in card
        # The classification columns must not be invented for a task without them.
        assert "val_accuracy" not in card

    def test_a_missing_cell_does_not_raise(self) -> None:
        """The old ':.4f' on the '?' fallback is what made this a 500."""
        data = {
            "experiment": "ragged",
            "metrics": {},
            "history": [
                {"epoch": 1, "train_loss": 0.5, "val_auroc": 0.7},
                {"epoch": 2, "train_loss": 0.4},  # no val_auroc this epoch
            ],
            "artifacts": {},
            "config": {},
        }

        card = _render_run_markdown(Path("."), data)

        assert "val_auroc" in card
        assert "—" in card  # the absent cell is shown as missing, not crashed on

    def test_non_numeric_history_values_survive(self) -> None:
        data = {
            "experiment": "custom",
            "metrics": {},
            "history": [{"epoch": 1, "note": "warmup", "train_loss": 0.9}],
            "artifacts": {},
            "config": {},
        }

        assert "warmup" in _render_run_markdown(Path("."), data)


class TestCustomTaskActionsAreRefused:
    def test_custom_run_is_rejected_by_name(self, tmp_path: Path) -> None:
        run_dir = _run_json(tmp_path, {"task": "custom:vlm_pseudo_label"})

        with pytest.raises(Exception) as excinfo:  # HTTPException
            _reject_custom_task_action(run_dir, "export_onnx")

        message = str(getattr(excinfo.value, "detail", excinfo.value))
        assert "vlm_pseudo_label" in message
        assert "export_onnx" in message

    def test_builtin_run_passes_through(self, tmp_path: Path) -> None:
        run_dir = _run_json(tmp_path, {"task": "multiclass"})

        _reject_custom_task_action(run_dir, "test")  # must not raise

    def test_missing_run_json_is_not_an_error_here(self, tmp_path: Path) -> None:
        """The endpoints already report a missing run; this guard stays quiet."""
        _reject_custom_task_action(tmp_path, "gradcam")


class TestGradcamClassNames:
    """Grad-CAM captions need the index -> name mapping ImageFolder implies."""

    @staticmethod
    def _dataset(tmp_path: Path, classes: list[str]) -> Path:
        for name in classes:
            (tmp_path / "train" / name).mkdir(parents=True)
        return tmp_path

    def test_names_come_from_sorted_training_subdirs(self, tmp_path: Path) -> None:
        root = self._dataset(tmp_path, ["peaberry", "defect", "premium"])
        data = {
            "config": {
                "task": "multiclass",
                "data": {"base_dir": str(root), "train_dir": "train"},
                "model": {"num_classes": 3},
            }
        }

        # Sorted, because that is the order ImageFolder assigns indices in.
        assert _gradcam_class_names(data) == ["defect", "peaberry", "premium"]

    def test_binary_run_with_one_output_unit_still_maps(self, tmp_path: Path) -> None:
        root = self._dataset(tmp_path, ["cat", "dog"])
        data = {
            "config": {
                "task": "binary",
                "data": {"base_dir": str(root), "train_dir": "train"},
                "model": {"num_classes": 1},
            }
        }

        assert _gradcam_class_names(data) == ["cat", "dog"]

    def test_a_count_mismatch_yields_no_names(self, tmp_path: Path) -> None:
        """Better an index than a confidently wrong class name."""
        root = self._dataset(tmp_path, ["a", "b"])
        data = {
            "config": {
                "task": "multiclass",
                "data": {"base_dir": str(root), "train_dir": "train"},
                "model": {"num_classes": 5},
            }
        }

        assert _gradcam_class_names(data) == []

    def test_non_classification_task_has_no_class_names(self, tmp_path: Path) -> None:
        root = self._dataset(tmp_path, ["a", "b"])
        data = {
            "config": {
                "task": "regression",
                "data": {"base_dir": str(root), "train_dir": "train"},
            }
        }

        assert _gradcam_class_names(data) == []

    def test_a_moved_dataset_yields_no_names(self) -> None:
        data = {
            "config": {
                "task": "multiclass",
                "data": {"base_dir": "/gone", "train_dir": "train"},
                "model": {"num_classes": 2},
            }
        }

        assert _gradcam_class_names(data) == []


class TestPreprocessingStepErrors:
    """An unknown filter is bad input, not a server fault (ADR-078)."""

    def test_unknown_step_names_the_available_ones(self) -> None:
        from PIL import Image

        from visionforge.core.preprocessing import apply_step

        with pytest.raises(ValueError) as excinfo:
            apply_step(Image.new("RGB", (8, 8)), "blur")

        message = str(excinfo.value)
        assert "'blur'" in message
        # "blur" is the natural guess; the message has to point at the real names.
        assert "gaussian_blur" in message
        assert "median_blur" in message

    def test_a_registered_step_still_applies(self) -> None:
        from PIL import Image

        from visionforge.core.preprocessing import apply_step

        out = apply_step(Image.new("RGB", (8, 8)), "grayscale")

        assert out.size == (8, 8)


class TestOneFolderTestDispatch:
    """Per-model test takes one labelled folder and dispatches per task (ADR-080)."""

    def test_task_is_read_from_the_run(self, tmp_path: Path) -> None:
        from visionforge.gui.api.routes import _run_task

        assert _run_task({"config": {"task": "segmentation"}}) == "segmentation"
        assert _run_task({"config": {"task": "multiclass"}}) == "classification"
        assert _run_task({"config": {"task": "binary"}}) == "classification"
        assert _run_task({"task": "custom:counting"}) == "custom:counting"
        # A run with nothing recorded is the original classification path.
        assert _run_task({}) == "classification"

    def test_missing_folder_is_rejected_before_loading_a_model(
        self, tmp_path: Path
    ) -> None:
        from visionforge.gui.api.routes import _evaluate_standalone_run
        from visionforge.gui.api.schemas import RunTestRequest

        run_dir = _run_json(tmp_path, {"config": {"task": "segmentation"}})

        with pytest.raises(ValueError, match="não encontrado"):
            _evaluate_standalone_run(
                run_dir,
                RunTestRequest(data_dir=str(tmp_path / "gone")),
                {"config": {"task": "segmentation"}},
                "segmentation",
            )

    def test_regression_refuses_a_folder(self, tmp_path: Path) -> None:
        """Its data model is a manifest, so the chosen path must be the .csv."""
        from visionforge.gui.api.routes import _evaluate_standalone_run
        from visionforge.gui.api.schemas import RunTestRequest

        folder = tmp_path / "some_split"
        folder.mkdir()
        run_dir = _run_json(tmp_path, {"config": {"task": "regression"}})

        with pytest.raises(ValueError, match=r"\.csv"):
            _evaluate_standalone_run(
                run_dir,
                RunTestRequest(data_dir=str(folder)),
                {"config": {"task": "regression"}},
                "regression",
            )

    def test_a_run_without_a_checkpoint_says_so(self, tmp_path: Path) -> None:
        from visionforge.gui.api.routes import _evaluate_standalone_run
        from visionforge.gui.api.schemas import RunTestRequest

        folder = tmp_path / "split"
        folder.mkdir()
        payload = {"config": {"task": "segmentation"}, "artifacts": {"model": None}}
        run_dir = _run_json(tmp_path, payload)

        with pytest.raises(FileNotFoundError, match="checkpoint"):
            _evaluate_standalone_run(
                run_dir, RunTestRequest(data_dir=str(folder)), payload, "segmentation"
            )
