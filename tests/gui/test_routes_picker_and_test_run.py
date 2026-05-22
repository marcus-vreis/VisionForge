"""Coverage for ``_open_native_folder_dialog`` and ``_execute_run_test``.

These two routes.py helpers carry the heaviest uncovered code in PR #32. The
folder picker uses tkinter; the test runner loads a saved checkpoint and runs a
fresh Evaluator pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from visionforge.gui.api.routes import (
    _execute_run_test,
    _open_native_folder_dialog,
)
from visionforge.gui.api.schemas import RunTestRequest

# ── dataset + checkpoint helpers ────────────────────────────────────────────


class _TinyBinaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def _make_dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        for cls in ("class_a", "class_b"):
            folder = root / split / cls
            folder.mkdir(parents=True)
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(folder / "image.png")
    return root


def _make_run_dir(tmp: Path, dataset_root: Path) -> Path:
    """Create a fake completed-run directory: best_model.pth + run.json."""
    run_dir = tmp / "models" / "exp1" / "20260522_120000_000000"
    run_dir.mkdir(parents=True)

    # Save a state_dict that matches ModelFactory("resnet18", num_classes=1).
    from visionforge.models.factory import ModelFactory
    from visionforge.utils.config import ModelConfig

    real_model = ModelFactory.create(
        ModelConfig(name="resnet18", num_classes=1, pretrained=False)
    )
    ckpt_path = run_dir / "best_model.pth"
    torch.save(real_model.state_dict(), ckpt_path)

    run_json = {
        "id": "exp1_20260522_120000_000000",
        "experiment": "exp1",
        "timestamp": "2026-05-22T12:00:00",
        "status": "completed",
        "device_used": "cpu",
        "run_dir": str(run_dir.resolve()),
        "config": {
            "name": "exp1",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "seed": 0,
            },
            "data": {
                "base_dir": str(dataset_root),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32},
            },
            "output": {
                "models_dir": str(tmp / "models"),
                "graphics_dir": str(tmp / "graphics"),
                "logs_dir": str(tmp / "logs"),
                "reports_dir": str(tmp / "reports"),
            },
            "device": {"kind": "cpu"},
        },
        "metrics": {
            "best_val_loss": 0.5,
            "best_epoch": 1,
            "total_epochs": 1,
        },
        "history": [],
        "artifacts": {"model": str(ckpt_path), "graphics": []},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return run_dir


# ── _open_native_folder_dialog ───────────────────────────────────────────────


class TestOpenNativeFolderDialog:
    def test_returns_resolved_path_on_success(self, tmp_path: Path) -> None:
        with (
            patch("tkinter.Tk"),
            patch("tkinter.filedialog.askdirectory", return_value=str(tmp_path)),
        ):
            resp = _open_native_folder_dialog()
        assert resp.cancelled is False
        assert Path(resp.path) == tmp_path.resolve()

    def test_returns_cancelled_when_user_dismisses(self) -> None:
        with (
            patch("tkinter.Tk"),
            patch("tkinter.filedialog.askdirectory", return_value=""),
        ):
            resp = _open_native_folder_dialog()
        assert resp.cancelled is True
        assert resp.path == ""
        assert resp.message  # human message present

    def test_returns_cancelled_when_tk_raises(self) -> None:
        with patch("tkinter.Tk", side_effect=RuntimeError("no display")):
            resp = _open_native_folder_dialog()
        assert resp.cancelled is True
        assert "Falha" in (resp.message or "")

    def test_returns_cancelled_when_tkinter_not_installed(self) -> None:
        """Drop ``tkinter`` from sys.modules and block re-import."""
        # Stash the real tkinter so we can restore it after the test.
        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "tkinter" or name.startswith("tkinter.")
        }
        try:
            with patch.dict(sys.modules, {"tkinter": None}):  # noqa: PTH118
                resp = _open_native_folder_dialog()
            assert resp.cancelled is True
            assert "tkinter" in (resp.message or "").lower()
        finally:
            sys.modules.update(saved)


# ── _execute_run_test ────────────────────────────────────────────────────────


@pytest.fixture
def configured_run(tmp_path: Path) -> tuple[Path, Path]:
    """Return (run_dir, dataset_root) for a completed-looking run."""
    dataset_root = _make_dataset(tmp_path / "ds")
    run_dir = _make_run_dir(tmp_path, dataset_root)
    return run_dir, dataset_root


class TestExecuteRunTest:
    def test_appends_test_record_to_run_json(
        self, configured_run: tuple[Path, Path]
    ) -> None:
        run_dir, dataset_root = configured_run
        req = RunTestRequest(
            base_dir=str(dataset_root),
            train_dir="train",
            val_dir="val",
            test_dir="test",
            label="smoke",
        )
        resp = _execute_run_test(run_dir, req)
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert len(data["tests"]) == 1
        rec = data["tests"][0]
        assert rec["label"] == "smoke"
        assert rec["test_id"] == resp.test_id
        assert Path(rec["base_dir"]) == dataset_root.resolve()
        assert "accuracy" in rec["metrics"]
        assert Path(rec["artifacts"]["confusion_matrix"]).exists()
        # Per-test plot set must mirror the training-time evaluation outputs.
        assert Path(rec["artifacts"]["confusion_matrix_normalized"]).exists()
        if "roc_curve" in rec["artifacts"]:
            assert Path(rec["artifacts"]["roc_curve"]).exists()
        if "precision_recall_curve" in rec["artifacts"]:
            assert Path(rec["artifacts"]["precision_recall_curve"]).exists()

    def test_label_defaults_to_dataset_name_when_omitted(
        self, configured_run: tuple[Path, Path]
    ) -> None:
        run_dir, dataset_root = configured_run
        req = RunTestRequest(base_dir=str(dataset_root))
        resp = _execute_run_test(run_dir, req)
        # Falls back to dataset folder name.
        assert resp.label == dataset_root.name

    def test_multiple_calls_accumulate_history(
        self, configured_run: tuple[Path, Path]
    ) -> None:
        run_dir, dataset_root = configured_run
        for label in ("a", "b", "c"):
            _execute_run_test(
                run_dir,
                RunTestRequest(base_dir=str(dataset_root), label=label),
            )
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        labels = [t["label"] for t in data["tests"]]
        assert labels == ["a", "b", "c"]


# ── /api/runs/{id}/test endpoint integration ─────────────────────────────────


class TestRunTestEndpoint:
    def test_404_on_unknown_run(self, tmp_path: Path) -> None:
        from visionforge.gui.server import app

        with patch("visionforge.gui.api.routes._MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                "/api/runs/does_not_exist/test",
                json={"base_dir": str(tmp_path)},
            )
        assert resp.status_code == 404

    def test_400_when_dataset_layout_invalid(
        self, tmp_path: Path, configured_run: tuple[Path, Path]
    ) -> None:
        """An ExperimentConfig validation error must surface as HTTP 400."""
        from visionforge.gui.server import app

        run_dir, _ = configured_run
        models_dir = run_dir.parent.parent  # tmp / models
        with patch("visionforge.gui.api.routes._MODELS_DIR", models_dir):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post(
                f"/api/runs/{run_dir.name}/test",
                json={"base_dir": str(tmp_path / "nonexistent")},
            )
        assert resp.status_code == 400


# ── /api/dataset/pick endpoint integration ───────────────────────────────────


class TestDatasetPickEndpoint:
    def test_returns_path_when_dialog_succeeds(self, tmp_path: Path) -> None:
        from visionforge.gui.server import app

        with (
            patch("tkinter.Tk"),
            patch("tkinter.filedialog.askdirectory", return_value=str(tmp_path)),
        ):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/dataset/pick")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cancelled"] is False
        assert Path(body["path"]) == tmp_path.resolve()
