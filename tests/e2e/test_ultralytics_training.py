"""The Ultralytics backend, trained for real (no mock in sight).

Every other detection test hands `DetectionTrainer` a fake YOLO class. That is
the right shape for testing *our* argument translation, and it proves nothing
about the integration itself — which is the default backend, and the one whose
failures have been about how Ultralytics resolves paths rather than about which
kwargs it receives (ADR-079: a relative `project` wrote the whole run under its
own `runs_dir`, leaving a run directory with no checkpoint).

Marked ``slow`` and skipped when the extra is missing, so the pre-commit suite
stays fast and a bare install still runs green. Run it with ``pytest -m slow``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.selftest_data import build_detection_dataset

ultralytics = pytest.importorskip("ultralytics", reason="detection extra not installed")

pytestmark = pytest.mark.slow


def _config(tmp_path: Path, weights: Path) -> DetectionConfig:
    base = build_detection_dataset(tmp_path / "ds", size=64, per_split=4)
    return DetectionConfig.model_validate(
        {
            "name": "ultra_smoke",
            "model": {
                "backend": "ultralytics",
                "name": "yolo11n",
                "num_classes": 1,
                # The checkpoint in the repo root, so the test never downloads.
                "weights_path": str(weights),
            },
            "data": {"base_dir": str(base), "image_size": 64},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.01,
                "workers": 0,
                "seed": 0,
            },
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )


@pytest.fixture(scope="module")
def weights() -> Path:
    """The repo's own yolo11n.pt; skip rather than download one."""
    path = Path(__file__).resolve().parents[2] / "yolo11n.pt"
    if not path.is_file():
        pytest.skip("yolo11n.pt is not in the repo root")
    return path


class TestUltralyticsTrainsForReal:
    def test_one_epoch_lands_where_the_run_directory_says(
        self, tmp_path: Path, weights: Path
    ) -> None:
        from visionforge.core.detection_trainer import DetectionTrainer

        events: list[dict[str, Any]] = []
        cfg = _config(tmp_path, weights)

        result = DetectionTrainer(cfg).fit(progress_callback=events.append)

        # ADR-079: Ultralytics resolves a *relative* project under its own
        # runs_dir, so this assertion is the one that catches the regression.
        assert result.run_dir.is_relative_to(tmp_path / "models")
        assert (result.run_dir / "weights" / "best.pt").is_file()
        assert (result.run_dir / "weights" / "last.pt").is_file()
        assert not (Path.cwd() / "runs").exists()

        # The epoch callback is our only window into its loop; without it the
        # history and every streamed event would be empty.
        assert [e["event"] for e in events][:2] == ["start", "epoch_end"]
        assert events[-1]["event"] == "end"
        assert len(result.history) == 1

        run_json = json.loads((result.run_dir / "run.json").read_text("utf-8"))
        assert run_json["status"] == "completed"
        assert len(run_json["history"]) == 1
        assert run_json["metrics"]["total_epochs"] == 1

    def test_a_stopped_run_can_be_continued_by_ultralytics(
        self, tmp_path: Path, weights: Path
    ) -> None:
        """ADR-093 hands it `last.pt` and `resume=True`; only a real run proves it."""
        from visionforge.core.cancellation import CancellationToken
        from visionforge.core.detection_trainer import DetectionTrainer

        cfg = _config(tmp_path, weights)
        cfg = cfg.model_copy(
            update={"training": cfg.training.model_copy(update={"epochs": 2})}
        )
        token = CancellationToken()

        def stop_after_first(event: dict[str, Any]) -> None:
            if event.get("event") == "epoch_end" and event.get("epoch") == 1:
                token.cancel()

        stopped = DetectionTrainer(cfg).fit(
            progress_callback=stop_after_first, cancel_token=token
        )
        assert stopped.total_epochs == 1

        finished = DetectionTrainer(cfg).fit(resume_dir=stopped.run_dir)

        assert finished.run_dir == stopped.run_dir
        assert [h.epoch for h in finished.history] == [1, 2]
