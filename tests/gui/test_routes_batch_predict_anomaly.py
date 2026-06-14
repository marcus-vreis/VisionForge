from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image


def _write_image(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=(70, 90, 110)).save(path)


def _make_anomaly_run(tmp_path: Path) -> Path:
    """Build a real autoencoder run.json + checkpoint on disk."""
    from visionforge.models.anomaly_factory import ConvAutoencoder

    base_dir = tmp_path / "ds"
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dir = tmp_path / "models" / "anom1" / "20260601_120000_000000"
    ckpt = run_dir / "best_model.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ConvAutoencoder(latent_dim=16).state_dict(), ckpt)

    config = {
        "name": "anom1",
        "task": "anomaly",
        "model": {"name": "autoencoder", "latent_dim": 16},
        "data": {"base_dir": str(base_dir), "image_size": 32},
        "training": {"epochs": 1},
    }
    run_json = {
        "id": "anom1_20260601_120000_000000",
        "experiment": "anom1",
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": config,
        "metrics": {"test_auroc": 0.9, "test_threshold": 0.05, "test_image_f1": 0.8},
        "history": [],
        "artifacts": {"model": str(ckpt.resolve()), "graphics": []},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return run_dir


class TestAnomalyBatchPredict:
    def test_writes_score_and_decision_csv(self, tmp_path: Path) -> None:
        from visionforge.gui.api.routes import _execute_batch_predict
        from visionforge.gui.api.schemas import BatchPredictRequest

        run_dir = _make_anomaly_run(tmp_path)
        inbox = tmp_path / "inbox"
        _write_image(inbox / "a.png")
        _write_image(inbox / "b.png")

        resp = _execute_batch_predict(
            run_dir, BatchPredictRequest(input_dir=str(inbox), recursive=False)
        )

        assert resp.total_processed == 2
        assert resp.failed_count == 0

        out = Path(resp.output_csv)
        assert out.exists()
        with out.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["filename", "anomaly_score", "is_anomaly"]
            rows = list(reader)
        assert len(rows) == 2
        for row in rows:
            float(row["anomaly_score"])  # parseable score
            assert row["is_anomaly"] in {"0", "1"}  # threshold decision
