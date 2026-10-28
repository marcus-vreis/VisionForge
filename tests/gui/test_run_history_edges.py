"""Behavioral edges retained while extracting API domain helpers."""

from __future__ import annotations

from pathlib import Path

from visionforge.gui.api.run_history import parse_run_summary, summary_metrics


def test_summary_preserves_custom_metric_order_and_invalid_values() -> None:
    """Custom metrics use the first three convertible values in insertion order."""
    metrics = {
        "total_epochs": 3,
        "bad": "n/a",
        "loss": "0.5",
        "best_epoch": 2,
        "accuracy": 0.0,
        "f1": 0.8,
        "unused": 0.9,
    }
    assert summary_metrics("custom:foo", metrics) == {
        "loss": 0.5,
        "accuracy": 0.0,
        "f1": 0.8,
    }


def test_summary_preserves_legacy_block_and_fingerprint_rules(tmp_path: Path) -> None:
    """A missing digest hides the fingerprint method and old runs infer block."""
    data = {
        "experiment": "legacy",
        "timestamp": "2026-08-05T10:00:00",
        "status": "completed",
        "config": {
            "task": "segmentation",
            "data": {"preprocessing": {"steps": [1, 2]}},
        },
        "metrics": {"total_epochs": 2, "val_miou": 0.4},
        "dataset_fingerprint": {"method": "manifest", "digest": ""},
    }
    summary = parse_run_summary(tmp_path / "20260805_090000", data)
    assert summary.block == "segmentation"
    assert summary.preprocessing_count == 2
    assert summary.final_metrics == {"val_miou": 0.4}
    assert summary.dataset_digest is None
    assert summary.dataset_method is None
    assert summary.started_at.isoformat() == "2026-08-05T09:00:00"
