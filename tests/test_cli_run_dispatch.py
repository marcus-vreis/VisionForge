from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visionforge.__main__ import build_task_block


def _write(tmp: Path, raw: dict) -> Path:
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def _cls_dataset(tmp: Path) -> str:
    for split in ("train", "val", "test"):
        for cls in ("a", "b"):
            (tmp / split / cls).mkdir(parents=True, exist_ok=True)
    return str(tmp)


class TestBuildTaskBlock:
    def test_classification_dispatches_to_classification_block(
        self, tmp_path: Path
    ) -> None:
        ds = _cls_dataset(tmp_path)
        cfg_path = _write(
            tmp_path,
            {
                "name": "c",
                "task": "multiclass",
                "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
                "data": {"base_dir": ds, "num_workers": 0},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "ClassificationBlock"

    def test_grid_search_block_still_dispatches(self, tmp_path: Path) -> None:
        ds = _cls_dataset(tmp_path)
        cfg_path = _write(
            tmp_path,
            {
                "name": "g",
                "task": "multiclass",
                "block": "grid_search",
                "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
                "data": {"base_dir": ds, "num_workers": 0},
                "grid_search": {"hyperparameters": {"training.learning_rate": [0.1]}},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "GridSearchBlock"

    def test_detection_dispatches_to_detection_block(self, tmp_path: Path) -> None:
        cfg_path = _write(
            tmp_path,
            {
                "name": "d",
                "task": "detection",
                "model": {"backend": "ultralytics", "name": "yolo11n"},
                "data": {"base_dir": str(tmp_path)},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "DetectionBlock"

    def test_regression_dispatches_to_regression_block(self, tmp_path: Path) -> None:
        cfg_path = _write(
            tmp_path,
            {
                "name": "r",
                "task": "regression",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {"base_dir": str(tmp_path), "target_columns": ["target"]},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "RegressionBlock"

    def test_segmentation_dispatches_to_segmentation_block(
        self, tmp_path: Path
    ) -> None:
        cfg_path = _write(
            tmp_path,
            {
                "name": "s",
                "task": "segmentation",
                "model": {"name": "fcn_resnet50", "num_classes": 3},
                "data": {"base_dir": str(tmp_path)},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "SegmentationBlock"

    def test_anomaly_dispatches_to_anomaly_block(self, tmp_path: Path) -> None:
        cfg_path = _write(
            tmp_path,
            {
                "name": "a",
                "task": "anomaly",
                "model": {"name": "autoencoder"},
                "data": {"base_dir": str(tmp_path)},
            },
        )
        _config, block = build_task_block(cfg_path)
        assert type(block).__name__ == "AnomalyBlock"

    def test_unknown_task_raises(self, tmp_path: Path) -> None:
        cfg_path = _write(tmp_path, {"name": "x", "task": "teleportation"})
        with pytest.raises(ValueError, match="Unknown task"):
            build_task_block(cfg_path)


class TestCommandLineLanguage:
    """``VISIONFORGE_LANG`` is bound for the whole of a ``visionforge run`` (ADR-116)."""

    @staticmethod
    def _run_cli(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env: str | None
    ) -> dict[str, object]:
        import contextvars
        from types import SimpleNamespace

        from matplotlib.figure import Figure

        import visionforge.__main__ as cli
        import visionforge.utils.logger as logger_mod
        from visionforge.core.plotter import MetricsPlotter
        from visionforge.utils.messages import bound_lang, current_lang, tr

        seen: dict[str, object] = {}
        monkeypatch.setattr(
            Figure,
            "savefig",
            lambda self, *_a, **_k: seen.update(title=self.axes[0].get_title()),
        )

        class FakeBlock:
            def run(self) -> None:
                seen["bound"] = bound_lang()
                seen["message"] = tr("scan.split_missing", split="val")
                MetricsPlotter.metric_curve(
                    [1], [0.5], tmp_path / "curve.png", label="AUROC"
                )
                seen["messages_in"] = current_lang()

            def report(self) -> str:
                return "report"

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(logger_mod, "setup_logger", lambda *a, **k: None)
        monkeypatch.setattr(
            cli,
            "build_task_block",
            lambda _path: (SimpleNamespace(name="x", task="t"), FakeBlock()),
        )
        monkeypatch.setattr("sys.argv", ["visionforge", "run", "cfg.yaml"])
        if env is None:
            monkeypatch.delenv("VISIONFORGE_LANG", raising=False)
        else:
            monkeypatch.setenv("VISIONFORGE_LANG", env)

        # The command binds a language in the context it runs in; a copy keeps it
        # from leaking into the tests that follow.
        contextvars.copy_context().run(cli.main)
        return seen

    def test_without_the_variable_figures_are_english_and_messages_portuguese(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = self._run_cli(monkeypatch, tmp_path, None)

        assert seen["bound"] is None
        assert seen["title"] == "AUROC over epochs"
        assert seen["message"] == "Split 'val' não encontrado."

    def test_portuguese_figures_on_request(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = self._run_cli(monkeypatch, tmp_path, "pt")

        assert seen["bound"] == "pt"
        assert seen["title"] == "AUROC por época"

    def test_english_binds_the_messages_too(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = self._run_cli(monkeypatch, tmp_path, "en")

        assert seen["bound"] == "en"
        assert seen["title"] == "AUROC over epochs"
        assert seen["message"] == "Split 'val' not found."

    def test_an_unknown_value_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = self._run_cli(monkeypatch, tmp_path, "klingon")

        assert seen["bound"] is None
        assert seen["title"] == "AUROC over epochs"
