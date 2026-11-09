from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from visionforge.utils.config import ExperimentConfig


def _kfold_run_json(models_dir: Path, name: str) -> Path:
    """The K-fold's group run.json, in ``<models>/<name>_cv/<group_id>/`` (ADR-123)."""
    found = sorted((Path(models_dir) / f"{name}_cv").glob("*/run.json"))
    assert len(found) == 1, found
    return found[0]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cv_pool(tmp_path: Path) -> Path:
    """Two-class ImageFolder pool: class_a=black (0), class_b=bright (200)."""
    for cls, fill in [("class_a", 0), ("class_b", 200)]:
        folder = tmp_path / cls
        folder.mkdir()
        for i in range(6):
            img = Image.fromarray(np.full((32, 32, 3), fill, dtype=np.uint8))
            img.save(folder / f"img{i}.png")
    return tmp_path


@pytest.fixture
def cv_config(cv_pool: Path, tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "cv_test",
            "task": "binary",
            "block": "cross_validation",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "early_stopping_patience": 1,
                "seed": 0,
            },
            "data": {
                "base_dir": str(cv_pool),
                "train_dir": "",
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32},
            },
            "output": {
                "models_dir": str(tmp_path / "models"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "cross_validation": {
                "n_folds": 3,
                "stratified": True,
                "shuffle": False,
                "fold_seed": 0,
            },
        }
    )


def _mock_report_success(self: Any) -> dict[str, Any]:  # noqa: ANN001
    return {
        "train": {"best_val_loss": 0.4, "best_epoch": 1, "total_epochs": 1},
        "eval": {
            "accuracy": 0.8,
            "f1": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "auc_roc": 0.85,
        },
    }


# ── setup ─────────────────────────────────────────────────────────────────────


class TestSetup:
    def test_raises_without_cv_config(self, cv_pool: Path, tmp_path: Path) -> None:
        """setup() must raise ValueError when cross_validation is None."""
        from visionforge.blocks.cross_validation import CrossValidationBlock

        config = ExperimentConfig.model_validate(
            {
                "name": "cv_test",
                "task": "binary",
                "block": "classification",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {"learning_rate": 0.001, "epochs": 1, "batch_size": 2},
                "data": {
                    "base_dir": str(cv_pool),
                    "train_dir": "",
                    "num_workers": 0,
                    "pin_memory": False,
                },
            }
        )
        block = CrossValidationBlock()
        with pytest.raises(ValueError, match="cross_validation"):
            block.setup(config)

    def test_valid_setup_does_not_raise(self, cv_config: ExperimentConfig) -> None:
        """setup() with valid config must not raise."""
        from visionforge.blocks.cross_validation import CrossValidationBlock

        block = CrossValidationBlock()
        block.setup(cv_config)


# ── fold splitting ────────────────────────────────────────────────────────────


class TestFoldSplitting:
    def _run_mocked(self, cv_config: ExperimentConfig) -> Any:
        from visionforge.blocks.cross_validation import CrossValidationBlock

        block = CrossValidationBlock()
        block.setup(cv_config)

        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.cross_validation.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.cross_validation.torch.load", return_value={}),
            patch(
                "visionforge.blocks.cross_validation.torch.nn.Module.load_state_dict"
            ),
        ):
            from unittest.mock import MagicMock

            mock_train_result = MagicMock()
            mock_train_result.stopped = False  # ran to its end (ADR-111)
            mock_train_result.total_epochs = 1
            mock_train_result.best_val_loss = 0.4
            mock_train_result.model_path = cv_config.output.models_dir / "dummy.pth"
            mock_trainer_cls.return_value.fit.return_value = mock_train_result

            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.8
            mock_eval_result.f1 = 0.75
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        return block

    def test_fold_count(self, cv_config: ExperimentConfig) -> None:
        """run() must produce exactly n_folds fold records."""
        block = self._run_mocked(cv_config)
        assert len(block._fold_results) == 3

    def test_train_val_sizes_sum_to_pool(self, cv_config: ExperimentConfig) -> None:
        """train_size + val_size must equal the full pool size for each fold."""
        block = self._run_mocked(cv_config)
        for rec in block._fold_results:
            assert rec["train_size"] + rec["val_size"] == 12

    def test_stratified_each_fold_has_both_classes(
        self, cv_config: ExperimentConfig
    ) -> None:
        """With stratified=True, each val fold must contain samples from both classes."""
        import numpy as np
        import torchvision.transforms as T
        from sklearn.model_selection import StratifiedKFold
        from torchvision.datasets import ImageFolder

        pool_dir = cv_config.data.base_dir / cv_config.data.train_dir
        ds = ImageFolder(str(pool_dir), transform=T.ToTensor())
        targets = np.array(ds.targets)

        cv = cv_config.cross_validation
        assert cv is not None
        splitter = StratifiedKFold(n_splits=cv.n_folds, shuffle=False)
        for _, val_idx in splitter.split(np.arange(len(ds)), targets):
            val_targets = targets[val_idx]
            assert len(np.unique(val_targets)) == 2


# ── normalization stats ───────────────────────────────────────────────────────


class TestNormalization:
    def test_stats_computed_from_train_indices_only(
        self, cv_config: ExperimentConfig
    ) -> None:
        """_compute_fold_stats must only see train-fold indices, not the full 12."""
        from visionforge.blocks.cross_validation import (
            CrossValidationBlock,
            _compute_fold_stats,
        )

        seen_lengths: list[int] = []
        original_fn = _compute_fold_stats

        def spy(dataset: Any, train_indices: list[int], num_workers: int) -> Any:  # noqa: ANN001
            seen_lengths.append(len(train_indices))
            return original_fn(dataset, train_indices, num_workers)

        block = CrossValidationBlock()
        block.setup(cv_config)

        with (
            patch(
                "visionforge.blocks.cross_validation._compute_fold_stats",
                side_effect=spy,
            ),
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.cross_validation.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.cross_validation.torch.load", return_value={}),
            patch(
                "visionforge.blocks.cross_validation.torch.nn.Module.load_state_dict"
            ),
        ):
            from unittest.mock import MagicMock

            mock_train_result = MagicMock()
            mock_train_result.stopped = False  # ran to its end (ADR-111)
            mock_train_result.total_epochs = 1
            mock_train_result.best_val_loss = 0.4
            mock_train_result.model_path = cv_config.output.models_dir / "dummy.pth"
            mock_trainer_cls.return_value.fit.return_value = mock_train_result
            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.8
            mock_eval_result.f1 = 0.75
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        # Every call must have seen fewer than 12 indices (the full pool).
        assert all(n < 12 for n in seen_lengths)
        assert len(seen_lengths) == 3

    def test_fold_mean_differs_for_different_class_distributions(
        self, tmp_path: Path
    ) -> None:
        """Per-fold mean must differ from global mean when classes have different intensities."""
        import torchvision.transforms as T
        from torchvision.datasets import ImageFolder

        from visionforge.blocks.cross_validation import _compute_fold_stats

        # class_a = black (0), class_b = white (255)
        for cls, fill in [("class_a", 0), ("class_b", 255)]:
            folder = tmp_path / cls
            folder.mkdir()
            for i in range(4):
                img = Image.fromarray(np.full((32, 32, 3), fill, dtype=np.uint8))
                img.save(folder / f"img{i}.png")

        ds = ImageFolder(str(tmp_path), transform=T.ToTensor())
        # train on class_a only (indices 0-3)
        mean, _ = _compute_fold_stats(ds, list(range(4)), 0)
        global_mean = [0.5, 0.5, 0.5]  # equal mix of 0 and 1 normalized
        # class_a is all zeros so fold mean should be ~0, not 0.5
        assert abs(mean[0] - global_mean[0]) > 0.1


# ── report and artifacts ──────────────────────────────────────────────────────


class TestReportAndArtifacts:
    def _run_block(self, cv_config: ExperimentConfig) -> Any:
        from unittest.mock import MagicMock

        from visionforge.blocks.cross_validation import CrossValidationBlock

        block = CrossValidationBlock()
        block.setup(cv_config)

        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.cross_validation.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.cross_validation.torch.load", return_value={}),
            patch(
                "visionforge.blocks.cross_validation.torch.nn.Module.load_state_dict"
            ),
        ):
            mock_train_result = MagicMock()
            mock_train_result.stopped = False  # ran to its end (ADR-111)
            mock_train_result.total_epochs = 1
            mock_train_result.best_val_loss = 0.4
            mock_train_result.model_path = cv_config.output.models_dir / "dummy.pth"
            mock_trainer_cls.return_value.fit.return_value = mock_train_result

            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.8
            mock_eval_result.f1 = 0.75
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        return block

    def test_report_aggregates_metrics(self, cv_config: ExperimentConfig) -> None:
        """report() must contain mean/std accuracy and f1."""
        block = self._run_block(cv_config)
        result = block.report()
        assert "mean_accuracy" in result
        assert "std_accuracy" in result
        assert "mean_f1" in result
        assert "std_f1" in result

    def test_cv_summary_json_written(self, cv_config: ExperimentConfig) -> None:
        """run() must write cv_summary.json with n_folds fold entries."""
        self._run_block(cv_config)
        summary_path = cv_config.output.reports_dir / cv_config.name / "cv_summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert len(data["folds"]) == 3
        assert data["n_folds"] == 3
        assert "aggregate" in data

    def test_report_fold_results_included(self, cv_config: ExperimentConfig) -> None:
        """report() must include fold_results list."""
        block = self._run_block(cv_config)
        result = block.report()
        assert "fold_results" in result
        assert len(result["fold_results"]) == 3

    def test_top_level_run_json_written(self, cv_config: ExperimentConfig) -> None:
        """run() must also write models_dir/{name}_cv/run.json so /api/runs
        lists the CV experiment in the history alongside classification runs."""
        self._run_block(cv_config)
        run_json_path = _kfold_run_json(cv_config.output.models_dir, cv_config.name)
        assert run_json_path.exists()
        data = json.loads(run_json_path.read_text(encoding="utf-8"))
        # Mirrors the keys RunSummary parser already expects.
        assert data["status"] == "completed"
        assert data["experiment"] == cv_config.name
        assert "timestamp" in data
        assert data["config"]["block"] == "cross_validation"
        # The folds score on held-out folds of train, so the means are
        # validation scores, never the test column (ADR-123).
        assert "val_accuracy" in data["metrics"]
        assert "val_f1" in data["metrics"]
        assert "test_accuracy" not in data["metrics"]
        # A History group whose children are the folds (tagging is covered in
        # tests/core/test_run_groups.py, with real fold runs on disk).
        assert data["group"]["kind"] == "kfold"
        assert [c["fold"] for c in data["group"]["children"]] == [0, 1, 2]
        assert "fold_results" in data["metrics"]
        assert len(data["metrics"]["fold_results"]) == 3
        agg = data["metrics"]["cv_aggregate"]
        assert agg["n_folds"] == 3
        assert agg["n_folds_ok"] == 3

    def test_top_level_run_json_inherits_provenance_from_a_fold(
        self, cv_config: ExperimentConfig
    ) -> None:
        """The K-fold's run.json names the device, environment and dataset of
        the folds it trained, as every other run does, instead of null/empty."""
        from unittest.mock import MagicMock

        from visionforge.blocks.cross_validation import CrossValidationBlock

        fingerprint = {"method": "content", "digest": "abc", "n_files": 12}
        environment = {"python": "3.13", "torch": "2.x"}
        calls: list[int] = []

        def fake_fit(*_args: Any, **_kwargs: Any) -> MagicMock:
            # What Trainer does: a run folder with its own run.json.
            run_dir = cv_config.output.models_dir / f"fold{len(calls)}" / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "device_used": "cuda:0",
                        "environment": environment,
                        "dataset_fingerprint": fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            calls.append(1)
            result = MagicMock()
            result.stopped = False
            result.total_epochs = 1
            result.best_val_loss = 0.4
            result.model_path = run_dir / "best_model.pth"
            return result

        block = CrossValidationBlock()
        block.setup(cv_config)
        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.cross_validation.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.cross_validation.torch.load", return_value={}),
            patch(
                "visionforge.blocks.cross_validation.torch.nn.Module.load_state_dict"
            ),
        ):
            mock_trainer_cls.return_value.fit.side_effect = fake_fit
            mock_eval_cls.return_value.evaluate.return_value = MagicMock(
                accuracy=0.8, f1=0.75
            )
            block.run()

        data = json.loads(
            (_kfold_run_json(cv_config.output.models_dir, cv_config.name)).read_text(
                encoding="utf-8"
            )
        )
        assert data["device_used"] == "cuda:0"
        assert data["environment"] == environment
        assert data["dataset_fingerprint"] == fingerprint
        assert all(f["run_dir"] for f in data["metrics"]["fold_results"])


# ── failure handling ──────────────────────────────────────────────────────────


class TestFailureHandling:
    def test_failed_fold_continues_sweep(self, cv_config: ExperimentConfig) -> None:
        """A failing fold must be recorded; remaining folds must complete."""
        from unittest.mock import MagicMock

        from visionforge.blocks.cross_validation import CrossValidationBlock

        call_count: list[int] = [0]

        block = CrossValidationBlock()
        block.setup(cv_config)

        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.cross_validation.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.cross_validation.torch.load", return_value={}),
            patch(
                "visionforge.blocks.cross_validation.torch.nn.Module.load_state_dict"
            ),
        ):
            mock_train_result = MagicMock()
            mock_train_result.stopped = False  # ran to its end (ADR-111)
            mock_train_result.total_epochs = 1
            mock_train_result.best_val_loss = 0.4
            mock_train_result.model_path = cv_config.output.models_dir / "dummy.pth"

            def fit_side_effect(*args: Any, **kwargs: Any) -> Any:
                idx = call_count[0]
                call_count[0] += 1
                if idx == 1:
                    raise RuntimeError("injected failure")
                return mock_train_result

            mock_trainer_cls.return_value.fit.side_effect = fit_side_effect

            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.8
            mock_eval_result.f1 = 0.75
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        statuses = [r["status"] for r in block._fold_results]
        assert statuses.count("failed") == 1
        assert statuses.count("success") == 2
        assert "injected failure" in block._fold_results[1]["error"]

    def test_all_folds_fail_raises_in_report(self, cv_config: ExperimentConfig) -> None:
        """report() must raise RuntimeError when all folds failed."""
        from visionforge.blocks.cross_validation import CrossValidationBlock

        block = CrossValidationBlock()
        block.setup(cv_config)

        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as mock_trainer_cls,
        ):
            mock_trainer_cls.return_value.fit.side_effect = RuntimeError("always fails")
            block.run()

        with pytest.raises(RuntimeError, match="all folds failed"):
            block.report()


# ── VRAM hygiene ──────────────────────────────────────────────────────────────


class TestVramHygiene:
    def test_empty_cache_called_per_fold(self, cv_config: ExperimentConfig) -> None:
        """torch.cuda.empty_cache must be called exactly n_folds times."""
        from unittest.mock import MagicMock

        from visionforge.blocks.cross_validation import CrossValidationBlock

        block = CrossValidationBlock()
        block.setup(cv_config)

        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.cross_validation.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.cross_validation.torch.load", return_value={}),
            patch(
                "visionforge.blocks.cross_validation.torch.nn.Module.load_state_dict"
            ),
            patch(
                "visionforge.blocks.cross_validation.torch.cuda.empty_cache"
            ) as mock_cache,
        ):
            mock_train_result = MagicMock()
            mock_train_result.stopped = False  # ran to its end (ADR-111)
            mock_train_result.total_epochs = 1
            mock_train_result.best_val_loss = 0.4
            mock_train_result.model_path = cv_config.output.models_dir / "dummy.pth"
            mock_trainer_cls.return_value.fit.return_value = mock_train_result

            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.8
            mock_eval_result.f1 = 0.75
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        assert mock_cache.call_count == 3

    def test_empty_cache_called_even_on_failure(
        self, cv_config: ExperimentConfig
    ) -> None:
        """torch.cuda.empty_cache must be called even when a fold fails."""
        from visionforge.blocks.cross_validation import CrossValidationBlock

        block = CrossValidationBlock()
        block.setup(cv_config)

        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as mock_trainer_cls,
            patch(
                "visionforge.blocks.cross_validation.torch.cuda.empty_cache"
            ) as mock_cache,
        ):
            mock_trainer_cls.return_value.fit.side_effect = RuntimeError("fail")
            block.run()

        assert mock_cache.call_count == 3


# ── stratified vs plain KFold ─────────────────────────────────────────────────


class TestKFoldVariants:
    def _run_variant(self, cv_pool: Path, tmp_path: Path, stratified: bool) -> None:
        from unittest.mock import MagicMock

        from visionforge.blocks.cross_validation import CrossValidationBlock

        config = ExperimentConfig.model_validate(
            {
                "name": "cv_variant",
                "task": "binary",
                "block": "cross_validation",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 2,
                    "early_stopping_patience": 1,
                    "seed": 0,
                },
                "data": {
                    "base_dir": str(cv_pool),
                    "train_dir": "",
                    "num_workers": 0,
                    "pin_memory": False,
                    "transforms": {"image_size": 32},
                },
                "output": {
                    "models_dir": str(tmp_path / "m"),
                    "graphics_dir": str(tmp_path / "g"),
                    "logs_dir": str(tmp_path / "l"),
                    "reports_dir": str(tmp_path / "r"),
                },
                "cross_validation": {
                    "n_folds": 3,
                    "stratified": stratified,
                    "shuffle": False,
                    "fold_seed": 0,
                },
            }
        )
        block = CrossValidationBlock()
        block.setup(config)

        with (
            patch("visionforge.blocks.cross_validation.ModelFactory.create"),
            patch("visionforge.blocks.cross_validation.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.cross_validation.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.cross_validation.torch.load", return_value={}),
            patch(
                "visionforge.blocks.cross_validation.torch.nn.Module.load_state_dict"
            ),
        ):
            mock_train_result = MagicMock()
            mock_train_result.stopped = False  # ran to its end (ADR-111)
            mock_train_result.total_epochs = 1
            mock_train_result.best_val_loss = 0.4
            mock_train_result.model_path = config.output.models_dir / "dummy.pth"
            mock_trainer_cls.return_value.fit.return_value = mock_train_result

            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.8
            mock_eval_result.f1 = 0.75
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        assert len(block._fold_results) == 3

    def test_stratified_true_completes(self, cv_pool: Path, tmp_path: Path) -> None:
        """StratifiedKFold variant must complete all folds without error."""
        self._run_variant(cv_pool, tmp_path, stratified=True)

    def test_stratified_false_completes(self, cv_pool: Path, tmp_path: Path) -> None:
        """Plain KFold variant must complete all folds without error."""
        self._run_variant(cv_pool, tmp_path, stratified=False)


# ── registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_registry_discovers_cv_block(self) -> None:
        """BlockRegistry must include CrossValidationBlock after import."""
        import visionforge.blocks.cross_validation  # noqa: F401
        from visionforge.blocks.registry import BlockRegistry

        registry = BlockRegistry.discover()
        assert "CrossValidationBlock" in registry


# ── fold transforms follow the config (ADR-083) ───────────────────────────────


class TestFoldTransformsFollowTheConfig:
    """A fold trains with the same pipeline as a single run of the same config.

    The fold module rebuilt the transforms by hand and skipped both
    ``transforms.augment`` and ``data.preprocessing``: a K-fold with augmentation
    off augmented, and one with a filter configured trained without it.
    """

    def _fold(self, cv_config: ExperimentConfig, **data: Any) -> Any:
        from torch.utils.data import Subset
        from torchvision.datasets import ImageFolder

        from visionforge.blocks.cross_validation import _FoldDataModule

        raw = cv_config.model_dump(mode="json")
        raw["data"]["transforms"].update(data.pop("transforms", {}))
        raw["data"].update(data)
        config = ExperimentConfig.model_validate(raw)
        pool = ImageFolder(str(config.data.base_dir))
        return _FoldDataModule(
            Subset(pool, [0, 1, 6, 7]),
            Subset(pool, [2, 8]),
            [0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2],
            config,
        )

    @staticmethod
    def _steps(fold: Any, split: str) -> list[Any]:
        ds = fold._train_ds if split == "train" else fold._val_ds
        return list(ds.transform.transforms)

    def test_augment_off_has_no_random_step(self, cv_config: ExperimentConfig) -> None:
        fold = self._fold(
            cv_config,
            transforms={
                "augment": False,
                "horizontal_flip": True,
                "rotation_degrees": 15,
                "color_jitter": True,
            },
        )

        names = [type(s).__name__ for s in self._steps(fold, "train")]
        assert not [n for n in names if n.startswith("Random") or n == "ColorJitter"]

    def test_augment_on_keeps_the_random_steps(
        self, cv_config: ExperimentConfig
    ) -> None:
        fold = self._fold(
            cv_config,
            transforms={
                "augment": True,
                "horizontal_flip": True,
                "rotation_degrees": 15,
            },
        )

        names = [type(s).__name__ for s in self._steps(fold, "train")]
        assert "RandomHorizontalFlip" in names
        assert "RandomRotation" in names

    @pytest.mark.parametrize("split", ["train", "val"])
    def test_preprocessing_runs_first(
        self, cv_config: ExperimentConfig, split: str
    ) -> None:
        from visionforge.core.data import _PreprocessingTransform

        fold = self._fold(cv_config, preprocessing={"steps": [{"kind": "grayscale"}]})

        first = self._steps(fold, split)[0]
        assert isinstance(first, _PreprocessingTransform)

    def test_the_fold_statistics_normalise_both_splits(
        self, cv_config: ExperimentConfig
    ) -> None:
        fold = self._fold(cv_config)

        for split in ("train", "val"):
            normalize = self._steps(fold, split)[-1]
            assert list(normalize.mean) == [0.5, 0.5, 0.5]
            assert list(normalize.std) == [0.2, 0.2, 0.2]


class TestFoldStatisticsOnMixedSizes:
    """Fold statistics are measured on the resized view the model sees.

    The pool was read with ToTensor only, so a dataset whose images differ in
    size could not even be batched, and the statistics described pixels the
    model never saw.
    """

    def test_images_of_different_sizes_are_measured(self, tmp_path: Path) -> None:
        from torchvision.datasets import ImageFolder

        from visionforge.blocks.cross_validation import (
            _compute_fold_stats,
            _statistics_view,
        )

        for cls, (width, height) in [("a", (40, 40)), ("b", (48, 40))]:
            folder = tmp_path / cls
            folder.mkdir()
            for i in range(3):
                pixels = np.full((height, width, 3), 60 * (i + 1), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{i}.png")
        config = ExperimentConfig.model_validate(
            {
                "name": "mixed",
                "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
                "data": {
                    "base_dir": str(tmp_path),
                    "train_dir": "",
                    "transforms": {"image_size": 32},
                },
            }
        )

        pool = ImageFolder(str(tmp_path), transform=_statistics_view(config.data))
        mean, std = _compute_fold_stats(pool, list(range(6)), 0)

        assert len(mean) == 3
        assert all(0.0 < m < 1.0 for m in mean)
        assert all(s > 0.0 for s in std)
