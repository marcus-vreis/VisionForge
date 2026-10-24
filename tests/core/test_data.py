import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionforge.core.data import (
    DataModule,
    _build_transforms,
    _make_preprocessing_transform,
)
from visionforge.utils.config import (
    ExperimentConfig,
    PreprocessingConfig,
    PreprocessingStep,
    TransformConfig,
)
from visionforge.utils.messages import using_lang


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Minimal ImageFolder structure with tiny RGB images."""
    for split in ["train", "val", "test"]:
        for cls in ["class_a", "class_b"]:
            folder = tmp_path / split / cls
            folder.mkdir(parents=True)
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(folder / "image.png")
    return tmp_path


@pytest.fixture
def binary_config(dataset_root: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "test",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "seed": 42,
            },
            "data": {
                "base_dir": str(dataset_root),
                "num_workers": 0,
                "pin_memory": False,
            },
        }
    )


class TestBuildTransforms:
    def test_train_includes_augmentation_by_default(self) -> None:
        """Default TransformConfig must include RandomHorizontalFlip in train."""
        tc = TransformConfig()
        transform = _build_transforms(tc, is_train=True)
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomHorizontalFlip" in names

    def test_val_excludes_augmentation(self) -> None:
        """Val transform must not include any augmentation."""
        tc = TransformConfig()
        transform = _build_transforms(tc, is_train=False)
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomHorizontalFlip" not in names
        assert "RandomRotation" not in names

    def test_color_jitter_disabled_by_default(self) -> None:
        """ColorJitter must be absent when color_jitter=False."""
        tc = TransformConfig(color_jitter=False)
        transform = _build_transforms(tc, is_train=True)
        names = [type(t).__name__ for t in transform.transforms]
        assert "ColorJitter" not in names

    def test_color_jitter_enabled(self) -> None:
        """ColorJitter must be present when color_jitter=True."""
        tc = TransformConfig(color_jitter=True)
        transform = _build_transforms(tc, is_train=True)
        names = [type(t).__name__ for t in transform.transforms]
        assert "ColorJitter" in names

    def test_no_rotation_when_degrees_zero(self) -> None:
        """RandomRotation must be absent when rotation_degrees=0."""
        tc = TransformConfig(rotation_degrees=0)
        transform = _build_transforms(tc, is_train=True)
        names = [type(t).__name__ for t in transform.transforms]
        assert "RandomRotation" not in names

    def test_empty_preprocessing_does_not_add_callable(self) -> None:
        """Empty PreprocessingConfig must not prepend any step."""
        pp = PreprocessingConfig()
        transform = _build_transforms(
            TransformConfig(), is_train=False, preprocessing=pp
        )
        # Default chain has Resize, CenterCrop, ToTensor, Normalize = 4 steps.
        assert len(transform.transforms) == 4

    def test_preprocessing_prepends_callable(self) -> None:
        """A non-empty preprocessing must add a leading callable to the pipeline."""
        pp = PreprocessingConfig(steps=[PreprocessingStep(kind="grayscale")])
        transform = _build_transforms(
            TransformConfig(), is_train=False, preprocessing=pp
        )
        # First step is now our pipeline callable (not a torchvision class).
        assert callable(transform.transforms[0])
        assert type(transform.transforms[0]).__name__ != "Resize"

    def test_preprocessing_transform_is_picklable(self) -> None:
        """The preprocessing callable must survive pickling.

        DataLoader workers (num_workers > 0) pickle the dataset transform to
        send it to worker processes. Under the Windows 'spawn' start method a
        closure or lambda fails to pickle, crashing training the moment a
        preprocessing pipeline is configured.
        """
        pp = PreprocessingConfig(
            steps=[
                # extra="allow" params go through model_validate (mypy can't see
                # dynamic kwargs on the constructor).
                PreprocessingStep.model_validate(
                    {"kind": "gaussian_blur", "radius": 1.5}
                ),
                PreprocessingStep(kind="grayscale"),
            ]
        )
        transform = _make_preprocessing_transform(pp)
        restored = pickle.loads(pickle.dumps(transform))
        img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
        assert restored(img).size == img.size

    def test_full_preprocessing_pipeline_is_picklable(self) -> None:
        """The whole composed transform must pickle when preprocessing is on."""
        pp = PreprocessingConfig(steps=[PreprocessingStep(kind="edges")])
        transform = _build_transforms(
            TransformConfig(), is_train=True, preprocessing=pp
        )
        # Round-trips without raising — this is what the DataLoader does.
        pickle.loads(pickle.dumps(transform))


class TestDataModulePreprocessing:
    def test_data_module_with_pipeline_loads(self, dataset_root: Path) -> None:
        """End-to-end: DataModule with a preprocessing pipeline returns tensors."""
        cfg = ExperimentConfig.model_validate(
            {
                "name": "pp_test",
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
                    "preprocessing": {
                        "steps": [
                            {"kind": "gaussian_blur", "radius": 1.0},
                            {"kind": "grayscale"},
                        ],
                    },
                },
            }
        )
        dm = DataModule(cfg)
        inputs, _ = next(iter(dm.train_loader()))
        # Shape: (batch, 3, H, W) — preprocessing kept RGB channel count.
        assert inputs.dim() == 4
        assert inputs.shape[1] == 3


class TestDataModule:
    def test_loaders_are_created(self, binary_config: ExperimentConfig) -> None:
        """DataModule must expose train, val and test loaders."""
        dm = DataModule(binary_config)
        assert dm.train_loader() is not None
        assert dm.val_loader() is not None
        assert dm.test_loader() is not None

    def test_train_loader_batch_size(self, binary_config: ExperimentConfig) -> None:
        """Batch size from config must be respected in train loader."""
        dm = DataModule(binary_config)
        batch = next(iter(dm.train_loader()))
        inputs, _ = batch
        assert inputs.shape[0] <= binary_config.training.batch_size

    def test_val_loader_not_shuffled(self, binary_config: ExperimentConfig) -> None:
        """Val loader must have shuffle=False."""
        dm = DataModule(binary_config)
        loader = dm.val_loader()
        assert loader.sampler.__class__.__name__ == "SequentialSampler"

    def test_test_loader_not_shuffled(self, binary_config: ExperimentConfig) -> None:
        """Test loader must have shuffle=False."""
        dm = DataModule(binary_config)
        loader = dm.test_loader()
        assert loader.sampler.__class__.__name__ == "SequentialSampler"

    def test_class_names_returns_sorted_classes(
        self, binary_config: ExperimentConfig
    ) -> None:
        """class_names must return sorted class names from the training dataset."""
        dm = DataModule(binary_config)
        names = dm.class_names
        assert names == ["class_a", "class_b"]

    def test_train_loader_is_shuffled(self, binary_config: ExperimentConfig) -> None:
        """Train loader must use RandomSampler (shuffle=True)."""
        dm = DataModule(binary_config)
        loader = dm.train_loader()
        assert loader.sampler.__class__.__name__ == "RandomSampler"


def _classification_config(
    root: Path,
    train_classes: tuple[str, ...],
    eval_classes: tuple[str, ...],
    *,
    task: str,
    num_classes: int,
) -> ExperimentConfig:
    for split in ("train", "val", "test"):
        for name in train_classes if split == "train" else eval_classes:
            folder = root / split / name
            folder.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(folder / "image.png")
    return ExperimentConfig.model_validate(
        {
            "name": "labels",
            "task": task,
            "model": {"num_classes": num_classes, "pretrained": False},
            "data": {
                "base_dir": str(root),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32},
            },
        }
    )


class TestLabelVocabulary:
    @pytest.mark.parametrize(
        ("task", "num_classes", "train_classes", "eval_classes", "targets"),
        [
            ("binary", 1, ("a", "b"), ("b",), [1]),
            ("multiclass", 3, ("a", "b", "c"), ("a", "c"), [0, 2]),
            ("multiclass", 3, ("a", "b", "c"), ("b", "c"), [1, 2]),
        ],
    )
    def test_missing_classes_keep_training_indices(
        self,
        tmp_path: Path,
        task: str,
        num_classes: int,
        train_classes: tuple[str, ...],
        eval_classes: tuple[str, ...],
        targets: list[int],
    ) -> None:
        config = _classification_config(
            tmp_path, train_classes, eval_classes, task=task, num_classes=num_classes
        )
        dm = DataModule(config)
        assert dm.class_names == list(train_classes)
        for loader in (dm.val_loader(), dm.test_loader()):
            dataset = loader.dataset
            assert dataset.classes == list(train_classes)
            assert dataset.class_to_idx == dict(
                zip(
                    train_classes,
                    range(num_classes if task == "multiclass" else 2),
                    strict=True,
                )
            )
            assert dataset.targets == targets
            assert [label for _, label in dataset.samples] == targets
            assert dataset.imgs is dataset.samples
            assert next(iter(loader))[1].tolist() == targets

    @pytest.mark.parametrize("split", ["val", "test"])
    def test_empty_known_class_directory_is_allowed(
        self, tmp_path: Path, split: str
    ) -> None:
        config = _classification_config(
            tmp_path, ("a", "b"), ("b",), task="binary", num_classes=1
        )
        (tmp_path / split / "a").mkdir()
        dm = DataModule(config)
        loader = dm.val_loader() if split == "val" else dm.test_loader()
        assert next(iter(loader))[1].tolist() == [1]

    @pytest.mark.parametrize("split", ["val", "test"])
    @pytest.mark.parametrize("remove_folders", [False, True])
    def test_split_without_images_is_rejected(
        self, tmp_path: Path, split: str, remove_folders: bool
    ) -> None:
        config = _classification_config(
            tmp_path, ("a", "b"), ("a", "b"), task="binary", num_classes=1
        )
        for cls in ("a", "b"):
            (tmp_path / split / cls / "image.png").unlink()
            if remove_folders:
                (tmp_path / split / cls).rmdir()
        with using_lang("en"), pytest.raises(ValueError, match=rf"{split}.*no images"):
            DataModule(config)

    @pytest.mark.parametrize("split", ["val", "test"])
    @pytest.mark.parametrize("empty", [False, True])
    def test_unknown_class_is_rejected(
        self, tmp_path: Path, split: str, empty: bool
    ) -> None:
        config = _classification_config(
            tmp_path, ("a", "b"), ("a", "b"), task="binary", num_classes=1
        )
        unknown = tmp_path / split / "unknown"
        unknown.mkdir()
        if not empty:
            Image.new("RGB", (8, 8)).save(unknown / "image.png")
        with using_lang("en"), pytest.raises(ValueError, match=rf"{split}.*unknown"):
            DataModule(config)

    @pytest.mark.parametrize(
        ("task", "num_classes", "train_classes", "expected"),
        [
            ("binary", 1, ("a",), 2),
            ("binary", 1, ("a", "b", "c"), 2),
            ("multiclass", 3, ("a", "b"), 3),
            ("multiclass", 2, ("a", "b", "c"), 2),
        ],
    )
    def test_training_class_count_must_match_outputs(
        self,
        tmp_path: Path,
        task: str,
        num_classes: int,
        train_classes: tuple[str, ...],
        expected: int,
    ) -> None:
        config = _classification_config(
            tmp_path, train_classes, train_classes, task=task, num_classes=num_classes
        )
        with (
            using_lang("en"),
            pytest.raises(
                ValueError, match=rf"Training.*{len(train_classes)}.*require {expected}"
            ),
        ):
            DataModule(config)

    @pytest.mark.parametrize(
        ("problem", "text"),
        [
            ("empty", "não tem imagens"),
            ("unknown", "Mova ou renomeie"),
            ("count", "exigem 2"),
        ],
    )
    def test_vocabulary_errors_use_portuguese(
        self, tmp_path: Path, problem: str, text: str
    ) -> None:
        classes = ("a",) if problem == "count" else ("a", "b")
        config = _classification_config(
            tmp_path, classes, classes, task="binary", num_classes=1
        )
        if problem == "empty":
            for name in classes:
                (tmp_path / "val" / name / "image.png").unlink()
        elif problem == "unknown":
            (tmp_path / "val" / "unknown").mkdir()

        with using_lang("pt"), pytest.raises(ValueError, match=text):
            DataModule(config)


class TestAugmentToggle:
    """Turning augmentation off must not touch resize or normalization."""

    def _names(self, *, augment: bool, is_train: bool) -> list[str]:
        from visionforge.core.data import _build_transforms
        from visionforge.utils.config import TransformConfig

        cfg = TransformConfig(
            augment=augment,
            horizontal_flip=True,
            rotation_degrees=25,
            color_jitter=True,
        )
        composed = _build_transforms(cfg, is_train=is_train)
        return [type(t).__name__ for t in composed.transforms]

    def test_augment_off_drops_the_augmenting_steps(self) -> None:
        names = self._names(augment=False, is_train=True)

        assert "RandomHorizontalFlip" not in names
        assert "RandomRotation" not in names
        assert "ColorJitter" not in names

    def test_augment_off_keeps_resize_and_normalize(self) -> None:
        """These apply to train, val and test alike — they are not augmentation."""
        names = self._names(augment=False, is_train=True)

        assert "Resize" in names
        assert "Normalize" in names

    def test_augment_defaults_to_on_and_still_augments(self) -> None:
        from visionforge.utils.config import TransformConfig

        assert TransformConfig().augment is True
        assert "RandomHorizontalFlip" in self._names(augment=True, is_train=True)

    def test_the_flag_changes_nothing_outside_training(self) -> None:
        on = self._names(augment=True, is_train=False)
        off = self._names(augment=False, is_train=False)

        assert on == off
