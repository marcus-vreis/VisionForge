"""Facts the guides state about the product, pinned to the code that decides them.

A guide that says "ResNet-50, 10 epochs" while the config defaults to something
else is wrong the day the default moves. These tests read the number from the
code, so the guide has to move with it.
"""

import re
from pathlib import Path

import pytest
from fastapi import FastAPI

from visionforge.gui.api.dataset_download import download_dataset
from visionforge.gui.api.routes import router
from visionforge.utils.config import ExperimentConfig, ModelConfig, TrainingConfig

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = _ROOT / "docs"
_EN_STRINGS = _ROOT / "frontend" / "src" / "i18n" / "en.ts"
_CUSTOM_PREFIX = "/api/custom/{key}"


def _read(*parts: str) -> str:
    return _DOCS.joinpath(*parts).read_text(encoding="utf-8")


def _default(field: str) -> ModelConfig | TrainingConfig:
    value = ExperimentConfig.model_fields[field].get_default(call_default_factory=True)
    assert isinstance(value, (ModelConfig, TrainingConfig))
    return value


def _step_names() -> list[str]:
    """The six configuration steps, as the English interface labels them."""
    source = _EN_STRINGS.read_text(encoding="utf-8")
    block = re.search(r"stepShell: \{\s*steps: \{(.*?)\}", source, re.DOTALL)
    assert block is not None
    return re.findall(r':\s*"([^"]+)"', block.group(1))


def test_quickstart_cites_the_real_default_architecture():
    model = _default("model")
    assert isinstance(model, ModelConfig)
    assert f"`{model.name}`" in _read("QUICKSTART.md")


def test_quickstart_cites_the_real_default_epochs():
    training = _default("training")
    assert isinstance(training, TrainingConfig)
    assert f"{training.epochs} epochs" in _read("QUICKSTART.md")


def test_quickstart_names_no_other_default():
    text = _read("QUICKSTART.md")
    assert "ResNet-50" not in text
    assert "10 epochs" not in text


def test_quickstart_walks_the_six_configuration_steps():
    steps = _step_names()
    assert len(steps) == 6
    text = _read("QUICKSTART.md")
    for step in steps:
        assert f"**{step}**" in text


def test_models_guide_sends_custom_model_through_yaml():
    # There is no selector in the interface; if one lands, this guide changes.
    assert "custom_model" in ModelConfig.model_fields
    # Form fields are labelled by their dotted path in the interface strings.
    assert "model.custom_model" not in _EN_STRINGS.read_text(encoding="utf-8")
    text = _read("custom", "MODELS.md")
    assert "**Modelo**" not in text
    assert "YAML" in text


def _custom_routes() -> set[str]:
    """``METHOD /api/custom/<key>...`` for every route of the custom tasks."""
    # Read from the OpenAPI schema of a mounted app, not ``router.routes``:
    # newer FastAPI releases no longer copy an included router's routes into
    # the parent's list, and CI installs the newest one.
    app = FastAPI()
    app.include_router(router)
    paths: dict[str, dict[str, object]] = app.openapi()["paths"]
    return {
        f"{method.upper()} {path.replace('{key}', '<key>')}"
        for path, operations in paths.items()
        if path.startswith(_CUSTOM_PREFIX)
        for method in operations
    }


def test_tasks_guide_lists_every_custom_task_endpoint():
    routes = _custom_routes()
    assert len(routes) >= 8
    text = _read("custom", "TASKS.md")
    missing = sorted(r for r in routes if f"`{r}`" not in text)
    assert missing == []


class _FakeMnist:
    """Stand-in for torchvision's MNIST: ten classes, 25 images each."""

    classes = [
        f"{i} - {n}"
        for i, n in enumerate(
            "zero one two three four five six seven eight nine".split()
        )
    ]
    per_class = 25

    def __init__(self, root: str, train: bool, download: bool):
        from PIL import Image

        self._items = [
            (Image.new("L", (4, 4), color=label), label)
            for label in range(10)
            for _ in range(self.per_class)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        return self._items[idx]


def test_datasets_guide_publishes_the_numbers_the_download_produces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("torchvision.datasets.MNIST", _FakeMnist)
    result = download_dataset(
        provider="torchvision",
        dataset="mnist",
        out_dir=str(tmp_path / "mnist"),
        limit=20,
    )
    # 10 classes x 20 per split, then 20% of each class of train moves to val.
    assert result.total_images == 400
    assert result.splits == {"train": 160, "test": 200, "val": 40}
    text = _read("DATASETS.md")
    assert f"total_images={result.total_images}" in text
    assert f"splits={result.splits}" in text


def test_install_guide_pins_no_package_release():
    text = _read("INSTALL.md")
    assert not re.search(r"(?<![\d.])0\.\d+\.\d+(?![\d.])", text)
    assert "actions/workflows/install-smoke.yml" in text
