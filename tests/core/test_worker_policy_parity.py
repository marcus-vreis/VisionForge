"""The worker count each task's DataModule resolves, pinned input by input.

The anomaly, regression and segmentation modules once carried their own copy
of the policy in ``core.data.resolve_num_workers``. These expectations were
recorded from those copies before they were deleted, so a drift in the shared
helper (or a wrong ``loader_pools`` passed to it) shows up as a changed count.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from visionforge.core.anomaly_data import AnomalyDataModule
from visionforge.core.regression_data import RegressionDataModule
from visionforge.core.segmentation_data import SegmentationDataModule
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

_SMALL = 12
_LARGE = 600
_REQUESTED = (-1, 0, 1, 2, 4, 16, 64)

# 24 GiB of commit on 8 CPUs: suggested_workers gives 24/2/2 = 6 for two loader
# pools and 24/2/3 = 4 for three, so the budget tells the modules apart.
_COMMIT = 24 * 1024**3


def _row(n_small: tuple[int, ...], n_large: tuple[int, ...]) -> dict[str, Any]:
    return {
        "small": dict(zip(_REQUESTED, n_small, strict=True)),
        "large": dict(zip(_REQUESTED, n_large, strict=True)),
    }


# One deliberate change since the copies were recorded: the automatic setting
# (-1) on a small dataset now loads in process like any explicit request; the
# copies handed it the whole budget (6 or 4).
_PINNED = {
    "anomaly": _row((0, 0, 0, 0, 0, 0, 0), (6, 0, 1, 2, 4, 6, 6)),
    "regression": _row((0, 0, 0, 0, 0, 0, 0), (4, 0, 1, 2, 4, 4, 4)),
    "segmentation": _row((0, 0, 0, 0, 0, 0, 0), (4, 0, 1, 2, 4, 4, 4)),
}


def _build_anomaly(root: Path, n: int, *, with_test: bool = True) -> None:
    del with_test  # the anomaly module always needs its test split
    (root / "train" / "good").mkdir(parents=True)
    (root / "test" / "good").mkdir(parents=True)
    (root / "test" / "bad").mkdir(parents=True)
    for i in range(n):
        (root / "train" / "good" / f"n{i}.png").touch()
    (root / "test" / "good" / "g0.png").touch()
    (root / "test" / "bad" / "b0.png").touch()


def _build_regression(root: Path, n: int, *, with_test: bool = True) -> None:
    splits = ["train", "val", *(["test"] if with_test else [])]
    for split in splits:
        lines = ["image,target", *(f"{split}/{i}.png,{i}.0" for i in range(n))]
        (root / f"{split}.csv").write_text("\n".join(lines), encoding="utf-8")


def _build_segmentation(root: Path, n: int, *, with_test: bool = True) -> None:
    splits = ["train", "val", *(["test"] if with_test else [])]
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)
        for i in range(n):
            (root / split / "images" / f"{i}.jpg").touch()
            (root / split / "masks" / f"{i}.png").touch()


def _anomaly(root: Path, requested: int, **_: Any) -> AnomalyDataModule:
    return AnomalyDataModule(
        AnomalyConfig.model_validate(
            {
                "name": "a",
                "model": {"name": "autoencoder"},
                "data": {
                    "base_dir": str(root),
                    "image_size": 32,
                    "num_workers": requested,
                },
                "training": {"epochs": 1, "batch_size": 2},
            }
        )
    )


def _regression(root: Path, requested: int, **kw: Any) -> RegressionDataModule:
    return RegressionDataModule(
        RegressionConfig.model_validate(
            {
                "name": "r",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {
                    "base_dir": str(root),
                    "target_columns": ["target"],
                    "num_workers": requested,
                },
                "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
            }
        ),
        **kw,
    )


def _segmentation(root: Path, requested: int, **kw: Any) -> SegmentationDataModule:
    return SegmentationDataModule(
        SegmentationConfig.model_validate(
            {
                "name": "s",
                "model": {"name": "fcn_resnet50", "num_classes": 3},
                "data": {
                    "base_dir": str(root),
                    "image_size": 32,
                    "num_workers": requested,
                },
                "training": {"epochs": 1, "batch_size": 2},
            }
        ),
        **kw,
    )


_MODULES: dict[str, tuple[Callable[..., None], Callable[..., Any]]] = {
    "anomaly": (_build_anomaly, _anomaly),
    "regression": (_build_regression, _regression),
    "segmentation": (_build_segmentation, _segmentation),
}


@pytest.fixture(autouse=True)
def _fixed_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "visionforge.utils.workers.available_commit_bytes", lambda: _COMMIT
    )
    monkeypatch.setattr("visionforge.utils.workers.os.cpu_count", lambda: 8)


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], Path]:
    built: dict[tuple[str, str], Path] = {}
    for kind, (build, _) in _MODULES.items():
        for size, n in (("small", _SMALL), ("large", _LARGE)):
            root = tmp_path_factory.mktemp(f"{kind}_{size}")
            build(root, n)
            built[(kind, size)] = root
    return built


@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize("requested", _REQUESTED)
@pytest.mark.parametrize("kind", sorted(_MODULES))
def test_resolved_workers_match_the_pinned_policy(
    kind: str,
    requested: int,
    size: str,
    roots: dict[tuple[str, str], Path],
) -> None:
    module = _MODULES[kind][1](roots[(kind, size)], requested)
    assert module._num_workers == _PINNED[kind][size][requested]


@pytest.mark.parametrize("kind", ["regression", "segmentation"])
@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize("requested", _REQUESTED)
def test_evaluation_only_sizes_the_test_split(
    kind: str,
    requested: int,
    size: str,
    roots: dict[tuple[str, str], Path],
) -> None:
    module = _MODULES[kind][1](roots[(kind, size)], requested, evaluation_only=True)
    assert module._num_workers == _PINNED[kind][size][requested]


@pytest.mark.parametrize("kind", ["regression", "segmentation"])
@pytest.mark.parametrize(
    ("requested", "expected"),
    [(-1, 4), (0, 0), (1, 1), (2, 2), (4, 4), (16, 4), (64, 4)],
)
def test_with_no_split_to_size_only_the_budget_applies(
    kind: str, requested: int, expected: int, tmp_path: Path
) -> None:
    # evaluation_only without a test split leaves nothing to count, so a small
    # dataset cannot downgrade the request.
    _MODULES[kind][0](tmp_path, _SMALL, with_test=False)
    module = _MODULES[kind][1](tmp_path, requested, evaluation_only=True)
    assert module._num_workers == expected
