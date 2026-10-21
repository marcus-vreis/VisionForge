from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from sklearn.model_selection import GroupKFold
from torchvision.datasets import ImageFolder

from visionforge.blocks.cross_validation import _split_pool
from visionforge.utils.config import CrossValidationConfig, DataConfig


def _pool(tmp_path: Path) -> tuple[ImageFolder, Path]:
    rows = ["path,group"]
    for label in range(2):
        for group in range(4):
            for sample in range(2):
                relative = f"train/class{label}/{group}_{sample}.png"
                path = tmp_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), (label * 100, group * 20, sample)).save(path)
                rows.append(f"{relative},subject{label}_{group}")
    other = tmp_path / "val/class0/image.png"
    other.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "blue").save(other)
    rows.append("val/class0/image.png,validation_subject")
    manifest = tmp_path / "groups.csv"
    manifest.write_text("\n".join(rows))
    return ImageFolder(str(tmp_path / "train")), manifest


@pytest.mark.parametrize("stratified", [False, True])
def test_groups_never_cross_fold_boundaries(tmp_path: Path, stratified: bool) -> None:
    pool, manifest = _pool(tmp_path)
    cv = CrossValidationConfig(
        n_folds=2, stratified=stratified, shuffle=False, group_manifest=manifest
    )
    folds = list(_split_pool(pool, DataConfig(base_dir=tmp_path), cv))
    assert len(folds) == 2
    seen: list[int] = []

    def group(index: int) -> tuple[int, str]:
        return (pool.samples[index][1], Path(pool.samples[index][0]).stem.split("_")[0])

    for train, val in folds:
        assert {group(index) for index in train}.isdisjoint(
            group(index) for index in val
        )
        assert set(train) | set(val) == set(range(len(pool)))
        seen.extend(val.tolist())
        if stratified:
            assert np.bincount(np.array(pool.targets)[val]).tolist() == [4, 4]
    assert sorted(seen) == list(range(len(pool)))


def test_rejects_insufficient_groups_before_training(tmp_path: Path) -> None:
    pool, manifest = _pool(tmp_path)
    manifest.write_text(
        "path,group\n"
        + "\n".join(
            f"{Path(path).relative_to(tmp_path).as_posix()},only_subject"
            for path, _ in pool.samples
        )
    )
    cv = CrossValidationConfig(n_folds=2, group_manifest=manifest)
    with pytest.raises(ValueError, match="distinct groups"):
        _split_pool(pool, DataConfig(base_dir=tmp_path), cv)


def test_relative_manifest_and_seed_reproduce_folds(tmp_path: Path) -> None:
    pool, _ = _pool(tmp_path)
    cv = CrossValidationConfig(n_folds=2, group_manifest=Path("groups.csv"))
    data = DataConfig(base_dir=tmp_path)
    first = list(_split_pool(pool, data, cv))
    second = list(_split_pool(pool, data, cv))
    assert all(
        np.array_equal(train_a, train_b) and np.array_equal(val_a, val_b)
        for (train_a, val_a), (train_b, val_b) in zip(first, second, strict=True)
    )


def test_rejects_unknown_manifest_path(tmp_path: Path) -> None:
    pool, manifest = _pool(tmp_path)
    with manifest.open("a") as stream:
        stream.write("\ntrain/class0/missing.png,missing")
    cv = CrossValidationConfig(n_folds=2, group_manifest=manifest)
    with pytest.raises(ValueError, match="Unknown"):
        _split_pool(pool, DataConfig(base_dir=tmp_path), cv)


def test_dataset_root_alias_preserves_manifest_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _pool(root)
    alias = tmp_path / "alias"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(root)],
            check=True,
            capture_output=True,
        )
    else:
        alias.symlink_to(root, target_is_directory=True)
    pool = ImageFolder(str(alias / "train"))
    cv = CrossValidationConfig(n_folds=2, group_manifest=Path("groups.csv"))

    folds = list(_split_pool(pool, DataConfig(base_dir=alias), cv))

    assert len(folds) == 2
    for train, val in folds:
        assert set(train) | set(val) == set(range(len(pool)))
        assert set(train).isdisjoint(val)


@pytest.mark.parametrize("shuffle", [False, True])
def test_legacy_group_kfold_signature(tmp_path: Path, shuffle: bool) -> None:
    pool, manifest = _pool(tmp_path)
    cv = CrossValidationConfig(
        n_folds=2, stratified=False, shuffle=shuffle, group_manifest=manifest
    )

    def legacy_group_kfold(n_splits: int = 5) -> GroupKFold:
        return GroupKFold(n_splits=n_splits)

    with patch("visionforge.blocks.cross_validation.GroupKFold", legacy_group_kfold):
        if shuffle:
            with pytest.raises(ValueError, match="scikit-learn>=1.6"):
                _split_pool(pool, DataConfig(base_dir=tmp_path), cv)
        else:
            assert len(list(_split_pool(pool, DataConfig(base_dir=tmp_path), cv))) == 2
