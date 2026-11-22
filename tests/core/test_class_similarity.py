"""Which classes look alike before training (ADR-239)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from visionforge.core import class_similarity as cs
from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_explorer import build_index

# 2-D features chosen by hand. dog/d3 points almost exactly where the cats are.
VECTORS = {
    "cat/c1": (1.0, 0.0),
    "cat/c2": (1.0, 0.1),
    "dog/d1": (0.0, 1.0),
    "dog/d2": (0.1, 1.0),
    "dog/d3": (1.0, 0.05),
    "fox/f1": (-1.0, 0.0),
    "fox/f2": (-1.0, 0.1),
}


def _report() -> dict[str, Any]:
    names = list(VECTORS)
    labels = [["cat", "dog", "fox"].index(n.split("/")[0]) for n in names]
    return cs.neighbour_report(
        np.array([VECTORS[n] for n in names]), labels, ["cat", "dog", "fox"], names
    )


class TestNeighbours:
    def test_nearest_other_row(self) -> None:
        vectors = np.array([VECTORS[n] for n in VECTORS])
        nn, score = cs.nearest_neighbours(vectors)
        names = list(VECTORS)
        # c1 (1, 0): c2 at cos 1/sqrt(1.01) = 0.99504, d3 at 1/sqrt(1.0025) = 0.99875.
        assert names[nn[0]] == "dog/d3"
        assert score[0] == pytest.approx(0.99875, abs=1e-5)
        assert [names[i] for i in nn[2:4]] == ["dog/d2", "dog/d1"]
        assert [names[i] for i in nn[5:]] == ["fox/f2", "fox/f1"]

    def test_blocks_give_the_same_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vectors = np.random.default_rng(0).normal(size=(50, 8))
        whole = cs.nearest_neighbours(vectors)
        monkeypatch.setattr(cs, "_BLOCK", 7)
        blocked = cs.nearest_neighbours(vectors)
        assert (whole[0] == blocked[0]).all()
        assert np.allclose(whole[1], blocked[1])


class TestReport:
    def test_purity_per_class(self) -> None:
        classes = {c["name"]: c for c in _report()["classes"]}
        # Both cats' nearest image is d3; d3's is c2; d1 and d2 point at each
        # other, f1 and f2 too.
        assert classes["cat"]["purity"] == 0.0
        assert classes["dog"]["purity"] == pytest.approx(2 / 3, abs=1e-4)
        assert classes["fox"]["purity"] == 1.0

    def test_the_cat_dog_pair_is_flagged(self) -> None:
        report = _report()
        assert report["pair_count"] == 1  # no image of fox is nearest to another class
        assert report["flagged_count"] == 1
        pair = report["pairs"][0]
        assert (pair["a"], pair["b"]) == ("cat", "dog")
        assert pair["a_to_b"] == 1.0  # 2 of 2 cats
        assert pair["b_to_a"] == pytest.approx(1 / 3, abs=1e-4)  # 1 of 3 dogs
        # Centroids: cat (1, 0.05), dog (1.1/3, 2.05/3); cosine 0.5162.
        assert pair["centroid_similarity"] == pytest.approx(0.5162, abs=1e-3)
        assert {e["path"] for e in pair["examples"]} == {"cat/c1", "cat/c2", "dog/d3"}
        # The least similar of the three crossings comes last.
        assert pair["examples"][-1]["path"] == "cat/c1"
        assert pair["examples"][-1]["neighbour"] == "dog/d3"

    def test_below_the_share_a_pair_is_listed_not_flagged(self) -> None:
        # Eleven cats in a tight bunch, one of them nearest to the single dog: the
        # cat side is 1/11 (< 10%), but the dog's nearest is that cat, 1/1.
        vectors = [(1.0, 0.01 * i) for i in range(10)] + [(0.0, 1.0), (0.05, 1.0)]
        labels = [0] * 11 + [1]
        report = cs.neighbour_report(
            np.array(vectors), labels, ["cat", "dog"], [str(i) for i in range(12)]
        )
        pair = report["pairs"][0]
        assert pair["a_to_b"] == pytest.approx(1 / 11, abs=1e-4)
        assert pair["b_to_a"] == 1.0
        assert pair["flagged"] is True  # either side counts
        no_dog_side = cs.neighbour_report(
            np.array(vectors + [(0.06, 1.0)]),
            [*labels, 1],
            ["cat", "dog"],
            [str(i) for i in range(13)],
        )
        # Now the two dogs point at each other and the stray cat at a dog: 1/11.
        assert no_dog_side["pairs"][0]["b_to_a"] == 0.0
        assert no_dog_side["pairs"][0]["flagged"] is False


def _dataset(root: Path, colours: dict[str, tuple[int, int, int]], n: int) -> Path:
    for split in ("train", "val"):
        for cls, colour in colours.items():
            for i in range(n):
                path = root / split / cls / f"{cls}{i}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                # A hint of the index keeps two images from being identical.
                Image.new("RGB", (8, 8), (colour[0], colour[1] + i, colour[2])).save(
                    path
                )
    return root


def _mean_colour(images: list[Image.Image]) -> np.ndarray:
    """Features = mean RGB: as alike as the colours of the classes."""
    return np.array(
        [np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in images]
    )


class TestAnalyze:
    def test_classes_of_the_same_colour_are_flagged(self, tmp_path: Path) -> None:
        root = _dataset(
            tmp_path, {"cat": (200, 0, 0), "dog": (200, 0, 0), "sky": (0, 0, 200)}, 3
        )
        report = cs.analyze_class_similarity(
            build_index(root, "classification"), embed=_mean_colour
        )
        assert report["described"] == 18
        assert report["images_per_class"] == {"cat": 6, "dog": 6, "sky": 6}
        assert report["sampled"] is False
        flagged = [(p["a"], p["b"]) for p in report["pairs"] if p["flagged"]]
        assert flagged == [("cat", "dog")]
        purity = {c["name"]: c["purity"] for c in report["classes"]}
        assert purity["sky"] == 1.0

    def test_per_class_limit_samples(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, {"a": (10, 0, 0), "b": (0, 0, 10)}, 5)
        seen: list[int] = []

        def embed(images: list[Image.Image]) -> np.ndarray:
            seen.append(len(images))
            return _mean_colour(images)

        report = cs.analyze_class_similarity(
            build_index(root, "classification"), per_class=4, embed=embed
        )
        assert report["sampled"] is True
        assert report["described"] == 8
        assert sum(seen) == 8

    def test_unreadable_images_are_counted(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, {"a": (10, 0, 0), "b": (0, 0, 10)}, 2)
        (root / "train/a/broken.png").write_bytes(b"not an image")
        report = cs.analyze_class_similarity(
            build_index(root, "classification"), embed=_mean_colour
        )
        assert report["unreadable"] == 1
        assert report["described"] == 8

    def test_one_class_compares_nothing(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, {"a": (10, 0, 0)}, 2)

        def never(images: list[Image.Image]) -> np.ndarray:
            raise AssertionError("no model needed for a single class")

        report = cs.analyze_class_similarity(
            build_index(root, "classification"), embed=never
        )
        assert report["pairs"] == []
        assert report["described"] == 0

    def test_cancel_returns_no_pairs(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, {"a": (10, 0, 0), "b": (0, 0, 10)}, 40)
        token = CancellationToken()
        calls: list[int] = []

        def embed(images: list[Image.Image]) -> np.ndarray:
            calls.append(1)
            token.cancel()
            return _mean_colour(images)

        report = cs.analyze_class_similarity(
            build_index(root, "classification"), embed=embed, token=token
        )
        assert report["cancelled"] is True
        assert report["pairs"] == []
        assert len(calls) == 1  # stopped before the second batch

    def test_progress_reaches_the_total(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, {"a": (10, 0, 0), "b": (0, 0, 10)}, 20)
        ticks: list[tuple[int, int]] = []
        cs.analyze_class_similarity(
            build_index(root, "classification"),
            embed=_mean_colour,
            on_progress=lambda done, total: ticks.append((done, total)),
        )
        assert ticks[0] == (0, 80)
        assert ticks[-1] == (80, 80)


def test_pretrained_embedder_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real network, with random weights in place of the ImageNet download."""
    from torchvision.models import resnet18
    from torchvision.models._api import WeightsEnum

    torch.manual_seed(0)
    state = resnet18(weights=None).state_dict()
    monkeypatch.setattr(WeightsEnum, "get_state_dict", lambda self, **kw: state)
    embed = cs.pretrained_embedder()
    out = embed([Image.new("RGB", (40, 30), (255, 0, 0)), Image.new("RGB", (64, 64))])
    assert out.shape == (2, 512)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_sampling_takes_the_full_quota_evenly() -> None:
    """250 images and a quota of 200 give 200, not every second image (125)."""
    from visionforge.core.dataset_explorer import DatasetIndex, ExplorerImage

    images = [ExplorerImage("train", f"a/{i:03}.png", ("a",)) for i in range(250)]
    index = DatasetIndex(Path("."), "classification", ["a"], images)
    chosen, totals = cs.choose_images(index, 200)
    assert totals == {"a": 250}
    assert len(chosen) == 200
    assert len({c.rel for c in chosen}) == 200
    # Evenly spread: the first and the last image are both in.
    assert chosen[0].rel == "a/000.png"
    assert chosen[-1].rel == "a/249.png"
