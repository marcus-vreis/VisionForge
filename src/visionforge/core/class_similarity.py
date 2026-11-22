"""Which classes of a classification dataset look alike, before training (ADR-239).

Each image (at most ``per_class`` per class, evenly spaced) is described by an
ImageNet-pretrained resnet18 without its last layer: 512 numbers that put images
of the same kind of thing close together. Two measures per pair of classes A, B:

- neighbour share: of the images of A, the fraction whose nearest image (cosine
  similarity, any split, never itself) is labeled B. It is what a 1-nearest-
  neighbour classifier on those features would confuse, so it reads as "how often
  an A looks most like a B";
- centroid similarity: cosine between the mean descriptions of A and B.

A pair is flagged when either neighbour share reaches ``FLAG_SHARE``. The images
behind it are listed, most similar first: they are where the two classes meet,
and sometimes an image filed under the wrong class.

The features come from ImageNet, not from this dataset, so they see what ImageNet
separates (objects, textures). Classes that differ by a detail ImageNet ignores
can look alike here and still be learned apart by training.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_explorer import DatasetIndex, ExplorerImage

DEFAULT_PER_CLASS = 200
MAX_PER_CLASS = 2_000
BATCH_SIZE = 32
FLAG_SHARE = 0.10
MAX_PAIRS = 30
MAX_EXAMPLES = 6
# Rows of the similarity matrix computed at once: 1024 x N floats.
_BLOCK = 1024
MODEL_NAME = "resnet18 (ImageNet)"

# RGB images in, one row of features per image out: [N, D] float32.
Embedder = Callable[[list[Image.Image]], np.ndarray]


def pretrained_embedder() -> Embedder:
    """resnet18 with ImageNet weights, last layer removed, on the CPU.

    The CPU leaves the GPU to a training that may be running. The weights come
    from torchvision's cache; without them torchvision downloads them (~45 MB).
    """
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = weights.transforms()

    def embed(images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([preprocess(im) for im in images])  # [B, 3, 224, 224]
        with torch.inference_mode():
            out = model(batch)  # [B, 512]
        features: np.ndarray = out.numpy().astype(np.float32)
        return features

    return embed


@dataclass(frozen=True)
class _Item:
    rel: str
    split: str
    label: int


def choose_images(
    index: DatasetIndex, per_class: int
) -> tuple[list[_Item], dict[str, int]]:
    """At most ``per_class`` images per class, evenly spaced; and how many each has."""
    by_class: dict[str, list[ExplorerImage]] = {c: [] for c in index.class_names}
    for item in index.images:
        if len(item.classes) == 1 and item.classes[0] in by_class:
            by_class[item.classes[0]].append(item)
    chosen: list[_Item] = []
    for label, name in enumerate(index.class_names):
        items = sorted(by_class[name], key=lambda it: (it.split, it.rel))
        chosen += [_Item(it.rel, it.split, label) for it in _spread(items, per_class)]
    return chosen, {c: len(v) for c, v in by_class.items()}


def _spread(items: list[ExplorerImage], count: int) -> list[ExplorerImage]:
    """``count`` items evenly spaced from first to last; all of them if fewer."""
    if len(items) <= count:
        return items
    if count == 1:
        return items[:1]
    last = len(items) - 1
    return [items[round(i * last / (count - 1))] for i in range(count)]


def _open_rgb(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def _normalized(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit: np.ndarray = vectors / np.maximum(norms, 1e-12)
    return unit


def nearest_neighbours(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index and cosine similarity of each row's nearest other row; [N] each.

    ``vectors`` is [N, D] with N >= 2. Computed in blocks of rows, so memory stays
    at ``_BLOCK`` x N whatever N is.
    """
    unit = _normalized(vectors.astype(np.float64))
    n = unit.shape[0]
    index = np.empty(n, dtype=np.int64)
    score = np.empty(n, dtype=np.float64)
    for start in range(0, n, _BLOCK):
        sims = unit[start : start + _BLOCK] @ unit.T  # [b, N]
        rows = np.arange(sims.shape[0])
        sims[rows, rows + start] = -np.inf  # never itself
        index[start : start + _BLOCK] = sims.argmax(axis=1)
        score[start : start + _BLOCK] = sims[rows, index[start : start + _BLOCK]]
    return index, score


@dataclass(frozen=True)
class _Neighbours:
    """Each described image's label, nearest other image and that similarity."""

    labels: np.ndarray  # [N] int, indexing ``names``
    nearest: np.ndarray  # [N] int, a row of the same arrays
    score: np.ndarray  # [N] cosine similarity to ``nearest``
    names: Sequence[str]
    paths: Sequence[str]

    def crossing(self, a: int, b: int) -> list[dict[str, Any]]:
        """The images of a or b whose nearest image is of the other, most similar first."""
        rows = [
            i
            for i in range(len(self.labels))
            if {int(self.labels[i]), int(self.labels[self.nearest[i]])} == {a, b}
        ]
        rows.sort(key=lambda i: -self.score[i])
        return [
            {
                "path": self.paths[i],
                "class": self.names[self.labels[i]],
                "neighbour": self.paths[self.nearest[i]],
                "neighbour_class": self.names[self.labels[self.nearest[i]]],
                "similarity": round(float(self.score[i]), 4),
            }
            for i in rows[:MAX_EXAMPLES]
        ]


def _centroid_similarity(vectors: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Cosine between the mean features of each pair of classes; [k, k]."""
    means = [
        vectors[labels == c].mean(axis=0)
        if (labels == c).any()
        else np.zeros(vectors.shape[1])
        for c in range(k)
    ]
    unit = _normalized(np.stack(means))
    sims: np.ndarray = unit @ unit.T
    return sims


def neighbour_report(
    vectors: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str],
    paths: Sequence[str],
) -> dict[str, Any]:
    """Purity per class and the pairs of classes, most confused first.

    ``vectors`` [N, D] and ``labels``/``paths`` of length N, labels indexing
    ``class_names``.
    """
    k = len(class_names)
    label_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(label_arr, minlength=k)
    nearest, score = nearest_neighbours(vectors)
    found = _Neighbours(label_arr, nearest, score, class_names, paths)
    confusion = np.zeros((k, k), dtype=np.int64)  # [own class, neighbour's class]
    np.add.at(confusion, (label_arr, label_arr[nearest]), 1)
    centroid = _centroid_similarity(vectors, label_arr, k)
    pairs: list[dict[str, Any]] = []
    for a in range(k):
        for b in range(a + 1, k):
            if confusion[a, b] + confusion[b, a] == 0:
                continue  # no image of either is nearest to the other
            a_to_b = float(confusion[a, b] / counts[a])
            b_to_a = float(confusion[b, a] / counts[b])
            pairs.append(
                {
                    "a": class_names[a],
                    "b": class_names[b],
                    "a_to_b": round(a_to_b, 4),
                    "b_to_a": round(b_to_a, 4),
                    "centroid_similarity": round(float(centroid[a, b]), 4),
                    "flagged": max(a_to_b, b_to_a) >= FLAG_SHARE,
                    "examples": found.crossing(a, b),
                }
            )
    pairs.sort(
        key=lambda p: (-max(p["a_to_b"], p["b_to_a"]), -p["centroid_similarity"])
    )
    return {
        "classes": [
            {
                "name": class_names[c],
                "used": int(counts[c]),
                "purity": round(float(confusion[c, c] / counts[c]), 4)
                if counts[c]
                else None,
            }
            for c in range(k)
        ],
        "flag_share": FLAG_SHARE,
        "flagged_count": sum(p["flagged"] for p in pairs),
        "pair_count": len(pairs),
        "pairs": pairs[:MAX_PAIRS],
    }


def analyze_class_similarity(
    index: DatasetIndex,
    *,
    per_class: int = DEFAULT_PER_CLASS,
    embed: Embedder | None = None,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Describe the chosen images and compare the classes; see the module docstring.

    ``embed`` defaults to :func:`pretrained_embedder`, built here so the model is
    only loaded when the analysis runs.
    """
    per_class = max(1, min(per_class, MAX_PER_CLASS))
    chosen, totals = choose_images(index, per_class)
    report: dict[str, Any] = {
        "model": MODEL_NAME,
        "per_class": per_class,
        "images_per_class": totals,
        "sampled": any(n > per_class for n in totals.values()),
        "cancelled": False,
        "unreadable": 0,
    }
    progress = on_progress or (lambda done, total: None)
    progress(0, len(chosen))
    if sum(1 for n in totals.values() if n) < 2:
        return {**report, "described": 0, **_empty()}
    embed = embed or pretrained_embedder()
    root = index.root.resolve()
    rows: list[np.ndarray] = []
    kept: list[_Item] = []
    for start in range(0, len(chosen), BATCH_SIZE):
        if is_cancelled(token):
            return {**report, "cancelled": True, "described": len(kept), **_empty()}
        batch = chosen[start : start + BATCH_SIZE]
        opened = [(it, _open_rgb(root / it.rel)) for it in batch]
        good = [(it, im) for it, im in opened if im is not None]
        report["unreadable"] += len(batch) - len(good)
        if good:
            rows.append(embed([im for _, im in good]))
            kept += [it for it, _ in good]
        progress(start + len(batch), len(chosen))
    report["described"] = len(kept)
    if len(kept) < 2:
        return {**report, **_empty()}
    return {
        **report,
        **neighbour_report(
            np.concatenate(rows),
            [it.label for it in kept],
            list(index.class_names),
            [it.rel for it in kept],
        ),
    }


def _empty() -> dict[str, Any]:
    return {
        "classes": [],
        "flag_share": FLAG_SHARE,
        "flagged_count": 0,
        "pair_count": 0,
        "pairs": [],
    }
