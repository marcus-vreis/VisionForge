"""Visual near-duplicates across dataset splits (issue 351, ADR-228).

A difference hash (dHash) of 64 bits: the image in grayscale at 9x8 pixels, one
bit per pair of horizontal neighbours, set when the left one is brighter. A
re-encoding, a resize or a small change of brightness keeps almost every bit;
two different photographs differ in about half of them.

Two images are near when their hashes differ in at most ``MAX_DISTANCE`` bits.
The hash is cut into 8 bands of 8 bits: with 7 or fewer differing bits at
least one band is equal, so comparing only the hashes that share a band finds
every near pair without comparing all of them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MAX_DISTANCE = 6
_BANDS = 8
_BAND_BITS = 64 // _BANDS
_BAND_MASK = (1 << _BAND_BITS) - 1
# A flat image (one colour) has no brighter neighbour anywhere: hash 0 says
# nothing about its content, and a red and a blue square must not match.
_NO_STRUCTURE = 0


@dataclass(frozen=True)
class NearImage:
    """One readable image of the audit, with its split and both hashes."""

    split: str
    path: str
    dhash: int
    sha256: str


def difference_hash(path: Path) -> int:
    """The 64-bit dHash of the image at ``path``.

    Raises:
        OSError: the image cannot be decoded (truncated, unknown format).
    """
    with Image.open(path) as image:
        small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(small.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            left, right = pixels[row * 9 + col], pixels[row * 9 + col + 1]
            value = (value << 1) | int(left > right)
    return value


def hamming(a: int, b: int) -> int:
    """How many bits differ between two hashes."""
    return (a ^ b).bit_count()


def near_duplicate_groups(
    images: list[NearImage], max_distance: int = MAX_DISTANCE
) -> list[dict[str, Any]]:
    """Groups of near images that span more than one split.

    A group joins images whose hashes are near, directly or through another
    image of the group. A group whose images are all the same bytes is left out:
    the audit already reports it as an exact duplicate.
    """
    if not 0 <= max_distance < _BANDS:
        raise ValueError(f"max_distance must be 0..{_BANDS - 1}, got {max_distance}")
    by_hash: dict[int, list[NearImage]] = defaultdict(list)
    for image in images:
        if image.dhash != _NO_STRUCTURE:
            by_hash[image.dhash].append(image)
    parent = {value: value for value in by_hash}

    def root(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for a, b in _candidate_pairs(list(by_hash)):
        if hamming(a, b) <= max_distance:
            parent[root(a)] = root(b)
    components: dict[int, list[NearImage]] = defaultdict(list)
    for value, members in by_hash.items():
        components[root(value)].extend(members)
    return sorted(
        (
            {
                "splits": sorted({m.split for m in members}),
                "paths": sorted(m.path for m in members),
            }
            for members in components.values()
            if len({m.split for m in members}) > 1
            and len({m.sha256 for m in members}) > 1
        ),
        key=lambda group: group["paths"],
    )


def _candidate_pairs(values: list[int]) -> set[tuple[int, int]]:
    """Distinct hashes that share at least one band."""
    pairs: set[tuple[int, int]] = set()
    for band in range(_BANDS):
        buckets: dict[int, list[int]] = defaultdict(list)
        for value in values:
            buckets[(value >> (band * _BAND_BITS)) & _BAND_MASK].append(value)
        for bucket in buckets.values():
            for i, a in enumerate(bucket):
                for b in bucket[i + 1 :]:
                    pairs.add((a, b) if a < b else (b, a))
    return pairs
