"""A stratified fraction of a dataset's train split; val and test copied whole.

The subset is drawn once, with its own fixed seed, and written to disk, so every
run of the study reads the same files. Drawing it inside each run would tie the
data to the training seed and mix the two sources of variance the study is
trying to tell apart.

Usage:
    python research/seed_variance/make_subset.py \
        "datasets/Classification - USK-COFFEE" \
        "datasets/Classification - USK-COFFEE-10pct" --fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def make_subset(
    src: Path, dst: Path, fraction: float, *, seed: int = 2026, val_dir: str = "val"
) -> dict[str, int]:
    """Copy ``fraction`` of each train class (at least one file) plus val and test.

    Writes ``subset.json`` beside the splits with what was drawn and how, so the
    folder carries its own provenance. Returns the per-class train counts.

    Raises:
        FileExistsError: ``dst`` already exists — a subset is drawn once.
        ValueError: ``fraction`` outside (0, 1].
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; a subset is drawn once.")

    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for class_dir in sorted(p for p in (src / "train").iterdir() if p.is_dir()):
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        k = max(1, round(len(files) * fraction))
        out = dst / "train" / class_dir.name
        out.mkdir(parents=True)
        for f in sorted(rng.sample(files, k)):
            shutil.copy2(f, out / f.name)
        counts[class_dir.name] = k

    for split in (val_dir, "test"):
        shutil.copytree(src / split, dst / split)

    (dst / "subset.json").write_text(
        json.dumps(
            {
                "source": str(src.resolve()),
                "fraction": fraction,
                "seed": seed,
                "train_counts": counts,
                "copied_whole": [val_dir, "test"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    counts = make_subset(args.src, args.dst, args.fraction, seed=args.seed)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
