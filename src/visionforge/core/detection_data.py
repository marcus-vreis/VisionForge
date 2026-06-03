"""Resolve an Ultralytics ``data.yaml`` for a detection run.

Either passes through an explicit ``data_yaml`` from the config, or synthesizes
one from a YOLO-layout ``base_dir`` by detecting the train/val/test image
folders and the class names. No ``ultralytics`` import here — this module only
produces the dataset spec file the Trainer (brick 3) hands to ``YOLO.train``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger

from visionforge.utils.detection_config import DetectionConfig

# Image folder location per split for the two common YOLO layouts, in priority
# order. The first existing candidate wins; its value is the path written into
# data.yaml, relative to ``path`` (the dataset root).
_LAYOUTS = (
    lambda split: f"images/{split}",  # <base>/images/<split>
    lambda split: f"{split}/images",  # <base>/<split>/images
)


def resolve_yolo_split(base: Path, split: str) -> tuple[Path, Path] | None:
    """Return ``(images_dir, labels_dir)`` for a YOLO split, or None if absent.

    Supports both ``images/<split>`` + ``labels/<split>`` and
    ``<split>/images`` + ``<split>/labels`` layouts. The single source of truth
    for split resolution, shared by the trainer and the dataset-stats probe.
    """
    candidates = (
        (base / "images" / split, base / "labels" / split),
        (base / split / "images", base / split / "labels"),
    )
    for images_dir, labels_dir in candidates:
        if images_dir.is_dir():
            return images_dir, labels_dir
    return None


class DetectionDataModule:
    """Produces the Ultralytics ``data.yaml`` for a `DetectionConfig`."""

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config

    def resolve_data_yaml(self, out_dir: Path | None = None) -> Path:
        """Return a usable ``data.yaml`` path.

        Args:
            out_dir: where a synthesized spec is written; defaults to
                ``output.models_dir / name``. Ignored when the config already
                points at an explicit ``data_yaml``.

        Raises:
            ValueError: if a base_dir layout is missing the train or val split.
        """
        data = self._config.data
        if data.data_yaml is not None:
            return data.data_yaml

        assert data.base_dir is not None  # config guarantees one source
        base = data.base_dir.resolve()
        splits = self._detect_splits(base)

        spec: dict[str, object] = {"path": str(base)}
        spec.update(splits)
        names = self._resolve_names(base)
        spec["names"] = names
        spec["nc"] = len(names)

        target_dir = out_dir or (self._config.output.models_dir / self._config.name)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "data.yaml"
        target.write_text(
            yaml.safe_dump(spec, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("Synthesized detection data.yaml at {}", target)
        return target

    # ── private ───────────────────────────────────────────────────────────────

    def _detect_splits(self, base: Path) -> dict[str, str]:
        """Map each present split to its image folder, relative to ``base``.

        train and val are required; test is optional.
        """
        found: dict[str, str] = {}
        for split in ("train", "val", "test"):
            for layout in _LAYOUTS:
                rel = layout(split)
                if (base / rel).is_dir():
                    found[split] = rel
                    break

        missing = [s for s in ("train", "val") if s not in found]
        if missing:
            raise ValueError(
                f"Detection base_dir '{base}' is missing required split(s): "
                f"{', '.join(missing)}. Expected images under "
                f"'images/<split>' or '<split>/images'."
            )
        return found

    def _resolve_names(self, base: Path) -> list[str]:
        """Class names from config, else a classes.txt, else generated."""
        if self._config.data.class_names:
            return list(self._config.data.class_names)

        for candidate in ("classes.txt", "names.txt"):
            path = base / candidate
            if path.is_file():
                names = [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if names:
                    return names

        return [f"class_{i}" for i in range(self._config.model.num_classes)]


__all__ = ["DetectionDataModule", "resolve_yolo_split"]
