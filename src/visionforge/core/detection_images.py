"""Whether a detection split has an image the trainer would read (ADR-195).

The submit check asks this before a run enters the queue. Ultralytics walks
subfolders and reads formats (``.tif``, ``.avif``...) the stats probe does not
count, so its own list decides when it is installed. Without the ``detection``
extra only the torchvision backend can train, and its reader's extensions do.
"""

from __future__ import annotations

from pathlib import Path


def detection_image_formats() -> frozenset[str]:
    """Ultralytics' image extensions when it is installed, else torchvision's."""
    from visionforge.utils.third_party_privacy import apply_opt_outs

    apply_opt_outs()  # before ultralytics' first import in a `visionforge gui`
    try:
        from ultralytics.data.utils import IMG_FORMATS
    except ImportError:
        from visionforge.core.detection_dataset import _IMAGE_EXTS

        return frozenset(ext.lstrip(".") for ext in _IMAGE_EXTS)
    return frozenset(IMG_FORMATS)


def has_detection_image(directory: Path) -> bool:
    """Any image the trainer would read under ``directory``, subfolders included."""
    formats = detection_image_formats()
    return any(
        p.is_file() and p.suffix.lower().lstrip(".") in formats
        for p in directory.rglob("*")
    )
