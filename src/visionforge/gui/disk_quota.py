"""Server mode: how much disk a person's folder holds, and what may grow it (ADR-225).

``visionforge gui --server --quota-gb N`` gives every signed-in person the
same ceiling. The folder measured is the profile's: runs, plots, logs,
reports, datasets and previews. The check is made when an action that writes
new data is asked for, not while it writes: a job already running may pass
the ceiling, and the next one is refused until the person deletes runs.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from threading import Lock

GIB = 1024**3

# A folder is walked again after this long. A person who just deleted a run
# does not wait: the gate forgets their figure after every DELETE.
STALE_AFTER_S = 15.0

# The last path segment of the routes that write new data into the person's
# folder: a training or a strategy, a resumed run, a downloaded or derived
# dataset, and the files an evaluation action leaves in the run.
_GROWING = re.compile(
    r"/(?:run|sweep|compare|cv|replicates|replicated-comparison|resume"
    r"|download|sample-dataset|apply|test|gradcam|batch_predict|export_onnx)$"
)


def grows(route: str) -> bool:
    """True when ``route`` (``"METHOD /api/template"``) writes new data to disk."""
    return route.startswith("POST ") and _GROWING.search(route) is not None


def folder_bytes(root: Path) -> int:
    """The bytes of every regular file under ``root``; links are not followed.

    A file or folder that vanishes or cannot be read while walking counts as
    nothing: a run being deleted must not fail the request that asked.
    """
    total = 0
    pending = [root]
    while pending:
        try:
            entries = list(os.scandir(pending.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


class DiskUsage:
    """The size of each person's folder, remembered for a few seconds.

    Walking a folder with a large dataset takes a while; the gate asks before
    every action that writes, and the page asks on every load.
    """

    def __init__(self, stale_after_s: float = STALE_AFTER_S) -> None:
        self._stale_after_s = stale_after_s
        self._seen: dict[Path, tuple[float, int]] = {}
        self._lock = Lock()

    def of(self, root: Path) -> int:
        """Bytes under ``root``, walked again when the last figure is stale."""
        now = time.monotonic()
        with self._lock:
            seen = self._seen.get(root)
        if seen is not None and now - seen[0] < self._stale_after_s:
            return seen[1]
        size = folder_bytes(root)
        with self._lock:
            self._seen[root] = (now, size)
        return size

    def forget(self, root: Path) -> None:
        """Drop the figure of ``root``: something was just deleted there."""
        with self._lock:
            self._seen.pop(root, None)


def size_text(size: int, lang: str) -> str:
    """``size`` bytes as gigabytes with one decimal, in the language's notation."""
    text = f"{size / GIB:.1f}"
    return f"{text.replace('.', ',') if lang == 'pt' else text} GB"
