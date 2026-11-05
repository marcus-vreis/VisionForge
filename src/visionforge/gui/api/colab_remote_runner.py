"""The script that trains on the Colab VM (ADR-163, phase 2).

It is not imported by the server. ``colab_jobs.build_script`` prepends a
``PAYLOAD`` and sends this file's text to ``colab exec``, so it must stand on its
own: the standard library, plus the ``visionforge-studio`` it installs at the
version the local machine runs, and only APIs that version already has.

The protocol is one JSON object per stdout line, prefixed with ``EVENT_PREFIX``
so the CLI's own output and a library's prints cannot be mistaken for an event.
The training events are the ones the local queue streams (``start``,
``epoch_end``, ``batch``...). The last line is always a ``job_end`` of its own,
``status`` ``ok`` or ``error``, because the exit code of ``colab exec`` says
nothing about the script (a failing script exits 0). For the same reason the
script never calls ``sys.exit``: in google-colab-cli 0.7.5 that hangs ``exec``.
"""

# No `from __future__` here: build_script puts the payload lines first.
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

EVENT_PREFIX = "@@vf@@ "
REMOTE_ROOT = Path("/content/vf")
PACKAGE = "visionforge-studio"
BUNDLE_NAME = "run-bundle.zip"


def emit(event: dict[str, Any]) -> None:
    print(EVENT_PREFIX + json.dumps(event, default=str), flush=True)


def install(version: str) -> None:
    """Install the package at the local version; Colab's own torch is kept."""
    try:
        from importlib.metadata import version as installed

        if installed(PACKAGE) == version:
            return
    except Exception:  # noqa: BLE001 - not installed yet
        pass
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"{PACKAGE}=={version}"],
        check=True,
        capture_output=True,
        text=True,
    )


def single_child(folder: Path) -> Path:
    """A zip of ``dataset/train...`` unpacks to one folder: step into it."""
    entries = [p for p in folder.iterdir() if not p.name.startswith((".", "__MACOSX"))]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return folder


def fetch_dataset(source: dict[str, Any], root: Path) -> Path:
    """Bring the dataset onto the VM from where it already is; return its base folder."""
    target = root / "dataset"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    if source["kind"] == "url":
        archive = root / "dataset.zip"
        urllib.request.urlretrieve(source["url"], archive)  # noqa: S310 - http(s) checked by the server
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        archive.unlink()
        return single_child(target)
    from visionforge.gui.api.dataset_download import download_dataset

    download_dataset(
        source["provider"],
        dataset=source["dataset"],
        out_dir=str(target),
        limit=source.get("limit"),
    )
    return target


def prepare_config(
    config: dict[str, Any], base_dir: Path, source_kind: str
) -> dict[str, Any]:
    """The local config, pointed at the VM's dataset, outputs and device."""
    import torch

    cfg: dict[str, Any] = json.loads(json.dumps(config))
    data = cfg.setdefault("data", {})
    data["base_dir"] = str(base_dir)
    if source_kind == "provider":
        # The provider download writes train/val/test, whatever the local names were.
        data.update(train_dir="train", val_dir="val", test_dir="test")
    data["num_workers"] = min(int(data.get("num_workers") or 0), os.cpu_count() or 1)
    outputs = REMOTE_ROOT / "outputs"
    cfg["output"] = {
        **(cfg.get("output") or {}),
        **{
            f"{sub}_dir": str(outputs / sub)
            for sub in ("models", "graphics", "logs", "reports")
        },
    }
    gpu = torch.cuda.is_available()
    cfg["device"] = {"kind": "cuda" if gpu else "cpu", "gpu_ids": None}
    return cfg


def match_num_classes(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Set ``model.num_classes`` to the class folders of the remote training split.

    The GUI reads the classes from a local dataset; for one that exists only on
    the VM it cannot, so the VM does the same here. Only a multiclass task by
    folders is touched: a binary one keeps the validation error it would raise.
    Returns the change, or None when there was nothing to change.
    """
    if cfg.get("task", "multiclass") != "multiclass":
        return None
    data = cfg.get("data") or {}
    train = Path(data.get("base_dir", ".")) / str(data.get("train_dir", "train"))
    if not train.is_dir():
        return None
    classes = [p for p in train.iterdir() if p.is_dir() and not p.name.startswith(".")]
    model = cfg.setdefault("model", {})
    old = model.get("num_classes")
    if len(classes) < 2 or old == len(classes):
        return None
    model["num_classes"] = len(classes)
    return {"old": old, "new": len(classes)}


def build_block(config: Any) -> Any:
    """The same block choice as the local queue makes for ``config.block``."""
    from visionforge.blocks.classification import ClassificationBlock
    from visionforge.blocks.cross_validation import CrossValidationBlock
    from visionforge.blocks.grid_search import GridSearchBlock
    from visionforge.blocks.model_comparison import ModelComparisonBlock
    from visionforge.blocks.random_search import RandomSearchBlock
    from visionforge.blocks.transfer_learning import TransferLearningBlock

    blocks: dict[str, type[Any]] = {
        "cross_validation": CrossValidationBlock,
        "transfer_learning": TransferLearningBlock,
        "model_comparison": ModelComparisonBlock,
        "grid_search": GridSearchBlock,
        "random_search": RandomSearchBlock,
    }
    return blocks.get(config.block, ClassificationBlock)()


def train(config_dict: dict[str, Any]) -> Any:
    from visionforge.utils.config import ExperimentConfig

    config = ExperimentConfig.model_validate(config_dict)
    block = build_block(config)
    block.setup(config)
    block._progress_callback = emit
    block.run()
    return block.report()


def pack_run(report: Any) -> dict[str, Any] | None:
    """Zip the trained run as an experiment bundle for ``colab download``.

    The bundle is the format the local import reads (ADR-163, phase 3), so the
    VM writes it with the library's own export. A block that leaves no single
    run folder (a search, a comparison) has nothing to bring back yet.
    """
    train_section = report.get("train") if isinstance(report, dict) else None
    run_dir = (train_section or {}).get("run_dir")
    if not run_dir:
        return None
    from visionforge.core.experiment_bundle import export_experiment_bundle

    target = REMOTE_ROOT / BUNDLE_NAME
    target.unlink(missing_ok=True)
    export_experiment_bundle(run_dir, target)
    return {"path": str(target), "size": target.stat().st_size}


def environment() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main(payload: dict[str, Any]) -> None:
    started = time.monotonic()
    try:
        REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
        emit({"event": "remote", "stage": "install", "version": payload["version"]})
        install(payload["version"])
        emit({"event": "remote", "stage": "dataset", "environment": environment()})
        base_dir = fetch_dataset(payload["source"], REMOTE_ROOT)
        config = prepare_config(payload["config"], base_dir, payload["source"]["kind"])
        change = match_num_classes(config)
        if change is not None:
            emit({"event": "remote", "stage": "num_classes", **change})
        emit({"event": "remote", "stage": "train", "device": config["device"]["kind"]})
        report = train(config)
        emit({"event": "remote", "stage": "pack"})
        bundle = pack_run(report)
        emit(
            {
                "event": "job_end",
                "status": "ok",
                "report": report,
                "bundle": bundle,
                "seconds": round(time.monotonic() - started, 1),
            }
        )
    except Exception as exc:  # noqa: BLE001 - every failure must reach the job_end line
        detail = getattr(exc, "stderr", None) or ""
        emit(
            {
                "event": "job_end",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "detail": (str(detail).strip().splitlines() or [""])[-1],
                "traceback": traceback.format_exc(limit=8)[-4000:],
                "seconds": round(time.monotonic() - started, 1),
            }
        )


if "PAYLOAD" in globals():
    main(globals()["PAYLOAD"])
