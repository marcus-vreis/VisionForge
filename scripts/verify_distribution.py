"""Exercise an installed wheel from a directory outside the source checkout."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def validate_package_origin(package_file: Path, prefix: Path, checkout: Path) -> Path:
    """Reject editable imports and packages outside the selected environment."""
    source = package_file.resolve(strict=True)
    if not source.is_relative_to(prefix.resolve(strict=True)):
        raise ValueError(f"Package is outside the installed environment: {source}")
    if source.is_relative_to(checkout.resolve(strict=True)):
        raise ValueError(f"Package still resolves inside the checkout: {source}")
    return source.parent


def validate_static_assets(package: Path) -> list[str]:
    """Verify that the installed HTML references packaged local JS/CSS assets."""
    static = package / "gui/static"
    html = (static / "index.html").read_text(encoding="utf-8")
    assets = re.findall(r'(?:src|href)=["\']/?(assets/[^"\']+)["\']', html)
    if not any(name.endswith(".js") for name in assets):
        raise ValueError("Installed SPA has no JavaScript entry point")
    for name in assets:
        asset = (static / name).resolve(strict=True)
        if not asset.is_relative_to(static.resolve()) or not asset.is_file():
            raise ValueError(f"Invalid installed asset: {name}")
    return assets


def _cli(workspace: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    )
    completed = subprocess.run(
        [sys.executable, "-m", "visionforge", *arguments],
        cwd=workspace,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    return completed.stdout


def validate_run_metadata(report: dict[str, Any]) -> None:
    """Check the recorded CPU run and the config schema version from ADR-039."""
    if report.get("device_used") != "cpu" or len(report.get("history", [])) != 1:
        raise ValueError("Installed training did not complete one CPU epoch")
    version = report.get("config", {}).get("schema_version")
    if not isinstance(version, int) or version < 1:
        raise ValueError("Installed training is missing its config schema version")
    if "torch" not in report.get("environment", {}):
        raise ValueError("Installed training is missing environment metadata")


def verify_workflow(workspace: Path) -> dict[str, Any]:
    """Run synthetic CPU training and evidence commands through installed CLI."""
    from PIL import Image

    workspace.mkdir(parents=True, exist_ok=False)
    dataset = workspace / "data"
    for split_index, split in enumerate(("train", "val", "test")):
        for label in range(2):
            folder = dataset / split / f"class{label}"
            folder.mkdir(parents=True)
            for sample in range(4):
                Image.new(
                    "RGB",
                    (32, 32),
                    (split_index * 60 + sample, label * 120, sample * 20),
                ).save(folder / f"{sample}.png")
    config = {
        "name": "wheel_smoke",
        "task": "multiclass",
        "block": "classification",
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "data": {
            "base_dir": str(dataset),
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {"image_size": 32, "augment": False},
        },
        "training": {"epochs": 1, "batch_size": 4, "seed": 17},
        "device": {"kind": "cpu"},
        "output": {
            f"{name}_dir": str(workspace / "outputs" / name)
            for name in ("models", "graphics", "logs", "reports")
        },
    }
    config_file = workspace / "config.yaml"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    help_text = _cli(workspace, "--help")
    if any(name not in help_text for name in ("audit-data", "bundle", "verify-bundle")):
        raise ValueError("Installed CLI is missing evidence commands")
    audit = json.loads(_cli(workspace, "audit-data", str(dataset)))
    if not audit["ok"]:
        raise ValueError(f"Synthetic dataset audit failed: {audit}")
    _cli(workspace, "run", str(config_file))
    reports = list((workspace / "outputs/models").rglob("run.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected one training report, found {len(reports)}")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    validate_run_metadata(report)
    if not Path(report["artifacts"]["model"]).is_file():
        raise ValueError("Installed training has no model artifact")
    bundle = workspace / "experiment.zip"
    _cli(workspace, "bundle", str(reports[0].parent), str(bundle))
    manifest = json.loads(_cli(workspace, "verify-bundle", str(bundle)))
    if "run.json" not in manifest["files"] or len(manifest["files"]) < 2:
        raise ValueError("Installed bundle is missing the run or model")
    return {"report": str(reports[0]), "bundle_members": sorted(manifest["files"])}


def main() -> None:
    """Validate a wheel installation without importing the checkout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    arguments = parser.parse_args()
    if os.environ.get("PYTHONPATH"):
        raise ValueError("Unset PYTHONPATH before testing the installed wheel")
    if Path.cwd().resolve().is_relative_to(arguments.checkout.resolve()):
        raise ValueError("Run the distribution check outside the checkout")
    import visionforge

    package = validate_package_origin(
        Path(visionforge.__file__), Path(sys.prefix), arguments.checkout
    )
    assets = validate_static_assets(package)
    result = verify_workflow(arguments.workspace)
    print(
        json.dumps(
            {
                "version": visionforge.__version__,
                "package": str(package),
                "assets": assets,
                **result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
