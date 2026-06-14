from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image

from visionforge.core.batch_predict import (
    BatchPredictionSummary,
    collect_image_paths,
    predict_folder_to_csv,
)


def _write_image(path: Path, width: int, height: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(path)


class TestCollectImagePaths:
    def test_filters_by_extension_and_sorts(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "b.png", 4)
        _write_image(tmp_path / "a.jpg", 4)
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

        paths = collect_image_paths(tmp_path, recursive=False)

        assert [p.name for p in paths] == ["a.jpg", "b.png"]

    def test_recursive_descends_into_subdirs(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "top.png", 4)
        _write_image(tmp_path / "sub" / "deep.png", 4)

        flat = collect_image_paths(tmp_path, recursive=False)
        deep = collect_image_paths(tmp_path, recursive=True)

        assert [p.name for p in flat] == ["top.png"]
        assert {p.name for p in deep} == {"top.png", "deep.png"}


class TestPredictFolderToCsv:
    def test_writes_one_row_per_image_with_caller_values(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "x.png", width=5)
        _write_image(tmp_path / "y.png", width=9)
        out = tmp_path / "out" / "preds.csv"

        # transform encodes the image width; infer echoes it back as a column.
        def transform(img: Image.Image) -> torch.Tensor:
            return torch.tensor([float(img.size[0])])

        def infer(tensor: torch.Tensor) -> dict[str, str]:
            return {"width": str(int(tensor.squeeze(0).item()))}

        summary = predict_folder_to_csv(
            input_dir=tmp_path,
            output_csv=out,
            transform=transform,
            infer=infer,
            fieldnames=["filename", "width"],
            device=torch.device("cpu"),
        )

        assert isinstance(summary, BatchPredictionSummary)
        assert summary.total_processed == 2
        assert summary.failed_files == []

        with out.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_name = {Path(r["filename"]).name: r["width"] for r in rows}
        assert by_name == {"x.png": "5", "y.png": "9"}

    def test_unreadable_image_is_recorded_not_fatal(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "good.png", width=5)
        (tmp_path / "broken.png").write_bytes(b"not a real image")
        out = tmp_path / "preds.csv"

        summary = predict_folder_to_csv(
            input_dir=tmp_path,
            output_csv=out,
            transform=lambda img: torch.zeros(1),
            infer=lambda t: {"ok": "1"},
            fieldnames=["filename", "ok"],
            device=torch.device("cpu"),
        )

        assert summary.total_processed == 1
        assert len(summary.failed_files) == 1
        assert Path(summary.failed_files[0]).name == "broken.png"
