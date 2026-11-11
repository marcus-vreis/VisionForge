"""``scripts/capture_docs_images.py --check`` guards the README images (#221).

No browser here: the capture itself needs Chrome and a live GUI; these tests
cover the check, the PNG reader and the provenance file it writes.
"""

import struct
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.capture_docs_images import (
    EXPECTED_IMAGES,
    IMAGES_DIR,
    PROVENANCE,
    main,
    png_size,
    provenance,
    remove_workspace,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capture_docs_images.py"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def write_png(path: Path, width: int = 1440, height: int = 960) -> None:
    """The first bytes of a PNG: enough for the IHDR the script reads."""
    ihdr = struct.pack(">II", width, height) + bytes([8, 6, 0, 0, 0])
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\0" * 4
    )


def complete_dir(tmp_path: Path) -> Path:
    for name in EXPECTED_IMAGES:
        write_png(tmp_path / name)
    text = provenance(tmp_path, COMMIT, "9.9.9", "2026-11-11")
    (tmp_path / PROVENANCE).write_text(text, encoding="utf-8")
    return tmp_path


def run_check(images_dir: Path) -> int:
    return main(["--check", "--images-dir", str(images_dir)])


def test_check_passes_with_every_image_and_its_record(tmp_path: Path) -> None:
    assert run_check(complete_dir(tmp_path)) == 0


@pytest.mark.parametrize("name", sorted(EXPECTED_IMAGES))
def test_check_fails_when_an_image_is_missing(
    tmp_path: Path, name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (complete_dir(tmp_path) / name).unlink()
    assert run_check(tmp_path) == 1
    assert f"missing image: {name}" in capsys.readouterr().err


def test_check_fails_on_an_empty_folder(tmp_path: Path) -> None:
    assert run_check(tmp_path) == 1


def test_check_fails_without_the_provenance_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (complete_dir(tmp_path) / PROVENANCE).unlink()
    assert run_check(tmp_path) == 1
    assert f"missing provenance: {PROVENANCE}" in capsys.readouterr().err


def test_check_fails_when_the_provenance_omits_an_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = complete_dir(tmp_path) / PROVENANCE
    text = readme.read_text(encoding="utf-8")
    readme.write_text(text.replace("`vf-code-tab.png`", "vf-code-tab"), "utf-8")
    assert run_check(tmp_path) == 1
    assert "does not list vf-code-tab.png" in capsys.readouterr().err


def test_check_fails_on_a_file_that_is_not_a_png(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (complete_dir(tmp_path) / "vf-colab.png").write_bytes(b"GIF89a not a png")
    assert run_check(tmp_path) == 1
    assert "vf-colab.png is not a PNG file" in capsys.readouterr().err


def test_png_size_reads_the_ihdr(tmp_path: Path) -> None:
    image = tmp_path / "x.png"
    write_png(image, 1280, 720)
    assert png_size(image) == (1280, 720)


def test_provenance_records_commit_version_date_and_size(tmp_path: Path) -> None:
    for name in EXPECTED_IMAGES:
        write_png(tmp_path / name, 1440, 960)
    text = provenance(tmp_path, COMMIT, "9.9.9", "2026-11-11")
    assert f"Commit: `{COMMIT}`" in text
    assert "Versão: `9.9.9`" in text
    assert "Data: 2026-11-11" in text
    for name in EXPECTED_IMAGES:
        assert f"| `{name}` | {EXPECTED_IMAGES[name]} | 1440×960 |" in text


def test_the_committed_images_pass_the_check() -> None:
    assert run_check(IMAGES_DIR) == 0


def test_cli_check_exit_status(tmp_path: Path) -> None:
    command = [sys.executable, str(SCRIPT), "--check", "--images-dir"]
    empty = subprocess.run(
        [*command, str(tmp_path / "empty")], capture_output=True, text=True
    )
    assert empty.returncode == 1
    assert "missing image: vf-classification.png" in empty.stderr
    full = subprocess.run(
        [*command, str(complete_dir(tmp_path))], capture_output=True, text=True
    )
    assert full.returncode == 0


def test_remove_workspace_deletes_the_folder(tmp_path: Path) -> None:
    workspace = tmp_path / "vf-docs-images"
    (workspace / "outputs" / "logs").mkdir(parents=True)
    (workspace / "outputs" / "logs" / "visionforge.log").write_text("x", "utf-8")
    remove_workspace(workspace, attempts=1)
    assert not workspace.exists()
