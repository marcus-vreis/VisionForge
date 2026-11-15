"""``scripts/capture_docs_images.py --check`` guards the README images (#221).

No browser here: the capture itself needs Chrome and a live GUI; these tests
cover the check, the PNG reader and the provenance file it writes, and the
film of a training run that is built from frames (small generated ones here).
"""

import struct
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts.capture_docs_images import (
    EXPECTED_IMAGES,
    FILM_GIF,
    FILM_GIF_WIDTH,
    FILM_MAX_FRAMES,
    FILM_WEBP_WIDTH,
    IMAGES_DIR,
    LEARN_STILL_WIDTH,
    LEARN_STILLS,
    PROVENANCE,
    build_film,
    film_frames,
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
    (tmp_path / FILM_GIF).write_bytes(b"GIF89a" + bytes(8))
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


def test_check_fails_without_the_film(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (complete_dir(tmp_path) / FILM_GIF).unlink()
    assert run_check(tmp_path) == 1
    assert f"missing image: {FILM_GIF}" in capsys.readouterr().err


def test_check_fails_on_a_film_that_is_not_a_gif(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (complete_dir(tmp_path) / FILM_GIF).write_bytes(b"RIFF....WEBP")
    assert run_check(tmp_path) == 1
    assert f"{FILM_GIF} is not a GIF file" in capsys.readouterr().err


def test_check_fails_when_the_provenance_omits_the_film(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = complete_dir(tmp_path) / PROVENANCE
    text = readme.read_text(encoding="utf-8")
    readme.write_text(text.replace(f"`{FILM_GIF}`", FILM_GIF), "utf-8")
    assert run_check(tmp_path) == 1
    assert f"does not list {FILM_GIF}" in capsys.readouterr().err


def write_frames(folder: Path, shades: list[int], size: tuple[int, int]) -> Path:
    """One grey PNG per shade, numbered as the browser script numbers them."""
    folder.mkdir()
    for index, shade in enumerate(shades, start=1):
        Image.new("RGB", size, (shade, shade, shade)).save(folder / f"{index:03d}.png")
    return folder


def test_film_frames_drops_a_frame_equal_to_the_one_before(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", [10, 10, 10, 90, 90, 10], (12, 8))
    kept = [path.name for path in film_frames(frames)]
    assert kept == ["001.png", "004.png", "006.png"]


def test_film_frames_keeps_the_first_and_the_last_when_it_thins(
    tmp_path: Path,
) -> None:
    shades = list(range(0, 2 * (FILM_MAX_FRAMES + 20), 2))
    frames = write_frames(tmp_path / "frames", shades, (12, 8))
    kept = film_frames(frames)
    assert len(kept) == FILM_MAX_FRAMES
    assert kept[0].name == "001.png"
    assert kept[-1].name == f"{len(shades):03d}.png"
    assert kept == sorted(kept)


def test_build_film_writes_the_gif_and_the_interface_files(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", [20, 120, 220], (1440, 960))
    images = tmp_path / "images"
    images.mkdir()
    for source in LEARN_STILLS.values():
        Image.new("RGB", (1440, 960), (30, 60, 90)).save(images / source)
    assets = tmp_path / "assets"

    build_film(frames, images, assets)

    with Image.open(images / FILM_GIF) as gif:
        assert gif.format == "GIF"
        assert gif.size == (FILM_GIF_WIDTH, round(960 * FILM_GIF_WIDTH / 1440))
        assert gif.n_frames == 3
    with Image.open(assets / "training-film.webp") as film:
        assert film.format == "WEBP"
        assert film.size == (FILM_WEBP_WIDTH, round(960 * FILM_WEBP_WIDTH / 1440))
        assert film.n_frames == 3
    for name in LEARN_STILLS:
        with Image.open(assets / name) as still:
            assert still.format == "WEBP"
            assert still.size == (
                LEARN_STILL_WIDTH,
                round(960 * LEARN_STILL_WIDTH / 1440),
            )


def test_build_film_refuses_a_single_frame(tmp_path: Path) -> None:
    frames = write_frames(tmp_path / "frames", [50, 50], (12, 8))
    with pytest.raises(RuntimeError, match=r"the film has 1 frame"):
        build_film(frames, tmp_path, tmp_path / "assets")


def test_the_interface_shows_the_stills_at_their_real_size() -> None:
    """LearnOverlay.tsx gives the images a width and height; they must be the
    ones this script writes, or the page jumps when an image loads."""
    overlay = (
        ROOT / "frontend" / "src" / "components" / "learn" / "LearnOverlay.tsx"
    ).read_text(encoding="utf-8")
    still_height = round(960 * LEARN_STILL_WIDTH / 1440)
    film_height = round(960 * FILM_WEBP_WIDTH / 1440)
    assert f"width: {LEARN_STILL_WIDTH}, height: {still_height}" in overlay
    assert f"width: {FILM_WEBP_WIDTH}, height: {film_height}" in overlay


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
