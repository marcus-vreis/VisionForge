"""Recapture the README images in docs/images from a throwaway GUI (issue #221).

    python scripts/capture_docs_images.py          # capture, then write README.md
    python scripts/capture_docs_images.py --check  # fail if an image is missing

The capture starts ``visionforge gui`` from this checkout on a free port, with
``--offline`` and without a GPU, in a fresh work folder and a temporary
``VISIONFORGE_HOME``: no runs, profiles or Colab login of the user's leak in.
The work folder has a short path that names no user (``vf-docs-images`` at the
drive root), since the dataset path shows in an image. The browser
part is ``frontend/scripts/capture-docs-images.mjs`` (Playwright with the
installed Chrome; run ``npm ci`` in ``frontend`` first). It hides the version
in the header before every shot and checks in the DOM that it is hidden.
``docs/images/README.md`` records commit, version, date and size of each image.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "docs" / "images"
FRONTEND = ROOT / "frontend"
NODE_SCRIPT = FRONTEND / "scripts" / "capture-docs-images.mjs"
PROVENANCE = "README.md"
WORKSPACE_NAME = "vf-docs-images"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Every image the script captures, with what it shows.
EXPECTED_IMAGES: dict[str, str] = {
    "vf-classification.png": "Classificação, etapa Dados com o dataset de exemplo",
    "vf-replicates.png": "Regressão, etapa Estratégia com Réplicas ligado",
    "vf-code-tab.png": "Aba Código do painel de treino, após 1 época na CPU",
    "vf-colab.png": "Canto do Colab aberto, sem CLI instalado nem conta Google",
    "vf-strategy.png": "Classificação, etapa Estratégia com as seis estratégias",
    "vf-guide-menu.png": "Menu de guias do cabeçalho aberto",
    "vf-review.png": "Classificação, etapa Revisão antes de treinar",
    "vf-training.png": "Painel de treino na aba Log, após 1 época na CPU",
    "vf-results.png": "Resultados de uma classificação: métricas e gráficos",
    "vf-run-detail.png": "Histórico, detalhe de uma execução com as ações de publicação",
    "vf-transfer.png": "Classificação, etapa Modelo com transfer learning ligado",
    "vf-preprocessing.png": "Etapa Imagem e aumento com um filtro e o preview dele",
    "vf-datasets-explorer.png": "Ferramentas de dataset, aba Explorar com o dataset de exemplo",
}


def png_size(path: Path) -> tuple[int, int]:
    """Width and height from a PNG's IHDR chunk."""
    with path.open("rb") as source:
        head = source.read(24)
    if len(head) < 24 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise ValueError(f"{path.name} is not a PNG file")
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


def problems(images_dir: Path) -> list[str]:
    """What is missing from ``images_dir``: images, or their provenance lines."""
    found: list[str] = []
    for name in EXPECTED_IMAGES:
        image = images_dir / name
        if not image.is_file():
            found.append(f"missing image: {name}")
            continue
        try:
            png_size(image)
        except ValueError as error:
            found.append(str(error))
    readme = images_dir / PROVENANCE
    if not readme.is_file():
        found.append(f"missing provenance: {PROVENANCE}")
        return found
    text = readme.read_text(encoding="utf-8")
    found.extend(
        f"{PROVENANCE} does not list {name}"
        for name in EXPECTED_IMAGES
        if f"`{name}`" not in text
    )
    return found


def check(images_dir: Path) -> int:
    """Exit status of ``--check``: 0 when every image and its record exist."""
    found = problems(images_dir)
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print("run: python scripts/capture_docs_images.py", file=sys.stderr)
        return 1
    print(f"{len(EXPECTED_IMAGES)} images in {images_dir}")
    return 0


def provenance(images_dir: Path, commit: str, version: str, date: str) -> str:
    """The text of docs/images/README.md for the images in ``images_dir``."""
    rows = []
    for name, shows in EXPECTED_IMAGES.items():
        width, height = png_size(images_dir / name)
        rows.append(f"| `{name}` | {shows} | {width}×{height} |")
    return "\n".join(
        [
            "# Imagens do README",
            "",
            "Geradas por `python scripts/capture_docs_images.py` (issue #221): a GUI",
            "deste checkout num workspace temporário, `--offline`, sem GPU, Chrome",
            "a 1440×960, tema escuro, pt-BR. A versão do cabeçalho fica oculta na",
            "captura. Não edite à mão: rode o script de novo e confira com",
            "`python scripts/capture_docs_images.py --check`.",
            "",
            f"- Commit: `{commit}`",
            f"- Versão: `{version}`",
            f"- Data: {date}",
            "",
            "| Imagem | Mostra | Dimensões (px) |",
            "| --- | --- | --- |",
            *rows,
            "",
        ]
    )


def free_port() -> int:
    """A TCP port nothing listens on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_health(base_url: str, timeout: float = 120.0) -> dict[str, str]:
    """The server's /api/health once it answers."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=5) as reply:
                info: dict[str, str] = json.load(reply)
                return info
        except (urllib.error.URLError, ConnectionError):
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def git_commit() -> str:
    """The checked-out commit the images were taken from."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def default_workspace() -> Path:
    """The work folder: its dataset path shows in an image, so it is short and
    names no user. ``vf-docs-images`` at the drive root on Windows, in the
    temporary folder elsewhere."""
    if os.name == "nt":
        return Path(Path.cwd().anchor) / WORKSPACE_NAME
    return Path(tempfile.gettempdir()) / WORKSPACE_NAME


def remove_workspace(workspace: Path, attempts: int = 20) -> None:
    """Delete the work folder this script created.

    On Windows the stopped server's log file stays locked for a moment after
    the process exits, so the removal is retried for a few seconds.
    """
    for _ in range(attempts):
        shutil.rmtree(workspace, ignore_errors=True)
        if not workspace.exists():
            return
        time.sleep(0.5)
    print(f"could not remove {workspace}; delete it by hand", file=sys.stderr)


def capture(images_dir: Path, workspace: Path) -> int:
    """Start a throwaway GUI, run the browser script, write the provenance."""
    node = shutil.which("node")
    if node is None:
        print("node not found on PATH", file=sys.stderr)
        return 2
    if not (FRONTEND / "node_modules" / "@playwright" / "test").is_dir():
        print("run `npm ci` in frontend first", file=sys.stderr)
        return 2
    if workspace.exists():
        # Never reuse or delete a folder this script did not create.
        print(f"{workspace} exists: remove it or pass --workspace", file=sys.stderr)
        return 2
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    workspace.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="vf-docs-home-", ignore_cleanup_errors=True
        ) as home:
            env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])
                ),
                "VISIONFORGE_HOME": home,
                "VISIONFORGE_NO_BROWSER": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            command = [sys.executable, "-m", "visionforge", "gui", "--port", str(port)]
            server = subprocess.Popen(
                [*command, "--no-browser", "--offline"], cwd=workspace, env=env
            )
            try:
                version = wait_health(base_url)["version"]
                script = [node, str(NODE_SCRIPT), "--base-url", base_url]
                status = subprocess.run(
                    [*script, "--out", str(images_dir)], cwd=FRONTEND, check=False
                ).returncode
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
    finally:
        remove_workspace(workspace)
    if status != 0:
        return status
    date = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    text = provenance(images_dir, git_commit(), version, date)
    (images_dir / PROVENANCE).write_text(text, encoding="utf-8", newline="\n")
    return check(images_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recapture the README images in docs/images."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that every image and its provenance line exist",
    )
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="folder the GUI works in; must not exist (default: vf-docs-images at the drive root)",
    )
    args = parser.parse_args(argv)
    if args.check:
        return check(args.images_dir)
    return capture(args.images_dir, args.workspace or default_workspace())


if __name__ == "__main__":
    sys.exit(main())
