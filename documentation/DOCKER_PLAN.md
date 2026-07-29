# Docker + `visionforge doctor` — analysis & plan

> Status: **Part 1 (`visionforge doctor`) shipped** (ADR-042 slice 1);
> **Part 2 (Docker image) is still planned**. It deliberately stays inside the
> local-only philosophy: this is about *removing install friction and making
> runs reproducible*, not about turning VisionForge into a cloud service.

## Problem

The single biggest onboarding friction is the PyTorch install: torch/torchvision
are user-managed per hardware (ADR-005) because no resolver can pick the right
CUDA build, and picking wrong silently lands a CPU-only or broken environment.
This bit us in this very session — the sandbox couldn't get the right wheel.
Two complementary fixes, neither of which compromises local-only:

1. **A Docker image** that bundles the correct torch build — one command, no
   install dance, reproducible across machines.
2. **A `visionforge doctor` command** for users who run bare-metal and want the
   *exact* pip command for their GPU instead of guessing.

Explicitly out of scope (and why): **Kubernetes / cloud orchestration.** For a
single local GPU this is complexity with no payoff — problem-driven tooling, not
resume-driven. If multi-job queueing on one machine is ever wanted, a lightweight
job queue (RQ/SQLite) is the right tool, not k8s.

## Part 1 — `visionforge doctor` ✅ shipped

A CLI subcommand: `visionforge doctor` (logic in `utils/doctor.py`, subparser in
`__main__.py`).

Behaviour:
- Detect GPU + driver via `nvidia-smi` (parse "CUDA Version: 12.x" from its
  header). Handle "no nvidia-smi" → recommend the CPU build.
- Map the detected driver CUDA version to the nearest supported wheel index
  (`cu118` / `cu121` / `cu124` / `cu126` / `cpu`) — the same set already wired in
  `pyproject.toml [tool.uv.sources]`.
- Print the **exact** install command, e.g.
  `pip install -e ".[cu124]"` (or the `uv pip` equivalent), and the matching
  `--index-url`.
- `--fix` flag: optionally run the install (with confirmation), never silently.
- Also report: Python version vs `requires-python>=3.13`, whether torch is
  importable and CUDA-visible, and a one-line verdict.

Why first: it's a self-contained CLI addition (no new heavy deps), CPU-CI
testable by mocking `nvidia-smi` output, and it directly serves the
"facilitate requirements" goal. Lives in `__main__.py` / a `utils/doctor.py`.

## Part 2 — Docker image

Goal: `docker run --gpus all -p 8000:8000 visionforge` boots the GUI with a
working GPU torch, no host install beyond the NVIDIA driver + container toolkit.

Design:
- **Base**: an `nvidia/cuda:<ver>-runtime-ubuntu22.04` image matching a supported
  wheel (pick one default CUDA, e.g. cu124; document how to rebuild for others).
- **Python 3.13** installed in-image (the project floor).
- **Multi-stage build**: stage 1 builds the React SPA (`npm run build` →
  `gui/static`); stage 2 is the runtime with the Python package + the matching
  torch wheel baked in. Keeps the final image lean (no Node in runtime).
- **GPU passthrough**: documented `--gpus all` (needs `nvidia-container-toolkit`
  on the host — that's the one thing that stays the user's responsibility, like
  the driver).
- **Volumes**: mount the user's datasets read-only and `outputs/` read-write so
  runs persist outside the container and no user data is baked into the image.
- **CPU image variant**: a `cpu` build target for machines without a GPU.
- **`docker-compose.yml`**: the common case (GUI on :8000, dataset + outputs
  volumes, `deploy.resources.reservations.devices` for the GPU) so the user runs
  `docker compose up`.

What stays the user's job: installing the NVIDIA driver and
`nvidia-container-toolkit` on the host. The image can't ship those.

## Implementation order
1. `visionforge doctor` (+ tests mocking `nvidia-smi`) — ship standalone.
2. `Dockerfile` (multi-stage, GPU default) + `.dockerignore`.
3. `docker-compose.yml` (GPU) + a documented CPU override.
4. README "Run with Docker" section; ADR-042 already records the decision.
5. (Optional, later) publish a built image to a registry so users skip the build.

## Open questions for the agent team
- Which CUDA base tag is the sensible default (cu124 vs cu126)? Pick one, document
  the rebuild path for the others — don't try to ship all in one image.
- Bake a default model/weights cache, or always download on first run? (Leaning:
  download on first run, keep the image lean.)
- Does `tkinter` (native folder picker, ADR-018) work headless in the container?
  If not, the GUI folder-picker degrades to manual path entry inside Docker —
  document it, or gate the picker on a "not containerized" check.

---

## Part 2 status — 2026-07-29

**Built, run and verified** (CPU variant). The first build failed and the
first run failed differently — see ADR-071 for what that exposed. The GPU
variant remains unverified: this machine's Docker has no GPU passthrough
configured.

Shipped: `Dockerfile` (multi-stage), `.dockerignore`, `docker-compose.yml`,
README section, and the containerized folder-picker behaviour with tests.

### The open questions, answered

- **Which CUDA base?** None, exclusively. Shipping one image per CUDA version is
  a maintenance tax, and picking a single one strands everybody else — so
  `CUDA_TAG` / `CUDA_IMAGE` are build args with `cu124` / `12.4.1` as the
  default, and `VARIANT=cpu` gives the CPU build from the same file.
- **Bake a weights cache?** No — download on first run, keep the image lean. A
  baked cache would grow the image by hundreds of MB to save a one-time
  download that most users make once per model anyway.
- **Does the tkinter picker work headless?** It cannot: there is no display in
  the container. It already degraded without crashing (Tk raises, the route
  catches), but the message read like a bug. The image sets
  `VISIONFORGE_CONTAINER=1` and the route now checks it *before* touching Tk,
  returning "o seletor nativo não abre dentro do container — digite o caminho
  montado, por exemplo /work/datasets/meu-dataset". Two tests pin both
  branches.

### Verified on the CPU image

| Check | Result |
|---|---|
| `docker build` (CPU variant) | ✅ 509 MB |
| `uv python install 3.13` in-image | ✅ |
| SPA copied from the web stage | ✅ 42 files |
| `visionforge selftest --quick` inside | ✅ 5/5 |
| `GET /api/system/info` on the published port | ✅ `version: 0.1.0`, `platform: Linux` |
| SPA served at `/` | ✅ 200 text/html |
| Headless folder picker explains itself | ✅ names the mounted path |
| Runs as uid 1000, workdir `/work` | ✅ |

The CPU image dropped from **6.01 GB to 509 MB** once `BASE_IMAGE` let it start
from `ubuntu:22.04` instead of inheriting the CUDA runtime — a 12x cut for a
variant that never loads a driver library.

### Still unverified

- The **GPU variant**: `--gpus all` and `torch.cuda.is_available()` inside the
  image. Needs a host with `nvidia-container-toolkit` configured for Docker.
- Files written into a mounted `outputs/` being usable from a Linux host (uid
  1000 maps cleanly there; Docker Desktop on Windows virtualizes ownership, so
  testing it here would not prove the Linux case).
