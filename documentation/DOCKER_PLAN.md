# Docker + `visionforge doctor` — analysis & plan

> Status: **planning / spec for the agent team**. Decision recorded in
> **ADR-042**. No code yet — this doc is the design Claude Code analyses and
> implements. It deliberately stays inside the local-only philosophy: this is
> about *removing install friction and making runs reproducible*, not about
> turning VisionForge into a cloud service.

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

## Part 1 — `visionforge doctor` (ship first; small, high value)

A new CLI subcommand: `visionforge doctor`.

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
