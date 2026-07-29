# VisionForge — GPU-ready image (ADR-042 part 2).
#
# The single biggest onboarding cost is the PyTorch install: torch is
# hardware-specific by design (ADR-005), and picking the wrong wheel silently
# lands a CPU-only environment on a GPU machine. This image bakes one correct
# combination so `docker run` is the whole install.
#
# Which combination is a build arg rather than a hardcoded base, because
# shipping one image per CUDA version is a maintenance tax and picking a single
# one strands everybody else:
#
#   docker build -t visionforge .                          # default: CUDA 12.4
#   docker build --build-arg CUDA_TAG=cu126 \
#                --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04 \
#                -t visionforge:cu126 .
#   docker build --build-arg VARIANT=cpu --build-arg CUDA_TAG=cpu \
#                --build-arg BASE_IMAGE=ubuntu:22.04 -t visionforge:cpu .
#
# The base is a whole image reference, not just a CUDA version, so the CPU
# build can drop the CUDA runtime entirely — it was ~2.5 GB of driver libraries
# that a CPU wheel never loads.
#
# The host still owns the NVIDIA driver and nvidia-container-toolkit. An image
# cannot ship those, and pretending otherwise would just move the failure.

# Declared before the first FROM on purpose: an ARG used *in* a FROM must be in
# the global scope. Declaring it next to the stage it belongs to reads better
# and does not work — the value is empty there and the tag fails to parse.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ── stage 1: build the SPA ────────────────────────────────────────────────────
# Node exists only here. The runtime stage copies the built assets, so the
# final image has no JavaScript toolchain in it.
FROM node:20-slim AS web

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# vite.config.ts writes to ../src/visionforge/gui/static — relative to the
# frontend folder, so from /build that resolves to /src/... The directory has
# to exist before vite writes into it.
RUN mkdir -p /src/visionforge/gui/static && npm run build


# ── stage 2: runtime ──────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS runtime

# Repeated after FROM on purpose: an ARG declared before FROM is out of scope
# in the stage body.
ARG CUDA_TAG=cu124
ARG VARIANT=gpu
ARG EXTRAS=detection,timm,optuna,tensorboard

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    # Tells the app it is containerized so the native folder picker can explain
    # itself instead of failing with a bare Tk error (there is no display here).
    VISIONFORGE_CONTAINER=1

# libgl/libglib: OpenCV's runtime deps, pulled in by the image pipeline.
# ca-certificates: dataset downloads over HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv brings its own Python 3.13 — no distro PPA needed, and the floor in
# pyproject.toml is met exactly rather than approximately.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install 3.13

WORKDIR /app

# Dependency layer first: application code changes far more often than the
# dependency set, and torch is a ~2 GB download nobody wants to repeat.
COPY pyproject.toml README.md LICENSE ./
COPY src/visionforge/__init__.py ./src/visionforge/__init__.py
RUN uv venv --python 3.13 /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
        "torch>=2.3" "torchvision>=0.18" \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

COPY src/ ./src/
COPY --from=web /src/visionforge/gui/static ./src/visionforge/gui/static
RUN uv pip install --no-cache ".[${EXTRAS}]"

# Runs land in a mounted volume, so the container writes nothing that matters
# to its own filesystem — and not as root, so files created in the mount stay
# usable by the host user.
RUN useradd --create-home --uid 1000 forge \
    && mkdir -p /work/outputs /work/datasets /work/user_models /work/user_tasks \
    && chown -R forge:forge /work
USER forge
WORKDIR /work

EXPOSE 8000

# Bind to 0.0.0.0: the default 127.0.0.1 is only reachable from inside the
# container, which would make the published port useless.
CMD ["visionforge", "gui", "--host", "0.0.0.0", "--port", "8000"]

# Stamped at build time from the package metadata rather than typed here, so
# there is no second place to update when the version changes.
ARG VF_VERSION=dev
LABEL org.opencontainers.image.title="VisionForge" \
      org.opencontainers.image.description="Local-first computer-vision experimentation platform" \
      org.opencontainers.image.source="https://github.com/marcus-vreis/VisionForge" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VF_VERSION}" \
      com.visionforge.cuda="${CUDA_TAG}" \
      com.visionforge.variant="${VARIANT}"
