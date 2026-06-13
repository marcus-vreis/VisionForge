# Dataset import (Roboflow & friends) — analysis & plan

> Status: **idea / not started.** This is design only; no code yet. It stays
> inside the local-only philosophy: the network is touched once, at the user's
> request, to fetch a dataset to disk — never on a training path.

## Where we are today

Detection already consumes a Roboflow export with **zero new code**: a Roboflow
"YOLOv8" export ships a `data.yaml` + `train/val/test` folders, and
`DetectionDataConfig.data_yaml` points straight at it (the GUI dataset source is
even labelled "Ultralytics / Roboflow"). Classification/segmentation consume any
ImageFolder/mask layout the same way. So the manual flow — *download the export,
point VisionForge at the folder* — works now.

The idea here is to remove the manual download step: pull a dataset **directly**
from Roboflow (and similar sources) into a local folder, then hand off to the
existing data flow.

## Why this is safe for a local-first tool

Fetching a dataset is a **setup action**, not a core dependency — the same shape
as downloading pretrained weights (already done on first train). The download is:
user-initiated, one-shot, writes to a local directory, and once on disk the
training path is fully offline. No core path gains a network round-trip, so
ADR-020 / the local-first value proposition holds.

## Design

A small **dataset-source** abstraction, started with two concrete sources:

1. **Roboflow** — via the official `roboflow` Python SDK (or its REST download
   URL). Inputs: workspace, project, version, format (`yolov8`/`coco`/`folder`),
   and an API key supplied by the user at call time. The key is **never stored**
   in config or `run.json`; it lives only for the request.
2. **Generic URL / zip** — any direct link to a `.zip`/`.tar` dataset archive;
   download, checksum (optional), extract.

Both resolve to a local dataset root, after which auto-detect
(`/api/dataset/detect`, `/detection/dataset/stats`) takes over unchanged.

### Surface
- **CLI**: `visionforge fetch-dataset roboflow --workspace … --project … --version N --api-key …`
  → prints the local path. Scriptable, no GUI needed.
- **GUI**: an "Importar dataset" action on the dataset section that opens a small
  form (source + fields + key), shows download progress, and on success fills the
  dataset path field. Reuses the existing dataset-stats preview.
- **API**: `POST /api/dataset/import` (source-tagged body) running the fetch in a
  worker thread, streaming progress over the existing SSE channel.

### Dependencies
`roboflow` is an **optional extra** (`pip install -e ".[datasets]"`), bound
lazily like `ultralytics` (ADR — detection extra). Missing extra → a clear
"install the datasets extra" error, never a hard import at startup.

## Out of scope / open questions
- **Uploading** to Roboflow, or any write-back — no. Read-only fetch only.
- **Credential storage** — keys stay request-scoped; revisit only if users ask
  for a saved-profile convenience, and then via an explicit opt-in local file.
- **License/ToS** — the user is responsible for the dataset's license; surface
  the source's license string when the API returns one.
- Which formats to support first beyond `yolov8`? (Leaning: `yolov8` + `folder`
  for classification, add `coco` only if a task needs it.)
