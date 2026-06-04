# Phase 9 — Anomaly Detection — Design Plan

> Branch: `feat/anomaly` (stacked on `feat/segmentation`; rebases onto
> `development` once the earlier task branches merge — anomaly shares no code with
> them beyond the reused `OutputConfig`/`DeviceConfig`/`TransformConfig`).
> Status: **In progress** — brick 1 (`AnomalyConfig`) landing first.
> Kickoff 2026-06-04 · author: tech-leader iteration
>
> Progress (see TASKS Phase 9 for the live checklist):
> - [ ] brick 1 — `AnomalyConfig` (`utils/anomaly_config.py`) + tests
> - [x] brick 2 — `AnomalyDataModule` (normal-only train, labelled test)
> - [x] brick 3 — `AnomalyModelFactory` (conv autoencoder + PatchCore memory bank)
> - [x] brick 4 — `AnomalyTrainer` (reconstruction / memory-bank fit, image AUROC)
> - [x] brick 5 — `AnomalyBlock` (`setup/run/report`) + ADR-038
> - [x] brick 6 — `/api/anomaly/{schema,run}` run path
> - [x] brick 7 — GUI Anomalia tab (AnomalyPanel, wired end-to-end)
>
> **Phase 9 complete** — anomaly detection runs end-to-end (config → normal-only
> train / labelled test → autoencoder|PatchCore → trainer → block → API → GUI)
> with image-level AUROC / threshold / F1 and a live monitor.

Anomaly detection is the fifth task. It is the first **unsupervised** task: the
model trains on **normal images only** and learns to flag anything that deviates
at inference. This diverges from every prior task — there is no per-image class,
continuous target, box, or mask in training; the only labels are on the *test*
set (normal vs anomalous), used purely to score image-level **AUROC**.

It still **reuses** the shared infrastructure: `OutputConfig`, `DeviceConfig`,
`TransformConfig`, `PreprocessingConfig`, `SchedulerConfig`, the `outputs/` run
layout, the ADR-013 `run.json` contract, and the SSE `start`/`epoch_end`/`end`
event shape so `TrainingOverlay`/`ResultsView` work with zero contract changes.

## 1. Why anomaly needs its own config/block (not an ExperimentConfig block)

Classification assumes an ImageFolder with one labelled subdir per class and a
softmax head trained with cross-entropy. Anomaly detection violates all of that:

- Training data is **a single normal class** — there is no second class to
  classify against, so cross-entropy is undefined.
- The "model" is either a **reconstruction autoencoder** (anomaly score = pixel
  reconstruction error) or a **memory bank of normal patch features** (anomaly
  score = distance to the nearest stored normal patch). Neither has a classifier
  head.
- The metric is **image-level AUROC** over the labelled test split plus a decision
  **threshold** (chosen from the normal-score distribution), not accuracy/F1/mAP.

Forcing this through `ExperimentConfig`/the classification ABC would break the
ABC's `setup(self, config: ExperimentConfig)` contract — the same Liskov argument
as ADR-033 (detection), ADR-036 (regression), ADR-037 (segmentation). Anomaly
therefore gets its own `AnomalyConfig` tree and a standalone `AnomalyBlock`
dispatched through a dedicated `/api/anomaly/*` run path. (→ **ADR-038**.)

## 2. Dataset format — MVTec-AD style

The de-facto standard anomaly layout (MVTec-AD) is **normal-only train + mixed
labelled test**:

```
<base_dir>/
├── train/
│   └── good/            # normal images only (no anomalies in training)
├── test/
│   ├── good/            # normal test images        → label 0
│   ├── <defect_a>/      # anomalous images          → label 1
│   └── <defect_b>/      # any non-"good" subdir is anomalous → label 1
└── ground_truth/        # optional pixel masks (deferred — v1 is image-level)
```

- `normal_dir` (default `"good"`) names the normal subfolder in both splits.
- In `test/`, the `normal_dir` subfolder is label 0; **every other subfolder is
  label 1** (anomalous), so multiple defect types collapse to a binary
  image-level label.
- Pixel-level localization (`ground_truth/` masks) is **deferred**; v1 reports
  image-level AUROC only. The config field names stay generic so the loader can
  grow.

## 3. Model support — autoencoder first, PatchCore second

- **Convolutional autoencoder** (hand-rolled, dependency-light): encoder →
  bottleneck (`latent_dim`) → decoder back to input resolution. Trained to
  minimize reconstruction MSE on normal images; anomaly score per image = mean
  reconstruction error. This is the simplest, fully-trainable baseline and needs
  no pretrained weights.
- **PatchCore** (simplified, memory-bank): a frozen ImageNet backbone
  (`resnet18`/`resnet50`/`wide_resnet50_2`) extracts mid-level patch features over
  the normal train set; a **coreset subsample** (`coreset_ratio`) is stored as the
  memory bank. Anomaly score per image = max over patches of the distance to the
  nearest memory-bank feature. No gradient training — "fit" is one forward pass +
  subsample. Implemented with torch + torchvision only (greedy coreset, no faiss).

## 4. New modules (mirrors the prior task layout)

```
src/visionforge/
├── utils/
│   └── anomaly_config.py     # AnomalyConfig + Model/Data/Training subtrees   ← brick 1
├── core/
│   ├── anomaly_data.py        # AnomalyDataModule — normal train, labelled test ← brick 2
│   └── anomaly_trainer.py     # AE recon loop / PatchCore fit, image AUROC      ← brick 4
├── models/
│   └── anomaly_factory.py     # ConvAutoencoder + PatchCore memory bank         ← brick 3
└── blocks/
    └── anomaly.py             # AnomalyBlock (setup/run/report)                 ← brick 5
```

- `task` Literal is `"anomaly"`; the block dispatches via the dedicated path.
- `AnomalyTrainer` emits the **same SSE events** with anomaly metric fields
  (`auroc`, `threshold`, `image_f1`), so the GUI overlay/results view need no
  contract change. For PatchCore (no epochs) it emits a single fit `epoch_end`.
- `run.json` stays the contract: metrics become `auroc`, `threshold`,
  `image_f1`; artifacts point at the score-histogram / ROC plots + `best.pt`.

## 5. Build sequence (each a small, tested, shippable brick)

1. **`AnomalyConfig` Pydantic models + tests** — model name (`autoencoder`/
   `patchcore`), backbone (PatchCore), `latent_dim` (AE), `coreset_ratio`,
   dataset dirs + `normal_dir`, `threshold_percentile`, loss/epochs. ← **this brick.**
2. **`AnomalyDataModule`** — train loads only `train/<normal_dir>`; test loads
   `test/*` with binary labels (normal_dir → 0, else 1); reuse the image pipeline.
3. **`AnomalyModelFactory`** — `ConvAutoencoder` + `PatchCore` memory bank.
4. **`AnomalyTrainer`** — AE reconstruction loop or PatchCore one-pass fit;
   per-image scores; image-level AUROC + threshold from normal-score percentile;
   best checkpoint; `run.json`.
5. **`AnomalyBlock`** — `setup/run/report`; dispatch via `/api/anomaly/run`. ADR-038.
6. **`/api/anomaly/{schema,run}`** — schema drives the form; run dispatches the
   block with `_progress_callback` → SSE; reuses shared single-run state.
7. **GUI**: activate the Anomalia tab; schema-driven form; results view reads
   anomaly metrics + plots.

## 6. Open decisions (to ratify as ADRs)

- **ADR-038** — Anomaly detection is a standalone config/block/run path (mirrors
  ADR-033/036/037). To be written when brick 5 lands.
- **Threshold strategy** — decision threshold = a high **percentile of the normal
  (train) reconstruction/score distribution** (`threshold_percentile`, default
  95). AUROC is threshold-free; the threshold only drives the reported `image_f1`
  and a usable operating point.
- **Pixel-level localization** — deferred. v1 is image-level AUROC; the
  `ground_truth/` mask path and pixel AUROC/PRO are a later PR.
- **PatchCore nearest-neighbor** — exact torch `cdist` over the coreset (no faiss
  dependency); fine for the modest memory banks a local research tool produces.
