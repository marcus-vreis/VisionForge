# Changelog

All notable changes to VisionForge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0, the config schema and the HTTP API may change
between minor releases. Configs carry a `schema_version` and are migrated on
load (ADR-039), so a config exported from an older release keeps working.

Every entry links the ADR that records *why* the change was made; the full
reasoning lives in [`documentation/DECISIONS.md`](documentation/DECISIONS.md).

## [Unreleased]

## [0.1.0] — 2026-07-29

First public release. Everything below already shipped on `main`; this is the
point at which it becomes a version other people can install and cite.

### Tasks

- **Classification** — ResNet 18/34/50/101, EfficientNet B1/B7, VGG 16/19,
  AlexNet, plus any `timm` backbone (ADR-051) or a drop-in custom model
  (ADR-048/049). Binary, multiclass and multilabel.
- **Object detection** — Ultralytics YOLOv8/9/10/11/12/26 and RT-DETR, plus a
  torchvision backend (Faster R-CNN, SSD, RetinaNet) with its own loop
  (ADR-035). Full Ultralytics hyperparameter surface (ADR-040).
- **Image regression** — CSV-manifest datasets, CNN backbone + linear head,
  MSE/RMSE/MAE/R².
- **Semantic segmentation** — DeepLabV3, FCN, LR-ASPP and a hand-rolled U-Net;
  mean IoU, Dice, pixel accuracy, with `ignore_index` respected everywhere.
- **Anomaly detection** — convolutional autoencoder and PatchCore, image-level
  AUROC on an MVTec-style layout.
- **Your own task** — define a whole new task family in one documented Python
  file (`visionforge new-task`), with sweeps, replicates and the live monitor
  for free (ADR-058).

### Strategies

- Single run, K-fold cross-validation (classification, regression,
  segmentation), grid / random / Optuna-TPE sweeps (ADR-052), transfer learning
  (ADR-046/047), model comparison, and multi-seed replicates reporting
  `mean ± 95% CI` (ADR-056).
- **Paired significance testing** (ADR-061) — compares N configurations over
  the *same* seeds, picks and justifies a paired t-test or Wilcoxon, reports
  Cohen's `d_z`, a bootstrap CI of the difference and Holm-Bonferroni
  correction. It refuses to compare runs whose seeds do not line up, and flags
  when the seed count makes significance unreachable, so "not significant" is
  never read as "no effect".
- **Paper-ready output** — every advanced report is also written as a
  `booktabs` LaTeX table with notes stating what each interval covers.

### Reproducibility

- Versioned `run.json` for every run (ADR-013) carrying the full config, seed,
  per-epoch history, `environment` (Python, torch, CUDA, cuDNN, GPU model —
  ADR-057) and a `dataset_fingerprint` (ADR-061), so "same data" is checkable
  rather than assumed.
- `training.deterministic` in **every** task and in the custom-task SDK
  (ADR-062). Detection defaults to `True` to mirror `YOLO.train`; the rest
  default to `False` because pinning cuDNN costs throughput.
- Config `schema_version` with migrations (ADR-039); YAML round-trips between
  the GUI and the CLI.

### Interface

- React SPA served by the same Python process — no separate frontend to run.
- Canonical panel layout across all tasks (ADR-059): experiment name, YAML
  export/import, strategy selector, model, training, dataset stats,
  preprocessing filters and augmentation with live preview.
- History grouped by task family with wrapping filters, multi-select delete and
  run comparison (ADR-063/064); every dropdown is drawn by the app, so no
  operating-system popup breaks the dark theme.
- Datasets is its own surface: one-shot download from torchvision, Roboflow,
  Kaggle or Hugging Face (ADR-055).
- Post-training: per-checkpoint testing on new data, batch prediction to CSV,
  Grad-CAM, ONNX export with a PyTorch-vs-runtime latency benchmark, and
  TensorBoard scalars per run (ADR-054).

### Verification

- `visionforge doctor` — detects GPU/CUDA and prints the exact torch install
  line for the machine (ADR-042).
- `visionforge selftest` — trains every task through the real API on synthetic
  data and asserts the run, the report shape and the live-progress contract
  (ADR-060). Offline, ~90 s, CI-ready.
- A full matrix on **real** datasets (ADR-065) is recorded in
  [`documentation/VALIDATION.md`](documentation/VALIDATION.md): 21 cases across
  five tasks and five strategies, all passing.
- 1274 backend tests, 102 frontend tests, ruff + mypy clean, gated in CI.

### Fixed in the run-up to this release

- Transfer learning trained correctly but streamed no live progress, leaving
  the GUI's progress bar dead for the whole run (ADR-065).
- Multi-trial strategies (K-fold, sweeps, replicates) emitted no progress
  events, so the bar advanced on wall-clock only.
- Opening the history after a K-fold returned a 500: cross-validation wrote
  timezone-aware timestamps while every other writer used naive local time.
- The preprocessing preview served a stale "final" image from the browser
  cache, one pipeline behind.
- The torchvision detection backend never seeded, so `seed: 42` was a claim
  nothing backed (ADR-062).
- Sweeps and replicates silently accepted a metric no trial reported, ranking
  every trial 0.0 and crowning an arbitrary winner (ADR-060).
- Replicated comparison ranked descending regardless of metric direction, so a
  MAE of 4.02 beat a MAE of 0.99 (ADR-061).

[Unreleased]: https://github.com/marcus-vreis/VisionForge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/marcus-vreis/VisionForge/releases/tag/v0.1.0
