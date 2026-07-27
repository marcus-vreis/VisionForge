# VisionForge

[![CI](https://github.com/marcus-vreis/VisionForge/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/marcus-vreis/VisionForge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A local-first computer-vision experimentation platform for researchers.**
Train, validate and compare models on your own GPU — no cloud, no notebooks,
no copy-pasted training loops. PyTorch + FastAPI + React in one Python process.

VisionForge replaces ad-hoc Jupyter workflows with a clean, testable, reproducible
system where the numbers you report are numbers you can defend: every run records
its full provenance, and every comparison can be replicated across seeds with
confidence intervals.

![VisionForge — classification panel](docs/images/vf-classification.png)

## Five task families, one interface

| Task | Models | Metrics |
|---|---|---|
| **Classification** | ResNet 18/34/50/101, EfficientNet B1/B7, VGG 16/19, AlexNet, timm, custom | Accuracy, F1, Precision, Recall, AUC-ROC, confusion matrix, ROC/PR curves |
| **Object detection** | Ultralytics YOLOv8/9/10/11/12/26, RT-DETR · torchvision Faster R-CNN, SSD, RetinaNet | mAP@50, mAP@50-95, box loss |
| **Image regression** | CNN backbones + linear head (CSV manifest datasets), timm, custom | MSE, RMSE, MAE, R² |
| **Semantic segmentation** | DeepLabV3, FCN, LR-ASPP, U-Net, custom | mean IoU, Dice, pixel accuracy |
| **Anomaly detection** | Convolutional autoencoder, PatchCore (unsupervised, MVTec-style) | image AUROC, threshold, F1 |
| **Your own task** (SDK) | any `nn.Module` — you write 4 hooks in one Python file | any metrics you declare (`higher`/`lower` direction-aware) |

Every task panel follows the same canonical layout: experiment name + YAML
export/import, a strategy selector, model, training, dataset (with pre-training
stats), preprocessing filters and augmentation with live preview.

## Built for defensible results

![Multi-seed replicates — same config, N seeds, mean ± 95% CI](docs/images/vf-replicates.png)

- **Multi-seed replicates** — train the same config N times under different
  seeds and report `metric = mean ± 95% CI` (Student-t) instead of a single
  point estimate.
- **K-fold cross-validation** — classification, regression and segmentation;
  per-fold metrics + mean ± std, with fold-leakage-safe transforms.
- **Hyperparameter sweeps** — grid, random, or Optuna TPE over any config field
  by dot-path; one-click architecture-comparison preset.
- **Paired significance testing** — compare N configurations over the *same*
  seeds and get the difference, its bootstrap CI, a paired t or Wilcoxon test
  (chosen and justified per comparison), Cohen's `d_z`, and Holm-Bonferroni
  correction across the family. It refuses to compare runs whose seeds do not
  line up, and flags when the seed count makes significance unreachable — so
  "not significant" is never mistaken for "no effect".
- **Paper-ready output** — every replicates / sweep / K-fold / comparison
  report is also written as a `booktabs` LaTeX table, with notes stating what
  each interval covers and which correction was applied.
- **Full provenance** — every run writes a versioned `run.json` with the exact
  config, seed, per-epoch history, environment (Python, torch/torchvision,
  numpy, CUDA, cuDNN, GPU model) and a **dataset fingerprint**, so "same data"
  is a checkable claim rather than a shared path.
- **Reproducibility knobs** — seeded runs, optional deterministic cuDNN mode,
  config schema versioning with migrations, YAML round-trip (export from the
  GUI, re-run from the CLI).
- **Post-training tooling** — run history with multi-run comparison and config
  diff, per-checkpoint testing on new datasets, batch prediction to CSV,
  Grad-CAM explainability, ONNX export with PyTorch-vs-runtime latency
  benchmark, TensorBoard scalars per run.
- **Dataset utilities** — split auto-detection, per-split stats (class balance,
  image/mask pairing, manifest checks with target distributions), one-shot
  download from torchvision / Roboflow / Kaggle / Hugging Face.

## Installation

Requirements: **Python 3.13+**, **Node.js 18+** (only to build the frontend once),
and [uv](https://github.com/astral-sh/uv) (recommended) or pip.

```bash
git clone https://github.com/marcus-vreis/VisionForge.git
cd VisionForge
uv venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
uv pip install -e ".[dev]"
```

PyTorch is intentionally **not** pinned as a dependency — its build must match
your hardware. Let the built-in doctor tell you the exact command:

```bash
visionforge doctor        # detects your GPU/CUDA → prints the right install line
# e.g.: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Build the web UI once (it is then served by the Python backend — end users
never need Node):

```bash
cd frontend && npm install && npm run build && cd ..
```

### Optional extras

| Extra | Enables |
|---|---|
| `detection` | Ultralytics YOLO / RT-DETR backends |
| `timm` | hundreds of extra backbones via `model.timm_model` |
| `optuna` | TPE-guided sweeps (`mode="optuna"`) |
| `tensorboard` | per-epoch scalars under `<run_dir>/tensorboard/` |
| `roboflow` / `kaggle` / `huggingface` | one-shot dataset download providers |

```bash
uv pip install -e ".[detection,optuna,tensorboard]"
```

## Quickstart

> New here? The step-by-step walkthrough — install → built-in dataset download
> → first run → replicates with confidence intervals → YAML re-run — lives in
> [`docs/QUICKSTART.md`](docs/QUICKSTART.md).

**GUI** (recommended):

```bash
visionforge gui           # opens http://127.0.0.1:8000
```

Pick a task tab, point the dataset picker at your data (stats render
immediately), choose a strategy — single run, K-fold, sweep or replicates —
and press *Treinar*. A live monitor streams epochs; results land in the run
history with plots, markdown model cards and artifact paths.

**CLI** (automation):

```bash
visionforge run configs/baseline.yaml        # classification
visionforge run configs/detection.yaml       # any task — dispatched by config
```

Configs exported from the GUI are the exact wire payload, so they re-run
identically from the CLI. All artifacts (checkpoints, plots, `run.json`,
reports) are written under `outputs/`.

## Custom models

Drop a Python file into `user_models/` and register it:

```python
from visionforge.models.registry import register_model

@register_model("my_net")
def build_my_net(num_outputs: int) -> nn.Module: ...
```

Select it via `model.custom_model` — works for classification, regression and
segmentation. See `user_models/README.md`.

## Custom tasks — define a whole new task family (ADR-058)

When your research doesn't fit the five built-in tasks, define your own in
**one documented Python file** — no React, no FastAPI, no training loop:

```bash
visionforge new-task cell_counting     # writes user_tasks/cell_counting.py
```

The generated template **trains out of the box** on synthetic data. Fill four
hooks — `build_model`, `build_loaders`, `compute_loss`, `compute_metrics` —
and a Pydantic `Config` whose fields become a validated form schema. You get,
with zero extra code:

- `GET /api/tasks` · `GET /api/custom/<key>/schema` · `POST /api/custom/<key>/run`
  (live SSE monitor, TensorBoard, versioned `run.json` provenance)
- `POST /api/custom/<key>/sweep` — grid/random/Optuna over **any** config
  field, including the ones you declared
- `POST /api/custom/<key>/replicates` — N seeds → mean ± std ± 95% CI

Training not epoch-shaped (GANs, EM loops)? Override `run(cfg, ctx)` and own
the loop while keeping every contract. A working example ships in
`user_tasks/example_counting/` (a CNN counting dots in synthetic images —
trains in seconds on CPU). Full walkthrough: [`user_tasks/README.md`](user_tasks/README.md) (PT + EN).

## Verifying the install

`visionforge doctor` checks your environment; **`visionforge selftest` checks
the pipeline** — it builds tiny synthetic datasets, starts the real API, and
trains every task through the same endpoints the browser uses, asserting that
each run completes, streams live progress, and stores its report:

```bash
visionforge selftest --quick     # one run per task (~15s, CPU, offline)
visionforge selftest             # every task x strategy: simple, K-fold, sweep, replicates, comparison
```

```
case                       result    time  detail
classification/replicates  PASS      2.7s  accuracy=1.0000±0.0000
segmentation/cv            PASS      2.0s  miou=0.0783
custom/sweep               PASS      0.6s  best mae=2.4231
regression/comparison      PASS      6.5s  best=baseline 1/1 signif.
...
27/27 cases passed
```

Filters: `--tasks classification,custom`, `--strategies sweep,replicates`,
`--json out.json`. Exit code is non-zero if any case fails, so it drops into
CI as-is. It verifies integrity, not model quality — one epoch on synthetic
data says nothing about accuracy.

## Architecture, decisions and contributing

- `documentation/ARCHITECTURE.md` — layers, modules, boundaries
- `documentation/DECISIONS.md` — every architecture decision as an ADR (001–061)
- `documentation/CONTRIBUTING.md` — dev setup, test/lint gauntlet, PR flow

Backend checks: `pytest` · `ruff check src/ tests/` · `mypy src/`.
Frontend: `cd frontend && npx vitest run && npx tsc --noEmit`.
End-to-end: `visionforge selftest` (or `pytest -m slow` for the harness's own
live cases — they are deselected from the default run).

## Citing

If VisionForge is useful in your research, please cite it — see
[`CITATION.cff`](CITATION.cff) (GitHub renders a “Cite this repository” button).

## License

[MIT](LICENSE)
