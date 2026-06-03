# Phase 6 — Image Regression — Design Plan

> Branch: `feat/object-detection-ultralytics` (continues the task sweep; a
> dedicated `feat/regression` branch is the eventual home if split out)
> Status: **In progress** — brick 1 (`RegressionConfig`) landing first.
> Kickoff 2026-06-03 · author: tech-leader iteration
>
> Progress (see TASKS Phase 6 for the live checklist):
> - [x] brick 1 — `RegressionConfig` (`utils/regression_config.py`) + tests
> - [x] brick 2 — `RegressionDataModule` (CSV → image/target tensors)
> - [x] brick 3 — regression head on `ModelFactory` (CNN → N continuous outputs)
> - [x] brick 4 — `RegressionTrainer` (MSE/MAE/Huber loss, MSE/MAE/RMSE/R²)
> - [x] brick 5 — `RegressionBlock` (`setup/run/report`) + ADR-036
> - [x] brick 6 — `/api/regression/{schema,run}` run path
> - [x] brick 7 — GUI Regressão tab (RegressionPanel, wired end-to-end)
>
> **Phase 6 complete** — image regression runs end-to-end (config → CSV data →
> model → trainer → block → API → GUI) with MSE/RMSE/MAE/R² and a live monitor.

Image regression is the second task that does **not** reuse the classification
engine wholesale, but it sits much closer to it than detection did: the same CNN
backbones, the same hand-written PyTorch epoch loop, the same `OutputConfig` /
`DeviceConfig` / `TransformConfig` / preprocessing pipeline. What changes is the
**target** (a continuous value, or vector of values, per image instead of a
class index), which cascades into the dataset format, the model head, the loss,
and the metrics.

## 1. Why regression needs its own config/block (not an ExperimentConfig block)

The classification stack assumes:

- **ImageFolder** datasets (one subdir per class) — there is no continuous label.
- A classifier head sized to `num_classes`, trained with cross-entropy.
- Metrics are accuracy / F1 / AUC — undefined for continuous targets.

Regression violates all three:

- Labels are **continuous values keyed by image** — supplied as a CSV
  (`image, target[, target2, …]`), not encoded in the folder structure.
- The head emits **`num_targets` raw linear outputs** (no softmax), trained with
  **MSE / MAE / Huber**.
- Metrics are **MSE, RMSE, MAE, R²** per target.

Forcing this through `ExperimentConfig`/`ExperimentBlock` would bloat the shared
config with mutually-exclusive fields (`num_classes` vs `num_targets`,
`target_columns`, `loss`) and break the classification `ExperimentBlock` ABC's
`setup(self, config: ExperimentConfig)` contract — the same Liskov argument that
produced ADR-033 for detection. Regression therefore gets its own
`RegressionConfig` tree and a standalone `RegressionBlock` dispatched through a
dedicated `/api/regression/*` run path, **reusing** `OutputConfig`,
`DeviceConfig`, `TransformConfig`, `PreprocessingConfig`, and `SchedulerConfig`
so output layout, device selection, and the preprocessing pipeline stay
identical across tasks. (→ to be ratified as ADR-036.)

## 2. Dataset format — the one real decision

Image regression needs a per-image continuous label, which the folder layout
cannot encode. The standard, tool-agnostic format is a **CSV manifest**:

```
<base_dir>/
├── images/                 # all images (flat or nested; paths are CSV-relative)
├── train.csv               # image,target  (+ extra target columns if multi-output)
├── val.csv
└── test.csv                # optional
```

- `image_column` (default `"image"`) holds a path **relative to `images_dir`**.
- `target_columns` (default `["target"]`) is one or more numeric columns;
  `len(target_columns)` must equal `model.num_targets` (cross-validated at the
  top level).
- A single combined `labels.csv` with a `split` column is a reasonable future
  alternative; v1 ships **per-split CSVs** because it mirrors the detection
  `base_dir` ergonomics and needs no split-resolution heuristics.

**Default chosen, flagged for review:** per-split CSV + `images/` root. If the
user's data is shaped differently (single CSV + split column, or absolute image
paths), brick 2's `RegressionDataModule` is where that flexes — the config field
names are deliberately generic (`image_column`, `target_columns`, `*_csv`) so the
loader can grow without a schema break.

## 3. New modules (mirrors the detection layout)

```
src/visionforge/
├── utils/
│   └── regression_config.py   # RegressionConfig + Model/Data/Training subtrees
├── core/
│   ├── regression_data.py      # RegressionDataModule — CSV → (image, target)
│   └── regression_trainer.py   # MSE/MAE/Huber loop, MSE/RMSE/MAE/R²
├── models/
│   └── factory.py              # +regression head (num_targets linear outputs)
└── blocks/
    └── regression.py           # RegressionBlock (setup/run/report)
```

- `task` Literal is `"regression"`; the block dispatches via the dedicated path.
- `RegressionTrainer` emits the **same SSE events** as classification
  (`start`/`epoch_end`/`end`) with regression metric fields, so `TrainingOverlay`
  and `ResultsView` work with zero contract changes.
- `run.json` stays the contract (ADR-013): metrics become `mse`, `rmse`, `mae`,
  `r2`; artifacts point at loss/prediction-scatter plots + `best.pt`.

## 4. Build sequence (each a small, tested, shippable brick)

1. **`RegressionConfig` Pydantic models + tests** — no heavy deps; validates
   backbone, `num_targets` ↔ `target_columns` coherence, CSV field names, loss
   choice, image size, epochs. ← **this brick.**
2. **`RegressionDataModule`** — read the CSVs, resolve image paths, return
   `(image_tensor, target_tensor)`; reuse `TransformConfig`/preprocessing.
3. **Regression head in `ModelFactory`** — reuse the backbones, replace the final
   layer with `Linear(in_features, num_targets)` and no activation.
4. **`RegressionTrainer`** — MSE/MAE/Huber loss, per-epoch MSE/RMSE/MAE/R²,
   best-by-val-loss checkpoint, `run.json`; mirror the classification `Trainer`.
5. **`RegressionBlock`** — `setup/run/report`; dispatch via `/api/regression/run`.
6. **GUI**: activate the Regressão tab (currently placeholder); schema-driven
   form for `RegressionConfig`; results view reads regression metrics + plots.

## 5. Open decisions (to ratify as ADRs)

- **ADR-036** — Image regression is a standalone config/block/run path (mirrors
  ADR-033). To be written when brick 5 lands.
- **Loss default** — `mse` (sensitive to outliers but the standard regression
  baseline); `mae`/`huber` selectable for robustness.
- **Target scaling** — whether to standardize targets (z-score) inside the
  DataModule and invert for reporting. Deferred to brick 2; the config will not
  pre-commit to it.
