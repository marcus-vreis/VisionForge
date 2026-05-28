# Domain

## Core concepts

### Experiment

An experiment is a single training run defined entirely by a YAML config file.
It has a unique `name`, a `task` type, and fully specified `model`, `training`, `data`,
and `output` sections. Every experiment produces a reproducible set of outputs.

An experiment config is always validated before execution — invalid fields are rejected
by Pydantic before any GPU computation starts. The same config can be created via YAML
or via the GUI, which generates the same Pydantic object either way.

### Task

A task defines the problem type. Each task has its own config schema, trainer, metrics,
and GUI tab. Tasks are independent modules — adding a new task does not modify existing ones.

| Task | Status | Config | Loss | Key metrics |
|---|---|---|---|---|
| `classification` | ✅ In development | `ClassificationConfig` | BCE / CrossEntropy | Accuracy, F1, AUC-ROC |
| `regression` | 📋 Planned | `RegressionConfig` | MSE / MAE | R², RMSE, MAE |
| `detection` | 📋 Planned | `DetectionConfig` | task-specific | mAP, IoU |
| `segmentation` | 📋 Planned | `SegmentationConfig` | CE / Dice | IoU, Dice |
| `anomaly_detection` | 📋 Planned | `AnomalyConfig` | reconstruction | AUROC, threshold |

### Classification subtypes

| Subtype | `num_classes` | Loss | Output |
|---|---|---|---|
| `binary` | 1 | `BCEWithLogitsLoss` | Sigmoid |
| `multiclass` | N ≥ 2 | `CrossEntropyLoss` | Softmax |

The subtype is declared in the config and enforced at validation time.

### ExperimentBlock

A block is a self-contained unit of experimentation. It encapsulates a strategy
and exposes a standard three-method interface:

- `setup(config)` — receives the experiment config and prepares internal state
- `run()` — executes the experiment strategy
- `report()` — returns a dict of results for logging and GUI display

Blocks are composable and interchangeable. The same config can be passed to any block.

The active block is selected by the `block` field in `ExperimentConfig`. The GUI
dispatcher (`gui/api/routes.py`) routes `block="classification"` to
`ClassificationBlock` and `block="cross_validation"` to `CrossValidationBlock`;
the other backend blocks (grid_search, random_search, transfer_learning,
model_comparison, batch_prediction, export_onnx) are reachable via YAML today
and gain a GUI surface incrementally (see TASKS.md, Phase 5.6).

### Preprocessing pipeline

A user-configurable ordered list of image filters applied **before** any
augmentation step. Each filter is one of:

- `gaussian_blur` (radius)
- `median_blur` (size, force-odd)
- `unsharp` (radius, percent, threshold)
- `edges` (Sobel via PIL FIND_EDGES)
- `emboss`
- `grayscale` (promoted back to RGB to preserve 3 channels)
- `equalize` (CLAHE approximation via `ImageOps.equalize`)
- `autocontrast` (cutoff)
- `wavelet` (1-level Haar, band ∈ {LL, LH, HL, HH})

The pipeline is part of `DataConfig.preprocessing.steps` and runs identically on
train/val/test splits. Because the schema's `PreprocessingStep` uses `extra="allow"`,
filter-specific parameters are spread flat on each step (`{kind: "gaussian_blur",
radius: 2.0}`) — the runtime extracts them as a dict and forwards to
`visionforge.core.preprocessing.apply_step`.

The pipeline is reproducible: it survives YAML export/import, appears in the
markdown model card, is shown in the run detail page, and is surfaced as a badge
(`⚗ N filtros`) in the run history list.

### Cross-validation

`CrossValidationBlock` (selected by `block="cross_validation"`) runs N
independent trainings, one per fold. Folds are produced by `KFold` or
`StratifiedKFold` (sklearn) using only the dataset's training split. Each fold:

1. Recomputes `normalize_mean`/`normalize_std` from the fold's training images
   only — preventing the classic data-leakage trap of computing dataset-wide
   statistics before the split.
2. Trains a fresh model with the same hyperparameters as the parent config.
3. Evaluates against the fold's validation images.

The aggregate output is `cv_summary.json` in `reports_dir/<experiment_name>/`,
with per-fold records (`fold`, `train_size`, `val_size`, `accuracy`, `f1`,
`best_val_loss`, `status`) and aggregate `mean_accuracy ± std_accuracy`,
`mean_f1 ± std_f1`.

The hold-out test split is **not** consumed by CV — it stays reserved for a
final test of the chosen model.

### Run

A run is a single execution of an `ExperimentBlock`. It is fully described by a `run.json` file written by the Trainer at completion. This file is the single source of truth for everything the GUI history tab displays.

```
outputs/
  <experiment_name>/
    <timestamp>/          # e.g. 20260321_194000
      run.json            # full metadata (see schema below)
      best_model.pth      # best checkpoint by val metric
      history.json        # per-epoch metrics (also embedded in run.json)
      loss.png
      confusion_matrix.png
      report.html
```

#### `run.json` schema

```json
{
  "id": "resnet50_binary_20260321_194000",
  "experiment": "resnet50_binary",
  "timestamp": "2026-03-21T19:40:00",
  "status": "completed",
  "device_used": "cuda:0 (NVIDIA RTX 4090)",
  "run_dir": "outputs/models/resnet50_binary/20260321_194000",
  "config": { "data": { "preprocessing": { "steps": [...] } }, "...": "..." },
  "metrics": {
    "best_val_accuracy": 0.94,
    "best_val_f1": 0.93,
    "best_epoch": 47,
    "total_epochs": 47,
    "test_accuracy": 0.92,
    "test_f1": 0.91
  },
  "history": [
    { "epoch": 1, "train_loss": 0.8, "train_accuracy": 0.65, "val_loss": 0.7, "val_accuracy": 0.71 }
  ],
  "artifacts": {
    "model": "outputs/.../best_model.pth",
    "graphics": [
      "outputs/.../loss.png",
      "outputs/.../accuracy.png",
      "outputs/.../confusion_matrix.png",
      "outputs/.../confusion_matrix_normalized.png",
      "outputs/.../roc_curve.png",
      "outputs/.../precision_recall_curve.png"
    ]
  },
  "tests": [
    {
      "test_id": "test_20260522_103000_000000",
      "label": "holdout_2026",
      "base_dir": "C:/datasets/coffee_v2",
      "timestamp": "2026-05-22T10:30:00",
      "metrics": { "accuracy": 0.89, "f1": 0.88 },
      "artifacts": { "confusion_matrix": "outputs/.../tests/...png" }
    }
  ]
}
```

The GUI **Run History** tab scans `outputs/` for all `run.json` files via
`GET /api/runs` and displays them as a list of cards. Selecting a run shows
metrics, plots (with Lightbox click-to-zoom), the applied preprocessing/
augmentation pipeline, and per-model test history. A **+ testar** button calls
`POST /api/runs/{id}/test` to evaluate the saved checkpoint against a new
dataset path and append a record to `tests[]`.

**Note:** Cross-validation runs don't currently write `run.json` — they write
`cv_summary.json` under `reports_dir`. Surfacing CV results in the run history
is a follow-up tracked in TASKS.md (Phase 5.5 open follow-ups).

Runs are stored in `outputs/` indexed by experiment name and timestamp.

### Device

VisionForge adapts to available hardware at runtime. The user installs the
correct PyTorch build for their hardware — the system never assumes GPU availability.

| Hardware | Behavior |
|---|---|
| CPU only | Full support, slower training |
| Single GPU | CUDA acceleration, automatic detection |
| Multi-GPU (single machine) | `DataParallel` in Phase 2, `DDP` in a future phase |

### Config and GUI relationship

Pydantic config models are the single source of truth for both YAML and GUI:

- **YAML** → `load_config(path)` parses and validates the file
- **GUI** → form fields are auto-generated from `ExperimentConfig.model_json_schema()`:
  - `Literal["a", "b", "c"]` → `<Select>` dropdown
  - `int` / `float` → `<Input type="number">`
  - `bool` → `<Switch>`
  - Nested Pydantic model (`$ref`) → `<Card>` section
  - `list[float]` → comma-separated `<Input>`
- **API** → `POST /api/experiment/run` accepts JSON validated by FastAPI using the same `ExperimentConfig`
- All three paths produce the same `ExperimentConfig` object

### Validation rules (Classification)

| Rule | Where enforced |
|---|---|
| `batch_size` must be a power of 2 | `TrainingConfig` field validator |
| `learning_rate` must be > 0 | `TrainingConfig` field validator |
| `base_dir` must exist and be a directory | `DataConfig` field validator |
| `binary` requires `num_classes=1` | `ExperimentConfig` model validator |
| `multiclass` requires `num_classes>=2` | `ExperimentConfig` model validator |
| YAML must be a mapping (not list or empty) | `load_config()` |
