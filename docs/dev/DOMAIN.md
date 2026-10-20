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
runner and GUI tab. Tasks are independent modules — adding a task does not modify existing
ones. Classification is an `ExperimentConfig` run through the `ExperimentBlock` registry;
the other four are standalone config/block/run paths (ADR-033/036/037/038, ADR-041) with
their own `/api/<task>/*` endpoints. All five write the `run.json` contract (ADR-013).

| Task | Status | Config (`utils/`) | Loss | Key metrics | Primary metric |
|---|---|---|---|---|---|
| `classification` | Implemented | `ExperimentConfig` (`config.py`) | `BCEWithLogitsLoss` / `CrossEntropyLoss` | Accuracy, F1, AUC-ROC | `accuracy` |
| `regression` | Implemented | `RegressionConfig` (`regression_config.py`) | `mse` / `mae` / `huber` | R², RMSE, MAE | `r2` |
| `detection` | Implemented | `DetectionConfig` (`detection_config.py`) | backend-owned (Ultralytics, torchvision) | mAP@50, mAP@50-95 | `map50_95` |
| `segmentation` | Implemented | `SegmentationConfig` (`segmentation_config.py`) | `cross_entropy` / `dice` / `combined` | mean IoU, Dice, pixel accuracy | `miou` |
| `anomaly` | Implemented | `AnomalyConfig` (`anomaly_config.py`) | autoencoder: reconstruction MSE; PatchCore: none (memory bank) | image AUROC, F1, threshold | `auroc` |

The task name is the URL segment (`/api/<task>/*`). In YAML, the `task` field is
`binary` or `multiclass` for classification (the subtype, below) and `detection`,
`regression`, `segmentation` or `anomaly` for the others; `visionforge run` dispatches
on it (`__main__.py`). The primary metric is what each `*Runner`
returns from `primary_metric()` (`blocks/<task>_runner.py`).

Strategies available per task (each is a route in `gui/api/routes.py`):

| Task | Run | K-fold | Sweep | Model comparison | Replicates | Replicated comparison |
|---|---|---|---|---|---|---|
| classification | `POST /api/experiment/run` | block `cross_validation` | blocks `grid_search`, `random_search` | block `model_comparison` | `/api/classification/replicates` | `/api/classification/replicated-comparison` |
| regression | `/api/regression/run` | `/api/regression/cv` | `/api/regression/sweep` | `/api/regression/compare` | `/api/regression/replicates` | `/api/regression/replicated-comparison` |
| segmentation | `/api/segmentation/run` | `/api/segmentation/cv` | `/api/segmentation/sweep` | `/api/segmentation/compare` | `/api/segmentation/replicates` | `/api/segmentation/replicated-comparison` |
| detection | `/api/detection/run` | none | `/api/detection/sweep` | `/api/detection/compare` | `/api/detection/replicates` | `/api/detection/replicated-comparison` |
| anomaly | `/api/anomaly/run` | none | `/api/anomaly/sweep` | `/api/anomaly/compare` | `/api/anomaly/replicates` | `/api/anomaly/replicated-comparison` |

Sweeps, comparisons and replicates are task-agnostic: they drive any task through the
`TaskRunner` Protocol (`core/task_runner.py`, ADR-041), so a new task gets them by
providing a runner.

Researcher-defined tasks (ADR-058) are a sixth, open-ended surface: a documented `.py` in
`user_tasks/` registered with `@register_task` gets a tab, `run.json` (stamped
`custom:<key>`), sweeps, replicates and replicated comparison via `CustomTaskRunner`, under
`/api/custom/{key}/*`. Config is a `BaseTaskConfig` subclass (`tasks/base.py`). Scaffold
with `visionforge new-task <key>`.

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

The active block is selected by the `block` field in `ExperimentConfig`
(classification only). `POST /api/experiment/run` (`_execute_experiment` in
`gui/api/routes.py`, ADR-026) instantiates the class for `classification`,
`cross_validation`, `transfer_learning`, `model_comparison`, `grid_search` or
`random_search`. `batch_prediction` and `export_onnx` are also valid `block` values
(YAML) but the GUI runs them through `POST /api/runs/{id}/batch_predict` and
`POST /api/runs/{id}/export_onnx`, which build a one-shot config for the block. The four
non-classification tasks do not use this registry: `DetectionBlock`, `RegressionBlock`,
`SegmentationBlock` and `AnomalyBlock` mirror `setup`/`run`/`report` without subclassing
`ExperimentBlock` (ADR-033/036/037/038).

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

K-fold is available for classification (`CrossValidationBlock`, `block="cross_validation"`),
regression (`blocks/regression_cv.py`) and segmentation (`blocks/segmentation_cv.py`);
detection and anomaly have none. It runs N independent trainings, one per fold. The
classification block produces folds with `KFold` or `StratifiedKFold` (sklearn) using only
the dataset's training split. Each fold:

1. Recomputes `normalize_mean`/`normalize_std` from the fold's training images
   only — preventing the classic data-leakage trap of computing dataset-wide
   statistics before the split.
2. Trains a fresh model with the same hyperparameters as the parent config.
3. Evaluates against the fold's validation images.

The classification report is `cv_summary.json` in `reports_dir/<experiment_name>/`,
with per-fold records (`fold`, `train_size`, `val_size`, `accuracy`, `f1`,
`best_val_loss`, `status`) and aggregate `mean_accuracy ± std_accuracy`,
`mean_f1 ± std_f1`. Regression and segmentation write their own reports under
`reports_dir`.

The hold-out test split is **not** consumed by CV — it stays reserved for a
final test of the chosen model.

### Run

A run is a single execution of a task (for classification, of an `ExperimentBlock`). It is fully described by a `run.json` file written by the trainer at completion (ADR-013). This file is the single source of truth for everything the GUI history tab displays. Layout for a classification run (the other tasks write their own artifacts next to `run.json`):

```
outputs/
  models/
    <experiment_name>/
      <timestamp>/            # e.g. 20260321_194000
        run.json              # full metadata (see schema below)
        best_model.pth        # best checkpoint by val metric
        loss.png  accuracy.png
        confusion_matrix.png  confusion_matrix_normalized.png
        roc_curve.png         precision_recall_curve.png
        tensorboard/          # per-epoch scalars (ADR-054)
```

The per-epoch history lives inside `run.json` (`history`); there is no separate `history.json`.

#### `run.json` schema

```json
{
  "id": "resnet50_binary_20260321_194000",
  "experiment": "resnet50_binary",
  "timestamp": "2026-03-21T19:40:00",
  "status": "completed",
  "stopped": false,
  "device_used": "cuda:0 (NVIDIA RTX 4090)",
  "environment": { "python": "...", "torch": "...", "cuda": "...", "gpu": "..." },
  "dataset_fingerprint": { "...": "..." },
  "run_dir": "outputs/models/resnet50_binary/20260321_194000",
  "class_names": ["healthy", "diseased"],
  "config": { "schema_version": 1, "data": { "preprocessing": { "steps": [...] } }, "...": "..." },
  "metrics": {
    "best_val_loss": 0.21,
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
    ],
    "report": null
  },
  "warnings": [],
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

A K-fold is a History group (ADR-123): one group `run.json` in
`<models_dir>/<name>_cv/<group_id>/`, with the folds as children (`fold`, `run_dir`) and
the mean ± std copied from the report. The same group mechanism (`core/run_groups.py`,
ADR-113, ADR-124) folds replicate sets and sweep/comparison trials. Fold metrics are
reported as `val_<name>`, never as test metrics.

Runs are stored under `output.models_dir` (default `outputs/models`) as `<experiment_name>/<timestamp>/`.

### Device

VisionForge adapts to available hardware at runtime. The user installs the
correct PyTorch build for their hardware — the system never assumes GPU availability.

| Hardware | Behavior |
|---|---|
| CPU only | Full support, slower training |
| Single GPU | CUDA acceleration, automatic detection |
| Multi-GPU (single machine) | `multi_cuda`: `DataParallel` in the classification, regression and segmentation trainers (ADR-007); detection passes the GPU ids to Ultralytics. `DistributedDataParallel` is not implemented |

`DeviceConfig.kind` is `cpu`, `cuda` or `multi_cuda`; `gpu_ids` is optional. The device
actually used is recorded in `run.json` as `device_used`.

### Config and GUI relationship

Pydantic config models are the single source of truth for both YAML and GUI:

- **YAML** → `load_config(path)` (`utils/config.py`) parses, migrates `schema_version` (ADR-039) and validates the file. `visionforge run <config.yaml>` runs it; `configs/` holds one example per task.
- **GUI** → each task panel reads its JSON Schema: `GET /api/schema` (classification), `GET /api/{detection,regression,segmentation,anomaly}/schema`, `GET /api/custom/{key}/schema` for researcher-defined tasks. Classification renders through `ParamPanel`; the other tabs through their own `*Panel` components.
- **API** → `POST /api/experiment/run` (classification) and `POST /api/<task>/run` accept JSON validated by the same Pydantic config.
- Every path produces the same config object, validated before any GPU work (ADR-002).

### Validation rules (Classification)

| Rule | Where enforced |
|---|---|
| `batch_size` must be a power of 2 | `TrainingConfig` field validator |
| `learning_rate` must be > 0 | `TrainingConfig` field validator |
| `base_dir` must exist and be a directory | `DataConfig` field validator |
| `binary` requires `num_classes=1` | `ExperimentConfig` model validator |
| `multiclass` requires `num_classes>=2` | `ExperimentConfig` model validator |
| YAML must be a mapping (not list or empty) | `load_config()` |
