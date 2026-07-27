# Architecture

## Overview

VisionForge is organized in four layers with strict boundaries:

```
┌─────────────────────────────────────────────────────┐
│               React SPA (browser)                    │
│   ConfigForm · ExperimentRunner · ResultsView        │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP (fetch)
┌─────────────────────▼───────────────────────────────┐
│           FastAPI (same Python process)              │
│   /api/schema · /api/experiment/* · /api/artifacts/* │
└─────────────────────┬───────────────────────────────┘
                      │ direct import
┌─────────────────────▼───────────────────────────────┐
│                 Plugin Blocks                        │
│  ClassificationBlock · GridSearch · KFold (planned)  │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                 Core Engine                          │
│   Trainer · DataModule · Evaluator                   │
│   ModelFactory · BlockRegistry · MetricsPlotter      │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                   Storage                            │
│  outputs/models · logs · graphics · reports          │
└─────────────────────────────────────────────────────┘
```

## Modules

### `utils/`

Foundational utilities with no dependencies on other VisionForge modules.

| Module | Responsibility |
|---|---|
| `logger.py` | Centralized loguru setup. Two sinks: colored terminal + rotating file. Call `setup_logger()` once from the entry point. |
| `config.py` | Pydantic v2 models for experiment configuration. `load_config(path)` reads a YAML file and validates all fields before any training starts. `model_json_schema()` drives the GUI config form. |
| `cuda.py` | Runtime device detection. `check_cuda()` returns a `CUDAInfo` snapshot including a `GPUDevice` per visible GPU. Adapts transparently to CPU, single GPU, or multi-GPU and logs device name + CUDA version on startup. |
| `<task>_config.py` | One standalone Pydantic config tree per non-classification task (detection, regression, segmentation, anomaly) per ADR-033/036/037/038, reusing the shared `OutputConfig`/`DeviceConfig`/`TransformConfig` blocks. |
| `environment.py` | The `run.json` environment block (ADR-057): Python, torch/torchvision, numpy, CUDA, cuDNN and the GPU model — what makes a recorded number reproducible. |
| `doctor.py` | `visionforge doctor` — inspects the machine and prints the exact torch install line for it (torch is user-managed per ADR-005). |
| `selftest.py` · `selftest_data.py` | `visionforge selftest` (ADR-060) — builds tiny synthetic datasets in each task's real on-disk layout, starts the real API on an ephemeral socket, and trains every (task, strategy) pair through the same endpoints the browser uses, checking the report shape and the SSE contract. The answer to "does this install actually train?". |

### `models/`

| Module | Responsibility |
|---|---|
| `factory.py` | `ModelFactory` — instantiates any supported CNN architecture from a `ModelConfig`. Handles pretrained weights and final layer replacement. Routes to the custom registry when `model.custom_model` is set. |
| `registry.py` | Custom-model registry (ADR-048/049) — `@register_model` + `load_user_models()` discover user architectures dropped into `user_models/`; `build_custom_model(name, num_outputs=…)` instantiates them. Serves the classification, regression and segmentation factories (builder takes the task's output dimension). Local-first: imports the user's own Python, nothing networked. |
| `timm_source.py` | Optional timm backbone source (ADR-051) — `build_timm_model(name, num_outputs=…, pretrained=…)` lazily wraps `timm.create_model`. Selected via `model.timm_model` (classification + regression); the `[timm]` extra is optional. |

### `core/`

| Module | Responsibility |
|---|---|
| `trainer.py` | Training loop with early stopping, best-model checkpointing, per-epoch JSON history (incl. train+val accuracy), and DataParallel multi-GPU support. `resolve_device()` honours `DeviceConfig` and records the actual device used in `run.json` (`device_used` field), with a CPU fallback when CUDA is requested but unavailable. |
| `data.py` | `DataModule` — wraps `ImageFolder`, applies transforms and augmentation, returns train/val/test `DataLoader`s. The preprocessing pipeline is bound via a top-level `_PreprocessingTransform` class so it stays picklable for `spawn` DataLoader workers (ADR-030). |
| `evaluator.py` | Computes Accuracy, F1, AUC-ROC, Precision, Recall, confusion matrix, classification report, and preserves per-sample `y_true` / `y_score` / `y_proba_full` arrays for downstream ROC and PR plotting. |
| `plotter.py` | `MetricsPlotter` — generates loss, accuracy, raw confusion matrix, normalized confusion matrix, ROC curve, and precision-recall curve PNGs via matplotlib/seaborn (Agg backend, no display needed). |
| `tracking.py` | `TensorBoardLogger` (ADR-054) — best-effort per-epoch scalar logging to `<run_dir>/tensorboard/`; no-op unless the optional `[tensorboard]` extra is installed. Wired into the classification/regression/segmentation trainer loops. |
| `replicates.py` | Multi-seed replicates over the `TaskRunner` handle (ADR-056) — trains the same config once per seed and aggregates every metric into n/mean/std/min/max/95% CI (Student-t). Exposed as `POST /api/{task}/replicates` for all five tasks; report persisted to `outputs/reports`. |
| `task_runner.py` | The `TaskRunner` Protocol (ADR-041) — `config_type` / `run(cfg)` / `metrics(result)` / `primary_metric()`. Every generic orchestrator (comparison, sweep, replicates, replicated comparison) drives tasks through this handle alone, which is why adding a task costs no orchestrator code. |
| `comparison.py` | Model comparison (ADR-044) — trains N architectures on one dataset and ranks them through the runner handle. |
| `sweep.py` | Hyperparameter sweep (ADR-045/052) — grid, random or Optuna TPE over any config field by dot-path; emits `trial_start`/`trial_end` so the live monitor tracks real progress. |
| `replicated_comparison.py` | N seeds x M variants, then a paired test (ADR-061) — the honest form of "A beats B". Every variant trains on the *same* seed list; failed variants stay visible in the report instead of vanishing from the matrix. |
| `significance.py` | Paired significance testing (ADR-061) — paired t / Wilcoxon chosen and *justified* per comparison, Cohen's `d_z`, percentile bootstrap CIs, Holm-Bonferroni across the family, and `min_achievable_p` so an underpowered "not significant" is never read as "no effect". |
| `latex_export.py` | Paper-ready `booktabs` tables (ADR-061) for replicates / sweep / K-fold / comparison reports, written beside the JSON and CSV. Table notes state what each interval covers and which correction was applied. |
| `dataset_fingerprint.py` | `dataset_fingerprint` in every `run.json` (ADR-061) — sha256 over the sorted path+size manifest of `data.base_dir`, so "same dataset" becomes checkable. Records the method used, because `manifest` cannot see a same-size edit and `content` can. |
| `preprocessing.py` | The filter pipeline (blur, edges, wavelet, CLAHE…) shared by every task, applied before augmentation and normalization. |
| `gradcam.py` · `onnx_export.py` · `batch_predict.py` | Post-training tooling: explainability heatmaps, ONNX export with a latency benchmark, and batch inference to CSV. |
| `<task>_data.py` · `<task>_trainer.py` | One pair per standalone task (detection, regression, segmentation, anomaly) per ADR-033/036/037/038 — a new task adds files here and never edits existing ones. |

### `tasks/` — researcher-defined tasks (ADR-058)

The "sixth task" surface: a researcher drops one documented `.py` into
`user_tasks/` and gets a real GUI tab, the live monitor, run history,
`run.json` provenance, sweeps and replicates — without writing React,
FastAPI or a training loop.

| Module | Responsibility |
|---|---|
| `base.py` | `BaseTaskConfig` (composes the shared training/data/output/device blocks) and the generic `TaskSpec[ConfigT]` ABC: four Level-1 hooks (`build_model`, `build_loaders`, `compute_loss`, `compute_metrics`) plus a `run(cfg, ctx)` escape hatch for training that is not epoch-shaped. Torch appears only under `TYPE_CHECKING`, so schema generation needs no hardware extra. |
| `registry.py` | `@register_task` (key, label, accent colour, metric names + directions) and `user_tasks/` discovery, mirroring the proven `user_models/` pattern. Built-in keys are reserved; a broken user file logs a warning and is skipped rather than crashing discovery. |
| `engine.py` | `GenericTaskEngine` — the loop VisionForge owns: seeding, device resolution, AMP, early stopping, direction-aware best checkpoint, SSE events, TensorBoard, metric curve and the `run.json` contract stamped `custom:<key>`. |
| `runner.py` | `CustomTaskRunner` — wraps the engine behind the `TaskRunner` handle so custom tasks get sweeps, replicates and replicated comparison from the same orchestrators the built-ins use. |
| `scaffold.py` | `visionforge new-task <key>` — writes a commented template that **trains out of the box** on synthetic data, so the tab is live before the researcher writes anything. |

### `blocks/`

Each block implements the `ExperimentBlock` ABC:

```python
class ExperimentBlock(ABC):
    def setup(self, config: ExperimentConfig) -> None: ...
    def run(self) -> None: ...
    def report(self) -> dict: ...
```

`BlockRegistry` auto-discovers all `ExperimentBlock` subclasses in `blocks/` — adding a new block requires no changes to existing code.

| Block | Backend | GUI surface | Description |
|---|---|---|---|
| `ClassificationBlock` | ✅ | ✅ | End-to-end classification: ModelFactory + DataModule + Trainer + Evaluator + Plotter. |
| `CrossValidationBlock` | ✅ | ✅ | K-Fold and Stratified K-Fold with per-fold metrics (selectable via `BlockSelector`). |
| `GridSearchBlock` | ✅ | ✅ | Exhaustive sweep over hyperparameter combinations (inline `+ valor ao grid` per field → Cartesian preview; see ADR-031). |
| `RandomSearchBlock` | ✅ | ✅ | Random sampling of the hyperparameter space — `uniform`/`log_uniform`/`choice` per param + `n_trials` + `seed`. |
| `TransferLearningBlock` | ✅ | ✅ | Feature extraction vs fine-tuning with configurable frozen layers and backbone LR multiplier. |
| `ModelComparisonBlock` | ✅ | ✅ | Trains N selected architectures with identical config, ranks by chosen metric (F1/AUC/accuracy/time). |
| `BatchPredictionBlock` | ✅ | ✅ | Inference on an image folder from `RunDetailPanel` ("+ inferência em lote"), outputs CSV. |
| `ExportONNXBlock` | ✅ | ✅ | Exports the trained checkpoint to ONNX from `RunDetailPanel`, with optional inference validation and latency benchmark. |

Block dispatch is config-driven (see ADR-026): `gui/api/routes.py:_execute_experiment` reads `config.block` and instantiates the matching class. All seven block types are now wired into both the dispatcher and the `ParamPanel` `BlockSelector`. Blocks that run N sub-trainings (`grid_search`, `random_search`, `model_comparison`, `cross_validation`) surface a `⛓ fila de treinos · N runs` banner in the `TrainingOverlay` so the user knows the single progress bar is tracking one trial at a time.

### `gui/`

The GUI uses React + shadcn/ui (frontend) served by FastAPI (backend) in the same Python process. Training runs via `asyncio.to_thread()` so the API stays responsive while PyTorch uses the GPU.

| Module | Responsibility |
|---|---|
| `server.py` | FastAPI application. Mounts API routes and serves the pre-built React SPA as static files. Entry point: `start_server()`. |
| `api/routes.py` | REST endpoints. Lifecycle: `GET /api/schema`, `POST /api/experiment/run`, `GET /api/experiment/status`, `GET /api/experiment/events` (SSE), `GET /api/experiment/result/{id}`. Runs: `GET /api/runs`, `GET /api/runs/{id}`, `DELETE /api/runs/{id}`, `GET /api/runs/{id}/export_md`, `POST /api/runs/{id}/test`, `POST /api/runs/{id}/batch_predict`, `POST /api/runs/{id}/export_onnx`. Dataset probes: `POST /api/dataset/detect`, `POST /api/dataset/pick`, `POST /api/dataset/stats`, `POST /api/dataset/samples`, `POST /api/dataset/preview_preprocess`, `GET /api/dataset/file`. System: `GET /api/system/info`, `GET /api/device/info`, `POST /api/checkpoint/pick`. Artifacts: `GET /api/artifacts/{path}`. |
| `api/schemas.py` | Pydantic response models: `RunStatus`, `RunResponse`, `RunResult`, `RunSummary` (now with `preprocessing_count`), `RunDetail`, `RunTestRequest/Response`, `DatasetDetectRequest/Response`, `DatasetPickResponse`, `DatasetStatsRequest/Response`, `DatasetSamplesRequest/Response`, `PreprocessPreviewRequest/Response`, `DeviceInfoResponse`, `GPUInfo`, `SystemInfo`, `CheckpointPickResponse`, `SplitStats`. |
| `static/` | Pre-built React SPA (HTML, JS, CSS). Generated by `npm run build` in the `frontend/` directory. Users never need Node.js. |

### `frontend/` (dev-time only)

React source code. Not part of the Python package.

| Path | Responsibility |
|---|---|
| `src/App.tsx` | Top-level shell. Owns `schema`, `formData`, `device`, `pipelineSummary`, history/overlay visibility, dispatches the submit to `useExperiment.submit()`. |
| `src/components/ParamPanel.tsx` | Schema-driven form for the active task. Sections: name/task, `BlockSelector`, `CrossValidationFields` (when CV), model (with `WeightsPathField`), training (with `SchedulerFields`), dataset (`DatasetPicker` + `DatasetStats` + `PreprocessingPanel`), augmentation. When `block = grid_search`, a `GridContext` turns each gridable field into a grid axis: `SchemaFieldVF` renders `GridAxisExtension` (the `+ valor ao grid` affordance) and a `GridSearchBanner` previews the trial count. Hosts YAML import/export with client-side schema validation. |
| `src/lib/grid-axis.ts` | Pure helpers for grid-axis editing — `isGridableField`, `coerceGridValue`, `validateGridValue` (schema + power-of-two/enum rules), `suggestNextGridValue`. Unit-tested in `grid-axis.test.ts` (ADR-031). |
| `src/components/PreprocessingPanel.tsx` | **Controlled** pipeline builder (steps + onChange). Backed by `formData.data.preprocessing.steps`. Per-step params, reorder, remove, live preview via `/api/dataset/preview_preprocess`. |
| `src/components/DatasetPicker.tsx` | Folder input + native `📁 Escolher pasta` button (`/api/dataset/pick`) + auto-detect splits (`/api/dataset/detect`) + manual fallback dropdowns. |
| `src/components/DatasetStats.tsx` | Per-split class distribution + imbalance flag + thumbnail strip from `/api/dataset/samples`. Exposes a `🎯 aplicar binary\|multiclass·N` button that injects the detected class count back into `formData.model.num_classes` + `formData.task`. |
| `src/components/ConfigForm.tsx` | Legacy fallback schema-driven form (used only when ParamPanel is not active — kept for historical compatibility). |
| `src/components/ResultsView.tsx` | Post-run summary with metric tiles, plot grid (human-labeled, click-to-zoom via `Lightbox`), and `↓ markdown` download button. |
| `src/components/TrainingOverlay.tsx` | Modal during training. Progress bar (SSE-driven), live log stream, "⚗ pipeline ativo" banner with the filter list, "⛓ fila de treinos · N runs" banner for multi-trial blocks (grid_search, random_search, comparison, CV), failure detail panel. |
| `src/components/Lightbox.tsx` | Full-screen image zoom. Image renders at `calc(100vw - 48px)` × `calc(100vh - 48px)` with `object-fit: contain`; caption and close button are absolutely positioned overlays so they never crop the image. |
| `src/components/HistoryOverlay.tsx` | History sheet over `/api/runs`. Lists `RunCard`s (with `⚗ N filtros` badge when preprocessing was applied). Comparison toggle selects 2+ runs → `CompareRunsPanel`; click opens `RunDetailPanel`. |
| `src/components/RunDetailPanel.tsx` | Full run drilldown: location/checkpoint paths, device used, `PipelineSection` (preprocessing + augmentation), metrics, plot grid with Lightbox, "+ testar" form (calls `/api/runs/{id}/test`), markdown download. |
| `src/components/CompareRunsPanel.tsx` | Multi-run comparison: legend, side-by-side metric table, `ConfigDiffTable` (16 hyperparameters; diffs highlighted), `PreprocessingCompare` (pipelines side by side), overlaid SVG line charts for val_loss / val_accuracy. |
| `src/components/DeviceSelector.tsx` | CPU/GPU/multi-GPU selector driven by `/api/device/info`. Lives in `BottomBar`. |
| `src/components/BottomBar.tsx` | Fixed action bar — History, central Treinar/reopen button, DeviceSelector. |
| `src/hooks/useExperiment.ts` | Lifecycle: submit config, open SSE, poll status, fetch results. `humanizeFieldPath` translates Pydantic error paths (including filter list indices as `#N`). |
| `src/api/client.ts` | Typed fetch wrappers for all API endpoints (schema, experiment/* lifecycle, runs/*, dataset/*, system/info, device/info, checkpoint/pick, artifacts). |
| `src/lib/yaml-config.ts` | YAML import/export with structural validation via `validateParsedConfig` against the live JSON Schema. |

## Device handling

The `check_cuda()` function in `utils/cuda.py` detects the device at runtime. The Trainer adapts transparently.

```
CPU            → torch.device("cpu")
Single GPU     → torch.device("cuda:0")
Multi-GPU      → torch.nn.DataParallel (phase 1), DDP (future)
```

PyTorch installation is the user's responsibility. VisionForge adapts to whatever is available.

## Task expansion model

Each new task (Detection, Segmentation, etc.) adds:

```
src/visionforge/
├── configs/schemas/<task>_config.py   # Pydantic models for the task
├── blocks/<task>/                     # task-specific blocks
├── core/<task>_trainer.py             # task-specific training loop
```

The frontend adds new tabs as the tasks are implemented. No existing code is modified when a new task is added.

## Boundaries

- `utils/` has no imports from other VisionForge modules.
- `core/` imports from `utils/` only.
- `blocks/` imports from `core/` and `utils/`.
- `gui/` imports from `blocks/`, `core/`, and `utils/`.
- `configs/` are plain YAML files — no Python imports.
- `outputs/` is never imported — write-only at runtime.
