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

### `models/`

| Module | Responsibility |
|---|---|
| `factory.py` | `ModelFactory` — instantiates any supported CNN architecture from a `ModelConfig`. Handles pretrained weights and final layer replacement. |

### `core/`

| Module | Responsibility |
|---|---|
| `trainer.py` | Training loop with early stopping, best-model checkpointing, per-epoch JSON history (incl. train+val accuracy), and DataParallel multi-GPU support. `resolve_device()` honours `DeviceConfig` and records the actual device used in `run.json` (`device_used` field), with a CPU fallback when CUDA is requested but unavailable. |
| `data.py` | `DataModule` — wraps `ImageFolder`, applies transforms and augmentation, returns train/val/test `DataLoader`s. |
| `evaluator.py` | Computes Accuracy, F1, AUC-ROC, Precision, Recall, confusion matrix, classification report, and preserves per-sample `y_true` / `y_score` / `y_proba_full` arrays for downstream ROC and PR plotting. |
| `plotter.py` | `MetricsPlotter` — generates loss, accuracy, raw confusion matrix, normalized confusion matrix, ROC curve, and precision-recall curve PNGs via matplotlib/seaborn (Agg backend, no display needed). |

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
| `GridSearchBlock` | ✅ | ✅ | Exhaustive sweep over hyperparameter combinations (dot-path → CSV editor + cartesian preview). |
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
| `api/routes.py` | REST endpoints. Lifecycle: `GET /api/schema`, `POST /api/experiment/run`, `GET /api/experiment/status`, `GET /api/experiment/events` (SSE), `GET /api/experiment/result/{id}`. Runs: `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/export_md`, `POST /api/runs/{id}/test`. Dataset probes: `POST /api/dataset/detect`, `POST /api/dataset/pick`, `POST /api/dataset/stats`, `POST /api/dataset/samples`, `POST /api/dataset/preview_preprocess`, `GET /api/dataset/file`. System: `GET /api/system/info`, `GET /api/device/info`, `POST /api/checkpoint/pick`. Artifacts: `GET /api/artifacts/{path}`. |
| `api/schemas.py` | Pydantic response models: `RunStatus`, `RunResponse`, `RunResult`, `RunSummary` (now with `preprocessing_count`), `RunDetail`, `RunTestRequest/Response`, `DatasetDetectRequest/Response`, `DatasetPickResponse`, `DatasetStatsRequest/Response`, `DatasetSamplesRequest/Response`, `PreprocessPreviewRequest/Response`, `DeviceInfoResponse`, `GPUInfo`, `SystemInfo`, `CheckpointPickResponse`, `SplitStats`. |
| `static/` | Pre-built React SPA (HTML, JS, CSS). Generated by `npm run build` in the `frontend/` directory. Users never need Node.js. |

### `frontend/` (dev-time only)

React source code. Not part of the Python package.

| Path | Responsibility |
|---|---|
| `src/App.tsx` | Top-level shell. Owns `schema`, `formData`, `device`, `pipelineSummary`, history/overlay visibility, dispatches the submit to `useExperiment.submit()`. |
| `src/components/ParamPanel.tsx` | Schema-driven form for the active task. Sections: name/task, `BlockSelector`, `CrossValidationFields` (when CV), model (with `WeightsPathField`), training (with `SchedulerFields`), dataset (`DatasetPicker` + `DatasetStats` + `PreprocessingPanel`), augmentation. Hosts YAML import/export with client-side schema validation. |
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
