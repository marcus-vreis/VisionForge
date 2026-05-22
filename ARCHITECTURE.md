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

| Block | Status | Description |
|---|---|---|
| `ClassificationBlock` | Implemented | End-to-end classification: ModelFactory + DataModule + Trainer + Evaluator + Plotter. |
| `GridSearchBlock` | Planned | Exhaustive sweep over hyperparameter combinations defined in YAML. |
| `RandomSearchBlock` | Planned | Random sampling of the hyperparameter space with `n_trials`. |
| `CrossValidationBlock` | Planned | K-Fold and Stratified K-Fold with per-fold metrics. |
| `TransferLearningBlock` | Planned | Feature extraction vs fine-tuning with configurable frozen layers. |
| `ModelComparisonBlock` | Planned | Trains N models with identical config, ranks by F1/AUC/time. |
| `BatchPredictionBlock` | Planned | Inference on an image folder, outputs CSV. |
| `ExportONNXBlock` | Planned | Exports trained model to ONNX with inference validation. |

### `gui/`

The GUI uses React + shadcn/ui (frontend) served by FastAPI (backend) in the same Python process. Training runs via `asyncio.to_thread()` so the API stays responsive while PyTorch uses the GPU.

| Module | Responsibility |
|---|---|
| `server.py` | FastAPI application. Mounts API routes and serves the pre-built React SPA as static files. Entry point: `start_server()`. |
| `api/routes.py` | REST endpoints: `GET /api/schema`, `POST /api/experiment/run`, `GET /api/experiment/status`, `GET /api/experiment/result/{id}`, `GET /api/artifacts/{path}`. |
| `api/schemas.py` | Pydantic response models: `RunStatus`, `RunResponse`, `RunResult`. |
| `static/` | Pre-built React SPA (HTML, JS, CSS). Generated by `npm run build` in the `frontend/` directory. Users never need Node.js. |

### `frontend/` (dev-time only)

React source code. Not part of the Python package.

| Path | Responsibility |
|---|---|
| `src/components/ConfigForm.tsx` | Recursive schema-driven form renderer. Fetches JSON Schema from `/api/schema` and maps types to shadcn/ui widgets. |
| `src/components/ResultsView.tsx` | Displays metrics grid and plot images from a completed run. |
| `src/components/ExperimentRunner.tsx` | Shows training status (spinner) and error alerts. |
| `src/hooks/useExperiment.ts` | Manages the experiment lifecycle: submit config, poll status, fetch results. |
| `src/api/client.ts` | Typed fetch wrappers for all API endpoints. |

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
