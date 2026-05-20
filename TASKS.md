# Tasks

## Phase 1 — Foundation ✅

- [x] Modular folder structure (`src/visionforge/`)
- [x] `pyproject.toml` with runtime and dev dependencies, tool configurations
- [x] Structured logger (`utils/logger.py`) — terminal + rotating file sinks
- [x] Unit tests for logger
- [x] YAML config manager with Pydantic v2 validation (`utils/config.py`)
- [x] Baseline experiment config (`configs/baseline.yaml`)
- [x] Unit tests for config manager
- [x] CUDA device detection (`utils/cuda.py`) — CPU / single GPU / multi-GPU
- [x] Unit tests for CUDA module
- [x] GitHub Actions CI — ruff, mypy, codespell, pytest, coverage artifact
- [x] GitHub Actions CD — build & release on tags
- [x] SonarCloud scan on pull requests
- [x] pre-commit hooks — ruff, mypy, pytest
- [x] Project documentation (README, ARCHITECTURE, DOMAIN, CONTRIBUTING, DECISIONS, TASKS)

---

## Phase 2 — Core Engine ✅

- [x] `seed` in `TrainingConfig` for full reproducibility
- [x] `weights_path` in `ModelConfig` — local .pth, ImageNet, or random weights
- [x] `TransformConfig` nested in `DataConfig` — per-flag augmentation control
- [x] `ModelFactory` — ResNet, EfficientNet, VGG, AlexNet with classifier head replacement
- [x] `ExperimentBlock` ABC — `setup / run / report` contract
- [x] `BlockRegistry` — auto-discovers concrete subclasses
- [x] `DataModule` — ImageFolder + configurable transforms pipeline
- [x] `Trainer` — early stopping, best-model checkpoint, `run.json` writer
- [x] DataParallel multi-GPU support in Trainer
- [x] Unit tests for all core modules (92 passing)

---

## Phase 3 — Evaluator + ClassificationTrainBlock + Plots ✅

- [x] `Evaluator` (`core/evaluator.py`) — Accuracy, F1, Precision, Recall, AUC-ROC, confusion matrix
- [x] `MetricsPlotter` (`core/plotter.py`) — loss curve, confusion matrix PNG, saves to run dir
- [x] `ClassificationBlock` (`blocks/classification.py`) — concrete ExperimentBlock wrapping ModelFactory + Trainer + Evaluator
- [x] `run.json` updated with test metrics and `graphics` artifact paths
- [x] Unit tests for Evaluator, Plotter, and ClassificationBlock

---

## Phase 4 — GUI (React + FastAPI) ✅ MVP

> **Decision change:** Replaced Gradio with React + shadcn/ui + FastAPI (see ADR-016).

- [x] `gui/server.py` — FastAPI app, serves React SPA + API, CORS middleware
- [x] `gui/api/routes.py` — REST endpoints: schema, run experiment, poll status, fetch results, serve artifacts
- [x] `gui/api/schemas.py` — Pydantic response models (RunStatus, RunResponse, RunResult)
- [x] `__main__.py` — subcommands: `run` (CLI) and `gui` (web server)
- [x] `frontend/` — React 18 + TypeScript + Vite + Tailwind CSS v4 + shadcn/ui
- [x] Schema-driven config form — auto-generated from `ExperimentConfig.model_json_schema()`
  - `Literal` → Select dropdown
  - `int` / `float` → number Input
  - `bool` → Switch
  - `$ref` → nested Card section
  - `array[number]` → comma-separated Input
- [x] Experiment runner with status polling
- [x] Results view — metrics grid + plot images
- [x] Training runs via `asyncio.to_thread()` (non-blocking, GPU in same process)
- [x] `GET /api/runs` — list completed run summaries (PR #18)
- [ ] Live training monitor (SSE/WebSocket) — deferred to next iteration
- [ ] Run history browser — wire `HistoryOverlay` to `/api/runs` (backend exists, UI is stub)
- [ ] Config export / import as `.yaml` from GUI — deferred to next iteration

---

## Phase 4.5 — GUI Redesign ✅ (PR #22)

Visual baseline now matches `frontend-design/`: oklch dark palette, per-task accents, Space Grotesk + JetBrains Mono, glass panels, animated wave background. Classification flow wired end-to-end.

- [x] Dark theme tokens + `[data-task]` accent palette in `index.css`
- [x] Custom controls: `NumberField`, `SelectField`, `Segmented`, `Toggle`, `TextField`, `FieldLabel`
- [x] Header / TabBar (4 tabs, only Classificação interactive) / TaskHero
- [x] Schema-driven `ParamPanel` in glass card with redesigned primitives
- [x] BottomBar (History · Treinar · DeviceIndicator)
- [x] `TrainingOverlay` with progress + synthetic log stream
- [x] `ResultsView` restyled (metric tiles + plot grid in glass aesthetic)
- [x] `Waves` SVG animated background
- [x] `HistoryOverlay` empty-state stub — needs `/api/runs` wire (tracked in Phase 4)
- [ ] Detection / Regression / Segmentation tabs → currently "em breve" placeholder; wire when backends land

---

## Phase 5 — Advanced Experiment Blocks

- [x] `GridSearchBlock`
- [x] `RandomSearchBlock`
- [x] `CrossValidationBlock` (K-Fold + Stratified)
- [ ] `TransferLearningBlock` (feature extraction + fine-tuning)
- [ ] `ModelComparisonBlock`
- [ ] `BatchPredictionBlock` (CSV output)
- [ ] `ExportONNXBlock` (+ inference validation + latency benchmark)
- [x] Unit tests for implemented blocks (grid_search, random_search, cross_validation, classification)

## Phase 6 — Regression task

- [ ] `RegressionConfig` Pydantic models
- [ ] `RegressionTrainer` with MSE/MAE/R² metrics (depends: RegressionConfig)
- [ ] Regression blocks (GridSearch, KFold, etc.) (depends: RegressionTrainer)
- [ ] Regression tab in GUI (depends: RegressionTrainer)

## Phase 7 — Object Detection task

- [ ] `DetectionConfig` Pydantic models
- [ ] Model support: YOLO, Faster R-CNN, SSD (depends: DetectionConfig)
- [ ] `DetectionTrainer` with mAP, IoU metrics (depends: DetectionConfig)
- [ ] Detection blocks (depends: DetectionTrainer)
- [ ] Detection tab in GUI (depends: DetectionTrainer)

## Phase 8 — Segmentation task

- [ ] `SegmentationConfig` Pydantic models
- [ ] Model support: U-Net, DeepLab (depends: SegmentationConfig)
- [ ] `SegmentationTrainer` with IoU, Dice metrics (depends: SegmentationConfig)
- [ ] Segmentation blocks (depends: SegmentationTrainer)
- [ ] Segmentation tab in GUI (depends: SegmentationTrainer)

## Phase 9 — Anomaly Detection task

- [ ] `AnomalyConfig` Pydantic models
- [ ] Model support: Autoencoder, PatchCore (depends: AnomalyConfig)
- [ ] `AnomalyTrainer` with AUROC, threshold metrics (depends: AnomalyConfig)
- [ ] Anomaly Detection blocks (depends: AnomalyTrainer)
- [ ] Anomaly Detection tab in GUI (depends: AnomalyTrainer)

---

## Backlog / ideas

- Optuna integration as alternative to `RandomSearchBlock`
- `timm` model library support as additional model source
- Grad-CAM visualization block
- ONNX inference benchmark in `ExportONNXBlock`
- TensorBoard / MLflow integration for experiment tracking
- Dataset augmentation preview in GUI
- Dark/light theme toggle in GUI
- Migrate `utils/config.py` → `configs/schemas/classification_config.py` when a second task is added (Phase 6+)
