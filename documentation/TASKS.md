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
- [x] `POST /api/dataset/detect` — auto-detect train/val/test split subdirs (PT + EN aliases)
- [x] DatasetPicker UI — folder input + detect button + manual fallback selectors
- [x] Structured API error handling — fixed `[object Object]` rendering, parses 422 detail arrays
- [x] Sensible defaults in `TrainingConfig`/`ModelConfig` (lr=0.001, epochs=10, batch=32, resnet50, num_classes=2)
- [x] Per-field validation error display in `ParamPanel` (humanized field paths)
- [x] TrainingOverlay shows full error message panel on failure
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
- [x] Detection tab → wired end-to-end (Phase 7). Regression / Segmentation tabs still placeholder; wire when those backends land

---

## Phase 5 — Advanced Experiment Blocks ✅ (backend) / 🟡 (GUI surface)

Backend implementation of all blocks is complete with tests; GUI surfacing
is partial — only `ClassificationBlock` and `CrossValidationBlock` are
selectable from the UI today.

- [x] `GridSearchBlock` — backend done; **no GUI surface yet**
- [x] `RandomSearchBlock` — backend done; **no GUI surface yet**
- [x] `CrossValidationBlock` (K-Fold + Stratified) — backend done + GUI selector + per-fold defaults
- [x] `TransferLearningBlock` (feature extraction + fine-tuning) — backend done; **no GUI surface yet**
- [x] `ModelComparisonBlock` — backend done; **no GUI surface yet**
- [x] `BatchPredictionBlock` (CSV output) — backend done; **no GUI surface yet**
- [x] `ExportONNXBlock` (+ inference validation + latency benchmark) — backend done; **no GUI surface yet**
- [x] Unit tests for all blocks (grid_search, random_search, cross_validation, classification, transfer_learning, model_comparison, batch_prediction, export_onnx)
- [x] Dynamic block dispatch in `_execute_experiment` via `config.block` (was hardcoded to `ClassificationBlock`)

---

## Phase 5.5 — Frontend ↔ backend gap closing ✅ (2026-05-22 → 2026-05-23)

The backend grew faster than the frontend during Phase 4-5. This phase
plugged the gaps so that features already in the engine were actually
reachable from the GUI.

### Preprocessing pipeline (was the biggest gap)
- [x] `PreprocessingPanel` made **controlled** — pipeline lives in `formData.data.preprocessing.steps` instead of panel-local state (previously the user built a pipeline, hit Train, and it never reached the backend even though `DataModule` was wired)
- [x] Helpers `toUIPreprocessingSteps` / `fromUIPreprocessingSteps` in `ParamPanel` convert between schema-flat (`{kind, ...params}`) and UI-nested (`{kind, params}`)
- [x] Green badge "**⚗ N filtros ativos no treino**" + empty-state explainer
- [x] `RunSummary.preprocessing_count` populated by `_parse_run_summary`; `RunCard` shows `⚗ N filtros` badge
- [x] `RunDetailPanel` ganhou `PipelineSection` (preprocessing ordenado + augmentation aplicado)
- [x] `CompareRunsPanel` ganhou `ConfigDiffTable` (cells diferentes destacadas) + `PreprocessingCompare` (pipelines lado a lado)
- [x] `TrainingOverlay` ganhou banner "⚗ pipeline ativo" com lista dos filtros antes dos logs
- [x] Markdown export (`_render_run_markdown`) inclui seções "Preprocessing pipeline" + "Augmentation / Normalization"
- [x] YAML round-trip do `preprocessing.steps` testado via Vitest

### Custom checkpoints (weights_path)
- [x] `POST /api/checkpoint/pick` endpoint — tkinter `askopenfilename` filtrado pra `*.pth *.pt`
- [x] `CheckpointPickResponse` schema
- [x] `WeightsPathField` component no `ParamPanel` (text input + `📁 Escolher` + `limpar`)
- [x] `weights_path` removido do `SKIP_FIELDS` no `field-renderer.ts`
- [x] Bind em `formData.model.weights_path` com normalização `""` → `null`
- [x] Tests: cancelled + chosen-path

### Cross-validation (K-Fold) MVP
- [x] Backend dispatch dinâmico por `config.block` (mantém ClassificationBlock para `classification`, dispatcha CrossValidationBlock para `cross_validation`)
- [x] `BlockSelector` segmented control no `ParamPanel` ("Treino simples / K-Fold (CV)")
- [x] `CrossValidationFields` renderiza `n_folds`, `stratified`, `shuffle`, `fold_seed`
- [x] Auto-populate de defaults sensatos quando alterna pra CV
- [x] Regression test: `test_post_experiment_run_dispatches_cross_validation_block`

### ResultsView polish (alinhado com RunDetailPanel)
- [x] Plots com labels humanizados (`loss.png` → "Loss (train + val)", etc.)
- [x] Lightbox click-to-zoom
- [x] Botão `↓ markdown` no header (chama `downloadRunMarkdown`)

### Validation UX
- [x] `humanizeFieldPath` entende `preprocessing`, `steps`, `scheduler`, `device` e renderiza índices de array como `#N`
- [x] YAML import faz `validateParsedConfig` client-side e mostra até 5 issues no banner antes do submit
- [x] Vitest novos: `useExperiment.test.ts` (5 casos) + `yaml-config.test.ts` (round-trip preprocessing)

### Open follow-ups (sub-tarefas remanescentes)
- [x] **Renderização específica de CV no `ResultsView`** — tabela com fold-a-fold + headline mean ± std implementada via `CrossValidationReport`.
- [x] **Surface `cv_summary.json` no `RunDetailPanel` / `HistoryOverlay`** — `CrossValidationBlock` agora emite também `models_dir/{name}_cv/run.json` compatível com o parser do `/api/runs`. `RunDetailPanel.CrossValidationDetail` renderiza fold-a-fold + mean ± std quando `metrics.fold_results` está presente.

---

## Phase 5.6 — GUI surface dos blocks restantes ✅

Backend já existe e está testado. UI mínima entregue:

- [x] **TransferLearningBlock**: extensão do `BlockSelector` + `TransferLearningFields` (`mode`, `unfreeze_from_layer`, `backbone_lr_multiplier`)
- [x] **ModelComparisonBlock**: UI multi-select de arquiteturas + métrica de ranking (`ModelComparisonFields` + `ModelComparisonReport`)
- [x] **GridSearchBlock**: editor de hyperparameter space — versão inicial dot-path → CSV (substituída na Phase 5.8 pelo editor inline `+ valor ao grid`)
- [x] **RandomSearchBlock**: editor de `search_space` (uniform/log_uniform/choice) + `n_trials` + `seed`
- [x] **BatchPredictionBlock**: form `input_dir` + `recursive` no `RunDetailPanel` chamando `/api/runs/{id}/batch_predict`
- [x] **ExportONNXBlock**: botão "↗ exportar onnx" no `RunDetailPanel` com form de opset/dynamic_axes/validate/benchmark

## Phase 5.7 — Classification polish (2026-05-23) ✅

Itens entregues por iteração de tech-leader:

- [x] **Lightbox** — caption/botão como overlays absolutos (não consomem altura) e imagem em `calc(100vh - 48px)`; fim do crop em plots altos (CM, ROC).
- [x] **Auto-detect de classes** — botão `🎯 aplicar binary|multiclass·N` no `DatasetStats` que injeta `task` + `model.num_classes` no formData (binary → 1, multiclass → N).
- [x] **Lock `num_classes=1` quando `task=binary`** — `LockedNumClasses` substitui o input no `ParamPanel` quando binary; flip de volta para multiclass restaura input com default 2.
- [x] **DeviceSelector único** — vive só no `BottomBar` (Header não duplica mais).
- [x] **Trial-queue banner** no `TrainingOverlay` — para `grid_search`, `random_search`, `model_comparison` e `cross_validation`, mostra `⛓ fila de treinos · N runs` antes do primeiro epoch.
- [x] **CV no histórico** — `CrossValidationBlock` emite `models_dir/{name}_cv/run.json` compatível com `_parse_run_summary`; `RunDetailPanel.CrossValidationDetail` renderiza fold-a-fold + headline mean ± std.
- [x] **Search + filter no `HistoryOverlay`** — busca por nome/arch/run_id + filter chips por task + badge `N/M` + empty-state.
- [x] **`block` field no `RunSummary`** — distingue CV/grid/random/etc. de classification simples na lista (badge `⛓ <block>` no `RunCard`).
- [x] **DELETE /api/runs/{id}** — endpoint com safety (path containment + bloqueio do run em execução) + trash button + modal de confirmação no `RunCard` + 3 testes regressão.

## Phase 5.8 — Bug fixes + Grid search UX (2026-05-31) ✅

Iteração de tech-leader a partir de feedback de uso real:

- [x] **Fix: treino com pré-processamento quebrava no Windows** — `DataModule` ligava o pipeline via closure/lambda, não-picklável sob `spawn`; os workers do `DataLoader` (`num_workers > 0`) falhavam com `AttributeError: Can't get local object`. Trocado por classe top-level `_PreprocessingTransform` (ADR-030). Regressão coberta por `test_preprocessing_transform_is_picklable` + `test_full_preprocessing_pipeline_is_picklable`.
- [x] **Grid search inline** — removido o editor dot-path → CSV (`GridSearchFields`/`GridSearchRow`). Agora cada hiperparâmetro gridável (Modelo/Treinamento) ganha `+ valor ao grid` com validação inline por tipo (int/float/enum/potência-de-2); banner com preview de trials (ADR-031). Lógica pura isolada em `lib/grid-axis.ts` + `grid-axis.test.ts` (15 casos).

---

## Phase 5.9 — Live logs for grid/random search (2026-06-01) ✅

Iteração de tech-leader a partir de bug de uso real.

- [x] **Fix: monitor de treino vazio em grid/random search** — `run_trial` criava o `ClassificationBlock` interno sem `_progress_callback` e `routes._execute_experiment` só ligava o callback para `ClassificationBlock` top-level; logo nenhum evento de epoch subia pelo SSE durante uma varredura. Agora `GridSearchBlock`/`RandomSearchBlock` recebem `_progress_callback`, emitem `trial_start`/`trial_end` por trial e um único `end` terminal; eventos internos do Trainer passam por `make_trial_progress_wrapper` (anota `trial_index`/`total_trials`, reescreve `end`→`trial_end` para o `EventSource` não fechar no primeiro trial). `TrainingOverlay` calcula progresso sweep-wide e mostra separador `── trial k/N ──` + tag `[tk/N]` por epoch (ADR-032).
- [x] Regressão: `TestTrialProgressWrapper`, `TestProgressStreaming`, e assert `has_callback` no dispatch de grid search.

---

## Phase 6 — Regression task

- [ ] `RegressionConfig` Pydantic models
- [ ] `RegressionTrainer` with MSE/MAE/R² metrics (depends: RegressionConfig)
- [ ] Regression blocks (GridSearch, KFold, etc.) (depends: RegressionTrainer)
- [ ] Regression tab in GUI (depends: RegressionTrainer)

## Phase 7 — Object Detection task (Ultralytics, hybrid backends)

Design: `documentation/PHASE7_DETECTION_PLAN.md`. Primary backend Ultralytics
(YOLO / RT-DETR), secondary torchvision (Faster R-CNN / SSD / RetinaNet).

- [x] `DetectionConfig` Pydantic models — standalone tree (`utils/detection_config.py`), backend↔model validation, non-power-of-two batch, dataset source (`data_yaml` or `base_dir`), reuses `OutputConfig`/`DeviceConfig`. Tests in `tests/utils/test_detection_config.py`.
- [x] `DetectionDataModule` — passthrough explicit `data.yaml` or synthesize one from a YOLO-layout `base_dir` (detects `images/<split>` vs `<split>/images`, class names from config/`classes.txt`/generated). `core/detection_data.py`, no ultralytics import. Tests in `tests/core/test_detection_data.py`. `[detection]` extra (`ultralytics>=8.3`) declared in pyproject.
- [x] `DetectionTrainer` — wraps `YOLO.train` (lazy ultralytics bind), hooks `on_fit_epoch_end` → SSE `start`/`epoch_end`/`end` (mAP fields + classification-overlay-compat fields), writes ADR-013-compatible `run.json`; torchvision backend raises `NotImplementedError` (scaffold). `core/detection_trainer.py`, tested with a mocked `YOLO` in `tests/core/test_detection_trainer.py`.
- [x] `DetectionBlock` — standalone `setup/run/report` over `DetectionConfig` (not an `ExperimentBlock` subclass — see ADR-033), `_progress_callback` slot, wraps `DetectionTrainer`. `blocks/detection.py`, tested in `tests/blocks/test_detection.py`.
- [x] `ultralytics` optional extra in `pyproject.toml` (done in brick 2) + ADR-033 (standalone detection path) + ADR-034 (Ultralytics owns the loop).
- [x] Detection run path — `GET /api/detection/schema` + `POST /api/detection/run` dispatching `DetectionBlock` with `_progress_callback` → SSE; reuses the shared single-run state and `/experiment/{status,events,result}` (one run at a time, one GPU). `gui/api/routes.py`, tests in `tests/gui/test_routes_detection.py`.
- [x] Detection tab in GUI — `DetectionPanel` (backend/model/dataset/training form over `DetectionConfig`), submits to `/api/detection/run`, reuses `TrainingOverlay` (SSE) + `ResultsView` (mAP metrics + Ultralytics plots). Model options mirror the backend in `lib/detection-models.ts` (tested via Vitest). `App.tsx` wires the Detecção tab + BottomBar.

**Phase 7 Ultralytics path complete (backend + GUI).**

### Torchvision backend (`backend="torchvision"`) — ✅ complete
- [x] `build_torchvision_detector` model factory — **all five families wired**: Faster R-CNN (R50/MobileNet-FPN), SSD300-VGG16, SSDLite320-MobileNetV3, RetinaNet-R50-FPN. Each head sized to `num_classes + 1` (background slot, matching `DetectionDataset` labels); `weights_backbone=None` when not pretrained (no downloads in CI). `models/detection_factory.py`, tests in `tests/models/test_detection_factory.py` (build + train-forward with the max label verifies each family's sizing).
- [x] `DetectionDataset` — YOLO-format labels → torchvision targets (`boxes` xyxy abs + `labels` = yolo class + 1), `detection_collate`, degenerate-box skip, empty-target for unlabeled images. `core/detection_dataset.py`, tests in `tests/core/test_detection_dataset.py`.
- [x] torchvision training loop in `DetectionTrainer` (`backend="torchvision"` seam) — loss-dict loop (`build_torchvision_detector` + `DetectionDataset` + `detection_collate`), per-epoch train/val loss, best by val loss (frozen-BN safe), `weights/best.pt`, SSE (`start`/`epoch_end`/`end`) + ADR-013 `run.json` with `box_loss`. Requires `data.base_dir`. mAP deferred (ADR-035). Tests in `tests/core/test_detection_trainer.py::TestTorchvisionPath`.
- [x] `mean_average_precision_50` — mAP@0.5 (VOC all-points AP, `torchvision.ops.box_iou`), torchvision-format preds/targets. `core/detection_metrics.py`, tests in `tests/core/test_detection_metrics.py`.
- [x] Wire mAP@50 into the torchvision loop — per-epoch val mAP@50 (`_eval_torchvision_map`), best checkpoint by mAP, streamed + in `run.json` (`map50` per epoch + best). Supersedes the val-loss selection of ADR-035.
- [x] SSD / RetinaNet head replacement in the factory (see factory item above — all families wired).
- [x] Opt-in real end-to-end smoke test — trains the real torchvision pipeline (fasterrcnn_mobilenet, no mocks/downloads) one epoch on a synthetic YOLO set and checks loss loop + mAP@50 + run.json. Skipped in CI; enable with `VF_RUN_DETECTION_INTEGRATION=1`. `tests/integration/test_detection_torchvision_e2e.py`.

**✅ Phase 7 complete: Ultralytics + torchvision backends, both end-to-end (config → datamodule → trainer → block → API → GUI), with mAP@50.**

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
