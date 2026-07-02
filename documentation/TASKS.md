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
- [x] Live training monitor (SSE) — delivered in Phase 5.9 (`TrainingOverlay` + SSE epoch/trial stream)
- [x] Run history browser — `HistoryOverlay` wired to `/api/runs` with search/filter (Phase 5.7)
- [x] Config import as `.yaml` from GUI (Phase 5.5) + markdown export; native YAML export still optional

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

## Phase 6 — Image Regression task

Design: `documentation/PHASE6_REGRESSION_PLAN.md`. Standalone config/block/run
path (mirrors detection ADR-033); CSV-manifest dataset (image → continuous
target[s]); reuses CNN backbones, `OutputConfig`/`DeviceConfig`/`TransformConfig`.

- [x] brick 1 — `RegressionConfig` Pydantic models — standalone tree
  (`utils/regression_config.py`): backbone choice, `num_targets` ↔
  `target_columns` coherence (top-level validator), CSV field names, loss
  (mse/mae/huber), non-power-of-two batch; reuses `OutputConfig`/`DeviceConfig`/
  `TransformConfig`/`PreprocessingConfig`/`SchedulerConfig`. Tests in
  `tests/utils/test_regression_config.py` (20 cases).
- [x] brick 2 — `RegressionDataModule` (CSV → image/target tensors) — reads
  per-split CSV manifests with the stdlib `csv` module (no pandas), reuses
  `core.data._build_transforms` so augmentation/preprocessing match
  classification; train/val required, test optional. Picklable dataset
  (Windows spawn). `core/regression_data.py`, tests in
  `tests/core/test_regression_data.py` (12 cases).
- [x] brick 3 — regression head — `RegressionModelFactory`
  (`models/regression_factory.py`): CNN → `num_targets` linear outputs (no
  activation). Extracted shared `build_backbone`/`replace_final_layer`/
  `load_local_weights` helpers in `models/factory.py` so both factories share
  the backbone + head-swap logic (DRY). Tests in
  `tests/models/test_regression_factory.py` (8 cases); classification factory
  suite still green.
- [x] brick 4 — `RegressionTrainer` (`core/regression_trainer.py`): MSE/MAE/Huber
  loss, streaming MSE/RMSE/MAE/R² (`_MetricAccumulator`), best-by-val-loss
  checkpoint + early stopping + scheduler, SSE start/epoch_end/end, ADR-013
  `run.json`. Reuses `resolve_device`/`_seed_everything` from `core.trainer`.
  Tests in `tests/core/test_regression_trainer.py` (10 cases incl. metric math,
  multi-target, mae/huber, run.json shape).
- [x] brick 5 — `RegressionBlock` (`blocks/regression.py`): standalone
  `setup/run/report` (not an `ExperimentBlock` subclass — ADR-036). Wires
  `RegressionModelFactory` + `RegressionDataModule` + `RegressionTrainer`,
  reloads best checkpoint, computes test-set MSE/RMSE/MAE/R² (added reusable
  `RegressionTrainer.evaluate`), renders the loss curve (reuses
  `MetricsPlotter.loss_curve`), updates run.json with `test_*`. ADR-036 written.
  Tests in `tests/blocks/test_regression.py` (5 cases, factory/datamodule
  patched).
- [x] brick 6 — `/api/regression/{schema,run}` run path — `GET
  /api/regression/schema` (drives the form) + `POST /api/regression/run`
  dispatching `RegressionBlock` with `_progress_callback` → SSE; reuses the
  shared single-run state and `/experiment/{status,events,result}` (one run at a
  time). Mirrors the detection endpoint. Tests in
  `tests/gui/test_routes_regression.py` (4 cases: schema, dispatch+callback,
  409 conflict, 422 invalid config).
- [x] brick 7 — Regression tab in GUI — `RegressionPanel` (backbone, derived
  num_targets, CSV-manifest dataset form, loss/optimizer segmented controls)
  submits to `/api/regression/run`, reuses `TrainingOverlay` (SSE) +
  `ResultsView`. `lib/regression-models.ts` mirrors the backend
  (`buildRegressionPayload` parses comma-separated `target_columns` and forces
  `num_targets` to match). `App.tsx`/`useExperiment` wire the Regressão tab +
  BottomBar. Vitest in `regression-models.test.ts` (4 cases). Verified live in
  the browser (renders, reactive num_targets, 0 console errors).

**✅ Phase 6 complete: image regression end-to-end (config → CSV data → model →
trainer → block → API → GUI), with MSE/RMSE/MAE/R² and a live training monitor.**

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

### Phase 7.1 — parity with the classification post-training surface

Brings detection runs up to the same browse/evaluate surface classification has,
reusing the existing endpoints/components by task dispatch (see PHASE7 plan §7).

- [x] brick A — detection runs in run history with mAP (`_parse_run_summary` task-aware projection + `block="detection"`; `HistoryOverlay` task-aware metric keys).
- [x] brick B — detection metric/plot labels in `RunDetailPanel` (mAP@50, Box loss, Ultralytics plot names). Brick C (`ResultsView`) already covered by brick 5b.
- [x] brick D1 — hide classification-only actions (test/batch/ONNX) for detection runs + backend 400 guard.
- [x] brick D2 — detection-native per-model test: `evaluate_detection_run` (torchvision rebuild + mAP via `detection_metrics`; Ultralytics `YOLO.val`), dispatched from `_execute_run_test`; "+ testar" form re-enabled. `gui/api/detection_testing.py`.
- [x] brick E1 — `POST /api/detection/dataset/stats` + `_collect_detection_dataset_stats` (per-class instance tallies, unlabeled-image + imbalance flags); shared `resolve_yolo_split` extracted into `core/detection_data.py`.
- [x] brick E2 — `DetectionDatasetStats` GUI panel in `DetectionPanel` (mirrors classification `DatasetStats`).
- [x] brick F — Ultralytics ONNX export for detection runs: `export_detection_run` (drives `YOLO.export`), dispatched from `_execute_onnx_export`; export card re-enabled for Ultralytics-backend detection. `gui/api/detection_export.py`.

**✅ Phase 7.1 complete: detection has full parity with the classification post-training surface (history, detail, results, per-model mAP test, YOLO dataset stats, ONNX export).**

### Phase 7.2 — full hyperparameter surface + complete YOLO family (ADR-040)

Phase 7 shipped a minimal 6-knob training surface and only YOLOv8/YOLO11/RT-DETR.
This phase exposes the whole Ultralytics tuning surface and every YOLO family.

- [x] `DetectionTrainingConfig` expanded with optimizer (`optimizer`/`momentum`/
  `weight_decay`), schedule (`lrf`/`cos_lr`/`warmup_*`), loss gains (`box`/`cls`/
  `dfl`), and regularization/mechanics (`label_smoothing`/`dropout`/`nbs`/`freeze`/
  `amp`/`close_mosaic`/`single_cls`/`rect`/`multi_scale`). Defaults equal
  Ultralytics' own → behaviour-preserving. `utils/detection_config.py`.
- [x] New nested `DetectionAugmentationConfig` — every Ultralytics augmentation
  knob (`hsv_*`, `degrees`, `translate`, `scale`, `shear`, `perspective`,
  `flipud`, `fliplr`, `bgr`, `mosaic`, `mixup`, `copy_paste`, `auto_augment`,
  `erasing`). Tests in `tests/utils/test_detection_config.py`.
- [x] Full Ultralytics model set: YOLOv8/9/10/11/12/26 + RT-DETR (Python tuple +
  `lib/detection-models.ts`, newest-first, default stays `yolo11n`).
- [x] `DetectionTrainer._ultralytics_train_kwargs()` forwards the whole tree to
  `YOLO.train`; `_build_torchvision_optimizer` honours the optimizer trio
  (was hard-coded SGD). Tests in `tests/core/test_detection_trainer.py`.
- [x] GUI: `DetectionPanel` gains optimizer fields + Ultralytics-only "Schedule &
  loss", "Regularização & mecânica", and "Data augmentation" cards; payload maps
  the `auto_augment` "none" sentinel → `null`. Vitest coverage in
  `lib/detection-models.test.ts`. Example `configs/detection.yaml` documents all knobs.

**✅ Phase 7.2 complete: detection exposes the full Ultralytics hyperparameter surface and every YOLO family (incl. YOLO26).**

## Phase 8 — Semantic Segmentation task

Design: `documentation/PHASE8_SEGMENTATION_PLAN.md`. Standalone config/block/run
path (mirrors detection ADR-033 / regression ADR-036); paired image+mask
per-split dataset (image → per-pixel class map); torchvision DeepLab/FCN/LR-ASPP
families + hand-rolled U-Net; reuses `OutputConfig`/`DeviceConfig`/`TransformConfig`.

- [x] brick 1 — `SegmentationConfig` Pydantic models — standalone tree
  (`utils/segmentation_config.py`): backbone choice (unet + torchvision DeepLab/
  FCN/LR-ASPP families), `num_classes`, paired image/mask dataset dir names,
  `ignore_index` ↔ class-id coherence (top-level validator rejects an
  `ignore_index` that collides with `0..num_classes-1`), loss
  (cross_entropy/dice/combined), non-power-of-two batch, `image_size` floor;
  reuses `OutputConfig`/`DeviceConfig`/`TransformConfig`/`PreprocessingConfig`/
  `SchedulerConfig`. Tests in `tests/utils/test_segmentation_config.py` (31 cases).
- [x] brick 2 — `SegmentationDataModule` (`core/segmentation_data.py`) — paired
  image/mask dataset keyed by filename stem; joint geometric transforms in
  `__getitem__` (image bilinear+normalized, mask nearest-neighbor as `long`),
  joint hflip+rotation (mask filled with `ignore_index`), image-only color
  jitter; picklable (Windows spawn); RGB/palette masks raise an informative
  error (deferred). train/val required, test optional. Tests in
  `tests/core/test_segmentation_data.py` (11 cases).
- [x] brick 3 — `SegmentationModelFactory` (`models/segmentation_factory.py`) —
  torchvision DeepLabV3/FCN/LR-ASPP families (head sized to `num_classes`,
  `weights_backbone=None` when not pretrained → no downloads) + hand-rolled
  `UNet` (4-level, bilinear decoder realign for arbitrary input sizes).
  `segmentation_logits` normalizes dict (`"out"`) vs tensor output so the trainer
  is model-agnostic. Tests in `tests/models/test_segmentation_factory.py` (15
  cases).
- [x] brick 4 — `SegmentationTrainer` (`core/segmentation_trainer.py`) —
  cross_entropy/dice/combined loss (all `ignore_index`-aware), streaming
  `SegmentationMetricAccumulator` (K×K confusion matrix → mIoU/Dice/pixel-acc
  averaged over present classes), best-by-val-**mIoU** checkpoint + early
  stopping + scheduler, SSE start/epoch_end/end, reusable `evaluate`, ADR-013
  `run.json` (`miou`/`dice`/`pixel_acc`). Model-agnostic via `segmentation_logits`.
  Tests in `tests/core/test_segmentation_trainer.py` (12 cases incl. metric math,
  ignore_index, dice/combined modes, run.json shape).
- [x] brick 5 — `SegmentationBlock` (`blocks/segmentation.py`) — standalone
  `setup/run/report` (not an `ExperimentBlock` subclass — ADR-037). Wires
  `SegmentationModelFactory` + `SegmentationDataModule` + `SegmentationTrainer`,
  reloads best checkpoint, computes test-set mIoU/Dice/pixel-acc, renders the
  loss curve (reuses `MetricsPlotter.loss_curve`), updates run.json with
  `test_*`. ADR-037 written. Tests in `tests/blocks/test_segmentation.py`
  (4 cases, factory/datamodule patched).
- [x] brick 6 — `/api/segmentation/{schema,run}` run path — `GET
  /api/segmentation/schema` (drives the form) + `POST /api/segmentation/run`
  dispatching `SegmentationBlock` with `_progress_callback` → SSE; reuses the
  shared single-run state and `/experiment/{status,events,result}` (one run at a
  time). Mirrors the regression endpoint. Tests in
  `tests/gui/test_routes_segmentation.py` (4 cases: schema, dispatch+callback,
  409 conflict, 422 invalid config).
- [x] brick 7 — Segmentação tab in GUI — `SegmentationPanel` (architecture,
  num_classes, paired image/mask dataset form, ignore_index with live collision
  warning, loss/optimizer segmented controls) submits to `/api/segmentation/run`,
  reuses `TrainingOverlay` (SSE) + `ResultsView`. `lib/segmentation-models.ts`
  mirrors the backend (`buildSegmentationPayload`, `ignoreIndexCollides`).
  `App.tsx`/`useExperiment`/`client` wire the Segmentação tab + BottomBar. Vitest
  in `segmentation-models.test.ts` (5 cases). Verified live in the browser
  (renders, all sections, 0 console errors).

**✅ Phase 8 complete: semantic segmentation end-to-end (config → paired
image/mask data → model → trainer → block → API → GUI), with mean IoU / Dice /
pixel accuracy and a live training monitor.**

## Phase 9 — Anomaly Detection task

Design: `documentation/PHASE9_ANOMALY_PLAN.md`. Standalone config/block/run path
(mirrors ADR-033/036/037); first **unsupervised** task — trains on normal images
only, scores image-level AUROC over a labelled test split. Models: conv
autoencoder (reconstruction error) + simplified PatchCore (normal-patch memory
bank). Reuses `OutputConfig`/`DeviceConfig`/`TransformConfig`.

- [x] brick 1 — `AnomalyConfig` Pydantic models — standalone tree
  (`utils/anomaly_config.py`): model name (autoencoder/patchcore), PatchCore
  backbone (resnet18/34/50/wide_resnet50_2), `latent_dim` (AE), `coreset_ratio`
  (0<r≤1), MVTec-style dataset dirs + `normal_dir`, `threshold_percentile`
  (0–100), non-power-of-two batch, `image_size` floor; reuses `OutputConfig`/
  `DeviceConfig`/`TransformConfig`/`PreprocessingConfig`/`SchedulerConfig`. Tests
  in `tests/utils/test_anomaly_config.py` (24 cases).
- [x] brick 2 — `AnomalyDataModule` (`core/anomaly_data.py`) — normal-only train
  loader (`train/<normal_dir>`, all label 0) + binary-labelled test loader
  (`test/<normal_dir>`→0, every other subdir→1); reuses `_build_transforms`
  (image_size synced from `data.image_size`); picklable `AnomalyImageDataset`;
  raises on empty normal-train or missing test. Tests in
  `tests/core/test_anomaly_data.py` (8 cases).
- [x] brick 3 — `AnomalyModelFactory` (`models/anomaly_factory.py`) —
  `ConvAutoencoder` (encoder/decoder, bilinear decoder realign for arbitrary
  sizes; reconstruction-error score) + `PatchCore` (frozen resnet backbone,
  layer2⊕layer3 patch features, greedy k-center coreset memory bank, max-patch
  nearest-neighbor distance score via `cdist`; `fit`/`score`/`extract`). Tests in
  `tests/models/test_anomaly_factory.py` (8 cases, pretrained=False → no
  downloads).
- [x] brick 4 — `AnomalyTrainer` (`core/anomaly_trainer.py`) — dispatches on model
  type: autoencoder = real MSE reconstruction loop (best by lowest train recon
  loss, early stopping); PatchCore = one-pass memory-bank fit (`total_epochs=1`).
  Image-level AUROC (sklearn `roc_auc_score`, 0.5 on single-class guard) + decision
  `threshold` from the normal-score percentile + image `image_f1`; streamed each
  epoch + ADR-013 `run.json`. Reusable `evaluate(model, train_loader, test_loader)`
  and module-level `compute_auroc`/`compute_threshold`. Tests in
  `tests/core/test_anomaly_trainer.py` (8 cases incl. AUROC math, AE + PatchCore
  fit, run.json shape).
- [x] brick 5 — `AnomalyBlock` (`blocks/anomaly.py`) — standalone `setup/run/report`
  (not an `ExperimentBlock` subclass — ADR-038). Wires `AnomalyModelFactory` +
  `AnomalyDataModule` + `AnomalyTrainer`, reloads best checkpoint, recomputes
  test AUROC/threshold/F1, renders the AUROC curve (new reusable
  `MetricsPlotter.metric_curve`), updates run.json with `test_*`. ADR-038 written.
  Tests in `tests/blocks/test_anomaly.py` (3 cases) + `metric_curve` plotter tests
  (2 cases).
- [x] brick 6 — `/api/anomaly/{schema,run}` run path — `GET /api/anomaly/schema`
  (drives the form) + `POST /api/anomaly/run` dispatching `AnomalyBlock` with
  `_progress_callback` → SSE; reuses the shared single-run state and
  `/experiment/{status,events,result}`. Mirrors the segmentation endpoint. Tests
  in `tests/gui/test_routes_anomaly.py` (4 cases: schema, dispatch+callback, 409
  conflict, 422 invalid config).
- [x] brick 7 — Anomalia tab in GUI — `AnomalyPanel` (method segmented control
  with conditional autoencoder/PatchCore fields, MVTec dataset form,
  threshold-percentile, normal-dir) submits to `/api/anomaly/run`, reuses
  `TrainingOverlay` (SSE) + `ResultsView`. `lib/anomaly-models.ts` mirrors the
  backend (`buildAnomalyPayload`, `isPatchCore`). Added a 5th `anomaly` task tab.
  `App.tsx`/`useExperiment`/`client` wired. Vitest in `anomaly-models.test.ts`
  (3 cases). Verified live in the browser (renders, PatchCore conditional fields,
  0 console errors).

**✅ Phase 9 complete: anomaly detection end-to-end (config → normal-only train /
labelled test → autoencoder|PatchCore → trainer → block → API → GUI), with
image-level AUROC / threshold / F1 and a live training monitor.**

---

## Config schema versioning ✅ (reproducibility infra)

Addresses CLAUDE.md §6.2/§7.3 — "freeze the schema early" so saved configs stay
loadable across schema changes.

- [x] `schema_version` field on `ExperimentConfig` (default `CURRENT_SCHEMA_VERSION
  = 1`) + `migrate_config_dict` run in `load_config` (legacy/missing → v1; future
  migration steps chain here); newer-than-supported version rejected with a clear
  upgrade error. Hidden from the GUI form (`SKIP_FIELDS`); added to
  `configs/baseline.yaml`. ADR-039. Tests in
  `tests/utils/test_config_schema_version.py` (11 cases).

## Environment capture in run.json ✅ (reproducibility infra)

- [x] `utils/environment.capture_environment()` records Python/platform/torch/
  torchvision/numpy/visionforge versions; `Trainer` writes them into `run.json`
  under `environment` (ADR-013 update). Additive (`RunDetail.environment` defaults
  to `{}`); never raises (`"unknown"` fallback). Surfaced in the GUI
  `RunDetailPanel`. Tests in `tests/utils/test_environment.py` (5 cases).

## Grad-CAM explainability (backlog item, in progress)

Design: `documentation/GRADCAM_PLAN.md`. Post-hoc explainability for trained
classification runs — a per-run action (mirrors test/batch_predict/onnx), not a
new task. Dependency-free (pure torch hooks).

- [x] brick 1 — `core/gradcam.py` — `GradCAM` (forward/backward hooks on the last
  conv, GAP-weighted ReLU-summed CAM, normalized + bilinearly upsampled to input
  size), `resolve_target_layer` (last `nn.Conv2d`, arch-agnostic), `overlay_cam`
  (dependency-free jet colormap + ImageNet de-normalization → PIL overlay). Tests
  in `tests/core/test_gradcam.py` (9 cases).
- [x] brick 2 — `POST /api/runs/{id}/gradcam` per-run action — rebuilds the run's
  classifier (random init + checkpoint, no ImageNet download), reads model +
  transform settings straight from run.json (no full ExperimentConfig validation,
  so a moved dataset path doesn't block explainability), overlays up to
  `num_samples` images from `input_dir`, writes PNGs to `<run_dir>/gradcam/`.
  Classification-gated (detection/regression/segmentation/anomaly → 400).
  `GradCamRequest`/`GradCamResponse`/`GradCamItem` schemas. Tests in
  `tests/gui/test_routes_gradcam.py` (5 cases).
- [x] brick 3 — GUI "🔥 Grad-CAM" action in `RunDetailPanel` — collapsible
  section (classification runs only) with folder picker + num_samples, calls
  `gradcamRun` (`client.ts`), renders the overlay grid (click → Lightbox) with
  per-image predicted class. Verified live in the browser: 4 overlays generated
  end-to-end (real resnet18 checkpoint → `<run_dir>/gradcam/`), 0 console errors.

**✅ Grad-CAM complete: explainability for trained classification runs
(core → API → GUI), dependency-free.**

- [x] **Grad-CAM for regression + segmentation (ADR-053)** — generalized
  `core.gradcam` with a `target_fn`; `gui/api/torch_gradcam.build_gradcam`
  dispatches classification / regression (output saliency) / segmentation
  (per-class CAM) through the same `/api/runs/{id}/gradcam`; GUI card gated to
  those three (detection/anomaly excluded).

## Dataset augmentation preview ✅ (backlog item done)

- [x] `POST /api/dataset/preview_augment` — renders N random variants of a sample
  image with the configured train-time augmentations (flip/rotation/jitter)
  applied, into `outputs/preview_cache/augment/`; reports which augmentations are
  `active`. `_render_augment_preview` + shared `_pick_preview_image` helper.
  `AugmentPreviewRequest`/`Response` schemas. Tests in
  `tests/gui/test_routes_augment_preview.py` (5 cases). GUI: `AugmentPreview`
  component (🎲 button + original/variant strip) wired into the `ParamPanel`
  Transformações section. `previewAugment` client wrapper. Verified live (4
  variants + `active: flip · rotation · jitter`, 0 console errors).

## ONNX PyTorch-vs-runtime speedup benchmark ✅ (backlog item done)

- [x] `ExportONNXBlock._benchmark` now also times the **PyTorch** model on the
  same dummy input (warmup-excluded) and reports `torch_mean_ms` + a `speedup`
  ratio (`torch_mean_ms / mean_ms`) alongside the existing onnxruntime
  mean/p50/p95. Surfaced in the GUI `ExportResultPanel` (onnx μ, onnx p95, torch
  μ, speedup ×, n_runs) — also fixed a stale frontend benchmark type that
  referenced `std_ms`/`n_runs` the backend never produced. Test in
  `tests/blocks/test_export_onnx.py::...test_benchmark_includes_torch_vs_onnx_speedup`.

## Integration smoke tests (opt-in)

Real end-to-end pipeline tests (no mocks), skipped in CI to keep it fast
(ADR-010); enable with the matching env var.

- [x] Detection (torchvision): `tests/integration/test_detection_torchvision_e2e.py`
  — `VF_RUN_DETECTION_INTEGRATION=1`.
- [x] Classification: `tests/integration/test_classification_e2e.py` — trains a
  real resnet18 (random init, no downloads) one epoch on a synthetic ImageFolder
  and checks the whole `ModelFactory → DataModule → Trainer → Evaluator →
  MetricsPlotter → run.json` path (metrics, `test_accuracy`, loss/CM plots,
  checkpoint). Enable with `VF_RUN_CLASSIFICATION_INTEGRATION=1`.

## Planned & specced (decisions recorded)

- **Cross-task strategy parity (ADR-041/044/045)** — generic `TaskRunner` handle +
  comparison + sweep extended to all four standalone tasks (backend + GUI +
  persistence), generic batch-predict (classification/regression/anomaly), and
  ONNX export (classification/detection/regression/segmentation). **Shipped.**
  Per-task transfer-learning shipped for regression (ADR-046) + segmentation
  (ADR-047). Tracker: `documentation/CROSS_TASK_PARITY_PLAN.md`.
- **Custom models (ADR-048/049)** — drop-in `user_models/` + `@register_model`,
  selected via `model.custom_model`. **Shipped** for classification, regression
  and segmentation (the registry's builder takes the task's output dimension).
  See `user_models/README.md`.
- **timm model source (ADR-051)** — `model.timm_model` builds any timm architecture
  via the optional `[timm]` extra (lazy `timm.create_model`). **Shipped** for
  classification + regression (mutually exclusive with `custom_model`).
- **Optuna sweep mode (ADR-052)** — `mode="optuna"` in `core/sweep` drives a TPE
  study over the random search-space grammar; optional `[optuna]` extra, lazy
  import. **Shipped** for all four standalone tasks (API + GUI SweepCard). Pruning
  deferred.
- **TensorBoard tracking (ADR-054)** — best-effort `core/tracking.TensorBoardLogger`
  writes per-epoch scalars to `<run_dir>/tensorboard/`; optional `[tensorboard]`
  extra (no-op without it). **Shipped** for classification + regression +
  segmentation; detection/anomaly trainers are a follow-up. (MLflow not chosen.)
- **Docker + `visionforge doctor` (ADR-042)** — `doctor` CLI (detect GPU/CUDA →
  exact torch install command) **shipped** (slice 1); the multi-stage GPU Docker
  image + compose (slice 2) is still planned. Design in
  `documentation/DOCKER_PLAN.md`. Local-only; k8s rejected.
- **Multi-seed replicates (ADR-056)** — `core/replicates.py` +
  `POST /api/{task}/replicates` for all five tasks: train the same config N
  times under different seeds, aggregate every metric into mean/std/min/max/95%
  CI (Student-t), persist `replicates_summary.json` + `replicates_ranking.csv`.
  **Backend + API shipped** with tests (`tests/core/test_replicates.py`,
  `tests/gui/test_routes_replicates.py`). Remaining brick: GUI `ReplicatesCard`
  (mirrors `SweepCard` — seeds/n_replicates form → report table with
  mean ± CI headline).
- **CUDA/cuDNN/GPU provenance (ADR-057)** — `capture_environment()` now records
  the CUDA build, cuDNN version and GPU name into the run.json `environment`
  block. **Shipped.**

## Backlog / ideas (triaged into tasks)

**In progress:**
- **Online dataset download (ADR-055)** — `POST /api/dataset/download`,
  provider-based (`gui/api/dataset_download.py`); one-shot local-first fetch.
  ✅ **torchvision built-ins** (→ ImageFolder, no extra); ✅ **Roboflow** (`[roboflow]`
  extra; api_key + workspace/project + version); ✅ **Kaggle** (`[kaggle]` extra;
  kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY; owner/slug, unzip); ✅ **Hugging Face**
  (`[huggingface]` extra `datasets`; materializes image+label splits → ImageFolder;
  optional token). ✅ **GUI** — `DatasetDownloadCard` (provider Segmented +
  conditional fields → `/api/dataset/download`) below the active panel. **COMPLETE.**
- **Cross-validation (K-fold)** for regression/segmentation — last cross-task
  parity slice. Regression **backend done** (`blocks/regression_cv.py`, ADR-050,
  KFold + per-fold train/eval + mean±std). Remaining: regression API endpoint +
  GUI card + report renderer, then the same for segmentation.

**Researcher-grade rigor (sequenced follow-ups to ADR-056):**
- GUI `ReplicatesCard` for all five task panels (see ADR-056 entry above).
- Bootstrap confidence intervals on single-run test metrics (`Evaluator`) —
  cheap resampling of per-sample `y_true`/`y_score`, surfaces "0.87 ± 0.02"
  even without replicates.
- Paired significance test between two replicate sets (same seeds → paired
  t-test / Wilcoxon) surfaced in the comparison report — "A > B" claims need
  a p-value.
- Determinism toggle (`torch.use_deterministic_algorithms` + cudnn.benchmark
  off) as an opt-in training knob, documented with its speed tradeoff
  (relaxes ADR-020 on demand).
- Dataset fingerprint (per-split file count + content hash) in run.json —
  "which data produced this number" is provenance, same as the env block.
- English README + CITATION.cff + versioned PyPI release — adoption blockers
  for researchers outside the PT-speaking circle.

**Larger — needs design or new dependencies (prefer a reviewed session):**
- **Dark/light theme toggle** — needs a coherent light palette for the dark-first
  blueprint design (design judgment).
- **More animated-SVG touches** where they carry signal (empty/loading states,
  per-`RunCard` sparklines, live-training progress), gated behind
  `prefers-reduced-motion`, no heavy animation dependency.

**Housekeeping:**
- Migrate `utils/config.py` → `configs/schemas/classification_config.py` (deferred;
  the standalone tasks already carry their own `*_config.py`).

## Cowork dev-experience (skills + setup)

- VisionForge skills authored (install the `.skill` files): `vf-new-task`,
  `vf-new-adr`, `vf-verify`, `vf-patterns`, `vf-brainstorm`.
- `.gitattributes` added (LF normalization) — run `git add --renormalize .` on
  Windows to clear the CRLF churn.
- Single canonical root `CLAUDE.md`; deep narrative preserved as
  `documentation/PROJECT_CONTEXT.md`; `documentation/CLAUDE.md` is now a redirect.
- Suggested MCPs for Claude Code: Context7 (live lib docs), Playwright (already in
  use), GitHub (optional, for PR/CI visibility).
