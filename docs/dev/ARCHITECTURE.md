# Architecture

## Overview

VisionForge is a Python package (`src/visionforge/`) plus a React SPA served by it. Import direction is one-way; see [Boundaries](#boundaries) for the rule and its exceptions.

```
┌─────────────────────────────────────────────────────┐
│               React SPA (browser)                    │
│  TabBar · one panel per task · History · Queue       │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP + SSE
┌─────────────────────▼───────────────────────────────┐
│     gui/  — FastAPI, same Python process             │
│  /api/schema · /api/experiment/* · /api/<task>/*     │
│  /api/runs/* · /api/dataset/* · /api/custom/{key}/*  │
└───────────┬─────────────────────────────┬───────────┘
            │ direct import               │ direct import
┌───────────▼───────────────┐   ┌─────────▼───────────┐
│ blocks/ — strategies      │   │ tasks/ — researcher │
│ ExperimentBlock subclasses│   │ tasks (user_tasks/) │
│ standalone task blocks    │   │ TaskSpec · engine   │
│ <task>_runner.py          │   │ CustomTaskRunner    │
└───────────┬───────────────┘   └─────────┬───────────┘
            │                             │
┌───────────▼─────────────────────────────▼───────────┐
│  core/ — trainers, data, evaluators, orchestrators   │
│  sweep · comparison · replicates · TaskRunner        │
└───────────┬─────────────────────────────────────────┘
            │
┌───────────▼───────────────┐
│ models/ — model factories │
└───────────┬───────────────┘
            │
┌───────────▼─────────────────────────────────────────┐
│ utils/ — config trees, logger, device, environment   │
└─────────────────────────────────────────────────────┘
        write-only:  outputs/models · logs · reports
```

## Modules

`__main__.py` is the `visionforge` command line entry point (`python -m visionforge`): argument parsing and dispatch to the commands described below.

### `utils/`

Foundational utilities. No module-level import of another VisionForge package; the lazy inward imports are listed under [Boundaries](#boundaries).

| Module | Responsibility |
|---|---|
| `logger.py` | Centralized loguru setup. Two sinks: colored terminal + rotating file. Call `setup_logger()` once from the entry point. |
| `config.py` | Pydantic v2 models for experiment configuration. `load_config(path)` reads a YAML file and validates all fields before any training starts. `model_json_schema()` drives the GUI config form. |
| `paths.py` | Portable run-name validation shared by all task configs and group identifiers. Names stay single directory components; configured output roots remain explicit paths. |
| `cuda.py` | Runtime device detection. `check_cuda()` returns a `CUDAInfo` snapshot including a `GPUDevice` per visible GPU. Adapts transparently to CPU, single GPU, or multi-GPU and logs device name + CUDA version on startup. |
| `<task>_config.py` | One standalone Pydantic config tree per non-classification task (`detection_config.py`, `regression_config.py`, `segmentation_config.py`, `anomaly_config.py`) per ADR-033/036/037/038, reusing the shared `OutputConfig`/`DeviceConfig`/`TransformConfig` blocks from `config.py`. |
| `environment.py` | The `run.json` environment block (ADR-057/135): library versions, CUDA/cuDNN, GPU model and the imported source checkout's Git revision/dirty flag. Unavailable source identity is explicit. |
| `doctor.py` | `visionforge doctor` — inspects the machine and prints the exact torch install line for it (torch is user-managed per ADR-005). |
| `cli_strategies.py` | `visionforge run` with a `strategy:` envelope — replicates, paired comparison, sweep and K-fold through the same `core/` orchestrators and `core/run_groups.py` as the GUI (ADR-184). |
| `messages.py` | Server-written texts in the language of the request (`tr(key, **params)`, ADR-116). |
| `workers.py` · `credentials.py` · `redact.py` | DataLoader worker budget; local store for dataset-provider API keys; masking of credentials before text is shown or logged. |
| `third_party_privacy.py` | Switches off the usage telemetry of the bundled third-party SDKs once, before any of them is imported; never overrides a value the user already set (ADR-120, ADR-170). |
| `wandb_names.py` | The W&B project and entity name check, kept out of `core/` so the profile settings validate names without importing torch (ADR-172). |
| `selftest.py` · `selftest_data.py` | `visionforge selftest` (ADR-060) — builds tiny synthetic datasets in each task's real on-disk layout, starts the real API on an ephemeral socket, and trains every (task, strategy) pair through the same endpoints the browser uses, checking the report shape and the SSE contract. The answer to "does this install actually train?". |

### `models/`

Imports `utils/` only.

| Module | Responsibility |
|---|---|
| `factory.py` | `ModelFactory` and the shared helpers `build_backbone`, `replace_final_layer`, `final_linear`, `load_local_weights`. Instantiates classification CNNs from a `ModelConfig`; handles pretrained weights and the final layer. Routes to the custom registry when `model.custom_model` is set. |
| `regression_factory.py` · `segmentation_factory.py` · `anomaly_factory.py` · `detection_factory.py` | Per-task model builders: CNN + linear head; DeepLabV3/FCN/LR-ASPP/U-Net (`segmentation_logits` normalizes outputs); `ConvAutoencoder` and `PatchCore`; torchvision detectors with a resized head (the Ultralytics backend is built inside `DetectionTrainer`). |
| `registry.py` | Custom-model registry (ADR-048/049) — `@register_model` + `load_user_models()` discover user architectures dropped into `user_models/`; `build_custom_model(name, num_outputs=…)` instantiates them. Serves the classification, regression and segmentation factories. Local-first: imports the user's own Python, nothing networked. |
| `timm_source.py` | timm backbone source (ADR-051) — `build_timm_model(name, num_outputs=…, pretrained=…)` wraps `timm.create_model` with a lazy import. Selected via `model.timm_model` (classification + regression). timm is a base dependency (ADR-106); the `timm` extra in `pyproject.toml` is empty and kept only so old install lines resolve. |

### `core/`

Imports `utils/`, plus `models/` from the four non-classification trainers (see [Boundaries](#boundaries)).

| Module | Responsibility |
|---|---|
| `trainer.py` | Classification training loop with early stopping, best-model checkpointing, per-epoch history (incl. train+val accuracy) and DataParallel multi-GPU support. `resolve_device()` honours `DeviceConfig` and records the actual device used in `run.json` (`device_used` field), with a CPU fallback when CUDA is requested but unavailable. |
| `data.py` | `DataModule` — wraps `ImageFolder`, applies transforms and augmentation, returns train/val/test `DataLoader`s. The preprocessing pipeline is bound via a top-level `_PreprocessingTransform` class so it stays picklable for `spawn` DataLoader workers (ADR-030). |
| `evaluator.py` | Computes Accuracy, F1, AUC-ROC, Precision, Recall, confusion matrix, classification report, and preserves per-sample `y_true` / `y_score` / `y_proba_full` arrays for downstream ROC and PR plotting. |
| `plotter.py` | `MetricsPlotter` — generates loss, accuracy, raw confusion matrix, normalized confusion matrix, ROC curve, and precision-recall curve PNGs via matplotlib/seaborn (Agg backend, no display needed). |
| `mlflow_tracking.py` | Optional local SQLite exporter for GUI classification runs. Profile settings are captured on submission, epoch history is reconciled on retry, and the block exports the evaluated record (ADR-170). |
| `hub_export.py` | Local preview and explicit upload of a bounded package (checkpoint, sanitised metrics, model card, manifest) to a verified private Hugging Face Hub repository. The token comes only from the selected profile (ADR-171). |
| `github_export.py` | A run's record without weights (README report, public run.json, charts) committed in one commit to an existing private GitHub repository with the profile's token (ADR-175). |
| `zenodo_export.py` | An unpublished Zenodo draft of a run: previews the public bundle, creates an empty deposition and uploads the bundle to its bucket with the profile's token, never publishing; zenodo.org and the sandbox are separate sites (ADR-174). |
| `dvc_export.py` | A run versioned with DVC in a local git repository: the ADR-175 folder plus `params.yaml`, `metrics.json`, `history.csv` and a stage-less `dvc.yaml` in git, the weights through `dvc add`, one commit of that folder and `dvc push` to the default remote; runs the real git and DVC (ADR-176). |
| `run_report.py` | A run as a folder of text shared by the GitHub and DVC exporters: README report from run.json, public run.json, charts (ADR-175, ADR-176). |
| `public_run_record.py` | Reduces every local path in a run record to its final name, for the public bundle sent to Zenodo (ADR-174). |
| `wandb_tracking.py` | Optional Weights & Biases exporter of GUI classification runs. Confirms or creates a private project over GraphQL with the profile's key, then logs public parameters, epoch metrics and the final public metrics, with git, code, console and machine metadata off (ADR-172). |
| `tracking_privacy.py` | The public projection every tracker shares: config parameters without secrets or paths, and the known metric names (ADR-170, ADR-171, ADR-172). |
| `code_tour.py` | The "Code" tab of the training sheet: the source of the functions a classification training runs, read with `inspect`, with notes anchored on line text and marked by the run's settings (ADR-173). |
| `run_trackers.py` | `RunTracker` protocol and `TrackerFactory`: how the trainer and the classification block drive any number of exporters. |
| `tracking.py` | `TensorBoardLogger` (ADR-054) — best-effort per-epoch scalar logging to `<run_dir>/tensorboard/`. TensorBoard ships with the package (ADR-106); the logger degrades to a no-op if the import fails. Wired into all five trainers and `tasks/engine.py`. |
| `replicates.py` | Multi-seed replicates over the `TaskRunner` handle (ADR-056) — trains the same config once per seed and aggregates every metric into n/mean/std/min/max/95% CI (Student-t). Exposed as `POST /api/<task>/replicates` for all five tasks and `POST /api/custom/{key}/replicates`; report persisted to `outputs/reports`. |
| `task_runner.py` | The `TaskRunner` Protocol (ADR-041) — `config_type` / `run(cfg)` / `metrics(result)` / `primary_metric()`. Every generic orchestrator (comparison, sweep, replicates, replicated comparison) drives tasks through this handle alone, which is why adding a task costs no orchestrator code. |
| `comparison.py` | Model comparison (ADR-044) — trains N architectures on one dataset and ranks them through the runner handle. |
| `sweep.py` | Hyperparameter sweep (ADR-045/052) — grid, random or Optuna TPE over any config field by dot-path; emits `trial_start`/`trial_end` so the live monitor tracks real progress. |
| `replicated_comparison.py` | N seeds x M variants, then a paired test (ADR-061) — the honest form of "A beats B". Every variant trains on the *same* seed list; failed variants stay visible in the report instead of vanishing from the matrix. |
| `significance.py` | Paired significance testing (ADR-061) — paired t / Wilcoxon chosen and *justified* per comparison, Cohen's `d_z`, percentile bootstrap CIs, Holm-Bonferroni across the family, and `min_achievable_p` so an underpowered "not significant" is never read as "no effect". |
| `latex_export.py` | Paper-ready `booktabs` tables (ADR-061) for replicates / sweep / K-fold / comparison reports, written beside the JSON and CSV. Table notes state what each interval covers and which correction was applied. |
| `dataset_fingerprint.py` | `dataset_fingerprint` in every `run.json` (ADR-061) — sha256 over the sorted path+size manifest of `data.base_dir`, so "same dataset" becomes checkable. Records the method used, because `manifest` cannot see a same-size edit and `content` can. |
| `dataset_audit.py` | Read-only cross-split SHA-256 duplicate checks and explicit group manifests (ADR-136). Classification CV consumes the same mapping with GroupKFold/StratifiedGroupKFold. |
| `experiment_bundle.py` | Contained experiment records (ADR-135): allowlisted run artifacts in a ZIP with a SHA-256/size manifest, atomic export without overwrite, verification without extraction. CLI: `bundle` and `verify-bundle`. |
| `resume_provenance.py` | Compare a resumed run's recorded dataset fingerprint before native training; reject confirmed changes and warn when legacy identity cannot be verified (ADR-134). |
| `preprocessing.py` | The filter pipeline (blur, edges, wavelet, CLAHE…) shared by every task, applied before augmentation and normalization. |
| `gradcam.py` · `onnx_export.py` · `batch_predict.py` | Post-training tooling: explainability heatmaps, ONNX export with a latency benchmark, and batch inference to CSV. |
| `<task>_data.py` · `<task>_trainer.py` | One pair per standalone task per ADR-033/036/037/038: `regression_data.py`, `segmentation_data.py` and `segmentation_trainer.py`, `anomaly_data.py` and `anomaly_trainer.py`, and for detection `detection_data.py` (Ultralytics `data.yaml`), `detection_dataset.py` (YOLO-format adapter for torchvision), `detection_metrics.py` (mAP@0.5) and `detection_trainer.py`. A new task adds files here and never edits existing ones. |
| `run_groups.py` | Replicate sets, K-folds, sweeps and comparisons as History groups (ADR-113, ADR-123, ADR-124): one group `run.json` per job in `<models_dir>/<name>_<suffix>/<group_id>/`, children tagged with `group_id`. |
| `training_loop.py` · `trial_name.py` | Shared training lifecycle helpers with explicit task-specific inputs; per-trial experiment names for sweeps that carry the value each trial varies (for example `_lr0.002`). |
| `live_confusion.py` · `live_insights.py` | Live "Results" tab while a classifier trains: the validation confusion of each epoch (ADR-161) and Grad-CAM of a few fixed validation images that leaves the training history unchanged (ADR-162). |
| `cancellation.py` · `resume.py` · `loader_lifecycle.py` | Cooperative stop signal; epoch-boundary resume including RNG and loader generators (ADR-134). Explicit resume rejects unusable state before training setup (ADR-138); optional capability probes return `None`. Deterministic shutdown of DataLoader worker pools. Native runs record effective worker counts in `run.json.runtime`. |
| `training_health.py` · `metric_ci.py` · `image_size.py` · `learning_rate.py` · `materialized_dataset.py` | Warnings for runs that produced a number instead of a result; bootstrap CIs for one run's test metrics (ADR-074/076); suggested resolution and learning rate; filtered on-disk dataset copies for backends that own their pipeline (ADR-084, ADR-130). |

### `tasks/` — researcher-defined tasks (ADR-058)

Imports `core/` and `utils/`; used by `gui/` and lazily by `utils/selftest.py`. The "sixth task" surface: a researcher drops one documented `.py` into
`user_tasks/` and gets a real GUI tab, the live monitor, run history,
`run.json` provenance, sweeps and replicates — without writing React,
FastAPI or a training loop.

| Module | Responsibility |
|---|---|
| `base.py` | `BaseTaskConfig` (composes the shared training/data/output/device blocks) and the generic `TaskSpec[ConfigT]` ABC: four Level-1 hooks (`build_model`, `build_loaders`, `compute_loss`, `compute_metrics`) plus a `run(cfg, ctx)` escape hatch for training that is not epoch-shaped. Torch appears only under `TYPE_CHECKING`, so schema generation needs no hardware extra. |
| `registry.py` | `@register_task` (key, label, accent colour, metric names + directions) and `user_tasks/` discovery, mirroring the proven `user_models/` pattern. Built-in keys are reserved; a broken user file logs a warning and is skipped rather than crashing discovery. |
| `engine.py` | `GenericTaskEngine` — the loop VisionForge owns: seeding, device resolution, AMP, early stopping, direction-aware best checkpoint, SSE events, TensorBoard, metric curve and the `run.json` contract stamped `custom:<key>`. |
| `runner.py` | `CustomTaskRunner` — wraps the engine behind the `TaskRunner` handle so custom tasks get sweeps, replicates and replicated comparison from the same orchestrators the built-ins use. |
| `manage.py` | Hide, unhide and delete a `user_tasks/` file (`/api/custom/{key}/hide`, `/unhide`, `DELETE /api/custom/{key}`). |
| `scaffold.py` | `visionforge new-task <key>` — writes a commented template that **trains out of the box** on synthetic data, so the tab is live before the researcher writes anything. |

### `blocks/`

Imports `core/`, `models/` and `utils/`. Three kinds of module:

**Classification strategies** implement the `ExperimentBlock` ABC (`blocks/base.py`) over `ExperimentConfig`:

```python
class ExperimentBlock(ABC):
    def setup(self, config: ExperimentConfig) -> None: ...
    def run(self) -> None: ...
    def report(self) -> dict: ...
```

`BlockRegistry.discover()` lists the concrete `ExperimentBlock` subclasses. Dispatch is not by registry: `_execute_experiment` in `gui/api/routes.py` reads `config.block` (ADR-026).

| Block | `config.block` | Dispatched by | Description |
|---|---|---|---|
| `ClassificationBlock` | `classification` | `POST /api/experiment/run` | End-to-end classification: ModelFactory + DataModule + Trainer + Evaluator + Plotter. |
| `CrossValidationBlock` | `cross_validation` | `POST /api/experiment/run` | K-Fold and Stratified K-Fold with per-fold metrics. |
| `GridSearchBlock` | `grid_search` | `POST /api/experiment/run` | Exhaustive sweep over hyperparameter combinations (ADR-031). |
| `RandomSearchBlock` | `random_search` | `POST /api/experiment/run` | Random sampling: `uniform`/`log_uniform`/`choice` per param, `n_trials`, `seed`. |
| `TransferLearningBlock` | `transfer_learning` | `POST /api/experiment/run` | Feature extraction vs fine-tuning with frozen layers and a backbone LR multiplier. |
| `ModelComparisonBlock` | `model_comparison` | `POST /api/experiment/run` | Trains N architectures with identical config and ranks them by a chosen metric. |
| `BatchPredictionBlock` | `batch_prediction` | `POST /api/runs/{id}/batch_predict` | Inference on an image folder, CSV output. |
| `ExportONNXBlock` | `export_onnx` | `POST /api/runs/{id}/export_onnx` | ONNX export with optional inference validation and latency benchmark. |

Blocks that run N sub-trainings (`grid_search`, `random_search`, `model_comparison`, `cross_validation`) forward each inner epoch with trial context, bracketed by `trial_start`/`trial_end` events.

**Standalone task blocks** (ADR-033/036/037/038) mirror `setup`/`run`/`report` over their own config tree and do **not** subclass `ExperimentBlock`: `DetectionBlock`, `RegressionBlock`, `SegmentationBlock`, `AnomalyBlock`. Each is dispatched by its `/api/<task>/run` endpoint.

**Runners and K-fold helpers** (ADR-041): `classification_runner.py`, `detection_runner.py`, `regression_runner.py`, `segmentation_runner.py` and `anomaly_runner.py` each hold a `*Runner` that satisfies the `TaskRunner` Protocol (`core/task_runner.py`). Their `primary_metric()` is `accuracy`, `map50_95`, `r2`, `miou` and `auroc`. `regression_cv.py` and `segmentation_cv.py` provide `run_regression_cross_validation` and `run_segmentation_cross_validation` (`/api/regression/cv`, `/api/segmentation/cv`); `_search_utils.py` is shared by the two search blocks and `_cv_common.py` holds the fold loaders, reporting and execution shared by the standalone task CV.

### Task support matrix

| Task | Config | Block | Runner | K-fold | Sweep / compare / replicates |
|---|---|---|---|---|---|
| classification | `utils/config.py` | `ClassificationBlock` + strategy blocks | `ClassificationRunner` | `CrossValidationBlock` | blocks `grid_search`, `random_search`, `model_comparison`; `/api/classification/{replicates,replicated-comparison}` |
| regression | `utils/regression_config.py` | `RegressionBlock` | `RegressionRunner` | `regression_cv.py` | `/api/regression/{sweep,compare,replicates,replicated-comparison}` |
| segmentation | `utils/segmentation_config.py` | `SegmentationBlock` | `SegmentationRunner` | `segmentation_cv.py` | `/api/segmentation/{sweep,compare,replicates,replicated-comparison}` |
| detection | `utils/detection_config.py` | `DetectionBlock` | `DetectionRunner` | none | `/api/detection/{sweep,compare,replicates,replicated-comparison}` |
| anomaly | `utils/anomaly_config.py` | `AnomalyBlock` | `AnomalyRunner` | none | `/api/anomaly/{sweep,compare,replicates,replicated-comparison}` |
| custom (`user_tasks/`) | `BaseTaskConfig` subclass | `GenericTaskEngine` | `CustomTaskRunner` | none | `/api/custom/{key}/{sweep,replicates,replicated-comparison}` |

### `gui/`

Imports `blocks/`, `tasks/`, `core/`, `models/` and `utils/`. React + shadcn/ui frontend served by FastAPI in the same Python process. Training runs off the event loop so the API stays responsive while PyTorch uses the GPU; jobs share a FIFO queue (ADR-075).

| Module | Responsibility |
|---|---|
| `server.py` | FastAPI application. Mounts `api/routes.py` and serves the pre-built SPA from `static/`, confined to the resolved SPA root (ADR-127). Also serves `GET /api/health` and applies `security.py`. Entry point behind `visionforge gui`. |
| `api/routes.py` | REST router, prefix `/api`, which includes the routers split out of it (ADR-132). Still defines: schema + lifecycle (`GET /api/schema`, `POST /api/experiment/run`, `GET /api/experiment/{status,events,result/{id}}`); per-task schema, run, sweep, compare, replicates and K-fold (`/api/{detection,regression,segmentation,anomaly}/*`, `/api/classification/{replicates,replicated-comparison}`); researcher tasks (`/api/tasks`, `/api/custom/{key}/*`); per-run actions (`/api/runs/{id}` and `/export_md`, `/resume`, `/reveal`); `/api/sample-dataset`, `/api/queue`, `/api/update`, `GET /api/artifacts/{file_path}`. |
| `api/schemas.py` | Pydantic request/response models for the routes above. |
| `api/evaluation_routes.py` | Per-run evaluation split out of `routes.py`: `POST /api/runs/{id}/test`, `/gradcam`, `/batch_predict`, `/export_onnx`, with the run lookup and profile injected (#204). |
| `api/profiles_routes.py` | Profile endpoints mounted by the shared router, with output directories resolved per request (ADR-132). |
| `security.py` | Refuses requests whose `Host` or `Origin` does not name this server (DNS rebinding and cross-site requests to the local server). |
| `api/datasets_routes.py` | Dataset, picker and credential endpoints split out of `routes.py` (`/api/dataset/*`, `/api/<task>/dataset/*`, `/api/checkpoint/pick`, `/api/credentials`) (ADR-132). |
| `api/runs_routes.py` · `api/run_history.py` | Run list and delete endpoints; pure parsing of `run.json` for the list, the detail and the resume rules, with no FastAPI import (ADR-132). |
| `api/run_events.py` | The events of one run kept for its whole life so any reader, including a reloaded page, walks them from the start (ADR-159). |
| `api/http_responses.py` | OpenAPI text of the error status codes shared by the routers (ADR-132). |
| `api/model_card.py` | Pure localized model-card sections read from the `run.json` contract. |
| `api/code_tour_routes.py` | `GET /api/code-tour` and `/api/code-tour/{task}`: the real code of a training for the "Código" tab (ADR-173). |
| `api/profile_tokens.py` | One private token per profile and service (Hugging Face Hub, Weights & Biases, Zenodo, GitHub), stored where only the server reads it and never sent back to the page. |
| `api/hub_routes.py` · `api/wandb_routes.py` · `api/zenodo_routes.py` · `api/github_routes.py` · `api/dvc_routes.py` | History export endpoints for the selected profile: private Hugging Face Hub (ADR-171), Weights & Biases (ADR-172), Zenodo drafts (ADR-174), GitHub without weights (ADR-175) and a local DVC repository (ADR-176). |
| `api/evidence_routes.py` | Read-only dataset audits and profile-confined run bundle downloads (ADR-140). A source reservation excludes active/queued training and concurrent resume; cancellation waits for the archive writer before deleting temporary files. |
| `api/system_routes.py` · `api/system_service.py` | Host/device information and model defaults; thin HTTP handlers delegate to a service with injected runtime probes (ADR-132). |
| `api/run_queue.py` · `api/profiles.py` · `api/gpu_health.py` | FIFO training queue (ADR-075), and `RunLanes`: the local queue plus one per Google account, running side by side (ADR-163, phase 4); profiles as output folders (ADR-114); whether the CUDA context still works (ADR-122). |
| `api/dataset_download.py` · `api/sample_dataset.py` · `api/folder_opener.py` | Provider downloads (ADR-055); synthetic sample datasets for the first-run guides; open a folder in the OS file manager. |
| `api/torch_gradcam.py` · `api/torch_onnx_export.py` · `api/torch_batch_predict.py` · `api/detection_export.py` · `api/detection_testing.py` | Per-task post-training helpers over the `core/` tools. |
| `core/dataset_explorer.py` · `api/explorer_routes.py` · `api/explorer_jobs.py` | Read-only dataset explorer: an index that only lists directories (counts, gallery), server-side thumbnails, and a background integrity scan with progress and cancel (corrupted files, SHA-256 copies, leakage between splits, sizes) (ADR-165). |
| `core/dataset_incorporate.py` · `api/incorporate_routes.py` · `api/incorporate_jobs.py` | Adding new images to a dataset as a new sibling dataset: checks (format, unreadable, SHA-256 copies, new classes), a seeded split stratified by class, a hardlink-or-copy mirror written under a temporary name and renamed once, and `visionforge_dataset.json`; background preview and apply with progress and cancel (ADR-166). |
| `core/dataset_merge.py` · `core/dataset_derived.py` · `api/merge_routes.py` · `api/merge_jobs.py` · `api/derived_jobs.py` | Merging two datasets into a new sibling dataset with a class mapping (rename, merge, discard): SHA-256 duplicates across both, ids renumbered, each image keeping its split, `visionforge_dataset.json` with `kind: "merged"`; `dataset_derived` holds what it shares with incorporate (hardlink-or-copy, temporary folder plus one rename, registry, class files) (ADR-167). |
| `core/dataset_resplit.py` · `api/resplit_routes.py` · `api/resplit_jobs.py` | Re-splitting a dataset into a new sibling dataset: train and val pooled and redistributed by ratio (test locked by default), stratified by class and seeded, exact copies and optional `path,group` groups kept in one split, anything that would leak from the locked test dropped and reported, `visionforge_dataset.json` with `kind: "resplit"` (ADR-168). |
| `core/dataset_coco.py` · `core/dataset_coco_import.py` · `api/coco_routes.py` · `api/coco_jobs.py` | Converting detection datasets between YOLO and COCO into a new sibling dataset: export writes `images/<split>` + `annotations/instances_<split>.json`, import reads a COCO folder (splits from file names, ids remapped to 0..N-1, crowd and degenerate boxes reported), one job at a time, `visionforge_dataset.json` with `kind: "coco_export"` or `"coco_import"` (ADR-169). |
| `api/updates.py` · `updater.py` | Newer-release notice and install (ADR-120). |
| `api/colab_connection.py` · `api/colab_routes.py` | Google Colab through the pinned `google-colab-cli`, in a venv of its own: install, login per profile (`USERPROFILE`/`HOME` confined to `~/.visionforge/colab/<profile>/`), account and compute-unit balance (ADR-163). |
| `api/colab_jobs.py` · `api/colab_remote_runner.py` | One classification job on a Colab VM: `colab new`, `colab exec` of the runner (prefixed JSON events on stdout, a final `job_end`), `colab stop` in a `finally`; sessions left alive on the account (ADR-163, phase 2). Before the stop, `colab download` of the run's experiment bundle, which `core/experiment_bundle.import_experiment_bundle` unpacks into the profile's `models/` (phase 3). |
| `api/colab_drive.py` | Datasets in the user's Google Drive for Colab runs: the local Drive path is translated to the VM's `/content/drive/MyDrive/...`; none of it talks to Google (ADR-164). |
| `api/model_notes.py` | The sentence explaining why a model's default is flagged. |
| `static/` | Pre-built React SPA. Generated by `npm run build` in `frontend/`. Users never need Node.js. |

### `frontend/` (dev-time only)

React + Vite + TypeScript source (`frontend/src/`). Not part of the Python package; the build output lands in `src/visionforge/gui/static/`.

| Path | Responsibility |
|---|---|
| `src/App.tsx` | Top-level shell. Renders `TabBar` and the active task's panel: `ParamPanel` (classification), `DetectionPanel`, `RegressionPanel`, `SegmentationPanel`, `AnomalyPanel`, `CustomTaskPanel` (researcher tasks, form from `SchemaForm`). Hosts the History, Datasets and Queue overlays. |
| `src/types/tasks.ts` | The built-in task definitions (`classification`, `detection`, `regression`, `segmentation`, `anomaly`), merged with `GET /api/tasks`. |
| `src/components/ParamPanel.tsx` | Classification form: strategy selector, K-fold fields, model, training, dataset, preprocessing, augmentation. Grid-search axes via `src/lib/grid-axis.ts` (ADR-031). |
| `src/components/{Detection,Regression,Segmentation,Anomaly}Panel.tsx` | The same panel shape for the other four tasks, with their model lists in `src/lib/*-models.ts`. |
| `src/components/{PreprocessingPanel,DatasetPicker,DatasetStats,TaskDatasetStats,DatasetsOverlay}.tsx` | Preprocessing pipeline builder with live preview (`/api/dataset/preview_preprocess`), folder picking and split detection, dataset statistics, dataset download. |
| `src/components/{CvCard,SweepCard,ReplicatesCard}.tsx` | Strategy cards for K-fold, sweeps and replicates. |
| `src/components/TrainingOverlay.tsx` · `src/components/training-story/` | Modal during training: SSE-driven progress and live log. |
| `src/components/{HistoryOverlay,RunDetailPanel,RunGroupDetail,CompareRunsPanel,LeaderboardView}.tsx` | History over `/api/runs`, run drilldown, replicate/K-fold/sweep group detail, multi-run comparison, leaderboard. |
| `src/components/{ResultsView,Lightbox,QueueOverlay,DeviceSelector,BottomBar}.tsx` | Post-run summary, image zoom, queue view, device selector (from `/api/device/info`), fixed action bar. |
| `src/hooks/useExperiment.ts` | Lifecycle: submit config, open SSE, poll status, fetch results; translates Pydantic error paths. |
| `src/api/client.ts` | Typed fetch wrappers for the API. |
| `src/lib/` | Pure helpers with `*.test.ts` beside them (grid axes, sweep space, run groups, compare metrics, schema defaults, form import). |

## Device handling

`check_cuda()` in `utils/cuda.py` detects the device at runtime; `DeviceConfig.kind` is `cpu`, `cuda` or `multi_cuda`, with optional `gpu_ids`.

```
cpu         → torch.device("cpu")
cuda        → torch.device("cuda:<gpu_ids[0] or 0>")
multi_cuda  → torch.nn.DataParallel over gpu_ids (classification, regression, segmentation trainers);
              gpu_ids passed to Ultralytics (detection)
```

`DistributedDataParallel` is not implemented (ADR-007). The device actually used is recorded in `run.json` as `device_used`. PyTorch installation is the user's responsibility (ADR-005): `torch`/`torchvision` come from a hardware extra (`cpu`, `cu118` … `cu128`); everything else ships in the base install (ADR-106).

## Dependency policy

`pyproject.toml` is canonical. Base dependencies include `ultralytics`, `timm`, `optuna`, `tensorboard`, `roboflow`, `kaggle` and `datasets` (ADR-106). The extras `detection`, `timm`, `optuna`, `tensorboard`, `roboflow`, `kaggle` and `huggingface` are empty and exist only so saved install lines keep resolving. The extras that install something are the hardware extras (`cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`), the optional exporters `mlflow`, `wandb`, `dvc` and `hub` (ADR-170, ADR-171, ADR-172, ADR-176), and `dev`.

## Task expansion model

Two routes, by who adds the task.

**Built-in task** (ADR-033/036/037/038, skill `vf-new-task`). A new task adds new files and edits no existing task code:

```
src/visionforge/
├── utils/<task>_config.py       # standalone Pydantic config tree
├── core/<task>_data.py          # datasets / loaders
├── core/<task>_trainer.py       # training loop, writes run.json
├── models/<task>_factory.py     # model builder (if needed)
├── blocks/<task>.py             # standalone block (setup/run/report)
├── blocks/<task>_runner.py      # TaskRunner: sweeps, comparison, replicates
└── gui/api/routes.py            # /api/<task>/* endpoints (shared router, extended)
frontend/src/components/<Task>Panel.tsx   # tab panel
```

The shared files (`routes.py`, `App.tsx`, `types/tasks.ts`) are extended, not restructured.

**Researcher task** (ADR-058). One `.py` in `user_tasks/` registered with `@register_task`; no package edit, no React. See `tasks/` above and `docs/custom/TASKS.md`.

## Boundaries

Layer rule: `utils/` ← `core/` ← `blocks/` ← `gui/`. Imports point left only; `outputs/` is write-only.

- `utils/` imports no other VisionForge package at module level.
- `models/` imports `utils/` only.
- `core/` imports `utils/` and `models/`.
- `blocks/` imports `core/`, `models/` and `utils/`.
- `tasks/` imports `core/` and `utils/` (never `blocks/` or `gui/`).
- `gui/` imports `blocks/`, `tasks/`, `core/`, `models/` and `utils/`.
- `configs/` holds plain YAML examples (`baseline.yaml`, `detection.yaml`, `regression.yaml`, `segmentation.yaml`, `anomaly.yaml`, `synthetic_test.yaml`) — no Python imports.

Real exceptions to the strict order (verified by grepping `visionforge` imports per package):

| Exception | Where | Why |
|---|---|---|
| `core/` → `models/` | `core/{anomaly,detection,regression,segmentation}_trainer.py` import their factory (`regression_trainer` imports `final_linear` from `models/factory.py`) | Trainers build their own models; `models/` depends on `utils/` only, so there is no cycle. |
| `utils/` → `gui/`, `tasks/` | `utils/selftest.py` imports `visionforge.gui.server` and `visionforge.tasks.registry` inside functions | `visionforge selftest` drives the real API end to end. |
| `utils/` → package root | `utils/environment.py` reads `visionforge.__version__` inside a function | Records the version in `run.json`. |
