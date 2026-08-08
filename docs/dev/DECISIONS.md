# Architecture Decision Records

## ADR-001 — src/ layout

**Date:** 2026-03  
**Status:** Accepted

**Decision:** Place all source code under `src/visionforge/` instead of a top-level `visionforge/` directory.

**Reason:** Without the `src/` layout, Python may import the local directory instead of the installed package, causing subtle differences between development and CI environments. The `src/` layout enforces that the installed package is always used.

---

## ADR-002 — Pydantic v2 for config validation

**Date:** 2026-03  
**Status:** Accepted

**Decision:** Use Pydantic v2 models to validate all experiment configuration loaded from YAML files.

**Reason:** Experiment configs have many interdependent constraints (e.g. `task=binary` requires `num_classes=1`, `batch_size` must be a power of 2). Catching these at load time with clear error messages is far better than failing silently mid-training after hours of GPU time. Pydantic v2 also provides free JSON schema generation, used by the GUI to auto-generate dropdowns for `Literal` fields and free inputs for primitive types.

---

## ADR-003 — ExperimentBlock plugin interface

**Date:** 2026-03  
**Status:** Accepted

**Decision:** All experiment strategies (GridSearch, KFold, TransferLearning, etc.) implement an `ExperimentBlock` ABC with `setup()`, `run()`, and `report()` methods. A `BlockRegistry` auto-discovers blocks at runtime.

**Reason:** The original notebook had experiment logic mixed with training code, making it hard to add new strategies. The plugin interface allows adding a new block by creating a single file, with no changes to existing code. It also makes each strategy independently testable.

---

## ADR-004 — Gradio for GUI

**Date:** 2026-03  
**Status:** Superseded by ADR-016

**Decision:** Use Gradio for the browser-based GUI instead of PyQt6 or Streamlit.

**Reason:** Gradio runs as a local web server but the Python process stays local — CUDA, file system, and all local resources are fully accessible. PyQt6 requires native desktop dependencies and is harder to style. Streamlit has server/client separation that complicates direct CUDA access. Gradio supports `gr.Tabs()` for task-based page separation, `gr.Dropdown()` for `Literal` fields, and `gr.Number()`/`gr.Slider()` for primitive types — making it the cleanest option for a research tool that needs both a modern UI and direct GPU access.

---

## ADR-005 — PyTorch and CUDA are user responsibility

**Date:** 2026-03  
**Status:** Accepted

**Decision:** PyTorch is not listed as a standard dependency in `pyproject.toml`. Users install the correct build for their hardware. The project auto-detects and adapts to CPU, single GPU, or multi-GPU at runtime.

**Reason:** PyTorch has hardware-specific builds (CPU, cu118, cu121, cu124, cu128, Nightly) that cannot be resolved automatically by a package manager. Installing the wrong build silently results in CPU-only execution. Placing the responsibility on the user with clear documentation is safer than attempting auto-resolution. The `device.py` module handles runtime adaptation transparently.

**Update (2026-06-03):** `pyproject.toml` had drifted from this ADR — `torch>=2.3` and `torchvision>=0.18` were listed in core `dependencies`, with `[tool.uv.sources]` defaulting win32/linux to the **cu121** index. cu121 torchvision wheels stop at cp312, so on Python 3.13 the resolver failed and **CI install broke** (both open PRs). Restored ADR-005: torch/torchvision are removed from core `dependencies` and live in the per-hardware extras (`cpu`/`cu118`/`cu121`/`cu124`/`cu126`), so the user selects a build explicitly — `pip install -e ".[cu121]"` (GPU) or `".[cpu]"` (CPU). The no-extra default index is now **CPU** (wheels exist for every supported Python, incl. 3.13), so a transitive torch (e.g. via the `detection`/`ultralytics` extra) resolves instead of breaking.

---

## ADR-006 — Task-based architecture for future expansion

**Date:** 2026-03  
**Status:** Accepted

**Decision:** The system is designed around a `task` abstraction from the start, even though only classification is implemented initially. Future tasks (Object Detection, Segmentation, Regression, Anomaly Detection) will each have their own config models, blocks, and GUI tab.

**Reason:** Adding task support after the fact requires invasive refactoring. Designing the abstraction early means new tasks slot in as new modules without touching existing code. The GUI tab-per-task pattern enforces this boundary visually.

---

## ADR-007 — Multi-GPU: DataParallel first, DDP later

**Date:** 2026-03  
**Status:** Accepted

**Decision:** Multi-GPU support starts with `torch.nn.DataParallel` (single machine, simple API) and will be extended to `DistributedDataParallel` in a future phase.

**Reason:** `DataParallel` covers the common single-machine multi-GPU case with minimal code changes. `DistributedDataParallel` is more robust and scalable but requires significant additional infrastructure. Starting with `DataParallel` delivers value sooner without blocking the core system.

---

## ADR-008 — ruff replaces black + flake8

**Date:** 2026-03  
**Status:** Accepted

**Decision:** Use `ruff` for both formatting and linting instead of separate `black` and `flake8` tools.

**Reason:** `ruff` is 10-100x faster than `black` + `flake8`, implements a superset of flake8 rules, and provides formatting equivalent to black. Replacing two tools with one simplifies the pre-commit config and CI pipeline with no loss of coverage.

---

## ADR-009 — loguru for logging

**Date:** 2026-03  
**Status:** Accepted

**Decision:** Use loguru instead of the Python standard library `logging` module.

**Reason:** Standard `logging` requires significant boilerplate to configure properly (handlers, formatters, propagation). loguru provides the same functionality with a simpler API, built-in color support, automatic file rotation, and structured output. The `setup_logger()` function is called once from the entry point — not at import time — to avoid side effects.

---

## ADR-010 — CI on CPU with torch stable

**Date:** 2026-03  
**Status:** Accepted

**Decision:** GitHub Actions CI installs the CPU-only stable PyTorch build.

**Reason:** GitHub Actions runners have no GPU. The CI environment validates code correctness, type safety, and test logic — not GPU performance. Using the stable CPU build keeps CI fast and avoids depending on Nightly builds in the CI pipeline, which can break without notice.

---

## ADR-011 — Gradio as the GUI framework

**Date:** 2026-03  
**Status:** Superseded by ADR-016

**Decision:** Use Gradio for the browser-based GUI.

**Reason:** Gradio runs in the same process as the training loop. There is no client/server separation, so the GUI has direct access to the GPU, the file system, logs, and training history without any IPC or API layer. It supports `gr.Tabs()` for task-based layout, auto-generates form fields from Pydantic schemas, and is sufficient for a local research tool. The CSS/JS customization API allows a polished interface within those constraints.

---

## ADR-012 — Documentation records decisions, not options

**Date:** 2026-03  
**Status:** Accepted

**Decision:** When a decision is made, document only the decision and its reason. Do not maintain comparison tables of alternatives that were not chosen.

**Reason:** Comparison tables become outdated and create false ambiguity. If a decision changes, the document is updated to reflect the new reality. Documentation and code must always agree.

---

## ADR-013 — `run.json` as the contract between Trainer and GUI

**Date:** 2026-03  
**Status:** Accepted

**Decision:** The `Trainer` writes a single `run.json` file per run, containing the full config, per-epoch history, final metrics, and paths to all artifacts (model, graphics, report). The GUI reads only this file to populate the Run History tab.

**Reason:** A well-defined file contract decouples the training backend from the GUI frontend completely. The history tab can be built and tested independently of the Trainer. The schema must be stable before accumulating real runs — changing field names later breaks historical data. Keeping everything in one file per run also makes runs portable and inspectable without any database.

**Update (2026-06-04):** added an `environment` block to `run.json` (`utils/environment.capture_environment` → Python/platform/torch/torchvision/numpy/visionforge versions). The config records *what* was requested; the environment records *what actually ran it* — both are needed to truly reproduce a result when library versions change numerical behavior (CLAUDE.md §7.4). It is an additive field (legacy run.json without it still loads — `RunDetail.environment` defaults to `{}`) and surfaces in the GUI `RunDetailPanel`. Version probing never raises (`"unknown"` fallback), so capturing it can't fail a run.

**Update (2026-06-04, integ):** the `environment` block is now written by **all** trainers — classification plus detection/regression/segmentation/anomaly — so every run, regardless of task, records its library versions (reproducibility parity). Done on `integ/all-features` once `capture_environment` and the task trainers coexist; covered by an `assert "torch" in run_json["environment"]` in each task trainer's run.json test.

**Date:** 2026-03  
**Status:** Accepted

**Decision:** `ModelConfig` exposes a `weights_path: Path | None` field. The `ModelFactory` loads weights in priority order: local file if provided, ImageNet weights if `pretrained=True`, random weights otherwise.

**Reason:** A researcher may want to fine-tune from a domain-specific checkpoint rather than ImageNet weights. The field accepts any `.pth` file on disk, which the GUI surfaces as a file picker or a dropdown of files found in `configs/weights/`. This keeps the config YAML as the single source of truth without requiring code changes to switch weight strategies.

---

## ADR-015 — User-configurable transforms via `TransformConfig`

**Date:** 2026-03  
**Status:** Accepted

**Decision:** A `TransformConfig` Pydantic model is nested inside `DataConfig`. It exposes individual boolean and numeric flags for each augmentation step (`horizontal_flip`, `rotation_degrees`, `color_jitter`, `normalize_mean`, `normalize_std`). `DataModule` builds the transform pipeline from these flags at runtime.

**Reason:** Augmentation choices significantly affect model generalization and are part of the experiment hypothesis. Hardcoding transforms hides an experimental variable. Exposing them in the config makes augmentation reproducible, diffable between runs, and configurable from the GUI without touching Python code.

---

## ADR-016 — React + FastAPI replaces Gradio for GUI

**Date:** 2026-04  
**Status:** Accepted  
**Supersedes:** ADR-004, ADR-011

**Decision:** Replace Gradio with a React + TypeScript + shadcn/ui frontend served by a FastAPI backend in the same Python process.

**Reason:** Gradio was never implemented beyond a dependency listing. When starting the GUI work, React + FastAPI was chosen because: (1) FastAPI has native Pydantic v2 integration — `ExperimentConfig` works directly as a request body type with automatic validation error responses; (2) React with shadcn/ui provides polished, accessible UI components and full control over layout; (3) `asyncio.to_thread()` runs PyTorch training in a background thread while keeping the API responsive — GPU access stays in the same process; (4) the pre-built React SPA is served as static files by FastAPI, so users need only `python -m visionforge gui` (no Node.js required); (5) this stack scales naturally to future features like SSE-based live training monitors and WebSocket connections.

---

## ADR-017 — DeviceConfig as part of ExperimentConfig

**Date:** 2026-05-22  
**Status:** Accepted

**Decision:** Add a `DeviceConfig` (`kind: cpu | cuda | multi_cuda`, optional `gpu_ids`) to `ExperimentConfig`, and have `Trainer.resolve_device()` produce both the runtime `torch.device` and a human label that is persisted to `run.json` as `device_used`.

**Reason:** The old GUI device toggle was cosmetic — `Trainer` always picked CUDA when available and DataParallel when there were ≥ 2 GPUs, regardless of what the user clicked. Worse, when CUDA was unavailable the header still claimed "CUDA" while training silently ran on CPU. Making the device explicit in the config, validating it at submit time, and recording the actual device used eliminates this ambiguity. The fallback path (CUDA requested, CUDA unavailable) is logged as a warning and recorded with a `"(fallback: CUDA unavailable)"` suffix so it's auditable after the fact.

---

## ADR-018 — Server-side native folder picker for dataset paths

**Date:** 2026-05-22  
**Status:** Accepted

**Decision:** Add `POST /api/dataset/pick` that opens a `tkinter` directory dialog on the server and returns the absolute path. The frontend "Escolher pasta" button calls this endpoint instead of the browser's `showDirectoryPicker`.

**Reason:** Browsers deliberately hide absolute filesystem paths behind the File System Access API — `showDirectoryPicker` returns only a folder name, never `C:/datasets/coffee`. The previous implementation papered over this by asking the user to manually paste the parent path, which broke the auto-detect flow and silently dropped paths during YAML export/import. Because VisionForge always runs locally (browser + Python on the same host), running a native dialog server-side is safe and gives the user a real absolute path with no friction.

---

## ADR-019 — Full plot set + per-model test history

**Date:** 2026-05-22  
**Status:** Accepted

**Decision:** Expand the per-run plot set to include train+val accuracy, normalized confusion matrix, ROC curve, and precision-recall curve, in addition to the existing loss and raw confusion matrix. Add `POST /api/runs/{run_id}/test` so a saved checkpoint can be evaluated against arbitrary new datasets, with each test appended to `run.json` under a `tests[]` array.

**Reason:** Research interpretation needs more than loss + raw confusion: ROC/PR are required for threshold tuning, accuracy curves reveal overfitting independent of loss scaling, and normalized confusion matrices expose per-class recall imbalance. Persisting test runs *per model* (not per experiment session) lets the researcher answer "how does this exact checkpoint behave on dataset X six months from now?" without re-loading the model into a notebook — the answer is already in the run directory and surfaces directly in the history overlay.

---

## ADR-020 — GPU performance: benchmark mode by default

**Date:** 2026-05-22  
**Status:** Accepted

**Decision:** Default `TrainingConfig.deterministic` to `False`. When False, `cudnn.benchmark = True` and `cudnn.deterministic = False`; when True, the reverse. Additionally: DataLoaders use `persistent_workers=True` and `prefetch_factor=2` when `num_workers > 0`, and all `.to(device)` calls use `non_blocking=True`.

**Reason:** The original `_seed_everything()` unconditionally set `cudnn.benchmark = False` and `cudnn.deterministic = True` for reproducibility. This caused GPU utilization to drop to ~10% on real workloads because cuDNN was forced to use its slowest algorithm instead of auto-selecting the fastest one for each input shape — the single largest factor in CNN throughput. Combined with synchronous CPU→GPU transfers and DataLoader workers being re-spawned every epoch (expensive on Windows), the GPU spent most of its time idle. The fix makes performance the default and leaves reproducibility as an opt-in flag (`deterministic: true` in config), which is the standard practice in PyTorch research workflows.

---

## ADR-021 — LR scheduler + AMP as opt-in training knobs

**Date:** 2026-05-22  
**Status:** Accepted

**Decision:** Extend `TrainingConfig` with `scheduler: SchedulerConfig` (`kind` ∈ `{none, cosine, step, plateau}`) and `mixed_precision: bool = False`. The Trainer wires these in at `fit()` time and silently falls back to fp32 on CPU even when `mixed_precision=True` (AMP requires CUDA).

**Reason:** Constant LR is rarely optimal for CNN classification — cosine/step/plateau all show consistent gains in the literature and add no risk when default is `none`. AMP gives 2-3× speedup on Ampere+ GPUs at negligible accuracy cost but only matters with CUDA, so silently downgrading on CPU avoids confusing the user with errors. Both are opt-in so existing experiments are unaffected.

---

## ADR-022 — Dataset stats + sample preview before training

**Date:** 2026-05-22  
**Status:** Accepted

**Decision:** Two backend endpoints power a pre-training dataset overview:
- `POST /api/dataset/stats` returns per-class image counts per split and flags imbalance (max/min ratio > 2.0).
- `POST /api/dataset/samples` returns the first N image paths per class for thumbnail rendering, served via `GET /api/dataset/file?path=...` (image-extension gate, file existence check).

**Reason:** Two of the most common sources of failed classification experiments are (a) class imbalance that the user didn't notice and (b) mislabeled folders (a "dog" folder with cats). Surfacing both before training takes seconds and avoids hours of wasted GPU time. The `/dataset/file` endpoint is acceptably permissive because VisionForge runs locally — the user can already read any file via the OS.

---

## ADR-023 — Multi-run comparison panel

**Date:** 2026-05-22  
**Status:** Accepted

**Decision:** Selecting 2+ runs in `HistoryOverlay` opens a `CompareRunsPanel` that renders a side-by-side metric table plus overlaid SVG line charts of `val_loss` and `val_accuracy` across epochs (one color per run). The component fetches each run's full detail via the existing `GET /api/runs/{id}` endpoint — no new backend route required.

**Reason:** Research workflows constantly need answers to "which of these N models won?" — the same question that drives table 3 in any classification paper. Building this from the existing run-detail endpoint keeps the backend surface area small while making the comparison a one-click operation in the UI.

---

## ADR-024 — Preprocessing pipeline (PIL/NumPy, opt-in)

**Date:** 2026-05-22  
**Status:** Accepted

**Decision:** Image preprocessing filters live in `src/visionforge/core/preprocessing.py` as a registry-of-functions (`gaussian_blur`, `median_blur`, `unsharp`, `edges`, `emboss`, `grayscale`, `equalize`, `autocontrast`, `wavelet`). The `wavelet` step is a 1-level Haar decomposition (LL/LH/HL/HH) implemented in pure NumPy. A new endpoint `POST /api/dataset/preview_preprocess` writes each step's output PNG to `outputs/preview_cache/<class>/` and returns the artifact paths so the frontend can render a strip of thumbnails.

**Reason:** The user wanted multiple preprocessing modes (wavelet, CLAHE, blur, edges) with a "before/after" preview button. Implementing every filter with PIL + NumPy avoids adding `opencv`, `pywavelets`, or `scikit-image` as dependencies — VisionForge already runs on Windows where opencv wheels are large and version-sensitive. Researchers needing tighter wavelet families can adopt `pywt` in a later PR; the registry pattern keeps that addition local.

---

## ADR-025 — `PreprocessingPanel` is a controlled component

**Date:** 2026-05-22  
**Status:** Accepted

**Decision:** `PreprocessingPanel` accepts `steps` and `onChange` props instead of keeping the pipeline in panel-local `useState`. The owner (`ParamPanel`) binds those props to `formData.data.preprocessing.steps`, converting between the schema's flat shape (`{kind, ...params}`) and the UI's nested shape (`{kind, params}`) with `toUIPreprocessingSteps` / `fromUIPreprocessingSteps` helpers.

**Reason:** As originally written, `PreprocessingPanel` was uncontrolled: the pipeline the user built (Gaussian blur → Wavelet → Grayscale) lived only in the panel's own React state. The Preview button worked, but `formData` never received the pipeline — so the POST to `/api/experiment/run` shipped an empty `preprocessing.steps`, the `DataModule` saw an empty list, and training silently ran without any of the filters the user had configured. The fix unblocks YAML export/import of the pipeline, surfaces the pipeline in the run history badges, the training overlay, the run detail page, the markdown export, and the multi-run comparison panel — all of which depend on `data.preprocessing.steps` actually being in the submitted config.

---

## ADR-026 — Block dispatch by `config.block` instead of hardcoded `ClassificationBlock`

**Date:** 2026-05-23  
**Status:** Accepted

**Decision:** `_execute_experiment` in `gui/api/routes.py` dispatches to a concrete `ExperimentBlock` based on `config.block`. The first two paths wired are `classification` → `ClassificationBlock` and `cross_validation` → `CrossValidationBlock`. The progress callback (`block._progress_callback`) is set only when the block streams epoch-level events (today: just `ClassificationBlock`).

**Reason:** Backend Phase 5 implemented eight blocks (GridSearch, RandomSearch, CrossValidation, TransferLearning, ModelComparison, BatchPrediction, ExportONNX, Classification) but the GUI dispatcher always instantiated `ClassificationBlock` — making seven of them unreachable through the UI. Switching to config-driven dispatch removes the need for a parallel `/api/cv-experiment/run` route and lets future blocks be added with one line of code in the dispatcher plus their own UI surface. The progress callback gate exists because not every block has a per-epoch concept: K-Fold runs N full trainings serially and reports per-fold summaries.

---

## ADR-027 — Custom checkpoint path exposed via dedicated file picker endpoint

**Date:** 2026-05-23  
**Status:** Accepted  
**Supersedes context:** ADR-014 (which only described the field; this ADR documents the GUI surface)

**Decision:** Add `POST /api/checkpoint/pick` that opens a native `tkinter.filedialog.askopenfilename` filtered to `*.pth *.pt` on the server. The `WeightsPathField` component renders an input + `📁 Escolher` button + `limpar` action. `weights_path` is removed from the field-renderer's `SKIP_FIELDS` so it appears in the Model section.

**Reason:** ADR-014 introduced `weights_path` in `ModelConfig` but the GUI's field-renderer was skipping it, so the field only existed for YAML users. Researchers iterate fine-tuning experiments (train on big dataset → fine-tune on small dataset → fine-tune again) and that loop demands picking previously-trained `.pth` files without leaving the UI. The picker is server-side for the same reason as the dataset picker (ADR-018) — browsers refuse to return absolute paths.

---

## ADR-028 — Cross-validation as the second GUI-exposed block

**Date:** 2026-05-23  
**Status:** Accepted

**Decision:** The second block surfaced in the GUI is `CrossValidationBlock` (K-Fold + Stratified). A `BlockSelector` segmented control in `ParamPanel` toggles between "Treino simples" (classification) and "K-Fold (CV)". When CV is selected, `CrossValidationFields` renders `n_folds`, `stratified`, `shuffle`, `fold_seed`. Defaults (`n_folds=5, stratified=true, shuffle=true, fold_seed=42`) are auto-populated when the user flips the selector.

**Reason:** Of the seven backend-only blocks, K-Fold has the highest research value for the classification workflow VisionForge is built around — papers asking "how robust is this architecture on this dataset?" almost always require stratified K-Fold. Surfacing it second (after the default classification path) validates the dispatch-by-`config.block` design for all future blocks without committing to a UI grammar that has to cover GridSearch parameter spaces or BatchPrediction file workflows up front.

---

## ADR-029 — Multi-run comparison shows config diff, not just metrics

**Date:** 2026-05-23  
**Status:** Accepted  
**Extends:** ADR-023

**Decision:** `CompareRunsPanel` renders a `ConfigDiffTable` that compares 16 hyperparameters across the selected runs (architecture, lr, optimizer, batch, seed, scheduler, image_size, augmentation flags, preprocessing pipeline). Cells whose value differs from the first run are highlighted in amber. A separate `PreprocessingCompare` block lays out each run's preprocessing pipeline side by side with full params.

**Reason:** ADR-023 added the comparison panel for metrics and epoch curves, but a researcher seeing "Run B beats Run A by 3% accuracy" still has to dig through two configs to find the variable that changed. Surfacing the diff inline makes that attribution one glance instead of a manual file diff. The preprocessing block is broken out separately because the pipeline is an ordered list, not a scalar — a row in the diff table would only show "gaussian_blur → grayscale" vs "grayscale → wavelet" without making the difference obvious.

---

## ADR-030 — Preprocessing transform must be a top-level picklable callable

**Date:** 2026-05-31  
**Status:** Accepted  
**Extends:** ADR-024, ADR-025

**Decision:** `DataModule` binds the preprocessing pipeline through a module-level `_PreprocessingTransform` class (`__call__` runs `apply_pipeline`) instead of a closure or lambda. Any future code that injects a custom callable into a dataset transform must follow the same rule: top-level class or function, never a closure.

**Reason:** The pipeline was previously bound with a nested closure (`_apply`) and a `lambda` identity. Under the Windows `spawn` start method, `DataLoader` workers (`num_workers > 0`) pickle the dataset's transform to ship it to worker processes — and closures/lambdas are not picklable (`AttributeError: Can't get local object`). Training therefore crashed the moment a preprocessing pipeline was configured, but only on datasets large enough to keep workers enabled (the `< 500 images` auto-downgrade to `num_workers=0` masked it on small sets). A top-level class instance pickles cleanly, so the same transform now survives the worker hand-off on every platform. Covered by regression tests in `tests/core/test_data.py` (`test_preprocessing_transform_is_picklable`, `test_full_preprocessing_pipeline_is_picklable`).

---

## ADR-031 — Grid search edits values inline on the hyperparameter fields

**Date:** 2026-05-31  
**Status:** Accepted  
**Supersedes context:** Phase 5.6 GridSearch GUI (dot-path → CSV editor)

**Decision:** When `block = grid_search`, the GUI no longer shows a separate dot-path/CSV panel. Instead each gridable field (number/enum controls under Modelo/Treinamento, except `task`/`num_classes`/`seed`) keeps its normal control — value #1 of the axis — and gains a `+ valor ao grid` button that appends values #2, #3… Each extra value reuses the field's control type and is validated inline against the field's schema plus known rules (`learning_rate > 0`, integer fields, `batch_size` power-of-two, enum membership). Removing extras down to a single value drops the field from the search space. The axis state is shared via a `GridContext`; `SchemaFieldVF` reads it and renders the affordance, deriving the dot-path from the field's `path`. The submitted shape is unchanged (`grid_search.hyperparameters: {dot.path: [values]}`), so the backend (`validate_dot_keys` + Cartesian product) needs no change. Pure logic lives in `frontend/src/lib/grid-axis.ts` for isolated testing.

**Reason:** The dot-path/CSV editor required the researcher to know the internal config path of every hyperparameter and hand-type comma-separated values with no type guidance — easy to typo a path or enter an invalid `batch_size`. Editing the values directly on the fields the user already understands ("learning rate, and also try these other two") matches the mental model, and inline validation enforces each parameter's int/float/enum/power-of-two rules at entry time instead of failing per-trial deep in the sweep.

---

## ADR-032 — Grid/random search stream live progress via trial-scoped SSE events

**Date:** 2026-06-01  
**Status:** Accepted  
**Extends:** ADR-016 (React + FastAPI same-process), Phase 5.7 (trial-queue banner)

**Decision:** Multi-trial blocks (`GridSearchBlock`, `RandomSearchBlock`) now stream live training progress to the GUI through the same `/api/experiment/events` SSE channel that `ClassificationBlock` uses. The sweep emits a `trial_start` banner before each inner training and a single terminal `end` after the whole sweep. Each inner trial's `Trainer` events are routed through `make_trial_progress_wrapper` (`blocks/_search_utils.py`), which (a) drops the inner `start` (the sweep emits its own `trial_start`), (b) annotates `epoch_end` with `trial_index` / `total_trials`, and (c) rewrites the inner `end` to `trial_end`. `run_trial` injects the wrapper onto the inner `ClassificationBlock._progress_callback`; `routes._execute_experiment` wires `_put_event` onto any `ClassificationBlock | GridSearchBlock | RandomSearchBlock`. The frontend `TrainingOverlay` derives sweep-wide progress as `(completed_trials + active_epoch_fraction) / total_trials` and prints a `── trial k/N ──` separator plus `[tk/N]` epoch tags.

**Reason:** Live logs only appeared for plain classification — under grid/random search the monitor sat on the synthetic crawl with no epoch output, because `run_trial` created an inner `ClassificationBlock` without a progress callback and `routes` only wired the callback for a top-level `ClassificationBlock`. Simply forwarding the inner events was not enough: the client closes the `EventSource` on a bare `end`, and every trial's `Trainer` emits one — so the stream would have died after the first trial. Rewriting inner `end` → `trial_end` and emitting exactly one terminal `end` keeps the stream alive for the full sweep while giving the overlay the trial context it needs to show meaningful progress. Covered by `tests/blocks/test_search_utils.py::TestTrialProgressWrapper`, `tests/blocks/test_grid_search.py::TestProgressStreaming`, and the `has_callback` assertion in the routes grid-search dispatch test.

---

## ADR-033 — Object detection is a standalone config/block/run path, not an ExperimentConfig block

**Date:** 2026-06-01  
**Status:** Accepted  
**Extends:** ADR-002 (Pydantic config), ADR-003 (ExperimentBlock), ADR-006 (task abstraction)

**Decision:** The detection task uses its own `DetectionConfig` tree (`utils/detection_config.py`) and a standalone `DetectionBlock` (`blocks/detection.py`) that mirrors the `setup`/`run`/`report` contract but is **not** an `ExperimentBlock` subclass. The classification `ExperimentBlock` ABC is typed `setup(self, config: ExperimentConfig)`; detection's fields diverge (no power-of-two batch, mAP not accuracy, boxes not ImageFolder, Ultralytics-style hyperparameters), so subclassing would be a Liskov/type violation and would force a union config on every existing block. Detection is therefore dispatched directly through a dedicated detection run path (its own API endpoints), not the `ExperimentConfig.block` registry dispatch. `DetectionConfig` reuses `OutputConfig` and `DeviceConfig` so output layout and device selection stay identical across tasks, and `DetectionTrainer` writes the same `run.json` contract (ADR-013) so detection runs surface in `/api/runs`.

**Reason:** ADR-006 anticipated one config/block/tab per task precisely so a new task slots in as new modules without touching existing ones. Bending the classification ABC to also carry detection would couple the two task families and bloat `ExperimentConfig` with mutually-exclusive fields. A parallel, self-contained detection path keeps each task independently testable and the classification surface untouched, at the cost of a little duplication (a second `load_*` and a direct dispatch) — a deliberate trade favouring isolation over premature unification. The backlog note ("migrate `utils/config.py` → `configs/schemas/` when a second task is added") is the eventual home for a shared base; that refactor is deferred until segmentation/anomaly make the shared shape obvious.

---

## ADR-034 — Ultralytics owns the detection training loop

**Date:** 2026-06-01  
**Status:** Accepted  
**Extends:** ADR-005 (PyTorch is user-managed), ADR-033

**Decision:** Detection training is delegated to Ultralytics (`YOLO(...).train(...)`) rather than reimplemented as a hand-written epoch loop like the classification `Trainer`. `DetectionTrainer` (`core/detection_trainer.py`) translates `DetectionConfig` into Ultralytics arguments, registers an `on_fit_epoch_end` callback to stream `start`/`epoch_end`/`end` events over the existing SSE shape (ADR-032), and writes `run.json`. `ultralytics` is an optional extra (`[detection]`) bound lazily — the module imports without it, and unit tests patch the module-level `YOLO`, so CI runs on CPU without installing ultralytics or downloading weights. The hybrid plan keeps a torchvision seam for Faster R-CNN/SSD/RetinaNet (`backend="torchvision"`), which currently raises `NotImplementedError`.

**Reason:** Ultralytics provides a maintained, batteries-included training/val/export loop with mAP metrics, augmentation, and checkpointing that would be costly and error-prone to reproduce. Wrapping it (instead of vendoring its internals) keeps VisionForge thin and lets users track upstream YOLO/RT-DETR releases by bumping one dependency. The lazy bind + mocked tests preserve the project rule that CI is CPU-only and never depends on large model downloads (ADR-010).

---

## ADR-035 — torchvision detection: hand-written loss loop, val-loss selection, mAP deferred

**Date:** 2026-06-01  
**Status:** Accepted  
**Extends:** ADR-033, ADR-034

**Decision:** The torchvision detection backend (`backend="torchvision"`) is built incrementally: a `build_torchvision_detector` factory (Faster R-CNN family first; SSD/RetinaNet raise `NotImplementedError`), a `DetectionDataset` adapting YOLO labels to torchvision targets, and a hand-written training loop in `DetectionTrainer._fit_torchvision` (`model(images, targets)` → loss dict → backward). The per-epoch **validation loss** is the selection/early-stop metric and the streamed/`run.json` value (`box_loss`); **mAP is deferred** to a dedicated follow-up module. Validation loss is computed with a no-grad train-mode forward, which is safe because torchvision detectors use **frozen** BatchNorm by default (no running-stat pollution). The torchvision path requires `data.base_dir` (a YOLO layout); `data_yaml`-only is rejected with a clear error.

**Reason:** A correct mAP implementation needs either `pycocotools`/`torchmetrics` (a heavy, Windows-fragile dependency surface, against ADR-005/ADR-010's lean-CI stance) or a carefully-tested hand-rolled metric — too much to bundle into the same change as the loop without risking correctness. Shipping a working, fully-tested loss-based loop first (mocked-model unit tests, no weight downloads) delivers a usable torchvision trainer now and isolates mAP as its own verifiable module. Validation loss is a legitimate selection signal for detection and mirrors the classification `Trainer`'s best-checkpoint contract, so `run.json`/history stay uniform across backends.

**Update (2026-06-01):** the deferred mAP module landed — `core/detection_metrics.py` (`mean_average_precision_50`, VOC all-points AP via `torchvision.ops.box_iou`, no extra deps). The torchvision loop now selects the best checkpoint by **validation mAP@50** and records `map50` per epoch in `run.json`; validation loss is kept as a secondary logged signal. SSD/RetinaNet head replacement remains the open follow-up.

---

## ADR-036 — Image regression is a standalone config/block/run path

**Date:** 2026-06-03  
**Status:** Accepted  
**Extends:** ADR-002 (Pydantic config), ADR-003 (ExperimentBlock), ADR-006 (task abstraction), ADR-033 (detection precedent)

**Decision:** The image-regression task uses its own `RegressionConfig` tree (`utils/regression_config.py`) and a standalone `RegressionBlock` (`blocks/regression.py`) that mirrors the `setup`/`run`/`report` contract but is **not** an `ExperimentBlock` subclass — the same standalone shape ratified for detection in ADR-033. Regression's fields diverge from classification: continuous targets supplied by a **CSV manifest** (`image,target[,…]`) instead of an ImageFolder class index, a linear head emitting `num_targets` raw outputs instead of a softmax classifier, and MSE/RMSE/MAE/R² instead of accuracy/F1. Forcing these through `ExperimentConfig`/the classification ABC would bloat the shared config with mutually-exclusive fields (`num_classes` vs `num_targets`, `target_columns`, regression `loss`) and break the ABC's `setup(self, config: ExperimentConfig)` type contract. Regression is therefore dispatched through a dedicated `/api/regression/*` run path. It **reuses** `OutputConfig`, `DeviceConfig`, `TransformConfig`, `PreprocessingConfig`, `SchedulerConfig` (config), `resolve_device`/`_seed_everything` (trainer), the CNN backbones + head-swap helpers (`models/factory.build_backbone`/`replace_final_layer`), and `MetricsPlotter.loss_curve`, and writes the same ADR-013 `run.json` so regression runs surface in `/api/runs`.

**Reason:** ADR-006 anticipated one config/block/tab per task so a new task slots in as new modules without touching existing ones, and ADR-033 already chose isolation over unification for detection. Regression is closer to classification than detection (same backbones, same hand-written epoch loop), so it reuses far more — but the *target* still diverges enough that a shared config would couple the families. Extracting the genuinely shared pieces (backbone builders, head swap, device/seed helpers, loss-curve plot) keeps duplication low while the standalone tree keeps each task independently testable and the classification surface untouched. The eventual shared base (`configs/schemas/`, per the backlog note) is still deferred until segmentation/anomaly make the common shape obvious.

---

## ADR-037 — Semantic segmentation is a standalone config/block/run path

**Date:** 2026-06-04  
**Status:** Accepted  
**Extends:** ADR-002 (Pydantic config), ADR-003 (ExperimentBlock), ADR-006 (task abstraction), ADR-033 (detection precedent), ADR-036 (regression precedent)

**Decision:** The semantic-segmentation task uses its own `SegmentationConfig` tree (`utils/segmentation_config.py`) and a standalone `SegmentationBlock` (`blocks/segmentation.py`) that mirrors the `setup`/`run`/`report` contract but is **not** an `ExperimentBlock` subclass — the same standalone shape ratified for detection (ADR-033) and regression (ADR-036). Segmentation's fields diverge from classification: the label is a **per-pixel class mask** (a paired mask image keyed by filename stem) rather than an ImageFolder class index, the model emits a **dense `num_classes`-channel map at input resolution** (torchvision DeepLabV3/FCN/LR-ASPP, or a hand-rolled U-Net) instead of a pooled classifier, the loss is **pixel-wise cross-entropy / Dice / combined** (all `ignore_index`-aware), and the metrics are **mean IoU / Dice / pixel accuracy** instead of accuracy/F1/AUC. The best checkpoint is chosen by **val mean IoU**, not val loss. Forcing these through `ExperimentConfig`/the classification ABC would bloat the shared config and break the ABC's `setup(self, config: ExperimentConfig)` type contract. Segmentation is therefore dispatched through a dedicated `/api/segmentation/*` run path. It **reuses** `OutputConfig`, `DeviceConfig`, `TransformConfig`, `PreprocessingConfig`, `SchedulerConfig` (config), `resolve_device`/`_seed_everything` (trainer), `load_local_weights` (factory), `MetricsPlotter.loss_curve`, and the `_PreprocessingTransform` image pipeline, and writes the same ADR-013 `run.json` so segmentation runs surface in `/api/runs`.

**Reason:** Same isolation-over-unification rationale as ADR-033/036. Segmentation reuses the image pipeline and device/seed/plot helpers, but the dense per-pixel target diverges enough (joint image+mask geometric transforms, a confusion-matrix metric accumulator, mIoU-based selection) that a shared config would couple the families. Two divergences are segmentation-specific and worth recording: (1) image and mask must transform **in lockstep** — the loader applies geometric ops jointly in `__getitem__` (image bilinear+normalized, mask nearest-neighbor as `long`, mask rotation filled with `ignore_index`) rather than reusing the image-only `_build_transforms` `T.Compose`; (2) model outputs are **normalized via `segmentation_logits`** so the torchvision `OrderedDict({"out": ...})` and the U-Net plain tensor are interchangeable to the trainer. RGB/palette-color masks (needing a colour→id map) are deferred — v1 assumes integer-id single-channel masks.

---

## ADR-038 — Anomaly detection is a standalone config/block/run path

**Date:** 2026-06-04  
**Status:** Accepted  
**Extends:** ADR-002 (Pydantic config), ADR-003 (ExperimentBlock), ADR-006 (task abstraction), ADR-033/036/037 (standalone-task precedents)

**Decision:** The anomaly-detection task uses its own `AnomalyConfig` tree (`utils/anomaly_config.py`) and a standalone `AnomalyBlock` (`blocks/anomaly.py`) that mirrors the `setup`/`run`/`report` contract but is **not** an `ExperimentBlock` subclass — the same standalone shape as ADR-033/036/037. Anomaly is the first **unsupervised** task and diverges hardest from classification: training sees **normal images only** (no second class, so cross-entropy is undefined), the "model" is either a reconstruction **autoencoder** (score = reconstruction error) or a **PatchCore memory bank** of normal patch features (score = nearest-neighbor distance) — neither has a classifier head — and the metric is **image-level AUROC** over a labelled test split plus a decision **threshold** taken from a percentile of the normal-score distribution. Anomaly is dispatched through a dedicated `/api/anomaly/*` run path. It **reuses** `OutputConfig`, `DeviceConfig`, `TransformConfig`, `PreprocessingConfig`, `SchedulerConfig` (config), `resolve_device`/`_seed_everything` (trainer), the `_build_transforms` image pipeline, and writes the same ADR-013 `run.json` so anomaly runs surface in `/api/runs`.

**Reason:** Same isolation-over-unification rationale as the prior standalone tasks. Two anomaly-specific divergences are worth recording: (1) the trainer **dispatches on model type** behind one `fit()` — the autoencoder runs a real gradient loop (best checkpoint by lowest train reconstruction loss) while PatchCore has **no gradient training** (one pass builds the coreset memory bank, `total_epochs=1`); both then score the test set the same way. (2) Anomaly has **no validation split** — train is normal-only and the labelled test split is the eval set; model selection uses train reconstruction loss (autoencoder) and the reported AUROC is threshold-free, with the percentile threshold driving only the reported image-level F1 and a usable operating point. Pixel-level localization (`ground_truth/` masks, pixel AUROC/PRO) and a faiss-backed nearest-neighbor are deferred; v1 is image-level AUROC with exact torch `cdist`. A small reusable `MetricsPlotter.metric_curve` was added (generic single-series curve) for the AUROC-over-epochs plot, since `loss_curve` assumes a `val_loss` anomaly does not produce.

---

## ADR-039 — Config carries a `schema_version` migrated forward on load

**Date:** 2026-06-04  
**Status:** Accepted  
**Extends:** ADR-002 (Pydantic config), ADR-013 (`run.json` contract)

**Decision:** `ExperimentConfig` carries an integer `schema_version` (default `CURRENT_SCHEMA_VERSION = 1`). `load_config` runs `migrate_config_dict(raw)` before validation: a config without an explicit `schema_version` is treated as v1 (every legacy YAML and `run.json` predates the field), and future breaking schema changes add a migration step there that rewrites older shapes forward. A config whose `schema_version` is **newer** than this build supports is rejected at validation with a clear "written by a newer version — please upgrade" error rather than silently mis-parsing. The field is injected into `run.json` automatically (it is a normal config field) and hidden from the GUI form (`SKIP_FIELDS`) since it is infra metadata, not a user knob.

**Reason:** Reproducibility is the project's core value — *"which exact config produced this result?"* (CLAUDE.md §6.2/§7.3). Once real experiments accumulate, a breaking change to a required config field would silently invalidate or fail to load saved configs with no traceability. Establishing the version field + a migration hook **now, while the schema is still small** ("congele cedo"), means later schema changes migrate old configs forward deterministically instead of breaking them, and the newer-version guard prevents a stale build from silently misreading a config from a future release. The migration is centralized in one pure, tested function so each future bump is a single, isolated, verifiable step.

**Update (2026-06-04, integ):** all standalone task configs (detection/regression/segmentation/anomaly) now carry the same `schema_version` field, run `migrate_config_dict` in their loaders, and reject a future version via the shared `check_schema_version` helper — so every task's saved YAML/`run.json` is versioned and forward-migratable, not just classification. Done on `integ/all-features`; covered by `tests/utils/test_task_config_schema_version.py` (12 parametrized cases across the four tasks).

---

## ADR-040 — Full Ultralytics hyperparameter surface + complete YOLO family

**Date:** 2026-06-06
**Status:** Accepted
**Extends:** ADR-033 (standalone detection path), ADR-034 (Ultralytics owns the loop)

**Decision:** `DetectionTrainingConfig` is expanded from the original 6 knobs
(epochs, batch_size, learning_rate, patience, seed, workers) to the full set of
documented Ultralytics `train` hyperparameters, grouped as: optimizer
(`optimizer`, `momentum`, `weight_decay`), LR schedule (`lrf`, `cos_lr`,
`warmup_epochs`, `warmup_momentum`, `warmup_bias_lr`), loss gains (`box`, `cls`,
`dfl`), and regularization/mechanics (`label_smoothing`, `dropout`, `nbs`,
`freeze`, `amp`, `close_mosaic`, `single_cls`, `rect`, `multi_scale`). A new
nested `DetectionAugmentationConfig` carries every Ultralytics augmentation knob
(`hsv_h/s/v`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`,
`fliplr`, `bgr`, `mosaic`, `mixup`, `copy_paste`, `auto_augment`, `erasing`).
Every field's default equals Ultralytics' own default, so an unmodified config
is **behaviour-preserving** — the richer surface adds knobs, it does not change
results. The trainer translates the whole tree into the `YOLO.train(...)` call via
a single `_ultralytics_train_kwargs()`; the `optimizer`/`momentum`/`weight_decay`
trio is also honoured by the torchvision backend (`_build_torchvision_optimizer`),
which previously hard-coded SGD(momentum=0.9, weight_decay=5e-4).

The Ultralytics model set grows from {YOLOv8, YOLO11, RT-DETR} to **every
detection family Ultralytics ships**: YOLOv8 (n/s/m/l/x), YOLOv9 (t/s/m/c/e),
YOLOv10 (n/s/m/b/l/x), YOLO11 (n/s/m/l/x), YOLO12 (n/s/m/l/x), YOLO26
(n/s/m/l/x), and RT-DETR (l/x). Variant letters follow each family's own
convention (YOLOv9 uses t/c/e; YOLOv10 adds a `b`). The GUI lists them
newest-first for discoverability, but the default model stays `yolo11n` — an
explicit (non-positional) default in both `DetectionModelConfig` and
`detection-models.ts` so list order never silently changes the default.

**Reason:** The detection task shipped a deliberately minimal training surface in
Phase 7 to stay shippable, but serious detection research needs the optimizer,
schedule, loss-balancing, and augmentation knobs Ultralytics exposes — tuning
`box/cls/dfl` gains, mosaic/mixup/copy-paste, and the optimizer is routine, and
their absence forced users back to the CLI, defeating the GUI's purpose. Keeping
Ultralytics' own defaults means the expansion is purely additive: existing runs
reproduce exactly. Validating the whole model set in Pydantic (not just at
download time) gives a fast, clear rejection before any weights are fetched. New
families (notably YOLO26, which is NMS-free and DFL-free) train through the same
`YOLO(name).train(...)` entry point, so adding them is a names-list change plus
config/GUI coverage — no new code path. `freeze` is exposed as an int (0 = none)
in the GUI for a clean numeric binding while the config keeps `int | None`.

---

## ADR-041 — Cross-task strategy parity via a generic task-run handle

**Date:** 2026-06
**Status:** Accepted — slice 1 shipped: `TaskRunner` protocol + `RunResult` (`core/task_runner.py`), `ClassificationRunner` adapter (`blocks/classification_runner.py`), `ModelComparisonBlock` refactored to consume the handle; remaining slices (regression/segmentation comparison, batch predict, generic sweep) stay planned.
**Extends:** ADR-003 (ExperimentBlock), ADR-006 (task abstraction), ADR-033/036/037/038 (standalone tasks)

**Decision:** The orchestration-style strategies that today exist only for
classification — **model comparison, batch prediction, and grid/random
hyperparameter search** — will be extended to the standalone tasks through a
single task-agnostic `TaskRunner` handle, not by copying each block per task and
not by folding tasks into `ExperimentConfig`. Each task exposes a thin handle
(`config_type`, `run`, `metrics`, `primary_metric`, `load_checkpoint`,
`predict`); generic runners (`GenericComparisonRunner`, `GenericSweepRunner`,
`GenericBatchPredictor`) consume the handle and reuse the existing trial-progress
SSE plumbing (ADR-032). Strategies that touch task internals — transfer-learning
freeze/fine-tune, K-Fold cross-validation, ONNX export, Grad-CAM — stay
**per-task** modules, extended only where they make sense (see the verdict matrix
in `CROSS_TASK_PARITY_PLAN.md`). Detection sweeps defer to Ultralytics' own
tuner; anomaly/detection K-Fold are deferred.

**Reason:** The current blocks (`ModelComparisonBlock`, `BatchPredictionBlock`,
the search blocks) are `ExperimentBlock` subclasses hard-wired to
`ExperimentConfig`, `ClassificationBlock`, and classification metric names
(`accuracy`/`f1`/`auc_roc`) — they cannot be pointed at a `RegressionConfig` or a
segmentation run as written. The honest options are (a) duplicate every
orchestrator per task (4× the surface, the exact bloat ADR-006 avoids) or (b)
extract the genuinely-shared orchestration behind a uniform handle while keeping
task-specific logic in the task. (b) matches the project's standing trade-off —
*isolation over premature unification* (ADR-033) for the parts that diverge, but
*reuse* for the parts that are identical (running a pipeline N times and ranking,
or loading a checkpoint and predicting, are task-independent). Recorded as
**Proposed** because it is a design ratification ahead of implementation; the
slice order and open questions live in `CROSS_TASK_PARITY_PLAN.md` and each
behavior-changing slice gets its own follow-up ADR.

---

## ADR-042 — Distribution: Docker image + `visionforge doctor`, not orchestration

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-005 (torch is user-managed), ADR-010 (CPU-only CI)

**Decision:** VisionForge will ship two complementary distribution aids, both
inside the local-only philosophy: (1) a **`visionforge doctor`** CLI subcommand
that detects the host GPU/driver via `nvidia-smi`, maps it to the correct torch
wheel index (`cu118`/`cu121`/`cu124`/`cu126`/`cpu` — the set already in
`pyproject.toml`), and prints (optionally runs, with confirmation) the exact
install command; and (2) a **GPU-enabled Docker image** (multi-stage: build the
SPA, then a CUDA-runtime image with Python 3.13 + the matching torch wheel baked
in) plus a `docker-compose.yml`, run with `--gpus all`, mounting datasets
read-only and `outputs/` read-write. **Kubernetes and any cloud orchestration are
explicitly rejected.** Design detail in `DOCKER_PLAN.md`.

**Reason:** The torch install is the largest onboarding friction (ADR-005 makes
it the user's job because no resolver can pick the CUDA build; picking wrong
silently degrades to CPU). `doctor` removes the guesswork for bare-metal users;
the image removes the install entirely and makes runs reproducible across
machines — directly serving the "facilitate requirements" goal without adding a
cloud dependency to any core path. Kubernetes is rejected on first principles:
for a single local GPU it is complexity with no payoff (problem-driven, not
resume-driven); if single-machine job queueing is ever needed, a lightweight
queue (RQ/SQLite) is the right tool. The NVIDIA driver +
`nvidia-container-toolkit` remain the user's responsibility — an image cannot
ship host kernel components. Recorded as **Proposed**; `doctor` ships first as a
self-contained, CPU-CI-testable slice (mock `nvidia-smi`), the image second.

Slice 1 (`visionforge doctor` command + full test coverage) shipped 2026-06; the Docker image (slice 2) remains planned.

## ADR-043 — Detection runs surface the full YOLO metric set and a per-epoch history plot

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-013, ADR-032, ADR-040

**Decision:** A detection run captures and persists the full per-epoch metric
set Ultralytics already computes — precision, recall, mAP@50, mAP@50-95, and the
box/cls/dfl loss components for both train and val — in the `epoch_end` SSE event
and in each `run.json` history row (additive to the ADR-013 contract; existing
fields keep their meaning). The live overlay renders these detection-native
metrics instead of the classification loss/accuracy triple. The torchvision
backend, which has no Ultralytics `results.png`, synthesizes the equivalent
loss + mAP@50 history chart via `MetricsPlotter.detection_results`, so every
detection run — both backends — shows charts in the history view.

**Reason:** Detection previously streamed only `box_loss` mirrored onto the
classification fields (`train_loss`/`val_loss`/`val_accuracy`), so the live log
and history showed a misleading classification-shaped view and torchvision runs
showed no plots at all. mAP and the per-component losses are the metrics a
detection practitioner actually reads; dropping them while the trainer already
has them in hand is pure loss of signal. Capturing them is free (they sit on
`trainer.metrics`), and surfacing them closes the "detection runs look empty in
history" gap without a new dependency or a network round-trip.

## ADR-044 — Generic model comparison over the TaskRunner handle

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-041

**Decision:** Model comparison (train N architectures on one dataset, rank them)
is a task-agnostic function — `core/comparison.run_model_comparison(runner,
base_config_dict, model_names, metric)` — that drives any `TaskRunner` and ranks
by the metric the runner declares, never by hard-coded classification names. Each
task exposes a thin adapter next to its block (`blocks/<task>_runner.py`)
carrying `config_type`, `run`, `metrics`, and `primary_metric`; regression
(`r2`) and segmentation (`miou`) ship first. The comparison overrides only
`model.name` per trial — the one field every task config shares — so it needs no
knowledge of task internals. The existing classification `ModelComparisonBlock`
is left as-is for now (it is GUI-wired with classification-shaped report columns);
unifying it onto the generic path is a later, behaviour-neutral cleanup.

**Reason:** Slice 1 (ADR-041) added the handle but the comparison orchestrator
still hard-coded `accuracy`/`f1`/`auc_roc` and instantiated `ClassificationBlock`,
so it couldn't rank a regression or segmentation sweep. Pulling the
*run-N-and-rank* logic into a function over the handle is the cheapest way to
give the standalone tasks comparison without folding them into `ExperimentConfig`
(which ADR-033 forbids) — task-specific metric names stay in each adapter, the
shared orchestration is written once. `r2`/`miou` are the natural ranking
defaults (higher-is-better, like accuracy) so the descending sort is uniform.
Backend-first per the Phase-5 norm; the GUI surface and the generic sweep/
batch-predict are the next ADR-041 slices.

## ADR-045 — Generic hyperparameter sweep over the TaskRunner handle

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-041, ADR-044

**Decision:** Grid and random hyperparameter search are a task-agnostic function
— `core/sweep.run_sweep(runner, base_config_dict, search_space, *, mode, metric,
n_trials, seed)` — that drives any `TaskRunner` and ranks trials by the runner's
declared metric. Overrides are applied by dot-path on a validated-then-dumped
base config; the search-space format matches classification's existing search
(grid: `{path: [values]}`; random: `{path: {"type": "uniform"|"log_uniform"|
"choice", ...}}`). Surfaced as `POST /api/{regression,segmentation}/sweep`
(background run over the shared single-run state; ranked report via
`/experiment/result`), with sweep paths validated up front (422 on an unknown
path). The classification grid/random blocks are left as-is.

**Reason:** Sweeping hyperparameters is the headline "more training methods"
gap for the standalone tasks, and it is the same *run-N-and-rank* shape as
comparison (ADR-044) — so it reuses the handle rather than re-coupling to a task.
Keeping the search-space grammar identical to classification's means one mental
model and a future unification path. Validating the base config and every
dot-path before running spends no GPU time on a typo. Backend-first per the
Phase-5 norm; the GUI lands with the deferred comparison panel on the new design.

## ADR-046 — Transfer-learning knobs as a per-task config field (regression)

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-036, ADR-041

**Decision:** Transfer learning for the standalone tasks is an **optional config
field on the task**, not a generic block. `RegressionConfig` gains
`transfer_learning: RegressionTransferLearningConfig | None` with `mode`
(`feature_extraction` | `fine_tuning`) and `backbone_lr_multiplier`. When unset
(the default) training is unchanged — full network, single LR.
`feature_extraction` freezes every child except the head (the model's last named
child) and optimizes only the head; `fine_tuning` trains everything but puts the
backbone in its own param group at `learning_rate × backbone_lr_multiplier`. The
freeze + param-group construction lives in `RegressionTrainer`
(`_apply_transfer_learning` / `_build_optimizer`, DataParallel-unwrapped). GUI:
a "Transfer learning" segmented control in `RegressionPanel`; the payload maps
"none" → `null`. Classification keeps its separate `TransferLearningBlock`.

**Reason:** ADR-041's verdict put transfer learning in the *per-task* column,
not the generic-runner column: freezing touches task-specific model internals
(which child is the head) and a shared CNN backbone, so it does not generalize as
cleanly as comparison/sweep/batch-predict. A nullable config field keeps it fully
behavior-preserving when absent (no migration, no GPU-time change for existing
runs) while exposing the two workflows researchers actually use on small image
sets — frozen-backbone feature extraction and discriminative fine-tuning. Reusing
the "last named child = head" convention from classification's
`TransferLearningBlock` keeps one mental model without coupling the standalone
task back to `ExperimentConfig`. Segmentation gets the same field next.

## ADR-047 — Transfer-learning knobs for segmentation (mirrors ADR-046)

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-037, ADR-046

**Decision:** Segmentation gets the same per-task transfer-learning field as
regression: optional `SegmentationConfig.transfer_learning`
(`SegmentationTransferLearningConfig`: `mode` ∈ {`feature_extraction`,
`fine_tuning`} + `backbone_lr_multiplier`). `feature_extraction` freezes every
child except the dense head and optimizes only the head; `fine_tuning` puts the
backbone in its own param group at `learning_rate × backbone_lr_multiplier`. The
freeze + split logic lives in `SegmentationTrainer`
(`_apply_transfer_learning`/`_split_named_params`/`_build_optimizer`), identical
to `RegressionTrainer`. The "head = last named child" convention was verified to
hold for every family: `classifier` for DeepLabV3 / FCN / LR-ASPP (with
`aux_loss=False`, no `aux_classifier` child) and `outc` for U-Net. GUI: a
"Transfer learning" segmented control in `SegmentationPanel`; payload maps
"none" → `null`.

**Reason:** Same rationale as ADR-046 — transfer learning is a per-task knob over
a shared/pretrained backbone, behavior-preserving when unset. Verifying the
head-is-last-child invariant for each segmentation family up front de-risks the
freeze split (a wrong head would silently train the backbone and freeze the head).
The knob is most useful for the torchvision families (ImageNet-pretrained
backbones); U-Net has no pretrained weights, so feature extraction there only
trains the final conv — allowed but documented as not recommended, mirroring
regression's "pretrained=False is the user's call" stance rather than adding a
special-case guard.

## ADR-048 — User-supplied custom models via a drop-in registry

**Date:** 2026-06
**Status:** Accepted

**Decision:** Researchers can register their own architectures without editing
VisionForge source. A `.py` file dropped under `user_models/` calls
`@register_model("name")` (from `visionforge.models.registry`) on a builder
`(num_outputs) -> nn.Module`; `ModelFactory.create` builds it when a config sets
`model.custom_model: "name"` (a new optional field on the classification
`ModelConfig`). The registry (`models/registry.py`, same layer as the factory) is
a module-global dict; `load_user_models(dir)` imports every non-underscore `.py`
in the directory (default `./user_models`) to trigger registration, called lazily
by `build_custom_model` on a name miss. When `custom_model` is set the builtin
`name`/`pretrained` are ignored; `weights_path` still loads a local checkpoint
(non-strict). A blank `custom_model` ("" from an untouched GUI text field) coerces
to `None`, so the builtin backbone path is the default. Initially classification
only (the `ModelFactory` path); extended to regression and segmentation in ADR-049.

**Reason:** "Bring your own model" is a recurring research need that the fixed
`Literal` backbone list can't serve, and forking the factory per experiment
doesn't scale. A drop-in directory + decorator is the lightest extension point:
purely additive (absent `user_models/` ⇒ no behavior change, no migration), needs
no plugin manifest, and keeps user code out of the package tree. Executing local
Python is acceptable because VisionForge is local-first and offline (ADR-005) — the
trust boundary is the user's own machine, so no sandboxing is imposed; this is
documented in `user_models/README.md`. The field is exposed through the existing
schema-driven GUI form automatically (a `str | None` renders as a text input), so
no bespoke frontend was required. Bad user files are logged and skipped rather
than aborting discovery, so one broken drop-in doesn't break the others.

## ADR-049 — Custom models for regression and segmentation

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-048

**Decision:** The custom-model registry (ADR-048) now serves the regression and
segmentation tasks too. `RegressionModelConfig` and `SegmentationModelConfig` each
gain the same optional `custom_model` field (blank → `None`), and
`RegressionModelFactory`/`SegmentationModelFactory` route to `build_custom_model`
when it is set — passing the task's output dimension (`num_targets` for regression,
`num_classes` for segmentation). To make that contract task-neutral, the registry's
builder argument was renamed `num_classes` → `num_outputs`: a builder receives one
int (the output dimension) and returns an `nn.Module`, so a single custom model can
serve any CNN-headed task. A segmentation custom model is expected to emit per-pixel
logits already (it bypasses the torchvision head-swap); the classification head-swap
contract is unchanged.

**Reason:** The registry was built generic in ADR-048; extending it to the other
CNN-headed tasks is a few lines per factory and avoids three divergent
"bring-your-own-model" mechanisms. Renaming to `num_outputs` keeps the builder
contract honest now that the int means num_targets for regression — a builder
written once works across tasks. Detection (Ultralytics owns the model) and anomaly
(autoencoder/PatchCore, no swappable head) keep their own paths and ignore
`custom_model`. Purely additive and behaviour-preserving when unset, like ADR-048.

## ADR-050 — K-fold cross-validation for regression (backend)

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-036, ADR-041

**Decision:** Regression gets K-fold cross-validation via
`blocks/regression_cv.run_regression_cross_validation(config, *, n_folds, shuffle,
seed)`. It mirrors the classification `CrossValidationBlock` shape but over the
CSV-manifest dataset: the pooled training rows are split with sklearn `KFold`; each
fold builds a fresh model + `RegressionTrainer`, trains on K-1 parts, reloads the
best checkpoint and scores the held-out part via `RegressionTrainer.evaluate`; the
per-fold MSE/RMSE/MAE/R² are aggregated to mean ± std (`CrossValidationReport`).
No dataset refactor: two `RegressionCsvDataset` instances (augmented train +
clean eval transforms) are sliced per fold with `torch.utils.data.Subset`. The
fold train loader sets `drop_last=True` when the fold has more than one batch, so a
size-1 trailing batch can't break BatchNorm (CV folds are small). CV params are
invocation arguments (like comparison/sweep), not a config field. Backend-first;
the API endpoint + GUI card + report renderer land in a follow-up slice, and
segmentation CV follows the same shape.

**Reason:** CV is the last cross-task-parity feature researchers expect, and it is
genuinely useful on the small datasets regression-on-images usually involves. It
does not fit the generic `TaskRunner` handle (comparison/sweep just vary config;
CV varies the *data split*), so it is a per-task orchestrator rather than a generic
runner — consistent with ADR-041's verdict that data-level features stay per-task.
Reusing `Subset` over two transform variants avoids touching the dataset class and
keeps the val fold un-augmented. Passing `n_folds/shuffle/seed` as call arguments
(not a new config field) matches the comparison/sweep precedent and needs no config
migration. The `drop_last` guard is standard BatchNorm hygiene that CV makes
necessary because folds shrink the training set.

## ADR-051 — timm as an optional model source

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-005, ADR-048/049

**Decision:** Researchers can use any `timm` architecture via a `timm_model` field
on `ModelConfig` and `RegressionModelConfig` (classification + regression — timm
provides backbones with a linear head, not segmentation heads). When set, the
factory builds the model through `models/timm_source.build_timm_model(name,
num_outputs, pretrained)` — a lazy wrapper over `timm.create_model(name,
pretrained=, num_classes=num_outputs)` — and ignores the builtin `name`;
`weights_path` still loads a local checkpoint. `timm_model` and `custom_model` are
mutually exclusive (a `model_validator` rejects both; blank strings coerce to
`None`). `timm` is a new optional extra (`pip install -e ".[timm]"`); the import is
lazy so VisionForge runs without it. The output-dimension contract is the same
`num_outputs` the custom registry uses, so all three sources (builtin / custom /
timm) are interchangeable.

**Reason:** The fixed `Literal` backbone list is small; timm is the de-facto
hub of hundreds of pretrained vision models and `create_model` already sizes the
head, so wiring it in is a few lines per factory with no new head logic. Keeping it
an optional, lazily-imported extra honours ADR-005 (heavy/optional deps stay out of
the core install, like ultralytics) and keeps the test suite offline (tests mock
`create_model`, ADR-010). Mutual exclusivity with `custom_model` avoids ambiguous
precedence. Segmentation is out of scope — timm yields classifiers/feature
extractors, not dense decoders.

## ADR-052 — Optuna as a third sweep mode

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-045

**Decision:** `core/sweep.run_sweep` gains a third `mode="optuna"` alongside grid
and random. It reuses the random search-space grammar
(`{path: {type: uniform|log_uniform|choice, ...}}`) but drives an Optuna TPE study:
each trial is suggested adaptively from prior results (`suggest_float` /
`suggest_float(log=True)` / `suggest_categorical`), run through the same shared
`_execute_trial` helper as the other modes, and recorded as a `SweepTrial`. The
study direction is `maximize` (the ranking metrics are higher-is-better, matching
the existing descending sort); failed trials are recorded then `TrialPruned` so the
study continues. Optuna is a new optional extra (`[optuna]`), imported lazily.
Exposed through the existing sweep API/GUI by adding `"optuna"` to `SweepRequest.mode`
and the `SweepCard` strategy selector (the random-style editor + `n_trials` apply
unchanged). Pruning of in-progress trials is deferred (would need per-epoch
reporting from the trainers).

**Reason:** Random search wastes budget re-sampling bad regions; TPE concentrates
trials where the metric improves, which matters on the small datasets these tasks
target. Folding it into `run_sweep` (rather than a separate block) means all four
standalone tasks get it for free and the search-space grammar/report stay shared —
one mental model. Keeping it an optional lazily-imported extra follows ADR-005, and
tests stay offline by exercising the study with a fake runner (`pytest.importorskip`
guards when the extra is absent). Mid-training pruning is left out of this slice to
avoid changing the trainer/SSE contract.

## ADR-053 — Grad-CAM for regression and segmentation

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR (classification Grad-CAM), ADR-041

**Decision:** Grad-CAM explainability extends from classification to regression and
segmentation. `core.gradcam.GradCAM.__call__` gains an optional `target_fn(output)
-> scalar` that selects what to back-propagate, keeping the classification default
(predicted/`target_class` logit). A new `gui/api/torch_gradcam.build_gradcam(data,
target_index, checkpoint)` rebuilds the run's model (`pretrained=False`, lenient
read so a moved dataset doesn't block) and returns the model + eval transform +
`target_fn` + a `describe` label callable, dispatching on task: classification =
class logit; regression = a continuous output column (`out[:, i]`, saliency);
segmentation = mean logit of a class channel (`out[:, c].mean()`, model wrapped to a
logits tensor). `_execute_run_gradcam` now drives any of the three through one loop;
detection (Ultralytics) and anomaly (no class/conv target) are rejected.
`GradCamItem` gains `prediction: str | None` and `predicted_class` becomes optional
(None for regression); the GUI shows `prediction` when present and the Grad-CAM card
is gated to classification + regression + segmentation.

**Reason:** The CAM machinery (hook the last conv, GAP-weight the gradients,
ReLU-sum) is task-agnostic — only the back-prop target differs — so generalizing is
a small seam rather than three implementations. Routing the per-task model build
through a `torch_gradcam` module mirrors the ONNX-export/batch-predict precedent and
keeps `routes` thin. Regression saliency and per-class segmentation CAMs are the
standard explainability views for those tasks; detection/anomaly genuinely lack a
single conv-logit target, so they stay out. Dependency-free (pure torch), offline
tests with tiny models (ADR-010).

## ADR-054 — TensorBoard experiment tracking (best-effort, opt-in by install)

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-005, ADR-013

**Decision:** Training writes per-epoch scalars to `<run_dir>/tensorboard/` via a
`core/tracking.TensorBoardLogger`. The logger lazily imports
`torch.utils.tensorboard.SummaryWriter`; if the optional `tensorboard` extra is not
installed it is a **no-op**, so there is no config flag — installing
`".[tensorboard]"` enables it, then `tensorboard --logdir outputs/models`. Wired
into the classification, regression and segmentation trainer epoch loops (create
after the run dir, `log_scalars(epoch, …)` each epoch with namespaced tags like
`loss/train`, `accuracy/val`, `r2/val`, `miou/val`, close at the end). TensorBoard
was chosen over MLflow (user decision). Detection (Ultralytics owns its loop) and
anomaly trainers are a follow-up.

**Reason:** TensorBoard is the lightest, most local-first tracker — `SummaryWriter`
ships with torch and writes plain event files under the run dir, needing no server
(MLflow's tracking store is heavier and more opinionated, against the local-first
stance of ADR-005). Best-effort-if-installed keeps it zero-config and avoids adding
a tracking field to all five task configs, while staying fully optional (no new
hard dependency; the run.json history of ADR-013 remains the canonical record).
Tests inject a fake `SummaryWriter` so they neither require the extra nor write real
event files (ADR-010).

## ADR-055 — One-shot dataset download (provider-based)

**Date:** 2026-06
**Status:** Accepted

**Decision:** A user-initiated `POST /api/dataset/download` fetches a dataset into a
local folder, after which the existing data flow takes over — local-first, one-shot,
nothing in the core training path touches the network. It is **provider-based**
(`gui/api/dataset_download.py`): `download_dataset(provider, dataset, out_dir, …)`
dispatches to a per-provider fetcher. The user chose four providers; they ship one
per commit, simplest first:
1. **torchvision built-ins** (this slice) — no extra; downloads CIFAR10/100,
   MNIST/FashionMNIST/KMNIST and **materializes them into an ImageFolder layout**
   (`<out>/<split>/<class>/*.png`) so classification trains on them directly. A
   `limit` caps images per class. The raw download goes to a temp dir; only the PNGs
   are kept.
2. **Roboflow** (`roboflow` extra, lazy, API key) — next.
3. **Kaggle** (`kaggle` extra, lazy, kaggle.json) — next.
4. **Hugging Face** (`datasets` extra, lazy, optional token) — next.
Missing extras/credentials raise a clear error → HTTP 400. The endpoint runs in a
worker thread (downloads are slow). **All four providers + the GUI are now in**: a
`DatasetDownloadCard` (provider selector + per-provider conditional fields →
`/api/dataset/download`) sits below the active task panel; the user picks the local
output folder and the existing data flow takes over from there.

**Reason:** A provider dispatcher keeps each source isolated and lets the heavy/auth
ones be optional lazy extras (ADR-005), so the core install stays lean. Materializing
torchvision sets to ImageFolder is what makes them immediately trainable — the raw
torchvision format is not what the DataModule consumes. Backend-first per provider
(API-reachable + tested via a mocked dataset, no network/credentials, ADR-010) keeps
each commit small and green; the GUI is added once at the end rather than rebuilt per
provider. The downloaded files live wherever the user points `out_dir`; nothing is
auto-committed.

## ADR-056 — Multi-seed replicates with aggregate statistics

**Date:** 2026-07
**Status:** Accepted
**Extends:** ADR-041, ADR-044, ADR-045

**Decision:** A generic replicate runner (`core/replicates.py`) trains the *same*
config N times under different seeds over the `TaskRunner` handle and aggregates
every reported metric into `n / mean / std / min / max / 95% CI` (Student-t via
scipy, already a scikit-learn transitive dependency; normal-approximation
fallback). Exposed as `POST /api/{task}/replicates` for all five tasks
(classification included, via `ClassificationRunner`). Explicit `seeds` win;
otherwise N consecutive seeds derive from the config's own `training.seed` —
the one field every task config shares. Each replicate suffixes `name` with
`_s{seed}` so it keeps its own run dir; trials are reported in seed order,
**never ranked**. The aggregate report persists to `outputs/reports` via the
same summary writer as comparison/sweep (`replicates_summary.json` +
`replicates_ranking.csv`). GUI card (mirroring `SweepCard`) is the next brick.

**Reason:** Seed-to-seed variance in deep learning routinely exceeds the gap
between two architectures, so any single-run comparison — which is what the
comparison/sweep rankings report today — is statistically indefensible. This is
the single feature that turns VisionForge results into something a researcher
can put in a paper ("accuracy = 0.87 ± 0.02, n=5") instead of a point estimate.
Building it over the TaskRunner handle (ADR-041) keeps it one module + thin
endpoints instead of five task-specific implementations, and reusing the
sweep/comparison report pipeline means persistence and GUI wiring cost nothing
new. Replicates are deliberately not ranked: they are samples of one
distribution, and sorting them would invite exactly the cherry-picking the
feature exists to prevent.

## ADR-057 — CUDA/cuDNN/GPU recorded in run.json environment

**Date:** 2026-07
**Status:** Accepted
**Extends:** ADR-013 (run.json contract), environment capture

**Decision:** `capture_environment()` additionally records `cuda` (torch CUDA
build version), `cudnn` (cuDNN version) and `gpu` (device 0 name). `"none"`
means probed-and-absent (CPU build / no GPU); `"unknown"` means the probe
failed. Best-effort — the probe never raises into a training run. Additive to
the `environment` block, so existing run.json parsers are unaffected.

**Reason:** The pip version string alone (`torch 2.5.1`) cannot distinguish a
CPU wheel from cu118/cu124 builds, and kernel selection differs across
CUDA/cuDNN releases and GPU models — all of which can shift metrics between
"identical" runs. A run record that claims reproducibility but omits the
compute substrate is incomplete provenance.

## ADR-058 — Researcher-defined custom tasks (`user_tasks/` SDK)

**Date:** 2026-07
**Status:** Accepted — **fully implemented** (all six bricks shipped 2026-07:
SDK package, engine, API, orchestrator adapter, `new-task` scaffolder +
example + README, and the GUI — dynamic tabs, schema-driven panel, generic
results/history; per-brick record in `docs/archive/CUSTOM_TASK_PLAN.md`)
**Extends:** ADR-048 (user_models), ADR-041 (TaskRunner), ADR-013 (run.json)

**Decision:** A sixth, user-defined task surface: the researcher drops one
documented Python file into `user_tasks/`, registers it with `@register_task`
(key, label, **accent color**, description, metric metadata), and defines a
Pydantic `Config` (extending `BaseTaskConfig`, which composes the shared
training/data/output/device blocks) plus four hooks — `build_model`,
`build_loaders`, `compute_loss`, `compute_metrics`. A VisionForge-owned
`GenericTaskEngine` drives the loop (seeding, device, early stopping, best
checkpoint, SSE, run.json, TensorBoard); a `run(cfg, ctx)` escape hatch exists
for non-epoch-shaped training. The GUI renders the task as a real tab (dynamic
`TASKS` merge from `GET /api/tasks`) with the schema-driven form — **no
user-supplied JavaScript, ever**; custom identity is name/color/description
only. A `CustomTaskRunner` adapter gives every custom task hyperparameter
sweeps (ADR-045) and multi-seed replicates (ADR-056) for free —
`/api/custom/{key}/{sweep,replicates}`. Model comparison (ADR-044) is
**deliberately not exposed** for custom tasks: it overrides `model.name`, which
`BaseTaskConfig` does not guarantee, and comparing alternatives is a one-axis
sweep over whichever field the task declares. `visionforge new-task <key>`
scaffolds the commented template.

**Reason:** This is the product thesis (facilitation) applied to the last rigid
boundary: today adding a task family requires writing ~500 lines of trainer +
API + React panel across four layers — fine for the maintainers, impossible for
a visiting researcher. Every seam the SDK needs already shipped and is tested
(registry discovery, schema-driven form, SSE/run.json contracts, TaskRunner
orchestration), so the SDK is packaging, not invention. Constraining custom
tasks to declarative descriptors + Python hooks (instead of arbitrary frontend
plugins) is what keeps them functional, upgrade-safe and reviewable — the
standardization the request asked for ("padronizar da melhor forma"). Trust
boundary is unchanged from ADR-048: the user's own Python on the user's own
machine.

## ADR-059 — Canonical task-panel contract (classification is the template)

**Date:** 2026-07
**Status:** Accepted — **fully implemented** (bricks A–F shipped 2026-07-02;
audit + per-brick record in `docs/archive/PANEL_PARITY_PLAN.md`)
**Extends:** ADR-025, ADR-028, ADR-044/045/056

**Decision:** The classification `ParamPanel` layout is ratified as the canonical
task-panel contract, and the four standalone panels (detection, regression,
segmentation, anomaly) are brought to it:
1. **Section order:** Nome → Estratégia → Modelo → Treinamento → Dataset
   (+stats) → Pré-processamento → Augmentação (+preview) → cards auxiliares.
   Refined after user review (2026-07-02): Nome and Estratégia are **one
   card** — the canonical `ExperimentHeader` (name + YAML export/import side
   by side; strategy selector below, same box), byte-for-byte the
   classification layout. YAML import validates against the task's live
   schema and rebuilds the form via tested round-trip converters
   (`formFromPayload(buildPayload(form)) == form`).
2. **Strategy is a first-class selector** (segmented control that morphs the
   form — like classification's `BlockSelector`), not stacked always-visible
   cards: `Treino simples | K-fold (onde existir) | Sweep | Réplicas`.
3. **"Comparar arquiteturas" stops being a separate card**: it becomes a
   one-click *preset* inside the Sweep mode (fills a `model.name` axis from an
   architecture multi-select). The backend comparison runner stays (it is the
   engine); only the duplicated GUI concept goes away.
4. **Full config-surface parity:** preprocessing pipeline, augmentation fields
   + preview, and YAML import/export in every task panel whose backend supports
   them (regression/segmentation/anomaly — their configs already carry
   `PreprocessingConfig`/`TransformConfig` today, unreachable from the GUI).
5. **Detection is the documented exception** where semantics differ:
   Ultralytics owns augmentation (its own aug card stays) and preprocessing
   does not apply; order and strategy selector still conform.
Shared section components (`TrainingSection`, `TransformsSection`,
`StrategyBar`, …) parametrized by config path replace copy-pasted panel JSX so
the panels cannot drift again — and become the generic panel of ADR-058.

**Reason:** An audit (2026-07-01) found the standalone panels diverged from
classification in section order (Dataset before Treinamento), lost reachable
features their backends already support — most seriously, **augmentation
defaults (`horizontal_flip=True`, `rotation_degrees=10`) are silently applied
to every regression/segmentation/anomaly GUI run** with no way to see or
disable them, which can materially change anomaly results — and bolted
strategies on as stacked cards, mixing "what to train" with "how many times to
train it". One canonical contract restores the facilitation thesis (the user
learns one layout, every task behaves the same), closes a real correctness
hole, and prevents the recurring backend-ahead-of-GUI debt (Phase 5.5 déjà vu).
Defaults are NOT changed (behavior-preserving); they are surfaced.

## ADR-060 — `visionforge selftest`: end-to-end verification through the real API

**Date:** 2026-07
**Status:** Accepted — shipped 2026-07-26
**Extends:** ADR-010 (CPU-only CI), ADR-013 (run.json), ADR-041 (TaskRunner)

**Decision:** A first-class self-test command that trains **every task through
the real GUI API**, not through the blocks. `visionforge selftest` builds tiny
synthetic datasets in the exact on-disk layouts the tasks consume
(`utils/selftest_data.py`), starts the real FastAPI app on an ephemeral socket,
and POSTs one case per (task, strategy) pair to the same endpoints the browser
uses. Each case is validated on three axes:

1. the run reaches `completed` and its stored report carries the keys that
   task's block actually returns (train/test sections for the
   regression-family tasks, `detection` for detection, `metrics` for custom);
2. the **SSE stream** delivers the live-monitor contract — `epoch_end` for
   single runs, `trial_start`/`trial_end` for every multi-trial strategy;
3. artifacts land on disk.

Everything is CPU-sized, `pretrained=False` and `num_workers=0`, so it runs
offline on a bare install in minutes and writes only inside a scratch dir.
Filters (`--tasks`, `--strategies`, `--quick`) narrow the matrix; `--json`
emits machine-readable outcomes; exit code is non-zero if any case failed.
The strategy list grows with the API: `comparison` (ADR-061's replicated
comparison) was added the day the endpoint shipped, because a strategy the
harness does not exercise is exactly where the last three defects hid.

Test layering: the dataset builders, case table and formatter are covered by
fast always-on unit tests (`tests/e2e/`); the live-training cases carry the
`slow` marker and are **deselected by default** (`addopts = -m 'not slow'`) so
the pre-commit suite stays seconds-fast, with `pytest -m slow` and the CLI as
the explicit gates.

**Reason:** The unit suite mocked exactly the seam that kept breaking. Three
defects reached the user in one week — a 500 opening History after k-fold, a
preview that composed filters correctly but served a cached image, and
multi-trial runs that streamed *nothing* while the progress bar crawled on
wall-clock — and **every one of them passed CI**, because the tests asserted
against mocked orchestrators and never drove a real run end to end. The gap was
structural: no test started a server, trained something, and looked at what the
browser would actually receive. This command closes it, and doubles as an
install verifier for a researcher who has VisionForge but no dataset yet
(`visionforge doctor` checks the environment; `selftest` checks the pipeline).

The server is a real uvicorn socket rather than an in-process test client on
purpose: the loop-per-request behaviour of `TestClient` outside a context
manager silently destroys genuinely-async background training (learned in
ADR-058 brick 3), so a fidelity gap there would hide precisely the class of bug
this exists to catch.

## ADR-061 — Paper outputs: significance testing, bootstrap intervals, LaTeX

**Date:** 2026-07
**Status:** Accepted — core shipped 2026-07-26 (Phase D slice 1)
**Extends:** ADR-056 (replicates), ADR-045 (sweeps), ADR-050 (K-fold)

**Decision:** Three additions that carry a result from "trained" to
"publishable":

1. **`core/significance.py`** — paired comparison between two configs over the
   seeds they **share**, with the test chosen and *justified* per comparison
   (paired t when the differences pass Shapiro-Wilk and there are ≥8 pairs;
   Wilcoxon signed-rank otherwise, with the reason recorded in the report),
   Cohen's `d_z`, a bootstrap CI of the difference, and Holm-Bonferroni
   control across the comparison family.
2. **Bootstrap intervals** alongside the Student-t interval in every replicate
   aggregate (`boot95_low`/`boot95_high`).
3. **`core/latex_export.py`** — every advanced report (replicates, sweep,
   K-fold, comparison) is written as a `booktabs` table next to its JSON/CSV.

**Reason:** ADR-056 answered "how uncertain is this number?" but not the
question a paper asks — *is A better than B, or is that gap seed noise?*
Pairing is what makes the answer sensitive: two configs trained under the same
seed share the split, the initialization and the augmentation stream, so their
difference isolates the change under study instead of drowning it in
between-seed variance. The helpers therefore **refuse** to compare runs whose
seeds do not line up rather than silently falling back to a weaker test, and
Holm correction is applied by default because K configs mean K(K-1)/2 tests,
where at α=0.05 roughly one in twenty "wins" is chance.

The bootstrap is not a nicety: the t interval assumes the sampling
distribution of the mean is normal, which a handful of seeds cannot establish.
A real run made the point concretely — with n=2 the t interval for MAE came
out `[-0.70, 3.76]`, crossing zero for a strictly non-negative metric.
**Both intervals are reported, and below 5 seeds the table itself carries a
caution**, because with tiny n the t interval is far too wide while the
percentile bootstrap is far too narrow (it can only resample the values it
has). A caveat that lives only in the docs never reaches the reader.

LaTeX export exists because the last mile of a result is a table in a
manuscript, and retyping numbers is where transcription errors enter — errors
no reviewer catches and no rerun reproduces. Table notes state what the
interval is over, that a sweep ranked on one run per config still reflects
seed noise, and which correction was applied.

**Slice 2 (shipped 2026-07-26):** `core/replicated_comparison.py` +
`POST /api/{task}/replicated-comparison` (all five built-ins and custom
tasks) run N seeds for each of M variants over the **same seed list** and
return the Holm-corrected matrix, with a `.tex` table beside the JSON. Two
guards the first real run forced:

- **Power floor.** Wilcoxon's statistic is discrete: with n pairs its smallest
  two-sided p is `2^(1-n)`, so at **n=5 nothing can reach α=0.05** — a real,
  perfectly consistent 0.10 gap came back "not significant". Every comparison
  now reports `min_achievable_p` and an `underpowered` flag, and the report
  and LaTeX note say that a non-significant verdict there means "too few
  seeds", not "no effect".
- **Ranking direction.** The first run crowned MAE 4.02 over MAE 0.99 because
  ranking always sorted descending. `infer_direction` reads the metric name
  (loss/mae/mse/rmse/error → lower-is-better) and callers may override it;
  getting this wrong is uniquely damaging because the wrong winner arrives
  with a p-value beside it.

**Slice 3 (shipped 2026-07-26) — dataset fingerprint.** Every `run.json` now
carries `dataset_fingerprint`: a sha256 over the sorted `(relative path, size)`
manifest of `data.base_dir`, written by all six trainers. A config records a
*path*, and paths lie — files get added, a split gets re-shuffled, the dataset
gets re-exported between runs — so "same base_dir" was never a checkable claim
and now is.

The method is recorded next to the digest because the two on offer guarantee
different things: `manifest` (default) only stats files, so it is cheap enough
to run before every training and catches added/removed/renamed/resized files
but **not** an edit that preserves byte count; `content` hashes the bytes too.
The `note` field states that limitation inside the artifact, and
`same_dataset()` returns `None` rather than `False` when comparing digests
produced by different methods — over-claiming here would be worse than not
fingerprinting at all. It never raises: a missing directory, an unreadable
file or a dataset above 200k files yields an `unavailable` entry with the
reason, because provenance must not be what fails a training run.

**Phase D is complete.**

---

## ADR-062 — `training.deterministic` in every task, and detection's two seeds

**Date:** 2026-07-28
**Status:** Accepted — shipped 2026-07-28
**Extends:** ADR-020 (cuDNN benchmark on by default), ADR-057 (provenance)

**Context:** `training.deterministic` existed only in the classification
`TrainingConfig` (and, undocumented, in the custom-task SDK). The four
standalone tasks called `_seed_everything(cfg.seed)` with no way to pin cuDNN,
so a regression or segmentation run could not be made bit-reproducible at all —
while its `run.json` recorded a seed, implying it could. Worse, the torchvision
detection backend forwarded `seed` **only** to Ultralytics: on that path nothing
was seeded, and `seed: 42` in the config was a claim nothing backed.

A seed alone does not make a GPU run reproducible. cuDNN's autotuner picks
kernels by benchmark, and non-deterministic kernels are selected by default
(ADR-020, for throughput). Reproducibility is the pair — seed *and* pinned
cuDNN — so exposing one without the other overstates what the artifact proves.

**Decision:** every task's training config carries `deterministic: bool`, wired
to `_seed_everything(seed, deterministic=…)` in its trainer and surfaced as a
"Determinístico" toggle beside Seed in all five panels. The description is a
single shared constant, `utils.config.DETERMINISTIC_DESCRIPTION`, because the
GUI renders it as form help text and four hand-copies would drift.

Two deliberate asymmetries:

- **Detection defaults to `True`**, everywhere else `False`. `DetectionConfig`'s
  stated contract is that an unmodified copy trains exactly like a bare
  `YOLO.train` call, and Ultralytics' own `deterministic` default is `True`.
  Defaulting it to `False` for symmetry would have silently changed how every
  existing detection config trains.
- **The knob keeps costing speed, and says so.** It is opt-in, not the default,
  because pinning cuDNN measurably reduces throughput; ADR-020 stands, and this
  relaxes it on demand rather than reversing it.

`_seed_everything` was also added to the torchvision detection path, which fixes
unseeded runs independently of the new knob.

**Rejected:** a global "reproducible mode" flag outside the config tree — it
would not travel with the exported YAML, so a re-run from the CLI could differ
from the GUI run that produced it, which is the exact failure the config
contract exists to prevent.

**Tests:** `tests/core/test_determinism_parity.py` (27) — one file on purpose,
parametrized over all six training configs, since the defect being guarded
against is a *gap*, and a per-task test file is how such a gap stays invisible.
It asserts the field, its documented default and its shared description exist
everywhere; that the value reaches `_seed_everything` for regression,
segmentation and anomaly; that it reaches Ultralytics' `train` kwargs; and that
the torchvision detection backend is seeded at all.

---

## ADR-063 — History by task family, selection-driven actions, Datasets as its own surface

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Extends:** ADR-059 (canonical panel contract), ADR-055 (dataset download)

**Context:** four defects reported from real use, all in the same area — the
history overlay had grown past what its layout could carry, and the dataset
download had been placed inside every task panel.

**Decisions:**

1. **A tab per task family, not a chip per task value.** The task filter was a
   chip row that clipped its own options off both edges once a fourth task
   existed. It is now a tab row above the list, one tab per family with its run
   count, and the task is navigation rather than a filter competing for space
   with status, block and sort.

   The grouping key is a *family*, derived by `taskFamily()`, not `run.task`.
   `run.task` is not the family: classification runs record their problem type
   (`binary`, `multiclass`, `multilabel`) because that is what the
   classification config's `task` field means, while the standalone tasks
   record the family itself. Grouping on the raw value produced a "BINARY" and
   a "MULTICLASS" tab — neither of which is a task anyone selects. Custom tasks
   (ADR-058) keep one tab each, labelled with their own key.

2. **Status and block wrap instead of clipping.** They stay chips — every
   option and the current one are readable without opening anything — but each
   dimension is its own `flexWrap` row, so a long list grows downward instead
   of running off the edge, which was the actual defect. Options are scoped to
   the active tab, and a dimension with a single value is hidden entirely
   because it filters nothing. Inside Classificação the raw `run.task`
   (`binary`/`multiclass`) becomes a `tipo` row of its own: the family tab
   collapsed a distinction that is still worth filtering by.

3. **One selection mode drives both delete and compare.** Separate "compare
   mode" and "delete mode" toggles would make the researcher declare intent
   before picking runs, which is backwards — you pick the runs, then decide.
   Selecting reveals `🗑 Excluir N` (1+) and `↔ Comparar N` (2+). The
   confirmation names every run being deleted, however many: "excluir 12 runs"
   without the list is a destructive action taken on trust. Deletion is
   sequential, and a partial failure keeps the runs that survived on screen
   with the reason.

4. **Deleting no longer closes the history.** The confirmation modal renders
   inside the overlay's backdrop, whose `onClick` is `onClose`. Every click in
   the dialog — including Cancel and Confirm — bubbled to it, so any delete
   attempt dismissed the whole history. The modal layer now stops propagation.
   The comment above it claimed the opposite behaviour, which is how it
   survived review.

5. **Datasets is a surface of its own.** The download form rendered at the
   bottom of all five task panels: five copies of one global action, each
   pushing the panel it belonged to further down. A dataset is not owned by a
   task — you fetch it once and then point whichever panel you like at the
   folder. It now opens from the bottom bar next to History
   (`DatasetsOverlay`), and `DatasetDownloadCard` takes a `collapsible` prop so
   the standalone surface skips a toggle that would hide its only content.

**Rejected:** a top-level "Datasets" entry in the task tab bar. That bar is the
task selector — its state drives which config is built, which schema is
validated and what Treinar submits. A tab there that is not a task would have
to be special-cased in each of those, to place one form.

**Verified live** against the researcher's own 92 runs: tabs read
Classificação 79 / Detecção 4 / example_counting 9 (was BINARY 61 /
MULTICLASS 18); selecting two runs offers compare and delete; the confirmation
lists both names; cancelling leaves the history open on the same tab.

---

## ADR-064 — Every dropdown is drawn by the app; history opens on the active task

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Extends:** ADR-063 (history surface)

**Context:** three follow-ups from using ADR-063 for real.

**Decisions:**

1. **No native `<select>` anywhere in the GUI.** A native select renders its
   popup with the operating system's own widget — grey list, system font, OS
   highlight — and inside a dark monospaced UI that popup is the one surface
   that looks borrowed. `color-scheme: dark` (the earlier mitigation) only
   repaints its background; it cannot touch the typography or the highlight.

   `SelectField` already drew its own menu for labelled form fields. The five
   remaining native selects were *toolbar* controls, which `SelectField` does
   not fit (it is full-width and carries a `FieldLabel`), so they got a
   sibling: `MenuSelect`, compact and sized to content. Positioning,
   outside-click and the portal escape hatch are now one hook,
   `useAnchoredMenu`, shared by both — the menu must be portaled because the
   form cards use `backdrop-filter`, which creates a stacking context per card
   and would otherwise paint the menu under the next card.

2. **The history opens on the active task's tab.** Opening it from the
   Classification panel means you want *its* runs; landing on "Todos" and
   making the researcher click again is a step nobody wants. It falls back to
   "Todos" when the active task has no runs yet, so the sheet is never empty on
   open.

3. **Filters within a tab are wrapping chip rows, not dropdowns.** ADR-063
   turned the clipped chip row into a dropdown; the clipping came from the row
   being a single non-wrapping line, and `flexWrap` fixes it without hiding the
   options behind a click. Verified at a 620px viewport: the five `bloco` chips
   render on two lines with no horizontal overflow.

**Rejected:** styling the native popup with `appearance: none` plus CSS. It
restyles the *closed* control only — the open list is drawn by the OS and is
not reachable from CSS, which is precisely the part that looked wrong.

---

## ADR-065 — Transfer learning streams progress; validated on real datasets

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Extends:** ADR-060 (selftest), ADR-062 (determinism parity)

**Context:** `visionforge selftest` proves the pipeline on synthetic data. A
full matrix was run on **real** datasets instead — five tasks × the strategies
each one has (simple, K-fold, transfer, grid, random), 21 cases on GPU, driving
the same endpoints the browser uses through the selftest's own `run_case`, so
the pass criteria were identical: the run completes, the report carries the
keys that task really returns, and the SSE stream delivers progress.

**What it found:** `TransferLearningBlock` trained correctly and wrote a
correct report, but emitted **no SSE events at all** — the GUI's progress bar
sat dead for the whole run. The block never accepted a `progress_callback`, and
`routes.py` left it out of the isinstance check that attaches the event pump,
with a comment stating the gap as if it were a decision. The synthetic selftest
did not cover it because transfer learning is a classification *block*, not one
of the five task strategies the selftest enumerates.

**Decision:** the block takes a `_progress_callback` like every other streaming
block and forwards it to `Trainer.fit`; `routes.py` attaches the pump. Two
regression tests pin both halves — one asserts `epoch_end` reaches the
callback, the other asserts `routes._execute_experiment` still names the block,
because a block that accepts a callback nobody attaches is the same defect.

**Real-data corpus** (`docs/dev/VALIDATION.md`): USK-COFFEE for
classification, a Roboflow cats/dogs export for detection, Oxford-IIIT Pet
trimaps for segmentation, IMDB-WIKI `wiki_crop` ages for regression, and
USK-COFFEE again for anomaly — `premium` as the normal class and `defect` as
the anomaly, which is the dataset's own labelling re-expressed in the MVTec
layout. Nothing is synthesised; the only local work is file arrangement.

**Not defects, recorded so they are not re-investigated:**

- Three initial failures were the harness asserting the wrong contract: the
  anomaly report is `train`/`test` (not `anomaly`), and its sweep metric is
  `auroc` (not `image_auroc`). The second surfaced as a loud
  `RuntimeError: No sweep reported the metric 'image_auroc' — available:
  ['auroc', 'image_f1']`, which is ADR-060's unreported-metric guard working.
- Age regression came back with a negative R² on some strategies. Two epochs
  over 1500 faces does not learn age; this validates that the pipeline runs,
  not that the model is good. The distinction is the whole point of the "what
  this is not" section in `TRAINING_PLAN.md`.

---

## ADR-066 — One source of truth for the version; v0.1.0 as the first release

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29

**Context:** the project had shipped 65 ADRs, five task families, a custom-task
SDK and 1274 tests while still declaring `version = "0.0.1"`, never tagged. The
string was also duplicated in four places — `pyproject.toml`, `CITATION.cff`,
`gui/server.py` and a hardcoded `v0.0.1` in the React header — and
`bump-my-version` was configured to rewrite only the first, so any bump would
have silently shipped three stale versions.

**Decisions:**

1. **`0.1.0`, not `1.0.0`.** Below 1.0 says the config schema and HTTP API may
   still change between minor releases, which is true and worth saying out
   loud. `schema_version` + migrations (ADR-039) already keep old configs
   loadable across that churn, so the promise costs users nothing.

2. **The package reads its own installed metadata.** `visionforge.__version__`
   comes from `importlib.metadata`, and `server.py`, the CLI and the
   `/api/system/info` payload all read that. Only two files can now hold a
   literal — `pyproject.toml`, which *is* the metadata, and `CITATION.cff`,
   which is a data file that cannot import anything — and `bump-my-version`
   rewrites both plus the changelog heading.

3. **`visionforge --version`, and the version in the GUI header from the API.**
   A bug report without a version costs a round-trip; a screenshot of the GUI
   now carries the version that produced it.

**Also fixed, found while checking the first-run path:** `visionforge doctor`
derived its whole recommendation from `nvidia-smi`. On a machine where that
binary is not on PATH but torch reports a working CUDA build, it announced "No
CUDA-capable GPU detected", recommended the **CPU wheel**, and then printed
"environment looks good" — the worst possible answer for a new user, who would
follow it, get CPU-speed training, and conclude the tool ignores their GPU. The
torch probe now runs first and is trusted when it says CUDA works. Two
regression tests pin both directions.

**For external testers:** `.github/ISSUE_TEMPLATE/` asks for
`visionforge --version` and `visionforge doctor` up front, plus the exported
YAML; `.github/CONTRIBUTING.md` is a pointer so GitHub's "Contribute" links
resolve (it only looks at the root, `.github/` or `docs/`);
`CHANGELOG.md` records what 0.1.0 contains; and the README gained a **Status**
section stating the known limits — single concurrent training, dataset download
covering classification only, no K-fold for detection/anomaly, the Windows
worker cap, dark theme only.

---

## ADR-067 — Published as `visionforge-studio`

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Extends:** ADR-066 (versioning)

**Context:** `visionforge` on PyPI is taken — an unrelated computer-vision
project by another author, version 1.0.0, uploaded 2024-05-05. Discovered while
setting up Trusted Publishing, before the first tag: a release under that name
was never going to succeed, and pointing users at `pip install visionforge`
would have installed someone else's package.

**Decision:** the **distribution** name is `visionforge-studio`. The import
name (`import visionforge`), the CLI command (`visionforge`), the repository
and the project's own name are unchanged — only the string PyPI indexes moves.

The rename touches more than `pyproject.toml`, and two of the places fail
*silently*:

- `__init__.py` looks the version up by distribution name. Left as
  `visionforge`, `importlib.metadata` would have raised `PackageNotFoundError`
  on every install and every user would have seen `0.0.0+unknown`.
- Three runtime error messages told the user to run
  `pip install 'visionforge[detection]'` — which, post-release, installs the
  other project.

**Also fixed, found by installing the wheel into a clean venv:**

- `doctor` printed `pip install -e ".[cpu]"` to everyone. That form only works
  inside a checkout; a user who installed from PyPI has no source tree, so the
  first command doctor gave them simply failed. It now detects an editable
  install (`direct_url.json` → `dir_info.editable`) and prints the matching
  form.
- `doctor` reported "Verdict: environment looks good" when torch was not
  installed at all. Torch is what trains; that verdict sent a new user to
  discover the missing dependency on their first run instead of here. Missing
  torch is now a `[FAIL]` with the install line and a non-zero exit.

**Verified end to end in a throwaway venv:** the wheel installs, `visionforge
--version` reports `0.1.0` (not `0.0.0+unknown`), `import visionforge` works
under the new distribution name, and the built SPA ships inside the wheel
(42 files) so a pip-installed user never needs Node.

---

## ADR-068 — The pip-installed workspace, and a validation split on download

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Extends:** ADR-048/049 (custom models), ADR-055 (dataset download), ADR-058
(custom tasks)

**Context:** with `visionforge-studio` live on PyPI, the first-run path was
walked from a clean venv with no repository — the position every tester is in.
Two questions had no good answer.

**1. Where does a pip-installed user put their own model?**

`user_models/` and `user_tasks/` resolve relative to the working directory, so
the answer is "a folder next to wherever you run it" — verified working from a
throwaway venv, including `visionforge new-task`. Nothing is broken, but
nothing *said* so: a pip user has no repo to look at, `user_models/README.md`
ships only in the source tree, and running from a different folder silently
loses both.

**Decision:** keep the cwd-relative resolution — it needs no configuration and
gives one folder per project — and make it visible instead. `visionforge
doctor` now prints the working directory and the resolved paths for both
folders, with a count of what it found or a hint to create them. The README
gained a Workspace section showing the layout, so the answer arrives before the
question.

Rejected: a fixed `~/.visionforge/` home. It would make two projects share one
model namespace, and "which of my experiments used which version of my model"
is exactly the question this project exists to keep answerable.

**2. Does the one-click dataset download produce something trainable?**

Almost. torchvision ships its built-ins as train/test, while every VisionForge
task expects train/val/test — so a downloaded dataset landed one split short
and the picker reported *"Detectado parcialmente. Faltando: validação"* on what
should be the smoothest possible first run.

**Decision:** `download_torchvision` carves the missing split out of train
(`val_fraction`, default 0.2). Stratified per class, so a rare class keeps
representation in both; taken by sorted filename rather than at random, so
running the download twice yields the same split. `val_fraction=0` keeps
torchvision's original two for anyone who wants them.

**Verified from the published package**, not the checkout: `pip install
visionforge-studio` into a clean venv, then a real MNIST download (400 images,
ImageFolder layout) and a config validated against it.

---

## ADR-069 — User-facing text stops citing ADRs; install hints adapt

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Extends:** ADR-067 (distribution name)

**Context:** `visionforge --help` described two commands as
"...on synthetic data (ADR-060)" and "...under user_tasks/ (ADR-058)". An ADR
number is a pointer into this file; to someone who installed a package it is
noise that suggests they are missing context they cannot get.

**Decision:** ADR references belong in code comments, docstrings and
`docs/` — never in `--help`, error messages, or GUI copy. The two help
strings now say what the command does in the reader's terms. The audit found
these were the only two leaks; the frontend's ADR mentions are all in JSDoc,
which is developer-facing and stays.

**Also fixed, same class as the `doctor` bug in ADR-067:** the three optional
dataset providers raised
`ImportError("... pip install -e \".[roboflow]\".")`. That form only works
inside a checkout, so a pip-installed user got a command that fails — while
being told it is the fix. The install hint moved into
`doctor.extra_install_hint(extra)`, shared with `build_install_command`, so
there is one place that knows how to phrase "install this extra" and it can
never again be written for only one audience.

**`docs/DATASETS.md`** now documents each provider: which extra it needs,
which credential and where it goes, what to type, how to validate it from the
command line, and what the result should look like. It also states the limits
plainly — Kaggle returns whatever layout the author published, and Hugging Face
only materializes datasets that expose an image plus a label column.

---

## ADR-070 — Stored provider keys; hide vs delete for custom tasks

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Extends:** ADR-055 (dataset download), ADR-058 (custom tasks)

### Stored provider keys

**Context:** typing a Roboflow key or a Kaggle token on every download is the
kind of friction that stops a feature being used at all.

**Decision:** one local store, `~/.visionforge/credentials.json`
(`VISIONFORGE_HOME` overrides), with three deliberate properties:

1. **Per user, not per project.** Custom models and tasks resolve next to the
   working directory precisely so two projects do not share them (ADR-068). A
   key is the opposite — it belongs to the person, and keeping it out of the
   project folder means it cannot be carried into a git repository or a synced
   drive by accident.
2. **Read back masked.** The API returns `rf_•••••••1234`, never the value. The
   GUI only needs to show that a key exists and *which* one — the common
   question is "is this my old key?", not "what is it?" — and the download runs
   server-side where the real value already is. A screenshot of the panel is
   therefore not a leak, which a test asserts by serializing the status and
   checking the secret does not appear.
3. **Explicit beats stored.** A key passed in the request wins, so a one-off
   key can be used without overwriting what is saved.

The file is written owner-only where the platform supports it; on Windows that
call is a no-op and the file inherits the profile ACL, which the module
docstring states rather than glosses over. The log records that a credential
was stored, never its value.

Kaggle is the odd one: its client authenticates *at import time* from
`kaggle.json` or the environment, so the stored value (`user:key`) is placed
into the environment before the import rather than passed as an argument.

### Hide vs delete for custom tasks

**Context:** two custom tasks were enough to crowd the task bar. "Remove this
task" was the request — but it means two different things, and only one of them
is recoverable.

**Decision:** both, separated by how much they cost to undo.

- **Hide** — the tab stops rendering; the file is untouched; one click; undone
  by unhiding. This is the honest answer to "my tab bar is full", which is the
  actual complaint. A hidden task stays registered, so past runs still resolve
  and a YAML still re-runs it.
- **Delete** — removes the `.py` the researcher wrote. The API requires
  `confirm` to repeat the task key exactly, and the GUI keeps the button
  disabled until it matches. A second click is not evidence of intent; typing
  the name is.

A packaged task (`user_tasks/<key>/task.py`) takes its directory with it — the
folder exists to hold assets beside the module, so removing only the `.py`
would orphan them. Deleting also clears any stale hidden entry, or re-creating
a task with the same key would resurrect it already hidden with nothing on
screen to explain why.

The hidden list lives at `user_tasks/.hidden.json`, next to the tasks rather
than in the per-user config, because tasks are per working directory: two
projects with a task of the same name must be able to disagree about whether it
is visible.

**Placement:** the management card sits at the *bottom* of the custom task's
own panel, behind an "⚙ opções" toggle. You reach it by scrolling past the
thing you came to configure, not by aiming near it — unlike a tab-bar × or a
bottom-bar trash, both of which put an irreversible action within a mis-click
of a frequent one.

---

## ADR-071 — The Docker image, and two defects the first real build exposed

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Completes:** ADR-042 (Docker + doctor)

**Decision:** a multi-stage image whose *base* is a build arg, not a hardcoded
CUDA tag. Shipping one image per CUDA version is a maintenance tax and picking
a single one strands everybody else:

```
docker build -t visionforge .                                   # CUDA 12.4
docker build --build-arg CUDA_TAG=cu126 \
             --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04 .
docker build --build-arg VARIANT=cpu --build-arg CUDA_TAG=cpu \
             --build-arg BASE_IMAGE=ubuntu:22.04 .
```

`BASE_IMAGE` is a whole image reference rather than a CUDA version because the
CPU variant should not inherit the CUDA runtime at all — driver libraries a CPU
wheel never loads. Measured on disk (`docker images`): **6.01 GB → 2.36 GB**.
The compressed download is 509 MB; the two numbers answer different questions
and must not be compared with each other.

Other choices: `uv` installs Python 3.13 in-image, so the `pyproject` floor is
met exactly and no distro PPA is involved; dependencies are layered before
application code, because torch is a ~2 GB download nobody wants to repeat for
a one-line change; the runtime stage has no Node (the SPA is built in a
separate stage and copied); the container runs as uid 1000 so files it writes
into the mounted `outputs/` stay usable from the host; and `datasets/` mounts
read-only, because training reads images and has no business modifying them.

### What building it actually found

Three of these would have survived any amount of reading.

1. **`ARG` used in `FROM` must be global.** Declared after the first `FROM` it
   is stage-scoped, resolves empty, and the build dies on
   `nvidia/cuda:-runtime-ubuntu22.04`. Reading the file, it looked right.

2. **`visionforge selftest` failed on a clean install.** Its `custom` cases
   target `example_counting`, which exists only in the repository. On a fresh
   `pip install` — and inside the image, where `user_tasks/` is a mount point —
   the command that exists to say "your install is fine" reported a failure for
   the normal state of having no researcher-defined task. The cases are now
   skipped, with a log line naming `visionforge new-task` as the way to get one.

3. **`visionforge selftest` crashed on Windows exactly when it found a
   problem.** The report contains `→`; the default console codepage (cp1252)
   cannot encode it, so printing a *failure* raised `UnicodeEncodeError` and
   the diagnostic command died mid-diagnosis. stdout/stderr are reconfigured to
   UTF-8 with `errors="replace"` before anything prints.

Defects 2 and 3 affect every pip user, not just Docker ones. They surfaced
because the image was actually built and run, which is the argument for doing
that rather than shipping a reviewed Dockerfile.

**Verified:** `visionforge selftest --quick` reports 5/5 inside the container;
`visionforge --version` and the `org.opencontainers.image.version` label both
read 0.1.0, stamped from the package metadata at build time rather than typed
into the Dockerfile; the built SPA (42 files) ships inside the image; the
container runs as `forge` with workdir `/work`.

**Unverified and stated as such:** the GPU variant. This machine's Docker has
no GPU passthrough configured, so `--gpus all` and
`torch.cuda.is_available()` inside the image remain untested.

---

## ADR-072 — cu128 is the supported floor for current hardware

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Extends:** ADR-005 (user-managed torch), ADR-042 (doctor + Docker)

**Context:** the project's hardware extras stopped at `cu126`, and `doctor`'s
driver→wheel thresholds stopped there too. Building the GPU image on an
RTX 5060 Ti exposed what that costs.

RTX 50-series is Blackwell — compute capability **12.0 / sm_120**. No PyTorch
wheel before `cu128` ships an sm_120 kernel. On such a card an earlier build
installs cleanly, imports cleanly, reports the GPU through
`torch.cuda.is_available()`, and then fails at the first kernel launch. It is
the exact silent misconfiguration ADR-042 created `doctor` to prevent — and
`doctor` was recommending it, because `cu126` was the highest tag it knew.

**Decision:** `cu128` joins the hardware extras (`pyproject.toml`, both
`[tool.uv.sources]` marker chains) and `doctor`'s thresholds, which now map
driver 12.8+ → `cu128`.

It is also the **Docker default**, replacing `cu124`. That is not "newer is
better": `cu128`'s kernels span sm_75 (Turing) through sm_120 (Blackwell), so
it covers strictly more hardware than `cu124`, and CUDA minor version
compatibility lets its runtime work on any CUDA 12 driver. Choosing `cu124` as
the default meant shipping an image that cannot run on a current card.

**Verified on the real card, not just by version arithmetic:**
`docker run --gpus all` sees the RTX 5060 Ti; `torch.cuda.is_available()` is
True; a 512×512 matmul executes **on the device** (availability alone would not
have caught a missing kernel); `arch_list` includes sm_120; and
`visionforge selftest --quick` reports 5/5 inside the container.

A detail worth recording: `doctor` inside that container prints "GPU usable via
torch (CUDA build 12.8); nvidia-smi not on PATH". The CUDA *runtime* image does
not ship `nvidia-smi`, so the ADR-066 fix — trusting the torch probe when the
binary is absent — is what keeps doctor from telling a working GPU container to
install the CPU wheel.

---

## ADR-073 — Releases are one command, and CD refuses a mismatched tag

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Extends:** ADR-066 (single source of truth for the version)

**Context:** ADR-066 made the version single-sourced — two literals
(`pyproject.toml`, `CITATION.cff`), everything else reading
`importlib.metadata`, and `bump-my-version` rewriting both. What it did not fix
is that cutting a release was still described as `git tag v0.1.0`, which
bypasses that machinery entirely.

`bump-my-version` is already configured with `commit = true` and `tag = true`,
so `bump-my-version bump minor` rewrites every literal, adds the changelog
heading, commits and tags **atomically**. The mechanism existed; nothing
pointed at it, and the instructions pointed elsewhere.

**Decision:**

1. **`docs/dev/RELEASING.md`** states the one command, what it rewrites,
   what to verify before tagging (including that the wheel carries the built
   SPA — a stale build would publish an old interface with new code), and what
   to check after publishing. Linked from the README and the contributing
   guide.

2. **CD fails on a mismatched tag, before publishing.** The tag and the
   packaged version live in different places and nothing made them agree:
   tagging `v0.2.0` while `pyproject.toml` says `0.1.0` builds
   `visionforge_studio-0.1.0` and either publishes it under a `v0.2.0` release
   or dies inside the upload when PyPI rejects a version that already exists.
   The workflow now compares the tag against the built wheel's filename and
   stops with both values named and the correct command in the error.

**Rejected: deriving the version from the git tag (`setuptools-scm`).** It is
the more automatic answer and it was the first instinct here, but it trades a
guard for a build-backend change immediately after a working PyPI publish, and
it makes a source tree without `.git` produce a version of `0.0.0`. The
remaining manual step is one command that already does everything; automating
it further buys little and risks the release path.

**Verified:** the tag-vs-version extraction was checked against the real wheel
filename (`visionforge_studio-0.1.0-py3-none-any.whl` → `0.1.0`) rather than
assumed, and `bump-my-version` correctly refuses to run from a dirty working
tree.

---

## ADR-074 — A single run's test metrics carry a bootstrap interval

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Complements:** ADR-056 (multi-seed replicates), ADR-061 (significance testing)

**Context:** ADR-056 answers "how uncertain is this number?" by training N times
under N seeds and aggregating to mean ± CI. That is the stronger answer, and it
costs N trainings. Most runs are not replicated, and those reported a bare
`accuracy: 0.7506` — a number whose precision is unknowable from the report.
There is a second, cheaper question that one training can answer: given this
fixed model, how much of the metric is an accident of *which images landed in
the test split*? Resampling the split with replacement answers exactly that.

**Decision:** every classification run reports a 95% percentile bootstrap
interval per test metric (accuracy, F1, precision, recall, AUC-ROC), written to
`run.json` as a `metric_cis` block and surfaced in the results tiles, the
run-detail panel and the markdown model card.

- **Always on, no config knob.** Provenance, like the `environment` block
  (ADR-057) and the dataset fingerprint (ADR-061): a researcher should not have
  to know to ask for it. This was only defensible once the cost was negligible
  (see below).
- **`metric_cis` is a sibling of `metrics`, not nested inside it.** Several
  readers treat `metrics` as a flat name → number map — the history projection,
  the markdown table — and a dict value there is either silently dropped or
  printed as a raw Python dict.
- **Seeded from `training.seed`.** Re-evaluating a finished checkpoint must not
  produce a different interval than the one already in `run.json`; an interval
  that moves between reads of the same result cannot be cited.
- **Applies to plain and transfer-learning runs**, the two paths that evaluate
  one test split. **Not** to cross-validation: its report is already μ ± σ over
  folds, and a per-fold interval would be uncertainty on top of uncertainty
  measuring a different thing.

**What the interval is not, stated in the report itself.** It holds the model
fixed and varies the evaluation sample, so it captures test-split sampling noise
only — not the run-to-run variance from initialization, data order and
nondeterministic kernels that replicates measure. A tight interval here says
nothing about whether retraining would land in the same place. The markdown card
and the GUI tooltip both say so and point at replicates, because the failure mode
of shipping this feature is a reader treating it as the stronger claim.

**Metrics are recomputed with vectorized confusion-matrix and rank arithmetic,
not by calling sklearn once per resample.** This is a deliberate exception to
"prefer the obvious implementation", and the reason is measured, not assumed:
sklearn's per-call overhead dominates (~5 ms regardless of split size), putting
1000 resamples at ~5 s per run and ~135 s across the 27-case selftest. That cost
would have forced the feature to be opt-in, which defeats the point. Vectorized
it is 0.007 s at n=500 and 0.069 s at n=5000 — 700x faster — so it can simply
always be there. The shortcut is only acceptable because the tests pin every
path against sklearn: accuracy/precision/recall/F1 for binary and macro
averaging over 2, 3 and 7 classes, and AUC including the heavy-ties case that
bootstrap resampling guarantees and naive ranking gets wrong. Agreement is
1e-16.

**Two guards the honest version needs:**

1. **A 20-sample floor.** Below it the interval is arithmetic, not evidence —
   resampling a handful of values produces a narrow interval that describes the
   sample, not the population. Under the floor no interval is written, so the
   report says "none" instead of something misleading. (This is the same trap
   ADR-061 documented from the other direction: with n=2 seeds a t interval for
   a MAE came out negative.)
2. **A resample that loses a class is dropped, and the count is reported.**
   Macro AUC over a resample-dependent set of classes would be a different
   statistic than the point estimate. `n_resamples` therefore reports how many
   resamples the metric was actually defined in — a split with a one-image class
   visibly loses ~37% of them.

**Verified on real data, not only synthetically:** one epoch of resnet18 on
USK-COFFEE (4 classes, 1600 test images) on the GPU gave accuracy
`0.7506 [0.7294, 0.7713]`. The analytic binomial interval for p=0.7506 at
n=1600 is `[0.7288, 0.7712]` — an independent confirmation the bootstrap is
calibrated rather than merely self-consistent. Rendering was confirmed in the
browser against that run (five intervals under the right labels, no console
errors).

---

## ADR-075 — Submissions queue instead of being refused

**Date:** 2026-07-29
**Status:** Accepted — shipped 2026-07-29
**Supersedes:** the "one run at a time, 409 otherwise" contract of ADR-016

**Context:** the GUI accepted one run at a time and answered `409 An experiment
is already running.` to anything else. The machine genuinely has one GPU, so
running one at a time is right — but *refusing the submission* forced the
researcher to sit at the keyboard and resubmit the moment each run ended. The
obvious overnight workflow (line up the evening's experiments and walk away) was
impossible for anything the sweep/replicates orchestrators do not already
parametrize, which is exactly the heterogeneous case: a detection run, then a
segmentation run, then the same classifier on a different dataset.

Worth being precise about the gap, because it is narrower than "there is no way
to queue work": sweeps, replicates, K-fold and comparison already run N
trainings from one submission. What was missing is queueing *unrelated* jobs.

**Decision:** `gui/api/run_queue.py` holds a FIFO of submissions and drains it
one job at a time. A submission returns `{"run_id", "status"}` where status is
`running` (the GPU was free) or `queued`. `GET /api/queue` lists the active job
and everything waiting; `DELETE /api/queue/{run_id}` drops a job that has not
started; `/experiment/status` gained a `queued` count.

**Two things had to change beyond "keep a list", and they are why this needed an
ADR rather than a few lines in the route layer:**

1. **A finished run's result has to survive the next one starting.** The route
   module kept exactly one `_current_run` dict and `/experiment/result/{run_id}`
   read straight from it. With jobs running back to back, job 2 overwrites job
   1's report before the browser fetches it — the report would simply be gone.
   Terminal snapshots are now recorded per `run_id` in the queue (bounded at 100;
   the durable record is `run.json` on disk).
2. **The SSE queue must be created when a job starts, not when it is submitted.**
   Every starter used to do `_event_queue = asyncio.Queue()` in the request
   handler, which was safe *only* because a second submit was refused. Under
   queueing, submitting job 2 would replace the live queue of job 1 and its
   progress stream would go dead mid-training. Creation moved into the queue's
   `on_start` callback.

**A related bug the queue would have introduced in the frontend, fixed here:**
`useExperiment` polled `/experiment/status` — which describes whatever is
*active* — and treated the answer as its own run. A submission still waiting
would therefore have read another job's completion as its own and fetched the
wrong result. The hook now tracks the run it submitted and only acts on that id;
while waiting it reports position in line, and it attaches the EventSource at
the moment its own job becomes active, so a queued run's live monitor starts by
itself.

**A running job is deliberately not cancellable.** The trainers own their loop
and have no cooperative stop point, so "cancel" would either lie or leave a
half-written run directory. `DELETE /api/queue/{run_id}` answers 404 for a job
that already started, and says so in the message.

**The queue is in-memory and not persisted.** It describes what this server
process is about to do; losing the pending list on restart is correct rather
than a gap. Persisting it would promise to resume work after a crash, which
nothing else in the design supports.

**One bad job must not strand the batch.** The drain loop catches anything the
executors let escape, logs it, and continues — a queue that stops at the first
failure would defeat the purpose of submitting several jobs unattended.

**Rejected: per-run event streams** (`/experiment/events?run_id=`). Cleaner in
principle, and it would let a client watch a job that is not the active one, but
it rewrites a contract five task panels and every strategy already consume, for
a case that does not arise while exactly one job can run. The single active
stream stays; the hook attaches when its turn comes.

**Contract change, called out:** a busy server no longer answers 409 to a
submission. Thirteen route tests asserted that and now assert `status: "queued"`
instead; the 409 branch is kept in the frontend for an older server. The
per-endpoint validation errors (422 for a bad config, 404 for an unknown custom
task) are unchanged and still happen before anything is enqueued.

**Brick 2 (2026-07-29) — the queue's own surface.** `QueueOverlay` sits beside
History and Datasets, and the bottom-bar button appears **only when something is
waiting**: a permanent "fila 0" would be a standing reminder of a queue most
sessions never form. It lists the active job and the pending ones in order with
task and strategy translated, how long each has waited, and a 🗑 per pending job;
the active row says "sem cancelar" and explains why in its tooltip.

Found by running it rather than by reading it: the badge went **stale**. The
count was seeded once on mount, and a tab that had submitted nothing had no run
of its own to poll — so cancelling inside the panel fixed the list and left the
number wrong. The overlay now reports its count upward (`onCountChange`, the
pattern `HistoryOverlay` already used) and the app runs a 5s poll that exists
only while the badge is showing and this tab has no run of its own, stopping by
itself at zero. A failed read keeps the last known depth instead of flickering to
zero, because ambient information that lies briefly is worse than information
that is a few seconds late.

---

## ADR-076 — The bootstrap interval reaches the other four tasks

**Date:** 2026-07-30
**Status:** Accepted — shipped 2026-07-30
**Extends:** ADR-074 (bootstrap CI for classification)

**Context:** ADR-074 gave classification `0.8734 [0.8412, 0.9021]` and left
regression, segmentation and anomaly reporting bare numbers, because their
`evaluate` returned only aggregates from a streaming accumulator — there were no
per-sample arrays to resample. A researcher comparing a segmentation run against
a published mIoU has the same need as one comparing an accuracy.

**Decision:** one entry point per task family in `core/metric_ci.py` —
`bootstrap_regression_cis` (MSE/RMSE/MAE/R²), `bootstrap_anomaly_cis`
(AUROC/F1) and `bootstrap_segmentation_cis` (mIoU/Dice/pixel accuracy) — each
wired into its block and written to `run.json` under the same `metric_cis` key
the GUI already reads. Each trainer gained a sibling of `evaluate`
(`evaluate_with_predictions`, `evaluate_with_scores`, `evaluate_with_confusion`)
that returns the aggregates **and** the arrays from one pass, with `evaluate`
delegating to it — so no caller changed and a report can never show a metric
computed differently from its interval.

**The image is always the resampling unit, never the smaller thing inside it.**
This is the decision that matters:

- **Regression** resamples rows, not the flattened (sample × target) elements the
  metrics pool over. A multi-target sample's columns come from one image and move
  together; drawing them independently would understate the spread.
- **Segmentation** takes one K×K confusion matrix *per image* and sums the drawn
  matrices. Resampling pixels would be arithmetically easy and statistically
  wrong: pixels inside one image are not independent draws, and the interval
  would come out far tighter than the evidence supports. Passing matrices instead
  of masks also keeps the cost at K×K per image rather than one array per pixel.
- **Anomaly** holds the decision `threshold` fixed instead of recomputing it per
  resample: it is derived from the *normal training* score distribution, which
  makes it part of the trained detector, not of the test sample being varied.

**Each entry point is pinned against the accumulator that actually produces the
reported number**, not against a textbook formula — `_MetricAccumulator`,
`SegmentationMetricAccumulator`, and the anomaly trainer's own F1/AUROC. An
interval computed by a second implementation is worse than no interval if the two
disagree, because it would bracket a value the report never shows. That test
immediately earned itself: the first segmentation implementation averaged IoU
over the classes present in *ground truth*, while the accumulator averages over
classes present in ground truth **or** prediction — a silent disagreement that
the parity test failed on.

**Found in the browser, not in review:** segmentation's `run.json` carries both
`miou` (the best *validation* score) and `test_miou`, and the frontend lookup
stripped a `test_` prefix when present but otherwise matched the bare name — so
the validation number was rendered with the *test* split's interval. `metricCi`
now requires the prefix, since every entry in `metric_cis` is by construction a
test-split measurement. Classification never exposed this because its metrics
block has no bare metric name to collide with.

**Verified on real datasets on the GPU**, one epoch each, point estimates checked
against the reported metric in every case: segmentation mIoU
`0.5526 [0.5255, 0.5806]` (Oxford-IIIT Pets, 100 test images), regression MAE
`18.42 [17.19, 19.72]` (wiki age, 400), anomaly AUROC
`0.7738 [0.7411, 0.8036]` (coffee, 800). Regression's R² came out negative with
its interval entirely below zero — correct for a model worse than predicting the
mean, and worth noting as a reminder that these intervals do not assume a
metric's sign.

---

## ADR-077 — The post-training surface stops being classification-shaped

**Date:** 2026-08-02
**Status:** Accepted — shipped 2026-08-02
**Extends:** ADR-043 (task-aware history), ADR-053 (Grad-CAM beyond classification)

**Context:** classification was built first and the surfaces around a *finished*
run quietly assumed its shape. Exercising every per-run action against a real
trained run of every task — rather than reading the code — turned up three
separate versions of the same mistake, two of them crashes users would hit on
their first try.

**1. The model card 500'd for six of the eight run types.** `_render_run_markdown`
hardcoded the classification epoch columns and formatted each with `:.4f`. A
missing key fell back to the string `"?"`, and `f"{'?':.4f}"` raises
`ValueError: Unknown format code 'f'` — so the guard could never have worked, and
detection, regression, segmentation, anomaly and both custom tasks all returned
500 from the "↓ markdown" button. Columns now come from the history itself, which
is the only thing that knows what a task streams.

**2. A researcher-defined task's run crashed all four actions.** `test`,
`gradcam`, `batch_predict` and `export_onnx` read `config.task` to dispatch — but
a custom run.json has no `config.task` and no `config.model`, because the task
owns those. Dispatch fell through to the classification default, tried to rebuild
a ResNet, and died inside `load_state_dict` with a raw key mismatch shown to the
user as a 500. The task's key has always been at the *top level* of run.json
(ADR-013), which is what makes an honest refusal possible: the actions now answer 400
naming the task and saying why.

**3. Only classification had test-set diagnostics.** It renders six plots;
regression and segmentation rendered a loss curve, anomaly an AUROC curve. So the
tasks whose numbers are hardest to interpret had the least to interpret them
with. Each now gets the diagnostic its own literature expects:

| task | added | what the aggregate metric cannot say |
|---|---|---|
| regression | predicted-vs-actual scatter, residual histogram | whether the model tracks the target or just predicts its mean; whether the error is bias or spread |
| segmentation | per-class IoU bars, confusion matrix | which class is never found, and what it is confused with |
| anomaly | score histogram with the threshold | where the two populations overlap, and what the chosen cut actually keeps |

The arrays these need already existed: ADR-076 made every trainer return
per-sample predictions alongside its aggregates, so this is wiring, not new
inference.

**4. Grad-CAM now shows the true class next to the predicted one.** A heatmap
answers "where did it look" and left "was it right?" to the reader. `run.json`
never stored class names, so both sides were unavailable: the caption said
`classe predita: 2`. Names are recovered from the training folder — `ImageFolder`
assigns indices by sorting the class sub-directories, so the sorted names *are*
the mapping the checkpoint was trained with — and the ground truth from each
image's parent folder.

**Neither is guessed.** A class-count mismatch between the folder and
`model.num_classes` yields no names rather than a confidently wrong one, and a
flat folder of unlabeled images gets `true_class: null` instead of a fabricated
label. The GUI outlines a wrong prediction in red, because the overlays worth
opening are the mistakes.

**Verified on real runs, not fixtures:** the eight-target action matrix went from
6 model-card 500s and 8 custom-task 500s to `ok` and clean 400s; regression,
segmentation and anomaly were retrained and their new plots inspected. The
regression scatter immediately explained a negative R² from an earlier run — a
flat cloud at ~20 for every true age, which is the "predicts the mean" failure
MAE and RMSE report identically to an unbiased one.

**Still classification-only, and left that way deliberately:** per-model `test`
on a folder, for regression/segmentation/anomaly. Those refuse today with a raw
Pydantic error about `ExperimentConfig.task` instead of a plain "not supported"
— the refusal is correct, the wording is not. Fixing it properly means a native
evaluator per task, which is a feature, not a message change.

---

## ADR-078 — The interface stops hiding what it was asked to show

**Date:** 2026-08-03
**Status:** Accepted — shipped 2026-08-03

Three user reports with one shape: a surface that discards or covers something
the researcher explicitly asked for.

**1. A grid axis could not reach a scheduler's dependent parameters.** The
scheduler card decides which fields to render from the *scalar* `kind`. Putting
`step` on the grid axis while the scalar stayed `none` left `step_size` and
`gamma` unrendered — so the sweep ran them on their defaults, with no way to see
or sweep them. The shown set is now the union over the scalar kind **and every
kind on the axis**.

Checked whether the same shape exists elsewhere, since a conditional group is
exactly the thing that breaks under a sweep. Only three exist: the scheduler,
transfer-learning's `unfreeze_from_layer`, and the anomaly panel's
autoencoder-vs-PatchCore fields. Neither of the other two can collide with a
grid axis — transfer learning is a *block*, mutually exclusive with grid search,
and the anomaly panel sweeps through the dot-path SweepCard, which has no reveal
step to get wrong.

**2. The lightbox covered the plot it existed to show.** An earlier fix moved the
caption and close button to fixed overlays "so they never steal vertical space
from the image" — which put them *on top of* it, over the strip matplotlib uses
for the x-axis and legend. Neither obvious layout works: a fixed `85vh` image
clips tall plots on short viewports, and floating the chrome occludes them. It
is now three flex rows with the image row absorbing the remainder
(`min-height: 0`, without which a flex item refuses to shrink below its content
size), so nothing is covered and nothing is cut.

**3. Clicking outside threw away the whole navigation stack.** From a plot,
inside a run, inside the history, one stray click outside the sheet returned to
the training screen — discarding two levels the user had navigated into. The
backdrop now pops **one** level: plot → run detail → list → closed. Esc mirrors
it, and the lightbox stands down while it is the top layer so one key press
closes one thing. The header's × still closes everything at once, which is the
distinction that makes both useful: the backdrop is a stray click, the × is a
decision.

**Verified by driving the real GUI**, not by reading the components: the layer
trail reads `grafico → treinamento → historico → gui` for both Esc and the
backdrop, the × from a run detail lands on the GUI in one step, and adding
`step` to the scheduler axis makes STEP SIZE and GAMMA appear with their own
grid buttons.

**A fourth, found by the GUI sweep this ADR triggered:** an unknown
preprocessing filter came back as `500 Internal Server Error`. A researcher
typing "blur" into the filter list — a reasonable guess for what is spelled
`gaussian_blur` or `median_blur` — got a server fault and no hint. That is bad
input, so it is now `422`, and the message lists every registered filter.

**Not verified visually — stated because it matters:** the lightbox geometry.
The browser pane in this environment reports a 0×0 viewport and does not
composite, so every `getBoundingClientRect` returned zero. The layout is
reasoned from the CSS and wants a human eyeball before it is trusted.

---

## ADR-079 — Detection and custom runs get their artifacts back

**Date:** 2026-08-04
**Status:** Accepted — shipped 2026-08-04

**Context:** ADR-077 standardised the *plots* across regression, segmentation and
anomaly and stopped there, which left the two task families the user actually
noticed: a detection run showed no graphics in the history, and neither did a
researcher-defined one. Auditing every `run.json` on disk made the gap
countable — classification 74 of 80 runs with plots, detection 0 of 4, custom 0
of 44.

**Two unrelated causes, both confirmed by running a fresh training rather than
by reading the code.**

**1. Ultralytics was writing everything to the wrong directory.** The trainer
passes `project=str(run_dir.parent)` — a *relative* path — and Ultralytics
resolves a relative project under its own `runs_dir` setting. So a run destined
for `outputs/models/<name>/<stamp>/` landed in
`runs/detect/outputs/models/<name>/<stamp>/`, and the real run directory held
exactly two files: `data.yaml` and `run.json`.

That is much worse than missing plots. **The checkpoint was not there either** —
which is the actual reason every post-training action on a detection run
reported "no usable checkpoint at artifacts.model". ADR-077 recorded that as a
stale run from July; it was systematic, and a training run today reproduced it.
`project` is now `run_dir.parent.resolve()`, and the collected list grew to
everything Ultralytics draws that answers a question: both confusion matrices,
all four Box curves, `results.png`, and the validation prediction sample.

**2. The custom-task engine rendered a plot and then declared it did not exist.**
`_render_primary_curve` wrote `<primary>_curve.png` into the run directory and
`_write_run_json` hardcoded `"graphics": []`, so every custom run left an
orphaned PNG on disk that no surface ever showed — 44 of them here. The engine
now returns the paths it wrote and the run.json declares them, plus a train-loss
curve that comes free from the same history and is what every built-in task
plots first.

**Verified on real trainings:** a two-epoch YOLO11n run now leaves 18 files in
its run directory, declares 8 graphics, and its `best.pt` exists — with
`export_md` and `export_onnx` both answering 200 where the latter previously
returned 400. A custom run declares `loss.png` and `auprc_curve.png`, both
present.

**Still not at parity, stated rather than quietly skipped:** detection and
custom runs carry no bootstrap confidence intervals (ADR-074/076). Both are
genuinely harder rather than overlooked — Ultralytics reports aggregate mAP with
no per-image predictions to resample without re-running validation, and a custom
task declares arbitrary metric names whose resampling unit only its author
knows. Neither is a one-line wiring job like the plots were.

---

## ADR-080 — "Testar modelo" takes one labelled folder

**Date:** 2026-08-04
**Status:** Accepted — shipped 2026-08-04
**Implements:** ADR-047, which decided this and was never built

**Context:** the per-run test action asked for a `base_dir` plus three split
names — `train`, `val`, `test` — to score a checkpoint. That is a description of
a *training layout*, and the researcher supplying it has already trained; what
they have is one evaluation set. Worse, the request was built into an
`ExperimentConfig` regardless of task, so regression, segmentation and anomaly
runs answered with a raw Pydantic error — `Input should be 'binary' or
'multiclass'` — and the action only ever worked for classification and
detection.

**Decision:** the request is one field, `data_dir`, and the endpoint dispatches
per task.

The folder is given **in the label shape the run was trained with** — class
sub-folders for classification, images plus YOLO `.txt` for detection, the
paired image/mask folders for segmentation, normal and defect folders for
anomaly. Everything else (which sub-folder holds masks, what the normal class
is called) is read from the run's own config instead of being asked again: the
model was trained under those conventions, and a test set that does not follow
them is not comparable anyway.

**Regression is the exception its data model forces.** It has no split folders —
it has CSV manifests — so `data_dir` points at the `.csv`, and choosing a folder
is refused with a message that says what to pick instead. The GUI's field
relabels itself and the hint under it names exactly what the chosen path must
contain, per task.

**No new evaluators were written.** Each task's trainer already knows how to
score a loader; the only thing missing was a way to aim its *test* split at the
chosen folder. For the folder-shaped tasks that is `base_dir=<parent>` +
`test_dir=<name>`; regression gets `test_csv=<file>`. The existing DataModules
then do the loading, which keeps the numbers identical to the ones training
reports.

**A folder of bare images is deliberately not accepted.** Metrics need labels;
without them there is nothing to be right or wrong about. Running a checkpoint
over unlabelled images is a different action that already exists — batch
prediction, which writes a CSV.

**Validation order matters and is chosen, not incidental:** the path the
researcher just typed is checked before the run's checkpoint, because
"Run X has no usable checkpoint" is a confusing answer to "I picked the wrong
folder". A test caught the original order.

**Verified on real data**, all four families through the same entry point:
classification `accuracy=0.7506` on the coffee test folder — identical to what
that run reported at training time, which is the consistency check that matters
— plus regression `mae=15.32` from a CSV, segmentation `miou=0.4122` from a
paired folder, and anomaly `auroc=0.601`. The three standalone ones had never
completed this action at all.

---

## ADR-081 — Data modules own their DataLoaders

**Date:** 2026-08-05
**Status:** Accepted — shipped 2026-08-05

**Context:** a classification run died on Windows with

```
OSError: [WinError 1455] O arquivo de paginação é muito pequeno para que esta
operação seja concluída. Error loading "...\torch\lib\curand64_10.dll"
```

The message points at torch, and at a CUDA DLL, and at neither of the two things
that actually mattered. Inspecting the machine while it was failing found the
`visionforge gui` process holding **19 orphaned worker processes from two earlier
attempts**, together about 22 GB of commit charge, against a commit limit of
58.6 GB that was already 95% consumed. Windows' auto-managed paging file had
also shrunk back to a smaller size once the pressure passed, so it could not grow
quickly enough for a burst of spawns.

Each DataLoader worker is a **separate process** — Windows spawns, it does not
fork — so every one of them re-imports torch and loads the CUDA DLLs. The cost
is roughly a gigabyte per worker, and it is paid per loader, not per run.

**Two defects were found, and they are not the same defect.**

**1. Every `*_loader()` call built a new DataLoader.** Calling `train_loader()`
twice gave the same split two independent worker pools. `AnomalyBlock` does
exactly that — it passes `data.train_loader()` to the trainer and again to the
scorer. Splits are now built once and cached, so the second call returns the
same object.

**2. Nothing ever stopped a pool.** A `persistent_workers=True` pool lives until
the DataLoader's iterator is garbage-collected, which is fine in a script and not
fine in `visionforge gui`, where a failed run's traceback keeps the frames — and
therefore the loaders — alive. Each data module now has `close()`, and every
block calls it in a `finally`.

**The measurement that shaped this, including the part that disagreed with the
first guess.** A probe that counts live child processes shows 8 workers during a
classification run and 0 after, for both a completed run and one forced to raise.
But running the *same probe against the pre-fix code* also showed 0 left behind:
in a standalone script, refcounting already collected the loaders. So the
teardown is not what rescued the reported failure, and claiming otherwise would
be wrong. The orphaning in the GUI happens when the spawn itself fails partway —
some workers are already up, the iterator is never assigned, and no reference to
them exists for anyone to clean up. That is a cascade: the first failure strands
workers, and the stranded workers make the next attempt fail sooner.

Which is why the third change is the one a user actually feels:

**3. The error explains itself.** WinError 1455 is now translated into what it
means and what to change — `data.num_workers` first, the Windows paging file
second — instead of a DLL name.

**Also reduced: the peak.** The test split asked for `persistent_workers=True`
despite being read exactly once, so a run kept a third pool alive for no reason.
It does not any more, which takes a classification run from 12 worker processes
to 8.

**Not done:** capping `num_workers` from available commit at run start. It would
need a per-platform estimate of the per-worker cost, and guessing it wrong either
throttles a machine that was fine or fails to save one that wasn't. The error
message tells the researcher which knob to turn; they know their machine.

---

## ADR-082 — The dataset of each run appears in the history

**Date:** 2026-08-05
**Status:** Accepted — shipped 2026-08-05
**Extends:** ADR-061 (dataset fingerprint), ADR-078 (history overlay)

**Context:** the history listed the experiment name, the architecture, the task,
the epochs and the metrics — and never said which dataset the run was trained on.
Comparing two rows meant opening each run and reading its config.

**Nothing new is measured.** Both sources were already in `run.json`:
`config.data.base_dir`, written by every run since the beginning, and
`dataset_fingerprint` (ADR-061), written since 2026-07-26.

**Decision:** `dataset_identity(run_json)` derives `(name, path)` with the
precedence `dataset_fingerprint.root → config.data.base_dir → None`, the API
exposes it on `RunSummary` and `RunDetail`, and three places render it: a
`🗂 <name>` badge on the history card, a "Dataset" block in the run detail, and a
same-data verdict when comparing runs.

**The fallback is the whole point.** Reading only the fingerprint would put the
badge on the handful of runs written after 2026-07-26. Reading `base_dir` puts it
on 69 of the 78 runs in this researcher's history.

**The consequence the design must not hide:** recognising and re-finding are
retroactive; *verifying* is not. A path proves nothing about the bytes, so two
old runs pointing at the same folder cannot be shown as "same data" — the
comparison reports `⚠ não verificável`, with the reason, rather than guessing.
`same_dataset` already returned `None` for exactly this, including when the two
runs used different fingerprint methods; the UI now surfaces that third answer
instead of collapsing it into a yes or a no.

**`PureWindowsPath`, not `PurePath`.** Runs are written on Windows and the tests
also run on Linux in CI, where a backslash path has no separator and the entire
string would become the "name".

**A caveat that measurement resolved.** The open question was what a custom task
that *synthesizes* its data would show, since it treats `base_dir` as a marker.
Measured: `custom:example_counting` writes `base_dir: "."`, whose last segment is
empty, and the `name or None` guard already leaves those 9 runs with no badge —
the right outcome, with no special case. Meanwhile `custom:vlm_pseudo_label`
points at a real folder and gets a badge, so "custom task" does not imply "no
dataset".

**Deliberate duplication.** The comparison rule is four lines and is rewritten in
TypeScript rather than served by an endpoint; `CompareRunsPanel` already fetches
each run's detail. What had to survive the translation is the "cannot tell"
branch, which is the common case rather than an edge one.

**Not done, and why.** Grouping the history by dataset is a navigation decision
worth making on its own. Backfilling fingerprints onto old runs was refused: the
hash of the folder *today* does not describe what it held *then*, so it would
answer "same data" about two runs that saw different data — the exact error the
fingerprint exists to prevent.

---

## ADR-083 — Data augmentation has an explicit on/off flag

**Date:** 2026-08-05
**Status:** Accepted — shipped 2026-08-05
**Extends:** ADR-059 (canonical task-panel contract)

**Context:** there was no way to turn augmentation off short of zeroing every
field by hand. In detection that is 15 fields, so "run a baseline without
augmentation" was work nobody did.

**Decision:** `augment: bool = True` on `TransformConfig` (classification,
regression, segmentation, anomaly) and on `DetectionAugmentationConfig`.
`_build_transforms` gates its `is_train` branch on it; the Ultralytics kwargs go
through `_augmentation_kwargs`.

**A flag, not inference from the values.** The alternative was to treat "all
knobs neutral" as off and write neutral values when switching off. Rejected
because the `run.json` then only *implies* the state — wrongly, for anyone who
zeroed a field for another reason — and because the researcher's tuning would be
destroyed by the round trip. Writing neutral values into an exported YAML loses
the tuning permanently; the "previous values" would live only in browser state.
Detection makes that concrete: 15 values to restore versus one boolean.

**Detection disables by sending neutral values, not by omitting keys.**
Ultralytics fills an omitted `train` argument with its own default, which
augments — so omission would silently do the opposite of what was asked. The
config's own values are left untouched, which is what makes switching back on
restore them.

**Off hides the fields rather than disabling them.** Keeping 15 greyed-out rows
on screen is the visual clutter that prompted the request in the first place. The
count (`15 parâmetros ocultos`) stays visible so the panel does not read as
broken.

**`image_size` and `normalize_*` left the augmentation section.** They are not
augmentation: they apply to train, validation and test alike, which
`_build_transforms` already reflected by keeping them outside the `is_train`
branch. The UI heading "Augmentação & normalização" was the only place claiming
otherwise, and hiding them with the toggle would have said they stop applying.
This is a labelling error being corrected, not a preference.

**Backward compatible without migration.** A new field with a default means an
old config loads as `augment=True` and trains exactly as before. The YAML import
path needed the same rule stated explicitly — `form-import.ts` treats a missing
key as on — which the type checker, not review, is what caught.

**Not done in this ADR:** presets ("leve / médio / agressivo"). Reasonable, and a
separate decision.

---

## ADR-086 — A change under `src/` needs the GUI server restarted

**Date:** 2026-08-05
**Status:** Accepted — shipped 2026-08-05

**Context:** a researcher rebuilt the SPA, reloaded, and saw none of the new
features. The build was correct — the bundle on disk provably contained the code,
and the browser was serving it. The cause was a `visionforge gui` process that
had been running for ten hours.

FastAPI reads `static/` from disk on every request, so a rebuild reaches the
browser at once. Python does not work that way: modules are imported once, at
process start. A server left running across a change therefore serves **new
JavaScript from old Python**. The page asks for fields the stale routes never
send, every guard on those fields is false, and the feature renders nothing.

**What makes this expensive is that it mimics a broken build.** The two natural
responses — rebuild, hard-reload — both appear to fail, because neither is the
problem. It cost the researcher an evening, and cost me a wrong diagnosis
(browser cache) that I only corrected by inspecting the running process.

**Decision: document it, and expose `/api/health`.** CLAUDE.md and the README now
state that a change under `src/` requires restarting the server; the README
previously said to build the web UI "once", which implied a one-time setup step.
`/api/health` reports the version and the SPA bundle the process booted against,
which is enough to diagnose the condition from a terminal:

```bash
curl http://127.0.0.1:8000/api/health
```

**An in-app banner was built and removed the same day.** It compared the bundle
the server booted against with the one the page was running from, and warned on a
mismatch. It worked, and the researcher's reaction to seeing it was to ask what
the message was and to have it taken out. A warning that reads as noise in the
product is worse than the documentation line it was meant to reinforce — the
people who hit this are developing the project, not using it, and the terminal is
where they already are.

---

## ADR-084 — Detection preprocesses by materializing a filtered copy

**Date:** 2026-08-05
**Status:** Accepted — backend shipped 2026-08-05
**Extends:** ADR-059 (canonical task-panel contract), ADR-081 (teardown discipline)

**Context:** detection was the only task without preprocessing filters, and the
reason was not an oversight. With the Ultralytics backend — the default —
`model.train(data=data.yaml)` hands the library the entire data pipeline. A
per-image PIL filter cannot be injected into its loader without subclassing its
internal dataset, which would pin the project to one version of it.

**Decision:** apply the filters once, write the result to a temporary folder, and
point the synthesized `data.yaml` at that folder.

**This is cheaper than the path it replaces, not a workaround with a cost.** The
on-the-fly path filters **per image, per epoch**; this filters once. Over 30
epochs with an expensive filter (CLAHE, bilateral) it is roughly 30x less CPU.

Four properties make it safe rather than a trap, each with a test:

**The fingerprint stays on the original.** If it followed the copy, `run.json`
would record the digest and path of something that was then deleted, and the
history from ADR-082 would show `🗂 a3f9c1…` instead of the dataset name.

**Removed by the context manager, swept at startup.** The `with` covers the
exception path, which is the one that matters — runs die for real (ADR-081). A
process killed outright never runs its `finally`, so the GUI sweeps the cache
directory at startup, where every folder present is by definition orphaned.

**PNG, not the source format.** Re-encoding a JPEG dataset as JPEG would stack
compression loss on top of the filter. PNG of photographic data commonly runs
5-10x larger, so the size is logged before the copy is made and the format is
configurable for anyone who would rather trade fidelity for disk.

**Keyed by content, not by run.** The key is the dataset fingerprint plus the
canonicalized pipeline, so a 20-trial sweep materializes once. Using the
fingerprint rather than the path means re-exporting a dataset to the same
location produces a different key instead of silently reusing a stale copy.

**Labels travel verbatim.** Every non-image file is copied byte for byte;
filtering an image without carrying its label produces a training run that is
wrong and says nothing about it. The extension changes with the format but the
stem does not, which is what YOLO matches on.

**A half-written copy is deleted rather than left.** If the build raises partway,
the folder is removed — otherwise the next run would find it and treat it as
complete.

**Registered consequence:** detection now uses a *different mechanism from
classification for the same feature* — on-the-fly there, materialized here. Two
mechanisms for one feature is a smell, and it is recorded as one. The reason is
specific and does not generalize: Ultralytics owns its pipeline. Migrating the
other four tasks for the CPU saving is a separate decision, deliberately not
taken here, because the on-the-fly path works and is covered by tests.

**Verified on the real dataset** (`cats-dogsv2.v1i.yolov8`): 277 images filtered,
279 label files byte-identical, stems preserved across the `.jpg`→`.png` change,
`data.yaml` pointing at the copy with `names`/`nc` unchanged, the copy removed on
exit, and no orphans left behind.

---

## ADR-085 — Hyperparameters are tiered and explained, not trimmed

**Date:** 2026-08-05
**Status:** Accepted — shipped 2026-08-05
**Extends:** ADR-059 (canonical task-panel contract), ADR-083 (collapsing pattern)

**Context:** the report was *"conheço apenas metade desses; tá bem difícil de
entender, muita coisa junta"* — and, when asked directly, explicitly **not** a
request to remove any of them. The problem is presentation, not count.

**Decision:** every hyperparameter gets a one-line explanation, and the panels
split into a basic tier that is always visible and an advanced tier that starts
collapsed, reusing the toggle pattern from ADR-083.

**The cut is by how often a value changes, not by how much it matters.** The
optimizer matters enormously and is still advanced, because it gets decided once
and then left alone. Epochs, batch size, learning rate and seed are what move
between one experiment and the next, so those four stay on screen.

**Nothing is removed.** An advanced parameter is one click away and travels in
the payload with the same value it always had.

**An unclassified parameter counts as basic.** A field nobody thought to tier
must stay visible rather than disappear by accident — the failure mode of the
opposite default is a knob that silently stops being reachable when someone adds
it.

**A non-default advanced value opens the section.** Hiding a setting the
researcher deliberately chose, or that arrived with an imported YAML, would be
worse than the clutter the collapsing removes.

**The explanations live in a data module** (`lib/param-help.ts`) rather than
spread through JSX, which is what makes "every field is explained" a test
instead of a promise: the suite iterates the tier map and fails on any entry
without text.

**They describe consequences, not definitions.** "Alto demais diverge, baixo
demais nunca chega" is usable by someone who does not already know what a
learning rate is; "taxa de aprendizado do otimizador" is not.

**`num_workers` carries the warning ADR-081 earned.** It is the only knob in the
list whose wrong value does not degrade training but *prevents* it — each worker
is a process reloading torch and the CUDA DLLs, about a gigabyte each, and the
failure surfaces as WinError 1455 naming an unrelated DLL.

---

## ADR-087 — Detection gets a bootstrap interval, at a different price

**Date:** 2026-08-07
**Status:** Accepted — shipped 2026-08-07
**Extends:** ADR-074 (per-run intervals), ADR-076 (the image is the resampling unit)

**Context:** four of the five tasks reported `0.7506 [0.7294, 0.7713]`; detection
reported a bare `map50`. ADR-079 recorded this as genuinely harder rather than
overlooked, and left it there.

**Why it is harder, precisely.** Every other task's metric decomposes. A
classification split is a confusion matrix that can be accumulated per sample
and summed; segmentation is one KxK matrix per image, which is what let ADR-076
resample images cheaply. mAP does neither: it ranks every detection in the split
by confidence and walks a precision/recall curve, so it is a property of the
**set**, not a mean of per-image numbers. There is no accumulator to sum.

**Decision:** recompute the metric on each resampled set. The image stays the
resampling unit, as ADR-076 requires.

**The default resample count drops from 1000 to 200.** A percentile interval at
200 draws is grainier at the tails, and that is the honest price of a metric
that cannot be decomposed. It still separates `0.72 ± 0.03` from `0.72 ± 0.20`,
which is the question the interval is asked.

**A draw that loses a class is discarded, not averaged in.** If a resample
happens to exclude every image containing a class, `mean_average_precision_50`
averages over the remaining classes — a different quantity, not a noisier
estimate of the same one. Those draws are dropped and the reported
`n_resamples` counts only survivors, so a rare class makes that number fall
visibly. That fall is the signal that the interval rests on less evidence than
its width suggests.

**A caveat found by testing, worth writing down.** mAP is order-dependent when
confidence scores tie. A synthetic split where every detection scored 0.9 and
every hit preceded every miss produced an interval sitting *entirely below* the
point estimate — the full set enjoyed a precision curve no resample could
reproduce. That is a property of mAP under ties, not a defect of the bootstrap,
but it means a model emitting constant confidences gets an interval that looks
wrong. Real detectors do not, and the fixture was corrected to interleave hits
and use distinct scores.

**Where it is wired:** the per-model detection test, which already held the
per-image predictions and ground truth it needs. The Ultralytics *training* path
computes mAP inside the library and never exposes per-image detections, so an
interval there would need a second inference pass over the validation set —
deliberately not done here, and the reason detection training still reports a
bare number.

---

## ADR-088 — A running job can be stopped, and keeps what it earned

**Date:** 2026-08-08
**Status:** Accepted — shipped 2026-08-08
**Revises:** ADR-075 (the run queue)

**Context:** ADR-075 refused to cancel a running job, and its reasoning was
sound at the time: the trainers owned their loops and had no point at which
stopping was safe, so "cancel" would either lie about having worked or leave a
half-written run directory behind.

What that left, in practice, is a researcher who starts 120 epochs with one
wrong parameter and whose only recourse is killing the server — which also
destroys the queue behind it. In a tool built to leave an evening of
experiments running, one bad job blocks the whole night.

**What changed is not the mechanism but the availability of a safe point.**
Every trainer already pauses between epochs to write its checkpoint and emit
progress. By then the run directory is consistent, so stopping there costs
nothing. The token is read at the top of each epoch, which is the same instant
as the end of the previous one, and it is read nowhere else.

**Cancelling keeps the work.** The best checkpoint so far, its metrics, its
plots and its `run.json` all survive, and `total_epochs` records how far the run
actually got. The alternative — discarding on cancel — makes a button people
avoid pressing, which defeats having it. A researcher usually cancels because
the curve already answered the question, not because the work is worthless.

**The queue's `cancel` now means "the request was delivered", not "training has
ended".** A pending job still disappears immediately; a running one stops at its
next boundary. The endpoint answers 200 for both, and the docstring says which
is which, because a caller that assumed the process was already dead would be
wrong.

**Ultralytics needed a different lever.** It owns its own loop, so the token is
checked in the `on_fit_epoch_end` callback and sets `trainer.stop` — the same
flag its own early stopping uses. That is an internal of theirs; the comment in
the code says so, because a version bump could move it.

**Verified on a real training rather than a mock:** a 20-epoch classification
run cancelled after epoch 3 stopped at the top of epoch 4, reported
`total_epochs=3`, and left a 43 MB checkpoint from its best epoch plus a written
`run.json`.

**Still not done:** resuming a cancelled run. That needs optimizer and scheduler
state in the checkpoint, not just weights, which changes the checkpoint format
and deserves its own decision.
