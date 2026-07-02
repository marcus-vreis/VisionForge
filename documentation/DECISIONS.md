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
**Status:** Proposed (design in `documentation/CUSTOM_TASK_PLAN.md`)
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
only. A `CustomTaskRunner` adapter gives every custom task comparison, sweeps
and multi-seed replicates for free. `visionforge new-task <key>` scaffolds the
commented template.

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
audit + per-brick record in `documentation/PANEL_PARITY_PLAN.md`)
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
