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
