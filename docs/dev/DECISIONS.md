# Architecture Decision Records

How to read and write this file.

**When an ADR is required.** Write one when a change picks between real alternatives that a later reader would otherwise re-debate: a new dependency or a removed one, a public contract (CLI, `run.json`, task or model extension point), a module boundary, a statistical or reproducibility rule, or a change to the repo's own process. A bug fix, a refactor that keeps the boundaries, and a doc edit do not need one.

**Fields.** Copy `docs/dev/ADR-TEMPLATE.md`. Each entry is `## ADR-NNN — Title`, then `**Date:**`, `**Status:**`, optionally `**Relates to:**`, then Context, Decision, Consequences and Alternatives considered (the older entries use Decision and Reason). Numbers are sequential; take the next free one and append at the end of the file.

**Status.** The line after `**Status:**` starts with exactly one of these:

- `Proposed`: written, not yet agreed.
- `Accepted`: in force. A note after a dash is allowed (`Accepted — shipped 2026-08-05`).
- `Superseded by ADR-NNN`: replaced as a whole by a later ADR.
- `Superseded in part by ADR-NNN`: still in force except for the part the later ADR replaces; say which part.
- `Amended by ADR-NNN`: still in force, with a change recorded in the later ADR.
- `Rejected`: considered and not adopted; kept so the question is not reopened.

An accepted ADR is not rewritten. To change a decision, add a new ADR and move the old one's status to one of the `Superseded` or `Amended` forms. Older entries are not rewritten to this layout; `tests/test_decisions_format.py` checks the heading and the status prefix of every entry.

---

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

**Update (2026-06-04):** added an `environment` block to `run.json` (`utils/environment.capture_environment` → Python/platform/torch/torchvision/numpy/visionforge versions). The config records *what* was requested; the environment records *what actually ran it* — both are needed to truly reproduce a result when library versions change numerical behavior. It is an additive field (legacy run.json without it still loads — `RunDetail.environment` defaults to `{}`) and surfaces in the GUI `RunDetailPanel`. Version probing never raises (`"unknown"` fallback), so capturing it can't fail a run.

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

**Update (2026-06-01):** the deferred mAP module landed — `core/detection_metrics.py` (`mean_average_precision_50`, VOC all-points AP via `torchvision.ops.box_iou`, no extra deps). The torchvision loop now selects the best checkpoint by **validation mAP@50** and records `map50` per epoch in `run.json`; validation loss is kept as a secondary logged signal. SSD/RetinaNet head replacement also landed (`_build_ssd` /
`_build_retinanet` in `models/detection_factory.py`, covered in
`tests/models/test_detection_factory.py`), so nothing from this ADR is open.
*(Corrected 2026-08-13: this line still called it a follow-up long after it
shipped, which is how a closed item stays on a backlog.)*

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

**Reason:** Reproducibility is the project's core value — *"which exact config produced this result?"*. Once real experiments accumulate, a breaking change to a required config field would silently invalidate or fail to load saved configs with no traceability. Establishing the version field + a migration hook **now, while the schema is still small** ("congele cedo"), means later schema changes migrate old configs forward deterministically instead of breaking them, and the newer-version guard prevents a stale build from silently misreading a config from a future release. The migration is centralized in one pure, tested function so each future bump is a single, isolated, verifiable step.

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

**Correction (2026-10-09):** uniform only while every ranking metric was
higher-is-better. The API lets the researcher rank by `rmse`, `mae` or `loss`,
and a custom task may declare its metric "lower"; the descending sort then
named the worst model best, and the sweep (ADR-045) did the same. Rankings now
take the direction from the task's declaration, else from `infer_direction`,
the rule ADR-061 introduced for the replicated comparison
(`runner_metric_direction`, `rank_by_metric`), and the reports carry
`metric_direction`. A unit without the metric ranks last instead of as 0.
Backend-first per the Phase-5 norm; the GUI surface and the generic sweep/
batch-predict are the next ADR-041 slices.

**Addendum (2026-11-25): the classification block compares architectures only, and never crowns a NaN.** `ModelComparisonBlock` swaps only `model.name` per trial, but `ModelFactory` lets `model.timm_model` and `model.custom_model` win over it, `model.weights_path` loads one checkpoint into every architecture (and turns ImageNet weights off), and `classification.mode` other than `train` trains nothing: each would publish the same model under N architecture names in `ranking.csv`. The `ExperimentConfig` validator now refuses them when `block` is `model_comparison` (`config.model_comparison_fixed_model`, PT and EN, naming the field). Refused rather than cleared per trial, because clearing would run a config other than the one written and hide that the field was ignored. The same fields stay valid in every other block. The ranking maps a non-finite metric to "missing" before `rank_by_metric`, since a NaN compares False both ways and `sorted` could leave it first; the History group's `best_index` is the first trial with a finite metric, or none (this also stops crowning a trial whose metric is missing when all are).

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

**Amended 2026-10-16: the classification grid trains every trial on the base seed.** `GridSearchBlock` gave trial *i* the seed `base + i`, so a gap between two grid points mixed the hyperparameter's effect with the seed's. All trials now share `training.seed` and `best_config.yaml` writes it back; the spread between seeds is what replicates measure (ADR-056). Random search keeps one seed per trial.

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

**Addendum (2026-11-25): the classification K-fold shares the rule.** `CrossValidationBlock._FoldDataModule.train_loader` had no `drop_last`, so at `image_size: 32` a stride-32 backbone with BatchNorm (resnet18) failed on every fold whose train size is 1 modulo the batch size, and the K-fold averaged only the surviving folds. The rule moved to `_cv_common.fold_drop_last(n_train, batch_size)` (`n_train > batch_size`) and both K-folds call it. As in the task K-fold, the trailing partial batch is dropped whatever its size; the loader shuffles each epoch, so a different remainder is left out each time. Classification K-fold numbers of folds with a remainder change slightly.

## ADR-051 — timm as an optional model source

**Date:** 2026-06
**Status:** Accepted
**Extends:** ADR-005, ADR-048/049
**Superseded by:** ADR-106 (timm ships with VisionForge).

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
study continues. *(Corrected 2026-10-09: the study now minimizes a
lower-is-better metric, the same direction the trials are ranked in; see the
correction under ADR-044.)* Optuna is a new optional extra (`[optuna]`), imported lazily.
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
**Superseded by:** ADR-106 (dependencies bundled).

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

**Update (2026-11-10, amendment, #190):** the Classification tab now offers
**Replicates** on its strategy step, launched through the existing
`POST /api/classification/replicates`; the backend did not change. It is a state
of the GUI, not a block: `config.block` stays `"classification"` (the replicate
runner trains each seed with the plain `ClassificationBlock`), so transfer
learning is locked while it is picked, as with the multi-run blocks. The card is
the shared `ReplicatesCard`, headline metric `accuracy`. Colab (ADR-163) trains
one `config.block`, so its train button is off, with the reason shown, while
Replicates is picked.

**Amended 2026-11-25: repeated seeds are refused in the core.** The GUI and the CLI already refused `[42, 42, 7]`, but `run_replicates` and `run_replicated_comparison` did not: the aggregate counted one sample twice (n=3, a narrower interval by pseudo-replication) while `VariantResult.per_seed` kept one value per seed, so the same variant showed mean 0.84 in its aggregate and 0.86 in its paired test. Both now raise `ValueError` (`replicates.seeds_unique`, `compare.seeds_unique`, pt/en) before any training.

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

**Amended 2026-10-17: one run button, and fields that line up.** A visual review of
the regression, segmentation, detection and anomaly panels against classification
found them drifting from the template. The K-fold, sweep and replicates cards each
had a run button of their own besides the bottom bar's, which already ran the same
strategy; the cards now show what they will run ("5 seeds", "3 trials") or why they
cannot, and that reason turns into a warning when the bottom button is pressed, so
the one button never fails silently. `FieldLabel` no longer lets a hint wrap the
label onto two lines (the hint is cut with an ellipsis and whole in its tooltip),
which had pushed fields below their row; segmented controls keep each option on one
line; the experiment name and the strategy control span the card as in
classification; the preprocessing panel no longer draws a second frame inside the
task card; the regression target count is a field-shaped box.

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

**Amended 2026-10-16: the t interval is clipped to the metric's domain.** The n=2 example above (`[-0.70, 3.76]` for a strictly non-negative MAE) was documented but still shipped: with 3 seeds a mean-squared error showed a negative low end and an accuracy a high end above 1. `aggregate_replicates` now clips `ci95_low`/`ci95_high` to `metric_domain(name)` (errors and losses >= 0; accuracy, precision, recall, F1, IoU, Dice, AUC and mAP within [0, 1]; R2 <= 1; any other name unbounded) and reports `ci95_clipped`. Clipping was chosen over swapping in the bootstrap because `ci95` keeps one meaning (the t interval) and the bootstrap stays beside it unchanged; the domain is matched on the metric's name, so a custom metric with an unknown name is never cut on a guess.

**Amended 2026-11-25: the name's domain yields to the values, and a non-finite value is not a measurement.**

- *Domain.* The name is a guess. A custom task reporting accuracy in percent (84, 85, 86) had its interval cut to `[82.52, 1.0]`, inverted and excluding its own mean; a negative `delta_f1` was cut the same way. When any observed value lies outside `metric_domain(name)`, nothing is clipped (`_observed_domain`). With every value inside, the mean is inside too, so a clipped interval still holds it and is never inverted. `reproduce.py` still clips its prediction interval on the name alone (backlog).
- *NaN and inf.* A test split with a NaN target gives a finished run with `rmse = nan`. That crashed `aggregate_replicates` (`statistics.stdev`) after every seed had trained, and in the helpers a NaN led the ranking (`sorted` leaves it where it is) and stopped the Holm step-down. **Decision: the seed leaves that metric, it does not fail the job.** `aggregate_replicates` leaves it out like a null and counts it in `n_non_finite` (a metric with no finite value keeps an entry with `n = 0` and null statistics); `VariantResult.per_seed` leaves it out of the variant's tests and of the ranking, the report names it in `non_finite_seeds` (`{variant: [seeds]}`, empty when clean) and the log warns; `paired_comparison` treats a non-finite value as a missing seed on that side; `holm_correct` orders a NaN p last, never flags it and keeps it in the family size; `_effect_label` returns `undefined` for a NaN `d_z`. Failing the whole comparison was rejected because one bad secondary metric would discard every other measured number, which is the same outcome the crash had; dropping the seed silently was rejected because a diverged seed is evidence about the variant, so the count stays in the artifact. A seed lost this way is like a failed one (ADR-182 already fails a run whose validation goes non-finite): the comparison is paired on the seeds that remain.

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

**Amended 2026-10-16: in a checkout, the version is the one in `pyproject.toml`.** Point 2 still holds for an installed package, but the metadata of an editable install is written once, at `pip install -e .`, and goes stale after a pull or a bump: the GUI then compared the old number with PyPI and offered "update available, git pull" for good. `visionforge.__version__` now takes the `version` of the checkout's `pyproject.toml` (`utils.doctor.checkout_version`, only when the package sits in this project's `src` layout) and falls back to the metadata everywhere else.

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

**Addendum (2026-11-25): binary keeps class 1 as the positive class without probabilities.** With `y_proba_full` absent, `n_classes` came from the largest label, so a binary split with only class 0 gave `n_classes = 1` and precision/recall/F1 scored class 0 (`1.0 [1.0, 1.0]`) where the Evaluator (sklearn `average="binary"`) reports `0.0`. `bootstrap_classification_cis` now takes at least 2 classes for `task="binary"`, as `bootstrap_anomaly_cis` already did.

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

**Amended 2026-10-16: segmentation's `run.json` names the split of the validation
score.** The block carried `miou`, `dice` and `pixel_acc` (the best-epoch *validation*
score) beside `test_*`, while the History card's `miou` was the *test* score, so one name
meant two splits. New runs write `val_miou`, `val_dice` and `val_pixel_acc`; the card
projection and the comparison table read `val_<name>` first and fall back to the bare
name, so a run written earlier lists as before. The bare names are no longer written.

**Addendum (2026-11-25): a non-finite target is refused at reading, and R² is never invented.** `float()` accepts `nan` and `inf`, so one such cell in a regression CSV reached the metrics. The constant-target guard (`ss_tot > 1e-12`, else 0.0) read a NaN `ss_tot` as "no variance": the test R² came out `0.0` and its interval `[0.0, 1.0]` while MSE was NaN. Both `_regression_metrics` and `_MetricAccumulator` now return NaN R² when `ss_tot` is not finite (parity kept, and NaN predictions already gave NaN), and `RegressionCsvDataset` refuses a non-finite target naming the file, line and column (`regdata.non_finite_target`, PT and EN). Refusing inside `bootstrap_regression_cis` was rejected: it runs after training, so it would fail a finished run instead of reporting the NaN that the accumulator reports.

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

**Decision: document it, and expose `/api/health`.** `CONTRIBUTING.md` and the
README now state that a change under `src/` requires restarting the server; the README
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
**Status:** Accepted — trainers shipped 2026-08-08; the GUI never delivered the
token until ADR-094
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
and deserves its own decision. *(Done in ADR-092/093.)*

**Correction (2026-08-13):** the verification above was run against the trainer
directly. The GUI path was never wired — the queue created a token and cancelled
it, and no block ever handed it to a trainer, so the button reported success and
training ran to the end. Fixed in ADR-094.

---

## ADR-089 — A finished run announces itself

**Date:** 2026-08-08
**Status:** Accepted — shipped 2026-08-08
**Extends:** ADR-075 (the run queue), ADR-088 (stopping a run)

**Context:** the tool exists to let a researcher queue an evening of experiments
and walk away. Until now the only way to learn that a run finished was to come
back and look, which quietly undoes the point of the queue.

**Decision:** two channels, chosen because they fail differently.

**The tab title always changes** — `✓ coffee_v2 — VisionForge`. It needs no
permission, cannot be suppressed, and is readable from the tab strip. The mark
goes first because that is the end the strip does not clip. This is the channel
that carries the guarantee.

**A browser notification fires on top**, but only when the page is hidden and
permission was granted. A toast for a window the researcher is already staring
at is noise, and a `Notification` constructor that throws — an OS focus mode, a
locked-down profile — changes nothing, because the title already said it.

**Permission is requested when a run starts, never on page load.** A prompt
before the user has done anything is the one people reflexively deny, and the
denial is sticky: getting it wrong once costs the feature permanently. Asking at
the moment a long job begins ties the prompt to a visible reason.

**Announced once per run, keyed by `run_id:status`.** The status is polled, so
the effect re-fires; without the key a completed run would toast on every tick.

**No sound, no webhook, no desktop-toast dependency.** Sound needs its own
consent dance and is worse than useless in a shared room; a webhook is a
different feature for a different user (CI, a phone) and would need a place to
configure it. The browser is already open — that is the cheapest true thing.

---

## ADR-090 — A first-run introduction that asks the researcher's name

**Date:** 2026-08-11
**Status:** Accepted — shipped 2026-08-11

**First visit:** the screen dims, "Bem-vindo" arrives and leaves, and "Qual é o
seu nome?" opens a single line in the middle of the screen. The name is saved
and from then on sits in the header as `Bem-vindo, <nome>`.

**Every visit after:** only the greeting, for about two seconds, and the app is
there. It never asks again. Changing the name is a click on the header chip,
which remounts the overlay and replays the whole sequence.

**The name lives in `localStorage`, not the backend.** It is an interface
preference belonging to whoever is at the machine, not server state — a shared
local install must not overwrite one person's name with another's on each visit.

**Reading and writing tolerate blocked storage** (private mode). With no saved
name the introduction simply runs again; nothing breaks and nothing is reported
as an error, because a greeting is not worth a failure.

**A first draft of this took the name from the OS account instead**, on the
reasoning that a local single-user tool already knows who is sitting at it. That
was wrong, and worth recording: an account name is not what someone wants to be
called. `marcu` is a login, not a name. Asking once costs one screen, and the
answer is the name they actually use — which is the entire point of greeting
someone by it. That version was reverted.

**The overlay unmounts when it finishes** rather than staying at zero opacity,
so it can never intercept a click meant for the app behind it.

**`prefers-reduced-motion` is honoured by the global rule already in
`index.css`:** the transitions collapse to a single static frame. Someone who
asked not to be moved still gets the greeting, just without the movement.

**Verified end to end in the running GUI:** typing a name saved it to
`vf.welcome.name`, the overlay left the DOM, and the header chip rendered
`Bem-vindo, Marcus`.

---

## ADR-091 — `run.json` records the class-name mapping

**Date:** 2026-08-12
**Status:** Accepted — shipped 2026-08-12
**Fixes a regression from:** the dataset rename of 2026-08-11

**What broke.** Grad-CAM stopped showing whether a prediction was right. Not a
Grad-CAM bug: `_gradcam_class_names` recovered the index→name mapping by
re-reading the run's *training folder*, because `run.json` never stored it.
`ImageFolder` assigns indices by sorting the class sub-directories, so the sorted
folder names genuinely are the mapping the checkpoint was trained with — as long
as the folder is still there.

Renaming `datasets/USK-COFFEE` to `datasets/Classification - USK-COFFEE` took it
away. The function returned an empty list, `true_class` could no longer match
anything, and the overlay quietly fell back to bare indices with no ground truth.

**Silence is what made it expensive.** The researcher reasonably concluded their
environment was out of date. A feature that degrades without saying so costs more
than one that fails loudly.

**Decision:** `run.json` records `class_names` at training time, and Grad-CAM
prefers it, falling back to the folder for runs written before this. The mapping
is a property of the trained checkpoint, not of where the data happens to live,
so storing it is what should have been done from the start.

**The seven existing runs were backfilled**, and that is defensible where
rewriting `base_dir` would not have been: the class list is invariant — the
dataset's *contents* never changed, only the folder's name — so recording it
states a fact rather than inventing a history. Rewriting the path would claim a
run trained somewhere that did not exist on that date.

**An empty recorded list does not shadow the folder**, so a future writer that
sets `class_names: []` degrades to the old behaviour instead of asserting a
model has no classes.

---

## ADR-092 — Resume keeps the training state beside the checkpoint, not inside it

**Date:** 2026-08-12
**Status:** Accepted — trainer shipped 2026-08-12; GUI action not yet wired
**Extends:** ADR-081 (runs die), ADR-088 (runs are stoppable)

**Context:** a run that stopped at epoch 90 of 120 cost all ninety. The only
thing on disk was the best weights, and weights cannot continue: an optimizer
that has forgotten its momentum and a scheduler that has forgotten its step
restart the search somewhere the loss curve never was.

**The deliverable checkpoint is deliberately not the resume file.** `best_model.pth`
stays a bare `state_dict`, because five separate things load it — evaluation,
Grad-CAM, batch prediction, ONNX export and the per-model test — every one with
`weights_only=True`. Making it a dict of dicts would break all five to serve a
feature none of them use. The training state lives beside it in `resume.pt`,
which is disposable: deleting it costs the ability to continue and nothing else.

**Written every epoch, not only on improvement.** The best-checkpoint write is
conditional because only improvements are worth keeping; resume state is not,
because a run dies on its worst epochs too, and state from three epochs ago
would resume into a past the optimizer has already left.

**Written atomically.** A half-written resume file is the one failure this
feature must not introduce — it would be found on the next attempt, refuse to
load, and leave the researcher believing resume is broken rather than that the
run died. `torch.save` to a temporary, then `os.replace`.

**Loading never raises.** An unreadable file, a payload that is not a dict, a
different `format_version`, missing fields — all return `None`, meaning start
fresh. A version mismatch is refused rather than partially applied: restoring
half a training state is worse than restoring none, because the run would look
continued and not be.

**The file's presence is the answer to "can this be resumed".** It is removed
when a run reaches its configured last epoch, and kept when a run was stopped —
so no separate flag can disagree with reality.

**Resuming continues the same run directory** rather than starting a new one.
The history is restored and appended to, so a resumed run has one continuous
curve and one `run.json`, not two halves a reader has to stitch together.

**Verified on a real training:** six epochs configured, stopped after two,
resumed — the second pass ran epochs 3, 4, 5 and 6 with no repeats and no gaps,
finished with a six-epoch history in the original run directory, and removed its
resume file.

**Not done: the GUI action.** The trainer accepts `resume_dir` and everything
above works from Python, but nothing in the interface offers a resumable run a
button yet. The standalone trainers gained the same ability in ADR-093.

---

## ADR-093 — Every task can be resumed, each with the state its loop actually owns

**Date:** 2026-08-13
**Status:** Accepted
**Extends:** ADR-092 (resume state beside the checkpoint)

**Context:** ADR-092 gave classification a resume file and left the four
standalone trainers without one. That is the wrong place to stop: a
segmentation run is no shorter than a classification run, and the researcher who
loses ninety epochs does not care which task they were training.

**Regression, segmentation and the autoencoder use the same `resume.pt`** as
classification — the same format, written every epoch, cleared when the run
reaches its last one. Only the field names differ (`best_val_miou` instead of
`best_val_loss`), and they map onto `ResumeState.best_metric`, which is
deliberately named for what it is rather than for one task's metric.

**Ultralytics resumes itself.** The YOLO/RT-DETR backend owns its loop, its
optimizer and its EMA, and it already writes `weights/last.pt` plus the
`args.yaml` that `resume=True` reads. Writing a second, worse copy of that state
beside it would be duplication with two ways to disagree. So the detection
trainer hands Ultralytics its own `last.pt` and lets it continue; only the
torchvision loop, which VisionForge does own, gets a `resume.pt`.

**The epochs already recorded come back from `run.json`.** Ultralytics restarts
its epoch counter from the checkpoint, so a resumed run would otherwise finish
with a history that begins mid-training. The recorded epochs are read back and
seeded, and an epoch that is re-reported replaces its own record rather than
appending a duplicate — the history stays monotone whatever the backend does.

**PatchCore is excluded, not silently ignored.** It has no gradient loop and no
epochs: there is nothing to continue. Resume state is therefore never even
loaded for it, so an accidental `resume_dir` cannot make a memory-bank fit look
resumable.

**The run directory is now resolved before anything is created.** The first
implementation made a fresh timestamped directory, pointed the TensorBoard
writer at it, and only then discovered it had state to continue — leaving an
empty orphan directory behind and sending the resumed run's events into it. The
state is loaded first, and the directory is the resumed one when there is
something to resume. This also fixes the same bug in the classification trainer
shipped with ADR-092.

**A resume file that will not load starts a fresh run** rather than continuing
into the old directory. Continuing there would mean training from epoch 1 on top
of another run's `run.json` and checkpoint — the one outcome worse than losing
the ability to resume.

**Verified per task:** each of the four loops is stopped at its first epoch and
handed its own directory back; all four continue in the same directory with a
`1, 2, 3` history, one `run.json`, and no resume file left behind. The
Ultralytics path is covered for the two things only it can get wrong: it asks to
resume from `last.pt`, and a fresh run never asks to resume at all.

---

## ADR-094 — The stop button had nothing on the other end

**Date:** 2026-08-13
**Status:** Accepted
**Fixes:** ADR-088 (a running job can be stopped)

**Context:** found while giving the standalone trainers resume (ADR-093).
ADR-088 gave every trainer a cancellation token and an epoch boundary to read it
at, and the run queue created a token per job and cancelled it on
`DELETE /api/queue/{id}`. Nothing connected the two. The endpoint answered 200,
the queue logged "asked to stop at the next epoch", the test asserted the flag
had flipped — and the trainer, which was never given the token, trained to its
last epoch. The one observable that mattered was the only one nobody checked.

**The token now travels the same road as the progress callback.** That road
already exists and is already understood: the route layer sets
`block._progress_callback` on the block it just built, and the block hands it to
its trainer. `_cancel_token` is assigned in the same place and passed in the same
call. Nothing new is invented, so there is one pattern for "things the GUI
injects into a run" rather than two.

**The executor reads the token from the module, not from a closure.** The
coroutine that runs a job is built when the job is *submitted*, and at that
moment the job — and its token — do not exist yet. `_begin_job` therefore
publishes the active token immediately before the queue awaits `job.start()`.
That ordering is guaranteed by the queue's own drain loop, not by timing.

**A cancelled sweep stops between trials.** Grid and random search run many
trainings inside one job: the trial that is training stops at its own epoch
boundary, and the loop then breaks rather than starting the next trial. Without
that, cancelling a 40-trial sweep would stop one trial and immediately begin
another, which is indistinguishable from the button doing nothing.

**Declared on the base class where there is one.** `ExperimentBlock` carries
`_cancel_token = None`, so the route layer can assign it without knowing which
block it built. The standalone task blocks are deliberately not subclasses
(ADR-033/036/037), so each declares it beside its own `_progress_callback`.

**The test asserts delivery, not intent.** ADR-088's test checked that the token
was cancelled, which was true the whole time it was broken. The new ones check
that the block hands its token to the trainer, and that the running job's token
is the one the executor can reach.

**Correction (2026-10-07):** "a cancelled sweep stops between trials" held for
the classification grid and random search only. K-fold, comparison, replicates,
the standalone sweeps and custom tasks never received the token. Fixed in
ADR-111.

---

## ADR-095 — The button that continues a run reads the run, not the form

**Date:** 2026-08-13
**Status:** Accepted
**Completes:** ADR-092/093 (resume), which shipped without an interface

**Context:** the trainers could continue a stopped run from Python, and nothing
in the interface offered it. A capability only reachable by writing a script is
not a capability for the person this tool is for.

**The config comes from the run's own `run.json`, never from the browser.**
Resuming with different hyperparameters would produce one directory whose
history describes two different experiments — and the history is the thing the
researcher will later read as if it were one curve. The endpoint takes a run id
and nothing else for exactly that reason.

**"Can this be continued" is computed from disk on every read.** ADR-092 made
the resume file's presence the answer; the API repeats that rather than storing
a flag, because a stored flag can disagree with what is there. It costs one
`torch.load` per listed run, which is why the check refuses early on anything
whose config has no epoch count.

**Ultralytics is judged by its own state.** A YOLO run writes no `resume.pt` — it
keeps `weights/last.pt` (ADR-093) — so asking the resume file about it would
answer "no" for every detection run there is. Its test is `last.pt` plus a
history shorter than the configured epochs.

**A sweep's directory is never offered.** Grid search, random search and K-fold
write their real trainings into sub-directories with their own `run.json`; the
parent holds no training, so "continue" there would continue nothing. The
whitelist is by block, and it errs toward not offering.

**A resumed job goes through the queue like any other**, with its own job id and
SSE stream, while the training itself lands in the original run directory. The
queue id identifies the job; the directory identifies the experiment, and those
are different questions.

**Verified end to end in the browser, not only in tests:** a classification run
was stopped after epoch 2 of 4, appeared in History with a `⏸ 2/4` badge, showed
`▶ RETOMAR 2/4` in its detail panel, and clicking it logged
`Resuming 20260817_191703_145266 at epoch 3 of 4`, ran epochs 3 and 4 in that
same directory, and left the run reporting `total_epochs: 4`, a `1,2,3,4`
history and `resumable: false`.

---

## ADR-096 — The frontend lint became a gate, which meant fixing what it found

**Date:** 2026-08-13
**Status:** Accepted
**Extends:** ADR-085 (the frontend joined CI)

**Context:** ADR-085 added a frontend CI job and deliberately left `npm run lint`
out of it, because six pre-existing errors would have made the job red the day it
arrived. A check nobody can turn on is not a check — and the six were not style
nits.

**One was a real bug.** `SchedulerFields` called `useContext` *after* an early
`return null`, so the first render where the schema had no properties would have
shifted every later hook by one. It never fired because that schema always has
properties, which is the kind of latent defect a lint rule exists to catch.

**Two were "setState synchronously in an effect", and both had a better shape
underneath.** The history sheet stayed mounted while closed and re-cleared twelve
pieces of state on the way in, reproducing by hand exactly what a fresh mount
gives for free; it is now mounted only while open, and its effect does nothing
but fetch. The debounced dataset stats cleared themselves when disabled, which is
a question about the current render (`enabled ? stats : null`), not a state
change.

**Three were Fast Refresh warnings that pointed at the right seam.**
`SchemaForm.tsx` exported three schema-walking helpers alongside its components,
so editing the form during development remounted it and threw away whatever the
researcher had typed. The helpers moved to `lib/schema-form.ts`, where they were
always going to belong.

**Verified in the browser, because two of these changed component lifetimes:**
history opens and lists its runs, a typed filter is gone when the sheet is
reopened (the behavior those twelve resets existed for), the count badge still
tracks, and the dataset-stats endpoint still answers. No console errors.

**The gate is now on**, so the next one of these is caught on the pull request
rather than accumulating into a list too big to turn on.

---

## ADR-097 — Training Ultralytics for real, once, found a phantom epoch in every YOLO run

**Date:** 2026-08-13
**Status:** Accepted
**Fixes:** ADR-034 (the Ultralytics backend), ADR-088 (stopping a run)

**Context:** every detection test handed `DetectionTrainer` a fake `YOLO` class.
That is the right shape for testing our argument translation, and it proves
nothing about the integration — which is the default backend, and whose failures
have been about how Ultralytics behaves rather than about which kwargs it gets
(ADR-079 was one: a relative `project` wrote the entire run somewhere else). The
self-test harness avoids it too, deliberately, so it can run offline.

So one test was written that actually trains: `yolo11n` from the checkpoint in
the repo, one epoch, four synthetic images, `@pytest.mark.slow`, skipped when the
extra is absent. **It failed on its first assertion**, and the bug it found was
in every YOLO run this project has ever produced.

**Ultralytics fires `on_fit_epoch_end` one extra time after the loop.** Its
`final_eval` re-validates `best.pt` and raises its own counter first — the source
comment reads "log best metrics at step epochs+1, not overwriting last epoch".
That callback is our only window into its loop, so the extra call was recorded as
an epoch. Every completed run reported one epoch too many: a 10-epoch run said
11, a 100-epoch run said 101.

**And sometimes it stole `best_epoch`.** The final pass reports the best
checkpoint's metrics, so when they came out at or above the best real epoch's,
the phantom won: two of the runs on this machine name `best_epoch=6` for a
5-epoch run, and one names `best_epoch=3` for a 2-epoch run. A number that points
at an epoch that never ran is worse than a wrong count.

**Comparing the epoch number against the configured total is not enough.** A
cancelled run stops early, so its phantom lands *within* the configured count and
looks exactly like the next epoch. What separates them is not the number but the
loop having ended — either by reaching the last epoch or by being asked to stop.
The callback now closes itself in both cases, and ignores everything after.

**The fake now fires the extra callback too,** with better metrics than any real
epoch, so the fast tests fail without the guard — they were verified failing
before the fix went in, on all three assertions.

**Old `run.json` files are left alone.** They record what was measured, and the
phantom's numbers are real (they are `best.pt`'s), only mislabelled as an epoch.
Rewriting them would invent a history; the runs affected are already known to be
from before several corrections.

**What this says about the deferral.** Not testing the real library was a
reasonable trade while the concern was "does it get the right arguments". It
stopped being reasonable the moment the questions became about behaviour, and one
26-second test answered a question three months of mocked tests could not.

**Follow-up (same day): it runs in CI, and so does everything else.** A
`detection` job installs the extra and trains that epoch on every pull request.
A `selftest` job runs the ADR-060 harness in full — six tasks × five strategies,
27 cases, every one a real training through a live server — which had until then
only ever run when somebody typed `visionforge selftest` by hand. Locally the
whole matrix takes 172 seconds, which is the actual reason it can be a gate
rather than a ritual. The test no longer requires the local
`yolo11n.pt` — that file is gitignored, so requiring it would have made the test
skip everywhere but this machine, and a skipped test in CI is indistinguishable
from a passing one. Without a checkpoint it builds `yolo11n.yaml` from scratch,
which needs no download and exercises the same integration.

---

## ADR-098 — Four findings from training the tool like a researcher would

**Date:** 2026-08-21
**Status:** Accepted
**Revises:** ADR-062 (the determinism knob), ADR-081 (worker caps), ADR-038 (the
anomaly threshold)

**Context:** twenty-seven runs across five tasks — ablations, folds, seeds, a
sweep, both anomaly families, both detection backends — driven through the HTTP
API the way the GUI drives it, then read back through History and checked
against independent recomputation. Four things came out of it, and three of them
were invisible from the code alone.

### 1. The default made runs unreproducible, for a speed-up that is not there

Two runs of a byte-identical config under the same seed came out **0.8263 and
0.7481**. The cause is `deterministic: False` — cuDNN auto-tuning plus
non-deterministic kernels. That was a documented trade, and the docstring priced
it at "3-5× slower".

Measured on an RTX 5060 Ti over 2-epoch runs:

| shape | `benchmark` | `deterministic` | |
|---|---|---|---|
| resnet18 @224 | 65s | 65s | even |
| resnet50 @224 | 96s | 78s | **-19%** |
| resnet18 @320 | 75s | 63s | **-16%** |

Deterministic was *faster* in two of three. Auto-tuning costs time up front and
only repays it over many epochs at a stable shape, which is not the shape of an
experiment. So the default is now `True` for every task (detection already was),
and with it two runs of one config produce bit-identical accuracy, F1 and loss.

**Why this is worth a default and not a note.** The variance is not academic:
across three seeds the spread was 0.117 and across five folds 0.135, while the
effects being measured — augmentation, transfer learning, freezing — were 0.117,
0.103 and 0.091. Run-to-run noise was the same size as every conclusion. A tool
whose *default* hides that is arguing against its own purpose. Long training may
still prefer `False`; it stays a knob.

### 2. The anomaly threshold was calibrated on images the model never sees

An autoencoder reported **AUROC 0.79 and F1 0.00** — a bootstrap interval of
[0.0, 0.0] over 1000 resamples, so not an accident. The decision threshold is a
percentile of the normal-score distribution, and that distribution was collected
from the **training loader**, which augments: rotation pads corners, flips move
structure, both raise reconstruction error. The percentile therefore sat above
every score the model produces on clean images, and nothing was ever flagged.

The threshold now comes from a `calibration_loader` — the same training normals,
read the way inference reads them. Retrained on the same data: **threshold
0.0323 → 0.0150, F1 0.0000 → 0.3373**. The AUROC is unchanged in meaning; it
never depended on the threshold, which is exactly why the run looked healthy
while its operating point was useless.

### 3. `num_workers` can be capped, because the cost is now measured

ADR-081 declined to cap workers: the per-worker cost was a guess, and guessing
wrong either throttles a healthy machine or fails to save a doomed one. It is not
a guess. A worker never holds the model — it loads images while the model sits on
the GPU — so what it costs is **commit**, and on Windows that is readable.
`utils/workers.suggested_workers` divides half the free commit by ~1 GB per
worker per loader pool, capped by CPU count and a ceiling of 8.

Scaling with model size or GPU, the intuitive heuristic, measures the wrong
quantity entirely. On the machine where the reported WinError 1455 happened —
10.7 GB free commit, three pools, 8 requested — this yields 1, and the run would
have trained instead of dying. Requests above budget are lowered with a warning
rather than silently, so the limit is the machine's, stated.

### 4. The history showed three of five tasks the wrong numbers

Segmentation and anomaly listed an empty metric cell and regression listed its
loss instead of its R², because the summary map only had classification and
detection and everything else fell through to keys it never writes. The same
fallback filed every standalone task under the "classification" block, in the
filter that exists to separate them. Fixed and covered; the numbers were in
`run.json` all along.

**Also corrected, minor:** the ROC figure now draws the macro-average its
docstring always promised — the one number `test_auc_roc` reports — and the loss
curve's x-axis is integers, because there is no epoch 1.5.

**What the audit did not find.** Worth recording, because it was checked: per-fold
normalization statistics computed from the training fold only, stratified folds,
augmentation confined to training, model and sweep-trial selection by validation
loss and never by test, macro-F1, one-vs-rest macro AUC, VOC mAP matching each
ground-truth box once, mIoU honouring `ignore_index`, bootstrap resampling images
rather than pixels, paired significance testing with Holm correction and an
explicit underpowered flag. Recomputing a run's metrics from its checkpoint with
plain sklearn reproduced accuracy, F1, precision and recall to the digit.

---

## ADR-099 — Accuracy 0.50 was not a weak result, it was a collapsed run

**Date:** 2026-08-27
**Status:** Accepted
**Extends:** ADR-098 (the researcher audit)

**Context:** a user reported that some trainings land on 0.5 accuracy as if the
model "guessed one class for everything", that it depends on the optimizer and
the model, and that longer training does not fix it. It does not fix it, and the
model really is guessing one class.

**The grid, same data and seed throughout** (USK-COFFEE, 4 classes, 3 epochs, at
the default `learning_rate = 1e-3`):

| model           | Adam              | SGD                  |
|-----------------|-------------------|----------------------|
| resnet50        | 0.72              | 0.53 (undertrained)  |
| vgg16           | **0.25 collapse** | 0.80                 |
| alexnet         | **0.25 collapse** | 0.81                 |
| efficientnet_b1 | 0.86              | 0.34 (undertrained)  |

The collapsed runs predict a single class for every image: the confusion matrix
has one occupied column, and the training loss barely moves (1.446 → 1.393).
Repeated on a balanced two-class problem, the same pair reports **exactly
0.5000** with a confusion matrix of `[[300, 0], [300, 0]]` — which is the number
the report showed and called a result.

**The split follows batch normalization.** VGG and AlexNet predate it and carry
very large fully-connected heads; an Adam step of 1e-3 saturates them in the
first iterations and they never recover. The normalized architectures take that
step comfortably and instead undertrain under plain SGD at the same rate. No
single default serves both halves, which is why one existed and half the grid
failed under it.

**Three changes, in order of what matters.**

**1. A collapsed run says so.** `core/training_health.py` checks, at the end of
every run, whether the model predicted one class for the whole validation set
and whether the loss ever meaningfully fell. Warnings ride in `TrainResult` and
in `run.json`, and are logged. The stagnation threshold is 10%, calibrated on
the grid above rather than chosen: runs that learned nothing fell 3.7% and 4.4%,
runs that learned fell 49% and 62%. This is the part that had to exist —
detection is what turns a misleading number into a message, and a default can
always be overridden back into the same hole.

**2. The suggestion knows the pair.** `core/learning_rate.py` maps architecture
and optimizer to a starting rate that trains it: 1e-4 for Adam on VGG/AlexNet,
1e-2 for plain SGD, 1e-3 otherwise. Offered, never imposed — a value silently
substituted into a config the researcher wrote is the same class of problem as a
number appearing without explanation.

**3. Verified end to end on the failing pair.** VGG16 + Adam, binary: at 1e-3,
accuracy 0.5000 with every image on one side and the collapse warning raised; at
the suggested 1e-4, accuracy 0.8800 with a balanced matrix and no warnings. The
detector stays quiet on a healthy run, which is what keeps it worth reading.

**What this was not.** Not a bug in the loss, the metrics or the data pipeline —
all of those were checked in ADR-098 and recomputed against plain sklearn. The
defect was a default that fits half the model list, and a report that presented
the consequence as a measurement.


---

## ADR-100 — Defaults measured per task, and three attention architectures

**Date:** 2026-08-27
**Status:** Accepted
**Extends:** ADR-099 (the collapse)

**Context:** with collapse detection in place, the remaining question was which
defaults a first run should start from — across all five tasks, not only
classification — and whether the model list should reach past the CNNs.

**The new architectures were measured before being offered, not after.**
`vit_b_16`, `swin_t` and `convnext_tiny` come from torchvision, so they cost no
new dependency. Run on the same data and seed as ADR-099's grid:

| model         | Adam 1e-3         | AdamW 1e-4 |
|---------------|-------------------|------------|
| vit_b_16      | 0.41              | 0.85       |
| swin_t        | **0.25 collapse** | 0.88       |
| convnext_tiny | **0.25 collapse** | **0.91**   |

Two of the three collapse at the rate that was the default until this week —
the same failure as VGG and AlexNet, for the same practical reason. Adding them
without measuring would have shipped three models that fail out of the box.
`convnext_tiny` at its own rate is now the strongest classifier in the list.

**Defaults, per task.** Classification and regression start at `resnet18`
instead of `resnet50` (a first run should be quick; the bigger model is one
click away) and at 20 epochs instead of 10, because the curves in the audit were
still climbing when the run ended. Segmentation starts at `unet` instead of
`deeplabv3_resnet50`, which at the default 512px exhausts a modest GPU.
**Anomaly (30 epochs) and detection (100, Ultralytics' own) were left alone** —
both are already calibrated for what those tasks need, and changing them for
symmetry would be worse than the asymmetry.

**Image size can be read from the data, and only suggested.** CIFAR-10 ships
32×32; training it at 224 upscales sevenfold and spends roughly fifty times the
computation on pixels the resize invented. `core/image_size.py` samples the
dataset's median short side and proposes a size, floored at 64 and rounded to
the stride of 32. It never decides, for two reasons: ImageNet weights expect
~224, so smaller is cheaper rather than better; and ViT and Swin build fixed
position embeddings, so for them another size is not slower — it is an error,
and the suggestion is pinned to 224 regardless of the data.

**The interface offers, the researcher consents.** `POST /api/model/defaults`
answers with the measured optimizer, rate and size, and the form shows it as a
note with a button. Silently rewriting the learning rate would repeat the
original sin of ADR-099 — a number arriving without the researcher knowing where
it came from — wearing a friendlier face.

**Verified in the browser:** choosing `swin_t` raises "prevê uma classe só:
medimos 0.25 de acurácia", clicking the button sets AdamW and 1e-4 in the form,
and the note disappears because the settings now match what was measured.

---

## ADR-101 — Feature extraction was not freezing the backbone on two architectures

**Date:** 2026-08-28
**Status:** Accepted
**Extends:** ADR-099/100 (collapse detection and measured defaults)

**Context:** the previous two ADRs were validated on classification only. Asked
whether they held for the other tasks, and whether transfer learning does what
it says, the answer to both was no.

**Feature extraction froze the wrong thing.** The rule was "freeze every named
child except the last", which is the head on ResNet (`fc`), ViT (`heads`), Swin
(`head`) and EfficientNet (a two-element `classifier`). On VGG and AlexNet the
last child is a `classifier` block of **three** Linear layers — 102M + 16M + 4M —
so the mode whose entire promise is that the backbone stays put left this much
trainable:

| architecture    | before | after |
|-----------------|--------|-------|
| vgg16           | **89.04%** (119M) | 0.01% (16,388) |
| alexnet         | **95.67%** (54M)  | 0.03% (16,388) |
| everything else | ≤0.08% | unchanged |

The head is the last `nn.Linear` — the layer `replace_final_layer` creates — not
the last child, and `models/factory.final_linear` now says so once for both the
classification block and the regression trainer, which had the same rule and the
same defect. Verified red-green: eight of the twenty-two tests fail with the old
rule restored.

**Segmentation was already right**, and it matters to say why rather than to
"fix" it: there the last child is a decoder (ASPP for DeepLab, 16M parameters),
and training the decoder over a frozen backbone is what segmentation transfer
learning *is*. Applying the Linear rule there would have broken a working thing.

**Fine-tuning does what it claims.** Two parameter groups, backbone at
`learning_rate * backbone_lr_multiplier` and head at the full rate — checked on
resnet18, vgg16 and vit_b_16, at 1e-4 and 1e-3 with the default multiplier.

**New: freezing weights that were never trained.** Feature extraction is worth
doing because the frozen weights know something. On a `unet` without pretrained
weights it freezes 31.4M random parameters and trains 195 against them. That is
now a warning rather than a silent waste of an evening.

**The health checks reached the other four tasks.** ADR-099 shipped collapse
detection in the classification trainer and stagnation everywhere; detection had
neither. Now each task is asked the same question in its own vocabulary:
classification predicts one class, regression outputs one value, segmentation
paints one class over every pixel, detection finds nothing (mAP@50 of zero).

**Measured, not assumed, in both directions.** A VGG regressor at 1e-3 reaches
R² = −0.157 with predictions spread over 5.95 — bad, but not collapsed, and the
warning correctly stays quiet. A detector that finds nothing and a segmenter
whose other classes sit at IoU zero do raise it.


---

## ADR-102 — The form had its own defaults, and they were the old ones

**Date:** 2026-08-28
**Status:** Accepted
**Completes:** ADR-098/099/100 (determinism, early stopping, per-task defaults)

**Context:** finishing the proposed work meant putting the basic/advanced split
in front of the researcher. Doing that exposed something worse: the four
standalone panels build their form state from their own literals in
`lib/*-models.ts`, so every default corrected over the last three ADRs was
still the old value for anyone using the interface.

| field | project | the form said |
|---|---|---|
| `deterministic` | True (ADR-098) | **false** |
| `early_stopping_patience` | 0 (ADR-099) | 10 |
| `epochs` (regression) | 20 (ADR-100) | 50 |
| `model.name` (regression) | resnet18 | resnet50 |
| `model.name` (segmentation) | unet | deeplabv3_resnet50 |

The determinism one is the reason this is an ADR rather than a line in the
changelog: ADR-098 flipped the default so that two runs of one config reproduce
each other, and the form kept sending `false`. Every GUI run since then was
still unreproducible, and the fix looked done from the backend's side.

**Also found while checking**: three panels capped the early-stop field at
`min={1}`, so typing 0 — the value ADR-099 introduced to disable it — was
silently rounded up to 1. Backend permission means nothing when the input
refuses the number.

**The basic/advanced split is now on.** `param-help.ts` has classified every
parameter since the "muita coisa junta" complaint, and nothing read it: the
panel showed all nine training fields at once. Four stay visible (epochs,
learning rate, batch size, seed — what changes between two experiments) and the
rest sit behind a collapsed section that opens by itself when any field inside
differs from its default.

That auto-open needed a fix of its own. `hasNonDefaultAdvanced` compared against
schema defaults, and a nested object like the scheduler carries no `default` at
its own level — so it always differed, and the section always opened, which is
the same as not having one. An unknown default is now not evidence of a change.

**Verified in the browser:** the training section shows four fields and
"avançado (6)" closed; the regression panel opens on ResNet-18; picking Swin-T
there raises the same measured warning classification shows.

## ADR-103 — The worker count decides itself

**Date:** 2026-08-28
**Status:** Accepted
**Supersedes:** the fixed `workers: 2` of the detection form

**Context:** `num_workers` is the only field in the interface whose right value
depends on the machine and not on the experiment. It also has the sharpest
failure mode we ship: on Windows every DataLoader worker is a spawned process
that re-imports torch and the CUDA DLLs, about 1 GB of commit charge each, per
loader pool. Too high does not make training slow — it stops training from
starting, with `[WinError 1455] The paging file is too small`.

The form already had an "auto" button, and it was the wrong shape: it copied
`min(cpu_count, 8)` into the box. From that moment on the value was a literal
frozen at whatever the machine looked like on the day the form was filled in,
and `cpu_count` is not the constraint anyway — free memory is.

**Decision:** `-1` is the resting default of every worker field, and it means
"decide at load time". `utils/workers.suggested_workers(loader_pools=N)` asks
Windows for the free commit charge through `GlobalMemoryStatusEx`, spends at
most half of it, divides by 1 GB per worker per pool, and caps the result at
`cpu_count` and at 8. Each data module declares how many pools it opens —
classification, regression and segmentation open three (train/val/test),
anomaly two — because the budget is spent per pool, not per run.

An explicit number is still honoured, and still capped with a warning when it
exceeds the budget: refusing to start is worse than running with fewer workers.
`0` keeps its own meaning (load in the main process) and is not "automatic",
which is why the bound is `ge=-1` rather than a nullable field.

**The field stays on screen, showing what it resolved to.** Removing it would
hide the one number a researcher debugging WinError 1455 needs to see. It reads
"automático ≈ 3 agora", with the count coming from `/api/system/info`, and the
"auto" toggle hands the number over to whoever wants it.

**Verified in the browser:** the field reads `automático · ≈ 3 agora` on this
machine, the toggle switches to a number input pre-filled with 3, and toggling
back returns to automatic. 22 tests cover the contract, including that `0`
survives validation and `-2` does not.

**Amended (2026-10-15):** the automatic setting now obeys the small-dataset rule of
`resolve_num_workers` as an explicit count always did: under 500 images it loads in
the main process (0 workers). Before, `-1` skipped that check and gave a 100-image
dataset the whole budget the machine could afford, paying the spawn cost the rule
exists to avoid. The worker-policy parity test records the change.

## ADR-104 — A guided tour, offered once

**Date:** 2026-08-28
**Status:** Accepted
**Builds on:** ADR-090 (the first-run welcome)

**Context:** the welcome asks for a name and then drops the researcher into a
screen with five task tabs, a strategy row, four form sections and a fixed
bottom bar. Everything on it is discoverable and nothing on it is announced —
the "eu nem sei o que é o anomalia" reaction was about exactly this.

**Decision:** after the name is set, and only on a machine that has never been
offered it, a card asks whether the researcher wants a quick tour. Seven stops:
the task tabs, the dataset folder, the basic/advanced split, GPU vs CPU, the
train button, the history, and the datasets overlay. Every stop can be left with
✕, "Pular" or Esc, and the header keeps a "guia" button so it is never lost.

**The stops point at real elements**, marked with `data-tour` where they are
rendered, and are found in the DOM at the time the step opens. A stop whose
element is not on screen — because the active task has no such field — becomes a
centred card instead of disappearing, which is what lets one script serve the
five tasks.

**Two implementation notes worth keeping:**

The darkness is four panels around the highlight, not a `box-shadow` with a
9999px spread on the highlight itself. The shadow is the shorter way to write
it, and it makes the browser rasterise a layer some 20000px on a side on every
frame of the transition; on this page it held up painting for seconds at a time.
Four panels interpolate the same way and cost what any four divs cost.

The card's position is clamped to the viewport, and that is not defensive
padding: the smooth scroll to the next target has not finished when the step
opens, so "above the target" is briefly a coordinate nobody can see. The
geometry lives in `lib/tour.ts` rather than in the component so it can be tested
against a viewport smaller than the card.

**Verified in the browser:** the invitation appears after the name on a fresh
`localStorage`, the first stop highlights the tab row with the card beneath it,
and the second scrolls to the dataset card and follows it. The placement is
covered by seven unit tests, including the two off-screen cases.

## ADR-105 — "Frozen" freezes weights, not BatchNorm statistics

**Date:** 2026-08-28
**Status:** Accepted
**Follows:** ADR-101 (feature extraction was leaving 89% of VGG trainable)

**Context:** the 27-run audit reported that `feature_extraction` does not fully
freeze the backbone, and the claim deserved measuring rather than repeating.
Frozen ResNet-18, five batches, `train()` mode:

| what | moved |
|---|---|
| parameters | **2** — `fc.weight`, `fc.bias` |
| buffers | **60** — 20 BatchNorm modules × mean, variance, batch counter |

The same model in `eval()` moves none of the 60.

The cause is not a bug in the freeze: `requires_grad = False` stops the
optimizer from moving a *parameter*, and `running_mean` / `running_var` are not
parameters. They are buffers, written inside the forward pass, and a module
whose weights are frozen still updates them as long as it is in training mode.

**Decision: document it, do not change it.** Two reasons.

It is the behaviour most transfer-learning code has, and it is usually the one
you want — the pretrained features stay put while the normalization stops being
ImageNet's and becomes this dataset's. Switching frozen modules to `eval()` is a
defensible alternative, but it is a different method, not a bug fix.

And it would silently invalidate comparisons. Every run recorded so far trained
with recalibrating statistics; flipping the behaviour would make a rerun of an
old config produce a different number for a reason nothing in the config
records. If it ever changes, it needs a config field, so a `run.json` says which
of the two produced it.

**What changed instead** is every place that described it: the `_apply_freeze`
docstring, the `TransferLearningConfig` docstring, the `mode` field description
that reaches the form schema, and the caption under the strategy selector. All
of them now say "frozen weights, recalibrated normalization" rather than
implying the backbone is untouched — the distinction matters in a paper, where
"the backbone was frozen" reads as "identical to pretrained".

## ADR-106 — Everything ships in one install

**Date:** 2026-08-28
**Status:** Accepted
**Supersedes:** the per-feature extras of ADR-005 (the hardware extras stay)
**Amended by:** ADR-170/171/172/176 (`mlflow`, `hub`, `wandb` and `dvc` are optional extras that install something; the seven extras named here stay empty)
**Superseded in part by:** ADR-195 (`ultralytics`, AGPL-3.0, is again the `detection` extra; the other six stay in `dependencies`), itself reverted by ADR-243 (`ultralytics` is back in `dependencies`)

**Context:** seven capabilities were optional extras — `detection`, `timm`,
`optuna`, `tensorboard`, `roboflow`, `kaggle`, `huggingface`. The reasoning was
that a plain install should stay small. In practice the split moved a cost from
the download to the researcher: the detection tab was fully present in the
interface, with its models, its form and its strategy selector, and training
failed at the last step with "install the detection extra". A capability you can
see and cannot use is worse than one that took a few seconds longer to download.

Measured, per extra, as what it actually adds on top of the base install:

| extra | adds |
|---|---|
| `detection` (ultralytics) | 3.9 MB |
| `timm` | 12.9 MB |
| `optuna` | 1.9 MB |
| `tensorboard` | 8.6 MB |
| `roboflow` | 1.5 MB |
| `kaggle` | 4.5 MB |
| `huggingface` (datasets) | ~50 MB, of which 46.8 is pyarrow |
| **total** | **~83 MB** |

Against the 2–3 GB a CUDA build of PyTorch downloads, the whole set is about 3%.
The number that justified the split does not survive being measured.

**Decision:** the seven move into `dependencies`. PyTorch stays out, for the
reason it always did (ADR-005): its build has to match the hardware and no
resolver can choose. The extra *names* stay defined and empty, so an install
line saved in a requirements file or someone's notes keeps resolving instead of
failing on an unknown extra.

**Two things this changed beyond the dependency list.** Seven error messages
told the reader to add an extra; that advice now names a flag that does nothing,
so they say the package should be there and the install looks damaged, and
`doctor.extra_install_hint` became `missing_package_hint`. And `kaggle`
authenticates *at import time*, which is why bundling it is safe only because
every provider import already sits inside its function behind `except ImportError`
and `except OSError` — checked before making the change, not after.

## ADR-107 — `doctor --fix` replaces torch instead of reinstalling the package

**Date:** 2026-09-23
**Status:** Accepted
**Fixes:** a regression introduced by ADR-106

**Context:** before a first batch of testers got the install line, the published
0.10.0 was installed into a clean venv the way they would. The plain install
works and trains every task — but on Windows it now always brings a CPU torch:
`ultralytics`, a base dependency since ADR-106, requires torch, and PyPI only
carries CPU builds for Windows. `doctor` noticed the mismatch correctly. The
`--fix` it offered did not fix it:

```
pip install "visionforge-studio[cu128]" --index-url https://download.pytorch.org/whl/cu128
→ Requirement already satisfied: torch>=2.3 ... (2.14.0)
```

The CPU torch satisfies `torch>=2.3`, so pip stops there. Before ADR-106 the
first install had no torch at all and the same command worked, which is why
nothing ever flagged it. The hardware extras do not help either: from PyPI they
are identical lists (`torch`, `torchvision`), and only a source checkout with
uv honours the `[tool.uv.sources]` that maps them to an index. `pip install
"visionforge-studio[cu128]"` — which the README recommended as the manual
route — also lands the CPU build.

`--upgrade` is not an answer. The CUDA index trails PyPI (2.11.0+cu128 against
2.14.0+cpu at the time), so the CUDA build looks older and pip keeps what is
there.

**Decision:** the fix only ever touches torch, in two steps — uninstall
`torch torchvision`, then install them from the PyTorch index for the detected
tag. Measured in the clean venv: the second step, dry-run after the first,
resolves `torch-2.11.0+cu128 torchvision-0.26.0+cu128` downloading 29 kB of
metadata; applying it leaves `cuda_available=True` and doctor reporting the
environment good.

Three details worth keeping:

- The steps run as `sys.executable -m pip`, not the first `pip` on PATH — with
  the venv not activated that is another interpreter and the CUDA build lands
  where nothing imports it. An environment without pip (made by `uv venv`, like
  the dev checkout) gets `uv pip --python` instead.
- They print as two lines, not one joined with `&&`: Windows PowerShell 5.1 has
  no `&&`, and that is where most of these users are.
- Because the fix now *starts* by uninstalling, doctor must not print it to a
  setup that is already right. "torch sees a GPU" is not enough to say so — an
  older CUDA build on an RTX 50 imports, reports the GPU and fails at the first
  kernel — so the test is the build: when `torch.version.cuda` is at least what
  the driver calls for, doctor says "keep the current install".

Upgrading the package keeps the torch in place: `pip install --upgrade
visionforge-studio` from 0.9.1 with a CUDA torch installed resolved to exactly
one package, `visionforge-studio-0.10.0` — pip's default only-if-needed
strategy leaves a satisfied torch alone.

## ADR-108 — PatchCore keeps 1% of the patches by default, not 10%

**Date:** 2026-09-23
**Status:** Accepted
**Follows:** ADR-038 (exact `cdist` coreset, no approximation)

**Context:** the 27-run audit timed PatchCore at ~114 min for 2000 training
images with `coreset_ratio=0.1`. The greedy coreset makes one `cdist` over every
patch per selection, so its cost is O(k·M) with k = ratio·M: linear in the
ratio, quadratic in the dataset. The phase-progress events (0.10.0) made the
wait visible; they did not make it shorter. The PatchCore paper's operating
point is 1%.

The paper's claim that 1% loses little was not taken on faith — it was measured
on the anomaly dataset in this repository (coffee beans: 300 normal training
images, the full test set of 400 good + 400 defective), ResNet-18 at 256 px, one
run per ratio on an RTX 5060 Ti, changing nothing but the ratio:

| ratio | bank | wall time | image AUROC | image F1 |
|---|---|---|---|---|
| 0.1% | 307 | 12.0 s | 0.685 | 0.388 |
| **1%** | 3,072 | **40.5 s** | **0.703** | 0.314 |
| 10% | 30,720 | 373.7 s | 0.686 | 0.299 |

10% cost 9.2× the time of 1% and bought nothing: its AUROC was the lower of the
two. The spread across all three rows (~0.02) is the same size as what changing
nothing but the device produced — the 0.1% configuration scored 0.666 on CPU
with torch 2.14 and 0.685 on GPU with torch 2.11, because small float
differences change which patches the greedy selection picks. So the defensible
reading is "no measurable difference", not "1% is better".

**Decision:** the default moves to 0.01 in the config, the model's own default
and the form. On the full 1200-image training set the quadratic term scales the
coreset by 16× from the 300-image run — roughly 10 minutes at 1% instead of an
hour and a half at 10% (an extrapolation, not a measurement).

The field also gained its info dot: "what is this coreset?" was the first
question the maintainer asked about it, which makes it the first question a
tester will ask.

**Limits of the evidence:** one dataset, one backbone, one subset size, one run
per ratio. It is enough to say the 10× slower default was not buying accuracy
here; it is not a general result about PatchCore. A config that sets 0.1
explicitly still trains exactly as before, and a run recorded with it keeps its
number — only the default moved.

**Found on the way:** the first attempt at this measurement ran on the CPU. The
development environment had been switched to `torch 2.14.0+cpu` (with a
non-editable 0.10.0 from PyPI shadowing `src/`), and the trainer's CPU fallback
said so only in `run.json`. The measurement was repeated in an isolated
environment with the CUDA build, and every row above carries its device.

## ADR-109 — Python 3.12 is supported, and CI tests on Windows

**Date:** 2026-09-24
**Status:** Accepted
**Follows:** ADR-106 (one install; the floor is what that install requires)

**Context:** `requires-python = ">=3.13"` was inherited from the first
scaffold, not from anything the code used. On 2026-09-23 the dependencies were
installed by hand into a 3.12.12 environment — pip refused the package itself —
and the full suite passed: 1702 passed, 2 skipped. Python 3.11 is a different
story: the source does not compile there.

CI ran only on `ubuntu-latest`, while the author and the first testers train on
Windows. A Linux-only CI protected the platform nobody here uses.

**Decision:** the floor moves to 3.12 in `pyproject.toml`, in ruff's and mypy's
targets, and in `visionforge doctor`, through a single `MIN_PYTHON` that a test
holds equal to `requires-python`, so the doctor can never approve an
interpreter pip refuses. The `tests` job becomes a matrix of Ubuntu and Windows
× 3.12 and 3.13 with `fail-fast: false` (a Windows-only and a 3.12-only failure
are different bugs), a 30-minute timeout (a Windows-only hang would otherwise
hold a runner for six hours), and the venv activated by `$RUNNER_OS` rather
than by trying one path and falling back, which would report the wrong error.
Coverage is still uploaded once, from Ubuntu/3.13. The Docker image,
`.python-version` and the release build stay on 3.13: they are environments,
not the support floor, and the wheel is `py3-none-any`.

**Verified:** after the change, `uv pip install -e ".[dev]"` on 3.12.12
succeeds, the suite passes (1705 passed, 2 skipped), `visionforge doctor`
reports `[OK] Python 3.12.12`, and `visionforge selftest` passes 27/27. mypy
targeting 3.12 finds no issues, and no 3.13-only standard-library API or syntax
is used in `src/`.

**Consequences:** four test legs instead of one, and Windows runners are
slower. `python-multipart`, which a `pip-audit` of the development environment
flagged, is not a dependency of the package — it is absent from `uv.lock`,
pulled in by a stray `gradio` — so it needs no floor.

## ADR-110 — The interface is bilingual, from typed dictionaries

**Date:** 2026-10-06
**Status:** Accepted
**Follows:** ADR-104 (a guided tour that a researcher cannot read is no guide)

**Context:** the GUI was written in Portuguese only. That closes it to most of
the researchers who could use it, and to a reviewer who does not read
Portuguese — a JOSS submission among them. The README already ships in both
languages; the screen a researcher actually spends their time on did not.

**Decision:** no i18n library. The GUI has two languages and about a thousand
entries (1081 at the time of writing), and what a library adds — plural rules
for dozens of languages, message catalogues, runtime loading — is not needed to
write two files. What is needed is that a missing translation cannot ship, and
a type does that for free:

- `src/i18n/pt.ts` is the source and its type is the contract: `Dict = typeof pt`
  and `en.ts` is declared `const en: Dict`, so a missing or extra key fails
  `tsc -b`. A test also compares the two trees leaf by leaf (same keys, same
  kind of entry, same arity for the functions).
- A text that carries values is a function, `(n: number) => …`. That types the
  interpolation, and it is how plurals are written: each language says
  `n === 1 ? … : …` in its own grammar instead of a rule table choosing for it.
- The language lives in a React context (`useT()`), persisted in
  `localStorage` under `vf.lang`. The first visit takes it from
  `navigator.language`: any Portuguese locale gets Portuguese, everything else
  English. The provider also sets `<html lang>`; the static value in
  `index.html` is `en`, which is what a crawler or a screen reader sees before
  the JavaScript runs.
- Modules that are not components (`lib/*.ts`) take the dictionary as an
  argument, `validateParsedConfig(t, …)`, instead of reaching for a global, so
  they stay pure and testable in either language. The one exception is
  `api/client.ts`: it words its own errors (cannot reach the server, validation
  failed) at the moment it throws them, from ~50 call sites that have no
  provider to ask, so it reads the stored language (`vf.lang`, else the
  browser's) the way the provider seeds it.
- Numbers and dates: metric readouts are written with a dot as the decimal
  separator in both languages, because they sit next to the CSV and JSON output
  a researcher will paste them against. What is meant to be read as prose
  follows the interface locale (`pt-BR` / `en-US`): the human-readable sizes
  from `formatBytes` ("117,7 MB" / "117.7 MB") and the dates and times in the
  header clock, the history and the test records.
- Sentences with marked words (`code`, **strong**, __em__) are written whole in
  the dictionary and rendered by a shared `Rich` component over a tested
  parser (`lib/rich.ts`). Splitting a sentence around a bold word into three
  keys is what makes translations read as if they were assembled, because the
  other language wants the bold word somewhere else.
- A vitest guard parses every source file with the TypeScript compiler and
  fails on visible text outside `src/i18n/`: JSX text, string literals in JSX
  expressions, visible props (`title`, `placeholder`, `aria-label`, …) and prose
  in config or state objects. Grepping would have counted comments, imports and
  CSS values; parsing tells them apart. The guard is what keeps the tree from
  drifting back to Portuguese one literal at a time.
- Node types are scoped to the tests through `tsconfig.test.json`: the guard
  reads files with `node:fs`, and the application code must not be able to.

**Consequences:** every screen's words live in two files, and adding a screen
means adding them to both before the build passes. Texts that come from the
server are shown as the server writes them — partly Portuguese, partly
English — until a second step decides between translating at the API edge and
sending message codes that the interface words itself. What is in each
language today, checked against the code:

- **English:** the `detail` of the HTTP errors that `gui/api/routes.py` writes
  itself, and the messages of the Pydantic validators in `utils/*config*.py`.
  A Portuguese-speaking user reads English there.
- **Portuguese:** the training-health warnings (`core/training_health.py`);
  the dataset-scan messages, which are the `message` of the split-detection,
  dataset-stats, samples and preview responses in `routes.py`; the titles of
  the native file and folder dialogs (`routes.py`, "Selecione …") and the
  dialogs' own failure messages ("Falha ao abrir o seletor", the hint that
  the picker does not open inside a container; a plain dismissal, "Cancelado.",
  is not among them: `lib/picker-feedback.ts` swaps it for the dictionary's
  text); the title and axis labels of some plots
  (`core/plotter.py`: the regression residuals, the anomaly score
  distribution); the WinError 1455 hint (`core/loader_lifecycle.py`,
  `core/detection_trainer.py`); the exceptions that the run-test, batch
  inference and ONNX-export endpoints relay as an HTTP `detail` ("não tem um
  checkpoint utilizável", "Caminho não encontrado"); and the `description` of
  the training fields in the config schema (`early_stopping_patience`,
  `num_workers`), which the custom-task form shows as written.
- **Portuguese but not shown:** the `note` of `/api/model/defaults` (the
  interface words the same two cases from the numbers in the response,
  `lib/model-advice.ts`) and the `detail` that hiding, restoring or deleting a
  custom task returns on success.
- **Not an issue:** the PatchCore phase labels. The trainer sends three, all
  Portuguese, and `lib/training-phase.ts` maps all three; there are no others.

Some messages are stored in component state when they happen (an import error,
the result of a download) and keep the language they were created in until the
next action; the ones that depend only on current state, such as the
"downloading…" notice, are worded at render time.

**Follow-up:** the "second step" that this leaves open is ADR-116: the page
sends its language and the server words its own messages. Of the inventory
above, it moves the health warnings, the dataset-scan messages, the dialogs'
titles and failure messages, the WinError 1455 hint and the exceptions the
run-test endpoints relay; the rest is listed there.

---

## ADR-111 — Stop honored by every executor; the queue says where it stops

**Date:** 2026-10-07
**Status:** Accepted
**Fixes:** ADR-094 (the token reached single runs and the classification
grid/random search, and nothing else)

**Context:** `DELETE /api/queue/{id}` answered 200 for any running job and set
its token. Only the single-run trainers and the classification grid/random
search ever read it. Classification K-fold and model comparison; the sweep,
K-fold, comparison, replicates and replicated comparison of the standalone
tasks; custom tasks; and PatchCore all ignored it. A 3-fold K-fold stopped
during its first fold was seen training all three folds to the end. That is the
ADR-094 failure mode again, one level up: the interface promised a stop that
did not happen.

A second defect sat under the first: a run stopped before its first epoch
crashed. The trainer broke out before epoch 1, and the block then loaded a
`best_model.pth` that was never written. Segmentation, torchvision detection
and the custom engine had "epochs=0" guards that saved the untrained weights
instead, so their run was scored (or, for detection, kept as `best.pt`) as if
it were a result; the autoencoder crashed on its empty history.

**One mechanism, one more carrier.** Blocks carry `_cancel_token` (ADR-094).
Runners now carry one too: the route executor hands the job's token to the
runner (`give_cancel_token`), the runner hands it to every block it builds, and
the orchestrator reads the same token off the runner between units
(`runner_cancel_token`). It is an attribute rather than a `run()` argument so
the `TaskRunner` protocol, and every test double that implements it, stays as
it was. The K-fold functions take `cancel_token=` the way their trainers do,
and the custom engine's `run()` takes it for its Level 1 loop.

**Where each job stops** is published as `stop_at` on every entry of
`GET /api/queue`: the boundary after which the job starts nothing new. Inside a
multi-unit job, the training in flight is still cut at its own epoch where it
has one.

| Job | `stop_at` |
|---|---|
| single run of any task (and a resumed run) | `epoch` |
| anomaly with PatchCore | `phase` |
| classification grid/random search, standalone sweep (grid/random/optuna) | `trial` |
| classification K-fold, standalone K-fold | `fold` |
| classification model comparison, standalone comparison | `model` |
| replicates, replicated comparison | `replicate` |
| custom task, Level 1 (the engine owns the loop) | `epoch` |
| custom task, Level 2 (its own `run` owns the loop) | `null` |
| custom-task sweep / replicates | `trial` / `replicate` |

The mapping lives beside `_submit_job`, keyed by the strategy the job is
queued under; the two cases decided by the config (PatchCore, Level 2) are
passed explicitly. A strategy missing from the table maps to `null`: the
default is to not claim a stop. The field is additive, so a client that does
not know it ignores it.

**`null` means not stoppable, and DELETE says so.** A Level 2 custom task owns
its loop and its `ctx` has no hook to read a token at, so there is no honest
way to stop it. Faking one (killing the thread) is not available in Python and
would leave a half-written run directory. `RunQueue.cancel` raises
`NotStoppableError` for such a running job, the token is left untouched, and
the endpoint answers 409 with a Portuguese detail. A pending job is dropped as
before, whatever its `stop_at`. A sweep or replicate set of a Level 2 task is
still stoppable between trials: its current trial runs to its end.

**A unit the stop cut is recorded as `stopped`.** It keeps whatever metrics
it reached, and it is left out of every aggregate and ranking: a fold cut at
epoch 2 averaged with folds that ran to epoch 30 gives a mean no configuration
produced, and a cut trial that happens to score well would be written out as
the best config. The aggregates are therefore computed over the units that ran
to the end, with their real `n`. K-fold's `cv_aggregate` gains
`n_folds_stopped` next to `n_folds_ok` and `n_folds_failed`, the replicate
aggregate's `n` counts only finished seeds, and the comparison report counts
`stopped_count` apart from `failed_count`. The classification grid and random
search, which already stopped between trials, follow the same rule; they used
to rank a cut trial with complete ones.

**Whether a unit was cut is the trainer's answer, not the token's.** The first
version of this decision marked as `stopped` whatever unit was running when
the token flipped, on the grounds that the orchestrator could not tell a unit
cut at epoch 2 from one already in its last epoch. That was wrong in two ways
review found: a Level 2 custom task runs its unit to the end whatever the
token says, and a stop that lands in a built-in's last epoch, or in its test
evaluation, leaves a finished unit. Both were labelled stopped and dropped; a
replicate set stopped during its first seed then failed with "stopped before
the first replicate finished" although that replicate had finished, and a
2-fold K-fold wrote a `failed` run.json. The loop knows: every trainer result
now carries `stopped`, true only when its loop broke on the token (a run that
stopped before its first epoch, and a PatchCore that skipped its scoring,
count as cut; an Ultralytics run counts only when epochs were still left). The
block reports pass it on (`report["train"]["stopped"]`, or
`report["detection"]["stopped"]`), the runners carry it as `RunResult.stopped`,
and only that marks a unit. The token still decides whether the next unit
starts, read once per unit: reading it once to mark and again to break let a
stop land between the two reads.

**A ranking is a comparison too.** In a replicated comparison, a variant
stopped after one seed was crowned `best_by_mean` over a variant with three,
on a mean of one sample, while the same report listed it as too small to test.
`ranked_by_mean` now holds only the variants the paired tests could take, and
their means are taken over the seeds all of them finished (`ranking_seeds`).
With every seed finished that is the plain mean, as before. This also changes
complete runs in which a seed genuinely failed for one variant: `best_by_mean`
used to compare each variant's own mean, over different seed sets, and is now
taken over the seeds every ranked variant finished, so it can name a different
winner than before. Variants that never started are listed in `not_run`
instead of vanishing, and the power note quotes the number of seeds actually
paired, not the number requested.

**K-fold reports `n`, and no spread for one fold.** A stopped K-fold often has
one finished fold, and `np.std` of one value printed "± 0.0", which reads as a
measured absence of variation. The standalone K-fold aggregate now carries `n`
beside `mean` and `std`; the classification K-fold report carries `n_folds_ok`.
The std is the sample std (n-1), the convention of the replicate and paired
statistics, and null below two folds. The change of ddof also applies to
complete runs: their std grows by a factor of sqrt(n/(n-1)). Older run.json
files hold the population std, and comparing runs puts the two side by side,
so every CV std written now carries `std_ddof: 1` beside it -- in
`metrics.cv_aggregate` and the `cv_summary.json` aggregate of the
classification K-fold, in the classification K-fold report, and in each metric
of the standalone K-fold `aggregate`. A std without `std_ddof` is ddof=0.

**The LaTeX tables say what ran** (ADR-061: they are the paper-ready file). A
K-fold table used to list every fold row, so a fold cut at epoch 2 printed its
half-trained metric as an ordinary fold under a caption claiming all of them.
Now a fold that did not finish shows its status (`2 (stopped)`, `2 (failed)`)
and dashes, and the caption counts the folds the mean is over ("1 of 5 folds
finished (1 stopped, 3 not run)"). The replicates note lists only the seeds
that finished, and how many were requested; a stopped sweep's note says how
many of the planned trials ran, which needs `planned_trials` in the sweep
report (`total_trials` counts the trials that ran). The SD label reads
"SD ($n-1$)", since it is the sample SD. The other exports follow the same
rule: the standalone `{kind}_ranking.csv` keeps only finished units, as the
classification `ranking.csv` always did (ranked for comparisons and sweeps),
and the Markdown model card adds "Stopped: yes (2 of 5 epochs)" -- or folds,
or the PatchCore phase -- when the run-level `stopped` is true.

**A stopped job ends like a completed one, with the counts actually done.** It
emits its normal `end`, writes its summary (and, for classification K-fold, its
top-level `run.json`) with the folds, models, replicates or trials that ran,
and its status is `completed`, the same convention as a stopped single run,
which is `completed` with fewer epochs than configured. The classification
grid and random search reports gain `failed_count` and `stopped_count`, as the
comparison report has, so a cut trial is not read as a failed one.

**Whether a run was stopped is a field, not a message.** A reader should never
have to recognise a stop from Portuguese text, so every run carries a boolean
`stopped`, true when a user stop cut it:

- a single run: its epoch loop broke on the token (or it never ran an epoch,
  or a PatchCore skipped its scoring). It is in `run.json` (top level), in the
  report (`report.train.stopped`; `report.detection.stopped` for detection;
  `report.stopped` for a custom task) and in the result endpoint;
- a multi-unit job: a unit was cut, or units were left unrun
  (`job_was_stopped`). It is in the report (`report.stopped`), in the K-fold's
  top-level `run.json`, and in the result endpoint. A stop that arrived during
  the last unit, which then finished, cut nothing: `stopped` is false;
- `GET /api/experiment/result/{id}` carries it as a top-level `stopped`,
  whichever of the above the run is.

`completed` keeps meaning only that the run ended cleanly.

**An error inside the stop window is a failure.** A unit that raised -- an
out-of-memory after the stop, an evaluation that failed after a cut -- is
`failed` with its error, never `stopped`: only a loop that broke on the token
is a cut. The runners return `stopped=False` with a failure, the orchestrators
mark `stopped` only on a successful result, and the K-fold blocks drop the cut
when the fold raised.

This holds even when the stop landed before any unit finished. The first
version failed such a job ("Parado antes de concluir a primeira dobra..."), and
the result endpoint serves a failed job as a 500, so a stop the researcher
asked for surfaced as a server error. A stop is not a failure: the job is
`completed` with zero finished units, no mean or ranking (`null`, an empty
`top_3`, no `best_trial`) and `stopped: true`, and a K-fold's `run.json` says
`completed`. `failed` is kept for jobs whose units genuinely all failed and
that the stop did not cut. A stopped replicated comparison is reported even
when no pair is left to test, because the variants that ran are still a
result.

**PatchCore stops between phases.** The token is read between
feature-extraction batches and after each phase. Stopped during extraction, it
keeps nothing: a memory bank built from part of the normals would be a
different model that looks like this one. The bank itself is one call and is
not interrupted; a stop that arrives while it is being built lands when it is
done, the bank (which is the model, and the expensive part) is saved, and
scoring is skipped. Its `run.json` then records one fit and a null AUROC, and
the run can be scored later from History.

**Ultralytics is checked once more before training starts.** It reads its stop
flag only when an epoch ends, so a stop that landed during setup would
otherwise still cost a whole epoch.

**A run stopped before its first epoch ends coherently.** Every block checks
`total_epochs == 0` (an empty history for anomaly) and skips the reload and the
evaluation, and the trainers no longer save untrained weights in that case.
`run.json` records 0 epochs and `null` for the best metric: the classification
and regression `inf` sentinel was written as `Infinity`, which is not JSON, so
the result endpoint could not serve it. Metrics that were never measured are
`null` instead of `0.0`, and `artifacts.model` is `null` when no checkpoint
file exists, so the actions that need one say "no checkpoint" instead of
failing to open a path (the classification "+ testar" now checks it like the
other families). Such a run is not resumable, by the rule that already
decides it (ADR-092): no `resume.pt` is written before the first epoch, so
`_resume_status` says no. `_RESUMABLE_BLOCKS` is unchanged, and no multi-unit
job became resumable.

**What the interface reads.** The GUI takes all three from the contract above
rather than from any message text: `stop_at` from the queue snapshot decides
whether a running job is offered a stop button and what it promises; a unit
whose `status` is `"stopped"` is shown as stopped, not as failed; and the
run-level `stopped` of the result says the run was stopped. `std_ddof` is
persisted for whoever compares an older K-fold with a newer one: a CV std
without it is the population std (ddof=0).

**A unit the stop cut cannot be continued alone.** A trainer keeps its
`resume.pt` when the token cuts it, which is what lets a stopped single run be
continued. A cut unit of a sweep, comparison, replicate set or K-fold is one
cell of a table that stays `stopped`: continuing it would finish that lone
training as a run detached from its job, while the summary still lists it as
`stopped`. So wherever a unit is marked `stopped` its resume state is dropped
(`discard_resume_state`), and `_resume_status` answers no by the rule that
already decides it (ADR-092). The sweep, comparison and replicate
orchestrators do it from the `RunResult.run_dir` their runner reports
(`discard_cut_unit_resume`); the regression and segmentation K-fold functions,
which drive the trainer without a runner, from the fold's `model_path.parent`.
For Ultralytics that state is `weights/last.pt`, which goes with it; `best.pt`
and the other artifacts stay. The classification K-fold needed nothing for
History, since its fold configs carry `block: cross_validation`, which
`_RESUMABLE_BLOCKS` does not list; its fold directory still holds the file.

**Not done:**

- PatchCore's memory bank cannot be interrupted while it is being built.

**Amended 2026-10-16: the run detail carries `stopped`, and `end.total_epochs` is the
epochs that ran.** `RunSummary` already said whether a user stop cut a run, but
`RunDetail` (`/api/runs/{id}`) dropped the field, so the detail page could only guess
from the epoch count. It is served now. The `end` event of a stopped run reports a
`total_epochs` smaller than the `epoch_end` ones: that is deliberate, not a drift. The
`epoch_end` events carry the configured total and `end` carries how far the history
got, and the overlay reads the difference as the evidence of a stop
(`lib/run-control.ts`). The trainer now says so where it emits the event.

---

## ADR-112 — The History comparison names no winner from a single seed

**Date:** 2026-10-09
**Status:** Accepted
**Complements:** ADR-056 (multi-seed replicates), ADR-061 (paired tests with
Holm), ADR-074 (what a single run's interval is not)

**Context:** The History comparison sets 2+ runs side by side: a metric table and
the epoch curves. Its rows were a fixed list of classification metrics, so a
detection, regression, segmentation or anomaly comparison was dashes in almost
every row; that was fixed by listing each task's own metrics and by never mixing
tasks in one table. Making the table readable made a second problem visible: a
run in History is one training under one seed, and the project's whole premise
(ADR-056) is that seed-to-seed variance routinely exceeds the gap between two
configurations. A table that marks the larger of two such numbers is answering
"which is better?" from one sample of each, with the authority of a table. It is
the caveat ADR-061 already writes into its LaTeX notes for a sweep ranked on one
run per configuration: the gap still reflects seed noise, and it reads as a
result because of where it is printed.

**Decision:** the comparison describes the numbers and does not rank the runs.

- **The mark is "highest value" or "lowest value", never "best".** In a row with
  a direction, the cell holding the highest value of a higher-is-better metric,
  or the lowest of a lower-is-better one, is marked with a neutral tint and that
  tooltip. A glyph beside the metric name (↑ or ↓) says which end is better, so
  the mark can be read without being taken for a verdict. Ties are all marked;
  equal values, a single measured value, and rows with no direction (epochs, a
  decision threshold) mark nothing.
- **One direction rule.** The direction is `infer_direction` of the backend
  (`core/significance.py`), copied into `lib/compare-metrics.ts`, and a custom
  task's declared direction wins over the name exactly as in
  `runner_metric_direction`. A third rule would let a metric read one way in a
  sweep ranking and another in this table.
- **A note under the table whenever it marks something.** One line, in both
  languages: each run has a single seed, the gap can be seed variation, and
  claiming a difference takes replicates with several seeds. It is tied to the
  mark rather than to the number of runs: a table with no highest or lowest cell
  implies no winner and warns of none (`lib/compare-seeds.ts`, `seedNote`).
- **Runs of different tasks are refused, not tabulated.** A detection's mAP and a
  regression's R² have no metric in common, and the config diff and the curves
  below the table are meaningless across tasks too, so the panel names each run's
  task and asks for a selection of one. The task comes from the run detail
  (`task`, by the same `_run_task` rule the resume check uses): `config.task`
  alone cannot say it, since classification keeps its problem type there and a
  custom task keeps nothing.

**Not done: a replicate group beside the runs.** The intent was to show, for a
compared run that is a replicate group, its mean ± CI95 instead of a single
value, and the paired test between compared variants. History has no such run.
`replicates` and `replicated-comparison` jobs write their report
(`replicates_summary.json`, `comparison_summary.json`, with `aggregates` and
`comparisons`) under `outputs/reports/<name>/<timestamp>/` and nothing in the
models directory; each seed is an ordinary run named `<name>_s<seed>`. The
K-fold is the only multi-unit run in History, and its numbers are means over
folds of one seed. So everything the panel can compare is a single seed, and the
note says so for all of it. Showing a group needs the group to exist as a
History entry (as a K-fold's `run.json` does) or its report to be served; either
is a decision about History itself, not about this table. The frontend computes
no statistics either way, and the report would hand it less than the intent
assumed: the paired result is a raw `p_value` and the Holm-corrected
`significant` flag, with no adjusted p.

*Resolved by ADR-113: the group now exists as a History entry.*

**Consequences:** a comparison of runs from one replicate set (`<name>_s42`
against `<name>_s43`) carries the same note, which is true of each of them even
if the advice to run replicates has already been taken; the note does not try to
recognise a replicate by its name. The curves and the config diff are unchanged,
and the curves still plot classification's validation accuracy and loss, so a
detection or anomaly comparison has no chart yet.

---

## ADR-113 — Replicate groups are History runs

**Date:** 2026-10-09
**Status:** Accepted
**Complements:** ADR-056 (multi-seed replicates), ADR-061 (paired tests with
Holm), ADR-111 (the run-level `stopped` marker), ADR-112 (resolves its "Not
done")

**Context:** A replicate set and a replicated comparison are one job made of
several trainings. Each seed is an ordinary run named `<name>_s<seed>`, and the
job's own result (mean, 95% interval, the paired tests) was written only to
`outputs/reports/<name>/<timestamp>/{replicates,comparison}_summary.json`.
History scans the models directory, so it showed N loose runs and nothing about
what they add up to: the number a paper would quote was not in the interface,
and the History comparison could only ever tabulate single seeds (ADR-112). The
classification K-fold had already solved the same shape by writing a top-level
`{base}_cv/run.json`.

**Decision:** a replicate job writes a `run.json` of its own, in the models
directory, and History lists it as one run.

- **Where and what.** `<models_dir>/<name>_replicates/<run_id>/run.json` for a
  replicate set and `<name>_comparison/<run_id>/run.json` for a replicated
  comparison, `run_id` being the job's id (`<name>_<timestamp>`). The sub-folder
  per group means a second run with the same name does not overwrite the first,
  which would orphan its seeds. The top-level keys are the ones every run
  carries and `_parse_run_summary` already reads (`id`, `experiment`, `status`,
  `stopped`, `timestamp`, `config`, `metrics.total_epochs`, `block`), plus
  `task`, `task_label`, `device_used`, `environment` and `dataset_fingerprint`
  taken from a child, since a researcher-defined task stamps its identity there
  and the group's own config cannot. `block` is `replicates` or
  `replicated_comparison`. `status` is `completed` and `stopped` says the job
  was cut, as for a K-fold (ADR-111).
- **A `group` section** holds what only a group has: `kind`, `metric`, `seeds`
  (requested), `n_requested` and `n_finished` trainings, `children` (per seed:
  `seed`, `variant`, `status`, `run_id`, `run_dir`, `metrics`, `error`), the
  per-metric `aggregates` (`n`, `mean`, `std`, `std_ddof: 1`, `ci95_low/high`,
  min/max and the bootstrap interval) and, for a comparison, per-variant
  aggregates, the Holm-corrected `comparisons` (`p_value`, `significant`,
  `effect_size`, `effect_label`, the bootstrap interval of the difference,
  `underpowered`), `ranking_seeds`, `ranked_by_mean`, `best_by_mean` (null when
  no ranking is possible), `not_run`, `skipped_variants` and `report_path`.
- **Nothing is recomputed.** Every statistic is copied from the report the job
  already built; a second computation could drift from the report a paper's
  table is made from. The only transformation is that a non-finite number
  becomes `null`, since a bare `NaN` is not JSON and no browser parses it. The
  one thing added at the source is `std_ddof: 1` in `aggregate_replicates`, so
  the convention of the `std` is stated where it is computed.
- **Children are tagged, not moved.** Each seed's own `run.json` gets an
  additive `group_id` (the group's `run_id`) when the group is written. A seed
  stays an ordinary run, openable and testable on its own; `ReplicateTrial`
  gained `run_dir` so the group knows where each one is. A seed that never got
  a directory (a failed one) is listed in `children` with `run_id: null`.
- **The API adds fields, removes none.** `RunSummary` gets `group_id` and a
  slim `group` (kind, metric, seeds, counts, `child_ids`, the aggregates of a
  replicate set, the variants and `best_by_mean` of a comparison); `RunDetail`
  gets `group_id` and the full `group` section. A replicate set's `metrics`
  carry the mean of each metric under the key a single run of the task writes
  (`test_r2`, `test_miou`, ...; the bare `map50` for detection; the declared
  name for a researcher's task), so the history projection and the comparison
  table find it in the row where it belongs, and `group.metric_keys` says which
  aggregate filled which key. A comparison has no single headline number, so
  its `final_metrics` are empty rather than the mean of whichever variant won.
  `RunGroupBrief.final_aggregates` hands the card the aggregate behind each of
  those numbers under the same label, read through `metric_keys`. A group is
  never resumable: it trained nothing itself.
- **History shows a group as one entry.** The list folds a seed under the group
  its `group_id` names (`lib/run-groups.ts`, `foldGroups`), collapsed by default
  and still openable, selectable, comparable and deletable as the run it is;
  tab counts, filters and the header count are over entries, so a group and its
  seeds are not counted twice. A seed whose group is no longer in the list (the
  group was deleted, which removes only its summary) stays visible as a loose
  run instead of vanishing with the folder that named it, and the delete dialog
  says so. A card prints `mean ± half-width` from the report's own interval, a
  comparison card its variants and `best_by_mean` or "no best mean".
- **The group's detail** shows per-metric mean ± CI95 with n, the interval, the
  sample SD (n−1, `std_ddof`) and min–max, a caution under five seeds, the seeds
  with their status and a link to each run, and for a comparison the variants,
  the paired tests with the raw p and "significant after Holm" (yes or no), the
  `not_run` variants, and `best_by_mean` as a dash with the reason when there
  is none (a stop left variants unrun, or fewer than two seeds are common to
  the variants). A seed's own detail links back to its group.
- **The comparison reads a group as a mean, not a seed.** Its cells show the
  mean with the interval (`metric_keys` says which row an aggregate fills, so a
  validation-score row is never given a test-split interval by name); a
  replicated comparison, which has no single value per row, shows a dash. The
  single-seed note of ADR-112 is judged on the cells the table highlights, group
  means included: a row warns when it marks a highest or lowest value and a
  single-seed run measured it. Two groups side by side raise none; one single
  run beside a group mean does, since the highlighted gap then rests on one
  seed, worded for "the runs that are not replicate means". The frontend
  computes no statistic: the one subtraction, the interval's half-width, prints
  the report's own bounds in the form the results view already used.
- **A group has no checkpoint, and says so.** Continue, test, batch prediction,
  ONNX export and Grad-CAM on a group's id answer 400 ("é um conjunto de
  réplicas: abra uma das seeds") through one guard, instead of failing by
  accident further in. Deleting a group (only its summary goes) and revealing or
  exporting it are unchanged. The caution under five seeds also shows for a
  replicated comparison, judged on its weakest variant.
- **A researcher-defined task's seeds reach the group.** `CustomTaskRunner`
  returns the run directory the engine trained into, the way the built-in
  runners do; without it every trial was recorded with no directory, so no seed
  was tagged, the group had no child to take `task: custom:<key>` from and
  History read it as classification.

**Consequences:** the group is written once, when the job ends, so a server
killed mid-job leaves its seeds as loose runs, as before; a failure to write it
is logged and does not fail a job whose training and report are already on disk.
Runs written before this ADR have no group and no `group_id`; they stay loose
and nothing infers a group from their names. `n_requested` counts trainings (a
comparison of 3 variants over 5 seeds asks for 15), including those a stop kept
from starting. The report files under `outputs/reports` are unchanged, and the
`.tex` table and CSV remain the paper-ready form. Tagging a seed rewrites its
`run.json`; one that cannot be rewritten (on Windows an antivirus or the indexer
can hold the file) is logged, its temporary file removed, and it stays a loose
run while the other seeds are tagged and the group is still written. Re-training
a seed rewrites its `run.json` and drops `group_id`: it reappears as a loose run
and the group's aggregates do not update. This is rare, since a seed a stop cut
is not resumable on its own (ADR-111).

## ADR-114 — Profiles are output folders; organisation, not access control

**Date:** 2026-10-10
**Status:** Accepted
**Complements:** ADR-075 (one run queue per server), ADR-090 (the first-run
welcome and its name), ADR-113 (History is the models folder)

**Context:** a lab can run one `visionforge gui --host` for several people. The
History is the models folder of the machine, so everyone sees everyone's runs,
and finding your own among a colleague's thirty is the daily friction. What was
asked for is narrow: on entering, say who you are, and see only your own
training history. Nothing about who may *read* anything.

**Decision:** a profile is a folder, and it changes where outputs go and which
runs the History reads, nothing else.

- **A profile IS a folder.** `outputs/profiles/<slug>/` holds `models`,
  `graphics`, `logs`, `reports` and a `profile.json` with the display name.
  Listing profiles lists those folders; creating one creates the folder. There
  is no user database and nothing to migrate.
- **Slug.** 1-40 characters of lowercase ascii letters, digits, `-` and `_`,
  starting with a letter or digit, derived from the display name (accents fold
  to their letter, other characters become `-`). The display name (up to 60
  characters, whitespace collapsed) lives only in `profile.json`. `default` and
  the Windows device names (`con`, `nul`, `com0`-`com9`, `lpt0`-`lpt9`...) are
  reserved; the superscript spellings (`com¹`) fold to `com1` before the check.
  Two names that fold to the same slug ("João", "Joao") are the same profile:
  the second create answers 409 naming the first. When the operating system
  refuses to make the folders (no permission, full disk, a file where
  `outputs/profiles` should be) the create answers 500 with a Portuguese message
  and removes the half-made folder, so the same name can be tried again.
- **The default profile is the layout that already exists.** It maps to
  `outputs/models` and its siblings, so nothing moves, an install that never
  creates a profile behaves as before, and runs started from the command line
  keep writing there. Its runs and a named profile's are disjoint folders, so
  the default History does not list a profile's runs.
- **The browser keeps the choice and sends it on every API call** as
  `X-VF-Profile: <slug>`, from `localStorage`. The server validates it where it
  is used: not a safe slug (`../x`, uppercase, too long) is 400, a safe slug with
  no folder is 404, an absent or `default` header is the default profile. A
  slug cannot hold a separator or a dot, and the resolved folder must also be a
  direct child of `outputs/profiles` named exactly like the slug, so a link
  planted there is refused rather than followed. The same goes for the profile's
  own `models`, `graphics`, `logs` and `reports`: one that is a link pointing
  elsewhere makes the profile invalid, because a run would write its files
  outside it.
- **Links are confined for a named profile only.** Looking a run up for a
  History action (`_find_run_dir`) skips a hit whose real path left the
  profile's models folder, so a link planted inside a profile cannot reach a
  folder elsewhere. The default profile does not apply that check: it is the
  layout from before profiles, where old runs moved to another drive and linked
  back under `outputs/models` are legitimate and the History already lists
  them. Zero change for an install that never creates a profile.
- **What a profile changes.** (1) *Where a submitted run writes*: at submit
  time `config.output.{models,graphics,logs,reports}_dir` are forced under the
  profile for every executor (a single run, a sweep, K-fold, a model comparison,
  replicates, a replicated comparison, a researcher-defined task, and the
  detection, regression, segmentation and anomaly runs); a resumed run is scoped
  again to the profile it is found in. A sweep applies its search-space paths,
  and a replicated comparison its variants' override paths, to that scoped
  config once per trial, so a path under `output` (`output.models_dir`, or
  `output` itself) would undo the scoping: both are refused with a 422 before
  anything is queued, for every profile, since no experiment varies where its
  files go. The default profile leaves the rest of the config alone. (2) *The root the History reads*: one per-request resolver
  (`current_profile` in `gui/api/routes.py`, built on `gui/api/profiles.py`)
  replaces the module-level `_MODELS_DIR` in the list, the detail, the reveal,
  resume, delete, the markdown export, testing a run, Grad-CAM, batch
  prediction and ONNX export. The comparison, the ranking and the replicate
  groups are built in the browser from the list and the detail, or written under
  the models folder, so they follow without code of their own.
- **What stays shared.** Datasets, credentials, custom tasks and the run queue.
  The queue snapshot shows each job's profile display name, so a colleague's job
  ahead of yours is visible.
- **No passwords, and the interface says so.** Anyone who can reach the server
  can send any profile's slug, and the welcome screen states plainly that
  profiles organise runs and do not protect them. The existing name prompt
  keeps working: with no profile besides the default the flow is unchanged
  except for an optional "criar perfil" link.

**Interface.** The welcome overlay becomes "Quem é você?": the existing
profiles (the default shown as "Padrão (sem perfil)") and "criar novo perfil",
and a small chip in the header switches. Text lives in both dictionaries.

- *Who is asked.* Only on a server that has a profile besides the default, and
  only while this browser has not chosen one that still exists; a folder deleted
  by hand leaves an orphan slug in `localStorage`, which does not count as a
  choice. A one-person install never sees the question: it keeps the name
  prompt and gains an optional "criar perfil" link under it. A server that does
  not know `/api/profiles` (one started before this ADR) is treated the same
  way, and a failed read of the list never erases a stored choice.
- *One question, not two.* Choosing a named profile also stores its display
  name as the name the header greets, since a shared server's profile is who the
  person is; "Padrão (sem perfil)" says no name and keeps the one already
  saved. The header's profile chip appears only on a server that has profiles,
  and stands in for the greeting chip when both would print the same name.
- *The choice travels in one place.* The API client adds `X-VF-Profile` to every
  request, including the model-card download, and sends nothing for the default
  profile or for a stored value the server would refuse. Switching profiles
  closes the History sheet and resets its count, because both describe the
  other profile's runs. The choice lives in `localStorage`, which every tab of
  the browser shares, so a switch in one tab moves the next calls of the others;
  each tab listens for the `storage` event on the profile keys and follows the
  same way (chip, closed History, the greeting name), instead of keeping a chip
  that names a profile its calls no longer use. The queue panel names each job's profile once a job of a
  profile other than the default is in it.

**Rejected:** accounts with passwords (a security claim this tool cannot back:
the server runs training code and reads any folder it is pointed at); one
server per person (they share one GPU, and the point of the queue, ADR-075, is
that they can); filtering the History by an `owner` field in `run.json` (it would
leave every run in one folder, and the runs written before this ADR have no
owner).

**Consequences:** `GET /api/profiles` lists and `POST /api/profiles` creates; no
rename or delete, which a researcher does in the file manager (the folder is the
profile). Profile-scoped means History and where outputs land, so a few things
are deliberately not scoped: `GET /api/artifacts/...` still serves any file
under `outputs/` (an `<img>` cannot send the header, and this ADR makes no
confidentiality claim), the live progress stream and `/experiment/result/{id}`
belong to the one running job, and the preview cache is shared. `profile.json`
is read leniently: a folder made by hand with a valid name is a profile whose
display name is its slug. The orphan sweep of filtered detection copies at
server start now also covers each profile's models folder.

## ADR-115 — Guides with gates; a synthetic sample dataset

**Date:** 2026-10-10
**Status:** Accepted
**Builds on:** ADR-104 (a guided tour, offered once), ADR-060 (the self-test's
synthetic datasets), ADR-114 (outputs are per profile)

**Context:** the tour of ADR-104 says where things are and waits for nothing: a
stop highlights a button and the card says what it does, but the researcher
reads it, clicks "Continuar" and is never asked to *do* it. A newcomer who
finishes it has seen the Train button and has still not trained. Two things are
missing for a guide that teaches by doing: steps that wait for the researcher's
action, and something to train on, since a first-time user usually has no
dataset on disk. The project already generates a synthetic dataset for every
task, offline and deterministically (`utils/selftest_data.py`, ADR-060), and a
short classification run on it takes seconds on CPU.

**Decision.** Four parts, shipped as the smallest slice that works end to end:
classification only.

**1. A sample dataset on request.** `POST /api/sample-dataset` with
`{"task": "classification"}` builds the self-test classification layout
(`train/val/test`, two classes, six 32px images per class and split) into
`datasets/exemplo-classificacao/`, relative to the folder the GUI was started
from, creating `datasets/` if needed. It answers 201 with
`{path, classes, counts}` where `path` is absolute and `counts` is images per
split. Local and offline: nothing is downloaded.

- *It never overwrites.* If the folder exists and holds anything, the answer is
  409 and the body carries `path` next to `detail`, so the guide can offer that
  folder instead of failing; an empty folder left by hand is taken. The 409 is
  not an error for the guide, it is "you already have one".
- *No half-written folder.* The builder runs in a hidden scratch folder beside
  the target and is renamed into place, so a crash cannot leave a partial
  `exemplo-classificacao/` that the next call would take for a finished one.
- *Only classification.* Any other known task is a 400 saying it has no sample
  yet, and an unknown task is a 400 too; neither creates a folder. `task` is a
  plain string for that reason: a `Literal` would turn both into a 422.
- *Datasets are not per profile* (ADR-114 left them shared), so the folder is
  not under a profile.
- The folder is added to the session's dataset allow-list, so its thumbnails
  load in the dataset card like any folder the researcher picked.

**2. Guides are a registry.** `lib/guides/` holds `GuideDefinition { id,
title(t), steps(t) }` for two guides: the existing tour, unchanged (its steps
are the ones of ADR-104 and have no gate), and "Primeiro treino". The header's
"guia" button opens a small menu with both, and the first-visit invitation
offers both (it used to offer one). A new guide is a new entry in the registry
and new text in the two dictionaries, nothing in the component.

**3. Steps may wait.** A step can carry `waitFor(facts)`, a predicate over
facts read from the real app state, and `action`, a button on the card that
does something.

- *The facts* are three booleans: the classification form has a dataset path,
  the training stream has produced a `start` event, and it has produced an
  `end` event. They come from what the screen already holds (the form data and
  the training hook's events), not from a second source of truth.
- *Events count only when they arrive after the guide is open.* The training
  hook keeps the previous run's events until the next submission or until the
  results are closed, so a researcher who opens the guide after a finished run
  would find the "training started" gate already open. The latch (`advanceFacts`,
  pure and tested) records whether `start` and `end` were present when the guide
  opened and only counts a transition to present. A run that is already in
  flight when the guide opens does not count either; the guide waits for its
  own.
- *A gate never traps.* "Pular", the ✕ and Esc always leave, and Back always
  works. While a gate is closed "Continuar" is disabled (the arrow key too) and
  the card says what it is waiting for in one line. Nothing advances on its own:
  the researcher is reading, and a card that changes under their eyes when the
  event arrives is worse than one more click.
- *Actions return a result the card shows.* The one action built here creates
  the sample dataset and fills the form's dataset field, which makes the
  existing auto-detection run exactly as if the folder had been picked; a 409
  fills the field with the existing path and says so. An unreachable server
  shows its message on the card, not a silent no-op.
- *Steps may also* run something when they open (`onEnter`), ask for a card
  that does not dim the page (`floating`) and ask for the target near the top of
  the screen (`align: "top"`). `onEnter` switches to the classification tab and
  puts the training sheet away before the History is pointed at. `floating` is
  for the two steps about watching the training sheet, where a 74% dim over it
  would hide the thing the card describes: the card is 340px wide and goes
  beside the sheet when a side has room, and otherwise in the bottom-left
  corner, because the sheet keeps its stop, minimise and "view results"
  buttons at the bottom right and the last of them is what the researcher must
  click when the run ends (found in the browser: a corner card on the right
  covered it at 1424px). `align: "top"` is for the steps with a tall card
  (the sample-dataset button): with the target in the middle of an 800px window
  the card fits neither above nor below it and, centred, covers the field it
  points at. The guide now sits above the training sheet (z-index 95 against
  90) and still below the History, Datasets and Queue sheets (100); the header
  rose from 3 to 4 so its menu is not painted over by the tab row.

**4. The "Primeiro treino" script.** Eight short steps: create the sample
dataset (and why there are three splits) → point the dataset (gate: a path is
set; the classes it found) → the few parameters that matter, with epochs at 3 →
train (gate: `start`) → what is on the training sheet (epochs, train versus
validation loss) → the result (gate: `end`; accuracy, the two losses, what
overfitting looks like) → the History (where the run is saved, `run.json`,
repeating with another seed) → "E nas outras tarefas?", one line per task on
what changes in the data format and the metric. It says plainly, in the result
step and again at the end, that this data is easy on purpose: a near-perfect
accuracy teaches the mechanics and says nothing about what to expect on a
researcher's own images, and the last step points at the download card in
Datasets for a real one.

**Rejected:**

- *Waiting by watching the DOM* (a step is done when an element appears). The
  DOM changes for reasons unrelated to what the researcher did, and the gates
  that matter (the training started, ended) are not DOM facts.
- *Advancing automatically when a gate opens.* See above; it also fights the
  researcher who wants to read the previous card once more.
- *Generating the sample in the browser.* It would put PNG encoding in the
  frontend and give the server no folder to train on; the training reads a path
  on the server's disk.
- *A real dataset (CIFAR-10) as the sample.* It needs the network, tens of
  megabytes and minutes on CPU; the first run should take seconds and work on a
  plane. A real dataset is the guide's last suggestion instead.
- *One guide per task now.* Detection, segmentation, regression and anomaly
  each need their own builder call, their own form anchors and their own gates;
  the registry and the gates are what they will reuse, so they are a later
  slice, and the last step says in one line what changes for each.

**Consequences:** `datasets/exemplo-classificacao/` appears in the working
folder of whoever uses the guide; deleting it is what lets the guide generate a
fresh one. The default model is pretrained, so the first run on a machine that
has never trained a ResNet-18 downloads its weights once; the parameters step
says so. The tour's script, its anchors and its tests are untouched.

**Verified in the browser** (headless Chrome over CDP, on CPU): the invitation
offers both guides; the header menu opens above the tab row; the sample button
created the folder and filled the field (splits detected, binary applied,
thumbnails shown) and a second click got the 409 and reused the folder;
"Continuar" is disabled at the train step and the arrow key does not move it;
with a finished run's events still on screen the gate stayed closed until a new
run started; on a 20-epoch run the result step stayed closed through all twenty
epochs and opened at the end; the tour plays as before (stop 1 of 7, nothing
gated). One thing the run corrected in the first draft of the text: it claimed
the training loss "usually sits below" the validation loss, and on this very
dataset the first epoch's training loss (0.68, averaged while the model is
still changing) is above the validation loss (0.20), so the step now says how
each one is measured instead.

## ADR-116 — The server words its messages in the language of the page

**Date:** 2026-10-10
**Status:** Accepted
**Completes:** ADR-110 (the bilingual interface left the server's own texts as
they were), reusing the per-request header of ADR-114

**Context:** the interface is bilingual since ADR-110, but a message that the
server composes arrived in whichever language its author wrote it in: the
training-health warnings, the dataset-scan messages and the dialogs' failures in
Portuguese, the `detail` of most HTTP errors in English. A researcher who picked
English read Portuguese in the middle of a screen, and one who picked
Portuguese read English. ADR-110 named the choice (translate at the API edge, or
send codes the page words itself) and postponed it.

**Decision.** Both, each where it fits.

- *The page says which language it wants.* `request()` in `api/client.ts`, next
  to `X-VF-Profile`, adds `X-VF-Lang: pt|en` to every call, read from the same
  place the error texts of that file already read it (`vf.lang`, else the
  browser). It is not `Accept-Language`: that is the browser's preference, and the
  header toggle exists exactly for the person whose browser and interface
  disagree.
- *The server resolves it once per request.* `resolve_lang` takes the primary
  subtag (`en-US` is `en`) and returns Portuguese for an absent, empty or
  unknown value: there is no negotiation and no quality values, because the
  caller is our own page. A router-level `async` dependency (`request_lang`)
  stores the result in a `ContextVar`. Router-level, so the profile lookup that
  runs before the endpoint writes its errors in the right language; `async`,
  because a sync dependency runs in a worker thread and a context variable set
  there never reaches the endpoint.
- *Messages come from one catalog.* `utils/messages.py` maps a key to
  `{pt, en}`, each a `str.format` template; `tr(key, **params)` reads the
  current language. No gettext: two languages and about seventy entries need a
  dictionary and a test. It is in `utils/` and not in `gui/` because `core/`
  writes some of these texts and must not import the web layer. A context
  variable and not a `lang` parameter, because the messages are written in a
  dozen helpers and in the trainers, and the language rides through
  `asyncio.to_thread` and tasks by itself.
- *A queued job keeps the language it was submitted under.* The queue starts a
  job from its own task, long after the request that queued it, so
  `_submit_job` captures the language and wraps the job's `start` in it. A
  health warning, written by a trainer in a worker thread, is therefore in the
  language of whoever pressed Train; it is also what lands in `run.json`, next
  to a `code` that does not change. The command line binds no language of its own
  and keeps printing Portuguese; the figures it saves are the exception (see the
  second slice) and `VISIONFORGE_LANG` is how it chooses one.
- *Where the page already words something from structured data, the catalog does
  not duplicate it.* The plain dismissal of a dialog stays the sentence
  "Cancelado." because `lib/picker-feedback.ts` replaces exactly that string; the
  409 and 404 of a stop request are worded by `TrainingOverlay` and
  `QueueOverlay` from the status code. Their catalog entries are for every other
  client of the API.
- *The catalog is checked from both ends.* Every key has both languages, takes
  the same placeholders in each and differs between them; every `tr("…")` call
  in `src/` names a key that exists; every key is used somewhere. A typo fails a
  test, not a request.

**Moved in the first slice** (key prefix, then where):

- `health.*`: the six training-health warnings (`core/training_health.py`).
- `winerror.*`: the paging-file hint (`core/loader_lifecycle.py`) and the
  worker-crash message (`core/detection_trainer.py`).
- `detect.*`, `scan.*`: split detection and the messages of the classification,
  detection, segmentation, anomaly and regression dataset-stats, samples and
  preview responses (`gui/api/routes.py`).
- `pick.*`: the titles of the three native dialogs, their failure message, the
  missing-tkinter message and the container hint.
- `testrun.*`, `export.*`: what the run-test, batch-prediction and detection
  ONNX-export endpoints relay (`routes.py`, `detection_testing.py`,
  `detection_export.py`).
- `run.*`, `queue.*`, `reveal.*`: a run not found, still running, failed, not
  deletable or with nothing to continue; the replicate-group refusal; the
  "(resuming)" label; the stop refusals; the Origin and loopback 403s of "open
  folder".
- `profile.*`: every error of `gui/api/profiles.py` and the refusal to vary
  `output` in a sweep or comparison.

**Moved in the second slice:**

- `plot.*`: title, axes and legends of every figure `core/plotter.py` saves (loss
  and accuracy curves, the detection history, confusion matrices, ROC,
  precision-recall, predicted against actual, residuals, per-class bars, anomaly
  scores). A figure is an artifact, so it is worded for whoever started the run:
  the plotters call `tr_artifact()` while the job runs, under the language
  `_submit_job` captured. `tr_artifact` differs from `tr` in one place: when no
  language is bound at all, it answers English. A script, a notebook or a bare
  `visionforge run` has no language, and its figures were English
  (`Training & Validation Loss`, `ROC Curve`, `Confusion Matrix`) before this
  slice; they are what a paper gets, and turning them Portuguese without being
  asked would have been a regression. So the context variable now defaults to
  `None` ("nobody chose"), which `current_lang()` still reads as Portuguese for the
  messages, and `bound_lang()` exposes. A GUI request always binds one (Portuguese
  without the header), so a GUI run draws in the page's language. The figures of
  ADR-077 were Portuguese and become English in that unbound case. A name the
  caller passes (`label`, `title`, `metric_label`) is used as given, and so are the
  words that read the same in both languages (`normal`, `macro`, `mAP@50`).
- `VISIONFORGE_LANG=pt|en`: the command line reads it once, right after parsing its
  arguments, and binds it for the whole run (`__main__._bind_language_from_env`).
  A Portuguese researcher who wants Portuguese figures from `visionforge run` sets
  `pt`; `en` also words the health warnings and the other messages in English. An
  empty or unrecognised value is ignored (a typo does not silently pick a language),
  and under `gui` it changes nothing, because every request rebinds the page's
  language.
- `run.*` (the rest), `file.*`, `compare.*`, `replicates.*`: the `detail` of an
  unreadable `run.json`, a task that cannot be resumed, a stored config that is no
  longer valid, a delete refused outside the models folder, a run folder that is
  gone, an action a researcher-defined task does not offer, repeated seeds, and the
  refusals of the image preview and of the artifact server.
- `custom.*`: what hiding, restoring and deleting a researcher-defined task
  answer.
- `model.*`: the note of `/api/model/defaults` (`gui/api/model_notes.py` and the
  small-images sentence in `routes.py`). Its numbers are the same in both
  languages; the interface still words its own from the response's numbers.

**Moved in the third slice** (2026-10-15): the exceptions raised below the route
layer that a route relays as its `detail`. They are worded where they are raised,
not translated at the route, because the route only sees the finished sentence; the
language reaches them through the same context variable, across
`asyncio.to_thread`.

- `download.*`: every refusal of `gui/api/dataset_download.py` a person can cause
  from the page: an unknown torchvision dataset, the Roboflow and Kaggle dataset
  formats, the missing Roboflow key and version, the old Kaggle `username:key`
  pair, Kaggle credentials not found, a Hugging Face dataset without image and
  label columns, and a download that finished with no image (its list of stray
  extensions is a sentence of its own, appended).
- `export.no_checkpoint`, `gradcam.*`: a run with no checkpoint in
  `artifacts.model` (`torch_onnx_export.py`), Grad-CAM on a task that has no CAM
  path (`torch_gradcam.py`) or on a model with no `nn.Conv2d` (`core/gradcam.py`).
- `custom.*` (the rest): the confirmation and the missing file of `delete_task`
  (`tasks/manage.py`) and the "not registered" of `get_task` (`tasks/registry.py`),
  whose list of available keys is now joined with commas instead of printed as a
  Python list. The 404 that relays it read `str(exc)`, which for a `KeyError` is the
  repr of the message, quotes included; it now reads `exc.args[0]`.
- `credentials.empty`: a blank provider key (`utils/credentials.py`).

Left in English in this slice, on purpose: the errors only a programmer can cause
(the `register_task` checks a researcher meets while writing a task file, an unknown
provider name that the request schema already refuses, `val_fraction` out of range,
`ExportONNXBlock` without `export_onnx`), and the `ImportError` of a damaged install,
whose hint is a command line.

**Moved in the fourth slice** (2026-10-15): the two report texts that are not
figures, under the rule the figures already follow (`tr_artifact`: the language of
whoever asked, English when nobody chose).

- `latex.*`: captions, column headers and notes of the four tables
  `core/latex_export.py` writes (replicates, sweep, cross-validation, paired
  comparison), including the small-n caution, the stopped-sweep note and the
  "underpowered" warning. The tables are written by the job that ran the study, so
  they are in the language it was submitted under. LaTeX markup stays in the
  templates with its literal braces doubled; metric names, statuses and test names
  are data and pass through unchanged. "Fold", "Seed", "Scheduler", `$p$` and `$d_z$`
  read the same in both languages and have no entry.
- `card.*`: the Markdown model card of `/api/runs/{id}/export_md` and the
  `_stopped_extent` phrases it uses. Unlike the other routes, this one binds the
  language from `X-VF-Lang` alone and falls back to **English** without it, not to
  Portuguese: the page always sends the header, so the page gets its own language,
  and a script that fetched the card before this change gets the same text as
  before. Config keys and values (`num_classes`, the transforms table) are
  identifiers and stay as they are.

**Moved in the fifth slice** (2026-10-15):

- `config.*`: every message the config models and loaders raise
  (`utils/config.py` and the anomaly, detection, regression and segmentation
  configs, plus the unknown preprocessing step): a path that does not exist or is
  not a file or a folder, `batch_size` not a power of 2, `gpu_ids` and
  `multi_cuda`, a newer `schema_version`, `num_classes` against the task, `auc_roc`
  on multiclass, an invalid detector, `target_columns`, `num_targets`,
  `ignore_index`, a config file not found or not a YAML mapping. The five copies of
  "does not exist / must be a file / must be a directory" share one key each, with
  the field name as a parameter. They reach the page inside the `msg` of a 422,
  after pydantic's own "Value error, " prefix, which stays; the command line, which
  binds no language, now prints them in Portuguese like its other messages. Field
  names are identifiers and read the same in both languages. Pydantic's built-in
  messages ("Input should be greater than 0") are the library's and stay English.
- `schema.*`: the two hints the researcher-task form shows that VisionForge writes
  itself, `early_stopping_patience` and `deterministic` of the shared training
  block. `GET /api/custom/{key}/schema` replaces them in the page's language; the
  researcher's own descriptions are theirs and stay as written. The descriptions in
  the built-in configs' schemas are not shown by any screen (the built-in forms word
  their fields from the page dictionary), so they are left as written.

**What the training sheet turned out to be.** The lines of the sheet are not server
text. The server streams progress events (`start`, `epoch_end`, `trial_start`,
`phase`, `end`), which carry numbers, and `TrainingOverlay` words every line from
them with the page dictionary; the `logger.*` calls of the trainers go to the log
file and the console, not to the page. The "TRAINING COMPLETE" in the header and the
"$ training complete" of the last line were `trainingComplete` in the Portuguese
dictionary, written in English, as were `trainingFailed` and the `training · <task>`
header; they now read `treino concluído`, `treino falhou` and `treinando · <tarefa>`.
Two things in the sheet are still the server's: the `error` of a failed run (an
exception message, below) and the label of a PatchCore phase, which
`lib/training-phase.ts` already maps for the three known ones.

**Still as written, on purpose or not yet:**

- The `logger.*` lines of the trainers and blocks. They are diagnostics for the log
  file, shared by every run and read by whoever maintains the installation, not by
  the person at the page; they stay in the language they were written in.
- The epoch and trial lines of the sheet (`> epoch 3/10 · loss=…`, `── trial 1/3
  ──`): the trainer's own vocabulary, the same one the command line prints, kept
  on purpose (the comment on `trainingOverlay` in `pt.ts`).
- `STOPPED_NOTE` (`core/cancellation.py`): the audit text in the `error` of a unit
  a stop cut. No screen shows it; the page words "stopped" from the status.
- The `description` of the fields of the built-in configs' schemas, which no screen
  shows, and pydantic's own validation messages.
- Exceptions raised below the route layer by the libraries themselves (torch,
  Ultralytics, the dataset providers' clients) and relayed as is, including every
  `str(exc)` that becomes the `error` of a failed run or the `500` of an action.
  Our own ones moved in the third slice.
- The built-in task labels and descriptions in `/api/tasks`: replaced by the page's
  dictionary (ADR-110).

**Consequences:** a call without the header gets Portuguese. For the messages
that used to be English and moved (a run not found, the profile errors, the
"open folder" refusals, the queue 404), a script that does not send
`X-VF-Lang` now reads Portuguese and has to send `en` to keep the old text; the
status codes are unchanged. Adding a server message means adding both languages
to the catalog before the test passes, which is the same rule ADR-110 set for
the page. The figures made outside the GUI keep their English text: the command
line and the library draw in English unless `VISIONFORGE_LANG` says otherwise, and
only the figures of a run started from the page follow the page. The non-figure
messages of a command-line run stay Portuguese, as before. One defect came out of
moving the texts: the "froze N weights" warning replaced every comma of its Portuguese sentence by a period while grouping the
number ("pré-treinados. ou use"), and now changes only the number.

**Amended 2026-10-16: the model card reads each task's own config.** The card printed
`?` for fields a task does not have (`num_classes` and `pretrained` for regression,
anomaly and custom tasks), the classification `transforms.image_size` for anomaly and
segmentation (they resize to `data.image_size`), the raw status, `None` for a missing
metric, and the `run.json` id (`<experiment>_<folder>`) where the History shows the
folder name. A row whose field is absent is now left out (a custom task shows its
label as architecture and task), the size is the one training used, the status goes
through the catalog, `None` is `—`, and the id is the folder name.

## ADR-117 — "Primeiro treino" for every built-in task

**Date:** 2026-10-10
**Status:** Accepted
**Completes:** ADR-115 (which shipped classification only and named the other four
tasks as a later slice), reusing the gates and the registry it built

**Context:** the guide of ADR-115 taught one task. A researcher who opens the
detection, regression, segmentation or anomaly tab and asks for the guide got a
classification script (ResNet, accuracy, class folders) pointed at a form that
has no such thing, and its last step said in one line what the other tasks
would have been. The gates and the registry were built to be reused; what each
task needed was its own sample dataset, its own text and a way to put the folder
in a form that is not classification's.

**Decision.**

**1. A sample dataset for every task.** `POST /api/sample-dataset` takes the five
tasks and writes `datasets/exemplo-<tarefa>/` with the self-test builders
(ADR-060), unchanged: `exemplo-deteccao` (`images/` and `labels/` with `train`
and `val`, four images each, one YOLO box per image), `exemplo-regressao`
(`train.csv`, `val.csv`, `test.csv` of 12, 6 and 4 rows, columns `image` and
`target`, and `images/`), `exemplo-segmentacao` (`<split>/images` and
`<split>/masks`, 6, 3 and 2 pairs, mask values 0, 1, 2) and `exemplo-anomalia`
(eight normal images in `train/good`, and `test/good` and `test/defect` with four
each). The semantics are those of ADR-115: 201, 409 with the existing folder's
`path` when it holds anything, a scratch folder renamed into place so no half
folder is ever mistaken for a finished one. The response gains `task` and
`targets`, and `counts` is items per split in the unit the task reads (images,
CSV rows, image and mask pairs); `classes` is what the layout declares (class
folders, the names in `classes.txt`, the pixel values found in the train masks,
the subfolders of `test/`). One addition to the builders' output: the detection
folder gets a `classes.txt` with the single name `object`, which is how a plain
YOLO dataset says what class 0 is, and what the panel's stats card reads. The
messages of this endpoint (unknown task, folder in use, write failure) now go
through the catalog of ADR-116 in both languages; the "not built yet" 400 is gone
because nothing is not built.

**2. One guide, played on the active tab.** `GuideDefinition.steps(t, task)`
takes the task. The guide opened from the menu, or chosen in the invitation, plays
the script of the tab that is active at that moment (`guideTaskFor`: a built-in
task's own tab, classification for a researcher's own task, which has no sample).
The task is latched when the guide starts: switching tabs mid-guide does not
change the script, and the dataset gate keeps reading the form of the guide's
task (`datasetPaths[task]`), not the form that happens to be on screen. The
eight steps, their gates (dataset set, `start`, `end`), their anchors and the
wording of the buttons are the same for all five; what is per task, in
`guides.firstTraining.tasks.<task>`, is what the data is, what was found, the
parameters that matter, how long it takes, what the sheet shows, the result
metric and what changes with data of one's own. Classification's texts moved
there unchanged, except that its last step now says how to do the same on another
task.

**3. Anchors in the four panels.** Only the classification panel carried
`dataset` and `training`. Each of the other four now marks the folder row
(`dataset`), the training grid (`training`) and, new, the stats card under the
dataset fields (`dataset-stats`). The second step of every task but
classification points at the stats card instead of the folder row: with the card
anchored on the folder row, the guide's card sat on top of the very counts the
step describes (seen in the browser on the regression tab). Classification's
picker shows what it found inside its own box, so it keeps `dataset`.

**4. The sample sets what the defaults would not train on.** The forms default to
photographs: images of 224 px (regression), 256 (anomaly), 512 (segmentation) and
640 (detection) for samples of 32 to 128 px, two classes for masks that hold
three values, which stops the run at the first batch. The guide's button, and the
409 that reuses an existing folder, therefore also set the image size
(`SAMPLE_SETUP`: 128 for detection, 64 for the others) and the model's class count
(1 for detection, 3 for segmentation), and the steps say so. How each form
receives a folder is a pure function per task (`lib/guides/sample-forms.ts`),
because each panel keeps its own form shape: detection also goes back to the
folder source if it was left on a `data.yaml`.

**5. The few numbers the guide asks the researcher to type were measured, not
guessed.** Each was run on CPU through the same API the page uses, and the
defaults were run too, because the text has to say what they do:

| Task | Asked | Result (best epoch) | Defaults would give |
|---|---|---|---|
| Classification | epochs 3 | unchanged from ADR-115 | |
| Detection | nothing (100 epochs) | mAP50-95 0.995, about 50 s | 5 to 60 epochs: 0.04 to 0.08 |
| Regression | epochs 30, batch 4 | val R² 0.88, RMSE 0.065 | 20 epochs, batch 32: R² -0.62 (one step per epoch) |
| Segmentation | epochs 8, batch 2 | mIoU 0.92, about 40 s | batch 8: one step per epoch |
| Anomaly | epochs 3 | AUROC 1.0, image F1 0.89 | |

Two of these are lessons the text states instead of hiding. Detection is the one
task where fewer epochs is the wrong advice: with four images and a few epochs YOLO
barely leaves its warm-up and the mAP stays near zero, so the step says to leave
the 100. Regression's
test split has four rows, so its R² ranged from -14 to 0.6 across the runs of
this calibration; the result step says that a low test R² there is sample noise,
and the watching step that the live R² swings from epoch to epoch while the
result keeps the best. The anomaly step also says what PatchCore does differently: it has
no epochs, and the sheet shows `phase` events (extracting features, building the
memory bank, scoring) with their progress in place of the epoch counter.

**Rejected:**

- *One menu entry per task.* Five entries for one script would crowd the menu and
  the invitation; the active tab already says which task the researcher means.
- *Filling epochs and batch size for the researcher too.* The folder, the image
  size and the class count are facts about the sample, which the form cannot
  know. Epochs and batch are what the researcher decides, and typing them is the
  exercise; a guide that does it for you teaches where the button is.
- *Changing the self-test builders* (a larger regression set, a named detection
  class). The builders also feed `visionforge selftest` and a dozen tests; the one
  addition lives in the sample endpoint, not in them.
- *A gate on the class count.* A forgotten click on "apply" in the segmentation
  card would fail the run, so the guide sets it; a gate would only turn that
  failure into a waiting card.

**Consequences:** a researcher on any of the five tabs gets a script written for
that tab, with a folder that loads in that task's own loader and a result worth
looking at. The response of `POST /api/sample-dataset` has two more fields, which
a client that ignores them does not notice. On Windows the anomaly run takes
about 13 to 16 s per epoch, almost all of it the start of the DataLoader workers
for the loaders that are not persistent (its per-epoch metrics are measured on
them), so the text says "about a minute" and asks for 3 epochs; that cost is the
loaders', not the guide's. A calibration slip is worth recording: the first
segmentation runs sent the image size where the regression config keeps it
(`transforms.image_size`), which the segmentation config ignores, so they ran at
512 px and took a minute per epoch; the numbers above are from `data.image_size`.

**Verified in the browser** (headless Chrome over CDP, on CPU, with the sample
folder deleted first so the button created it): regression, anomaly, detection and
segmentation were played from the first step to the last. On each the button
created the folder and filled that tab's own field, with the image size (and class
count) set; the stats card was highlighted at step 2 and showed the counts the text
names; the typed epochs and batch reached the run (`run.json` says 30 and 4 for
regression); "Continuar" was disabled at the train step until `start` and at the
result step until `end`; the sheet's log lines carry the metrics the watching step
names. A second click on the button, with the folder already there, got the 409,
filled the field and applied the image size as well. One thing the run corrected: a
first draft put the second step on the folder row, and the card covered the counts.

**Amended 2026-10-17: the sample dataset is one click away in every task panel.** The sample could only be created by the guide, and only for the open tab, so the segmentation one was hard to find. Each panel (classification, detection, regression, segmentation, anomaly) now has a small "usar dataset de exemplo" button under its folder row (`SampleDatasetButton`). It calls `createSample(task)`, the same call the guide's button makes, and applies the result with the `sample-forms` functions the guide uses, so the folder, the class count and the image size land exactly as in the guide; a 409 reuses the existing folder. A failure shows as a red line under the button. The guide only changed to share `createSample`.

**Update (2026-10-24):** The sample-dataset endpoint declares HTTP 409 in
OpenAPI when the sample folder already exists, matching its existing response
with `detail` and `path`. This case returns `JSONResponse` rather than raising
`HTTPException`, so the exception-documentation check did not expose the missing
declaration. The HTTP contract snapshot records the added status explicitly.

## ADR-118 — Training story: an illustrative animation driven by real stages

**Date:** 2026-10-13
**Status:** Accepted
**Builds on:** ADR-111 (the stop outcome and the units of a multi-unit job),
ADR-115 and ADR-117 (the guides, whose "what is happening now" step opens it)

**Context:** the training sheet is a log. A researcher who does not yet know what
training is sees `epoch 3/10 · loss=0.41` and has nothing to attach it to. The didactic
ideas list (kept outside the repository) asked for an
animation of the network learning, with one hard rule: it must never teach wrong.
A moving picture synchronised to "training now / validating now" cannot keep it,
because the stream has no such events (there is no per-batch progress, and the
validation between two epochs is over before the screen could show it). The user
decided the rest on 2026-10-13: one drawing per architecture family, a neon-circuit
look on an organic-network topology, three flows in three colours, text kept short,
and a tab next to the log.

**Decision.**

**1. What is real and what is not.** Two things on the tab are real: the macro
*stage* of the run, and the *numbers* the trainer reported (epoch, training loss,
validation loss, the task's headline metric, and the unit of a multi-unit job).
Both are computed from the events the log already reads (`start`, `epoch_end`,
`phase`, `trial_start`, `trial_end`, `end`) and the run status. The motion of the
particles is illustrative and says so on the tab: "animação ilustrativa, em câmera
lenta". No backend change and no new event: the `stage` event the plan first
proposed is not needed, because the stage the viewer can use is the macro one.

**2. The stage rules** (`lib/training-story.ts`, a pure function of the events):

| Stage | When |
|---|---|
| queued | the server reports the run as waiting for the GPU |
| preparing | running, and no epoch reported yet — or, between two units of a multi-unit job, the next unit has not reported one |
| training | epochs short of the unit's last planned epoch are being reported; for a run that reports phases (PatchCore), a phase is under way |
| testing | the unit's last planned epoch was reported and the unit has not closed (`trial_end`); the last unit has closed; or the stream's `end` arrived and the result is not out yet |
| done | the run completed |
| interrupted | the run completed and the sheet's stop verdict is `stopped` |
| failed | the run failed |

A stop is not re-derived: the sheet already computes `stopOutcome`
(lib/run-control.ts) for its closing log line, and the story takes
`stopped = outcome === "stopped"` from it, so the two cannot disagree. A run that
stops on its own early (patience) goes from `training` straight to `done`: the
stream shows no last planned epoch, and the story does not pretend there was. A
multi-unit job names the unit that is running, with the word the stop point already
uses (`fold`, `trial`, `replicate`, `model`), from `trial_start`, and the numbers of
the previous unit are cleared when the next one starts. A job of one unit shows
none.

**3. One drawing per family, chosen once, when the run is submitted**
(`lib/story-family.ts`). The sheet outlives the form it came from, and a tab
switched mid-run must not change the picture of the run in flight, so `App` records
the spec (`storySpecFor(task, config)`) at each submission, next to the block kind.

| Family | Models | Drawing |
|---|---|---|
| `cnn` | the ten convolutional built-ins of classification and regression (`resnet18/34/50/101`, `efficientnet_b1/b7`, `vgg16/19`, `alexnet`, `convnext_tiny`); a `timm` name that is not an attention model; a model of the researcher's own; a researcher's own task; a comparison of mixed families | columns narrowing towards a head; the head is class scores, or one neuron for regression |
| `transformer` | `vit_b_16`, `swin_t`; a `timm` name with `vit`, `deit`, `swin`, `beit`, `cait`, `xcit`, `pvt`, `twins`, `volo` or a leading `eva` (so `maxvit` and `mobilevit`) | columns of patch tokens joined by attention arcs, ending in a class token |
| `encoder_decoder` | every segmentation model: `unet`, `deeplabv3_*`, `fcn_*`, `lraspp_*` | shrinking, a bottleneck, growing back, with skip arcs |
| `detector` | every detection model: the Ultralytics YOLOs, RT-DETR, and torchvision's detectors | backbone, neck, three heads that end in boxes |
| `autoencoder` | the anomaly autoencoder | encoder, bottleneck, decoder |
| `patchcore` | PatchCore | feature extractor, memory bank, scoring: phases, not epochs |

Regression is a CNN (or a transformer, for `vit_b_16` and `swin_t`) with a
one-neuron head, as the user asked. The headline metric is the task's own:
accuracy, mAP50, R², mIoU, AUROC; a researcher's own task has only the compat field
`val_accuracy` (ADR-058), shown as "main metric" because the name of its metric is
not in the event.

**4. Three flows, three colours.** Data entering and the forward pass go in the
task's accent (`TASK_ACCENT`, the colours the rest of the interface uses); the
gradient coming back is amber; validation is cyan, a sparse pulse that fires once
per `epoch_end` and a slow loop while `testing`. PatchCore has no gradient: it shows
the accent while it extracts and stores features and cyan while it scores, and
fills the memory bank with the progress of its `phase` events. Particle counts are
small (three forward, two backward) and slow, on purpose.

**5. Where it lives.** A tab "Como funciona" next to "Log" in the training sheet.
The log stays the default, and a new run goes back to it. The tab state is lifted
into `App` so the "Primeiro treino" guide can select it: its "what is happening
now" step does it on entering (`GuideContext.showTrainingStory`).

**6. Motion, accessibility and cost.** SVG and SMIL `animateMotion`, no video, no
Lottie, no dependency. `storyMotion` decides, as a pure function: particles when
the run is going and the drawing is seen; arrows on the curves, still, under
`prefers-reduced-motion: reduce` (the CSS rule of the app that zeroes animation
durations does not reach SMIL, so the preference is read in script); nothing when
the browser tab is hidden or the sheet is minimized, where the particles are not
mounted at all. A run that is not going (queued, finished, failed) is a still
drawing. Each flow has a shape as well as a colour (a filled dot for data and the
forward pass, a ring for the gradient, a diamond for validation), because the
gradient's amber is close to the anomaly task's own accent and the validation's
cyan to the regression's blue. The interface has a single, dark theme and the
sheet paints its own dark background, so the neon look needs no light variant; the
drawing never uses a CSS variable for a colour it sets as an SVG attribute. Every
word is in `i18n/pt.ts` and `i18n/en.ts`.

**Rejected:**

- *A `stage` event from the trainers* (the original plan). It buys "validating
  now", which the screen cannot follow, at the price of a contract change in five
  trainers. The macro stage is already in the stream.
- *One drawing per model.* Twelve-plus pictures to maintain for a teaching point
  that is the family's. A ResNet-18 and a VGG-19 teach the same thing.
- *Reading the model from the queue snapshot.* It carries a label, not the config,
  and arrives after the run starts; the form at submission has the answer.
- *Deriving the stop in the story.* `stopOutcome` has its own tests and its own
  edge cases (resumed runs, searches, PatchCore); a second copy would drift.
- *A charting or animation library.* The interface draws its charts by hand
  already, and a few circles on a few curves do not justify a dependency.

**Consequences:** the sheet gains a tab and no request. The cost of the drawing is
bounded (a few dozen circles and curves, and at most eight pulses), and it is zero
while the sheet is minimized or the tab hidden. The stage `testing` is honest about
what the stream shows, not about what the server is doing every second: after the
last epoch the server evaluates, draws figures and writes the report, and the
stream cannot tell those apart. A model comparison streams no epochs, so its story
stays on `preparing` until the job ends; the drawing is the family its models
share, or the generic network.

**Verified in the browser** (headless Chrome over CDP, the GUI served from the
worktree's own `src` on CPU, in Portuguese and once in English). Classification was
run through the "Primeiro treino" guide: the sheet opened on the log, the guide's
"O que está acontecendo" step switched it to "Como funciona", and the strip went
`dados` to `treino` to `teste` to `pronto` with the epoch, both losses and the
accuracy changing as `epoch_end` arrived; the `teste` stage lasted about six
seconds, which is the server evaluating the test split and writing the report after
the last epoch. Regression drew the CNN in blue with a one-neuron head and an R²
tile; segmentation drew the U-Net in violet with its skip arcs and an mIoU tile, and
a stop confirmed in the middle of it went to `testing` (the server still evaluates a
stopped run) and then to `interrompido`, the drawing turning amber. PatchCore drew
the extractor, the memory bank and the scoring zone, showed `extraindo features ·
0/1` and `pontuando · 1/2` in place of an epoch, and lit the cyan pulses while
scoring. A K-fold showed `dobra 1/5`, `2/5` and `3/5` on the same drawing, and a
run with no dataset went from `dados` straight to `falhou`, in red. With
`prefers-reduced-motion: reduce` emulated, no `animateMotion` was mounted and
chevrons were drawn instead; minimizing the sheet took the animated elements from 17
to 0 and reopening it brought them back, as did hiding and showing the document.
One thing the check changed: the first drawing was 190 units tall and the tab
scrolled inside the sheet, hiding the caption, so it was reduced to 166 and the
spacing tightened. Two things it left as they are: the guide's floating card covers
the left edge of the sheet at 1280 px wide, as it already did with the log, and
the amber of the gradient is close to the anomaly accent, which the ring shape
settles.

**Amended (2026-10-15): comparing models streams each model's epochs.** In the GUI,
comparing architectures is a sweep over `model.name` (ADR-059); for the standalone
tasks that sweep (`core.sweep`) announced each trial and its end but carried none of
its epochs, because the task runners had no way to pass the trainer's events on, and
the classic comparison loops (`ModelComparisonBlock`, the task-agnostic
`run_model_comparison`) streamed nothing at all. So the story sat on `preparing`, or
on a unit with no numbers, for the whole job. Now every task runner has a
`_progress_callback` slot it hands to the block it builds, next to `_cancel_token`,
and the three loops stream each unit through `core.task_runner.UnitProgress`: a
`trial_start` with what the unit varies in `overrides`, the unit's own epochs
annotated with its index, and exactly one `trial_end` carrying the outcome (`status`)
and, when the trainer reported its end, its `total_epochs`; the trainer's own `end` is
held back for that, so it never closes the stream and a unit that died first is
closed too. The comparison loops add a single terminal `end`. The epoch-annotating
wrapper moved from `blocks/_search_utils` to `core` so the task-agnostic loops can
reach it. A unit whose only override is `model.name` is named a model on the tab
(`modelo 2/3`) and in the log (`── model 2/3 · resnet34 ──`), not a trial over a
parameter called `name`. Verified in the browser on 2026-10-15, with the guides' sample
datasets: a regression sweep over three ResNets showed `trial 2/3`, then (after the
naming change) `modelo 2/2`, the training stage and each model's epochs and losses;
also the detector, transformer (`swin_t`) and autoencoder drawings, which the first
check had not covered, and a detection run stopped mid-way going to `interrompido`.

**Amended 2026-10-16: replicates and the replicated comparison stream their epochs too.** They were the two multi-unit jobs left out: their sheet sat on "preparing" for the whole run. `run_replicates` now drives a `UnitProgress` with one unit per seed (`overrides` = `training.seed`), and `run_replicated_comparison` one unit per (variant, seed), numbered across the variants, with the variant and its overrides in `overrides`; the job emits a single `end`. The variant-level `trial_start`/`trial_end` pair of the comparison is gone: its `total_trials` is now variants x seeds. The frontend needed no change (`replicate` was already a unit kind).

**Amended 2026-10-17: a finished run can replay the stages.** The tab was left on "done" with no way to see the drawing move again. A button "rever animação" (only when the stage is `done` and the family is not PatchCore) walks `preparing`, `training` and `testing` in about 13 s (3 + 6 + 4) and returns to the real stage; it can be clicked again once it ends. It is a generic replay, not the run played back: the numbers stay the run's final ones, and the caption "animação ilustrativa" stays on screen. It is local state (`useReplay`, over `lib/training-replay.ts`) that overrides the stage the tab shows; the story model is untouched. `storyMotion` decides the motion as before, so under `prefers-reduced-motion` the stages are highlighted with still arrows and no particles.

**Amended 2026-10-17: the detector's heads end in a thumbnail.** The three boxes floated at the right with nothing joining them to the network. The drawing now ends in a small stylized image (a frame with three plain shapes as objects, no external image) and each head is linked to its left edge by a curve in the style of the other flows; the three boxes (small, medium, large, one per head, in the task's accent) are drawn over the objects. The forward and validation pulses go head, edge of the thumbnail, box, so they animate with the other flows and the still arrows follow the same route. The thumbnail is illustrative like the rest: where the boxes sit says nothing about a real prediction. Other families are unchanged (`Drawing.thumbnail` is `null`).

## ADR-119 — The training story names overfitting when the losses show it

**Date:** 2026-10-15
**Status:** Accepted
**Builds on:** ADR-118 (the training story and its rule that only the stage and the
numbers are real)

**Context:** the original plan for the training tab included a line on "what to watch
for": validation loss rising while training loss falls. ADR-118 kept the tab to one
sentence per stage and the live numbers, so a beginner watching the numbers had to
notice the pattern alone, and the pattern is the single most useful thing a first
training run can teach.

**Decision.** A sentence under the stage's, in the gradient's amber, shown when the
reported losses of the unit in flight show the pattern, and only then:

- **The rule** (`overfittingSince` in `lib/training-story.ts`, a pure function):
  over the last `OVERFIT_RISES` (3) epochs the validation loss rose at every epoch,
  and the training loss ended lower than it started. The answer is the first epoch
  of the climb, so the sentence says since when.
- **Why three rises and a falling training loss.** One or two rises are ordinary
  noise between epochs; a validation loss that rises together with the training loss
  is a run that diverges (learning rate too high), not one that memorises, and
  calling it overfitting would teach the wrong fix.
- **What it reads.** Only `epoch_end` events of the current unit (a new unit of a
  K-fold, sweep or comparison starts clean), and only when both losses are finite.
  PatchCore, which reports no losses, never shows it; nor does a failed or queued
  run. A finished or stopped run keeps it, because that is when the reader looks
  back at what happened.
- **What it says.** That the validation loss rises while the training loss falls,
  and that the model *may* be memorising the training set. It does not claim the
  best checkpoint is protected: classification keeps the lowest validation loss, but
  YOLO picks its best weights by mAP, and one sentence for every task cannot promise
  the same thing.

**Rejected:**

- *A loss chart on the tab.* The plan's mini-chart repeats what the log and the
  History already plot, and the user chose numbers over charts for the tab
  (ADR-118).
- *Detecting it on the server.* The health warnings (`core/training_health.py`)
  judge a finished run; this is a live cue that comes and goes with the epochs, and
  the page already has every number it needs.
- *A threshold on the size of the rise.* Losses of different tasks live on different
  scales; consecutive rises are scale-free.

**Consequences:** the tab gains one conditional sentence and no request. The rule can
miss slow overfitting that zig-zags (a dip in the climb resets it) and can fire on a
short, noisy plateau of three small rises; both are acceptable for a cue that says
"may", and the threshold is one constant. Verified by unit tests on the rule (three
rises fire, two do not, a dip does not, both losses rising does not, no validation
loss does not, units read separately, PatchCore never), and seen in the browser on
the regression sample dataset, whose validation loss is noisy: the sentence appeared
during a sweep's second model ("desde a época 6") and went away with the next unit.
That run is also the false-positive case the rule accepts: on six validation images a
three-epoch climb can be noise, which is why the sentence says "may".

## ADR-120 — The GUI offers updates, and installs one when asked

**Date:** 2026-10-15
**Status:** Accepted

**Context:** VisionForge ships on PyPI, and the people it is meant for (researchers,
the testers of the usability study) install it once and keep it. Nothing told them a
newer release existed; updating meant knowing the package name, the environment and
the right command, and on Windows it fails outright while the GUI is open, because a
running program cannot replace its own files (the reason the pre-commit hook already
runs `uv run --no-sync`).

**Decision.** A small notice in the corner of the header, under the version, and a
button that installs the release and brings the GUI back.

- **The check.** `GET /api/update` (`gui/api/updates.py`) reads the latest version
  from PyPI's JSON API with a 3-second timeout, off the event loop, cached for six
  hours per server process. Offline, behind a proxy or with an odd answer, it says
  nothing. Only plain `X.Y.Z` releases count, so a pre-release or a local build
  (`0.0.0+unknown`) is never offered. `VISIONFORGE_NO_UPDATE_CHECK=1` turns the
  lookup off entirely, for a lab that must not reach the internet or a study that
  pins its version. The page can also skip one version (`localStorage`); the next one
  shows again.
- **When the button may install, and when it may not.** The answer carries
  `can_apply` and, when false, a code the page words itself:
  - `source`: a checkout (`pip install -e .`, detected as by `doctor`) updates with
    `git pull`; reinstalling a wheel over it would detach it from the repository.
  - `no_installer`: neither `pip` in this interpreter nor `uv` on the PATH.
  - `remote`: the client is not this machine. Reinstalling the server is its owner's
    decision, the same rule as "open folder".
  - `busy`: a job is running or queued; the restart would kill it.

  In every blocked case the notice shows the command to run by hand.
- **Applying.** `POST /api/update/apply` repeats every check (and refuses a foreign
  `Origin`, as "open folder" does), then starts `gui/updater.py` in a process of its
  own and stops the server about a second later, after the response went out
  (`server.request_shutdown`: the way Ctrl+C does under `visionforge gui`, SIGINT
  under a bare `uvicorn`, a hard exit after 15 s if neither worked). The updater is
  copied out of the package first and imports only the standard library, because
  the install replaces the package's files while it runs. It waits for the server's
  processes to end (this Python, and the `visionforge.exe` launcher that started it
  when `argv[0]` is that .exe) and for the port to free, then installs the exact
  version the page announced
  (`pip install --upgrade visionforge-studio==X.Y.Z`, or `uv pip install --python
  <this python> …` for an environment `uv` created without pip), appends the output
  to `update.log` and writes `status.json` in the config folder (next to the stored
  credentials), and starts `python -m visionforge gui` again in the same working
  directory, with `VISIONFORGE_NO_BROWSER=1` so no second tab opens. An environment
  variable rather than the new `--no-browser` flag, because the relaunch runs the
  *new* version: a flag it did not know would stop it, an unknown variable is
  ignored.
- **The page waits.** It polls `/api/health` until the server answers with the new
  version, then reloads. A server that went away and came back with the *old*
  version means the install failed and the updater relaunched what was there; the
  page stops waiting at once and the notice turns into "the update failed, the
  previous version is still running", with the log's path. After five minutes it
  gives up and says to reload later (a first install of new dependencies can be
  slow).
- **Torch is untouched.** It is not a dependency of the package (ADR-005), so the
  install never swaps the CUDA build a researcher chose; pip's default
  `only-if-needed` upgrades other dependencies only when the new release requires it.

**Rejected:**

- *Updating on its own, at start-up.* A study in progress must not change version
  between two runs because a release came out overnight. The runs record their
  version, but the researcher decides when.
- *Installing in place from the server process.* It fails on Windows (locked
  `visionforge.exe` and extension modules) and would leave a half-replaced package
  running.
- *Upgrading to "latest" in the updater.* A release published between the notice
  and the click would install a version the page never showed.
- *A desktop launcher or an installer app.* More to build and sign, for the same
  outcome.

**Consequences:** one request to PyPI per server process every six hours, which
tells PyPI the address of a machine running VisionForge (the same as `pip` does); the
environment variable removes it. The button cannot help a checkout, a pip-less
environment or a remote client, and says so with the command to run. A failed install
leaves the old version running and the reason in `update.log`.

**What the first real run taught.** The flow was run end to end on Windows: a wheel of
this code labelled 0.13.9 in a separate environment (`uv venv --seed`, so the `pip`
path), its GUI on port 8010, the button pressed in the browser, PyPI's 0.14.0
installed. The first attempt waited for the port only. The server closes its socket
and lives on for seconds while it shuts down, and the launcher with it; pip found
`Scriptsisionforge.exe` locked (`WinError 32`), had already removed the package by
then, and did not put it back -- the relaunch found no VisionForge at all. Three
changes came out of it: the updater waits for the processes, not the port; an
install whose log shows `WinError 32` is retried (five times, five seconds apart);
and a failed install is followed by reinstalling the version that was running
(`restore`, recorded as `restored` in the status) before the GUI starts again. The
page also offers "try again" on a failed attempt when the server can still install,
and the updater deletes its temporary copy when it is done. The second attempt, from
the page's failed state, installed 0.14.0 with no retry needed, the GUI came back on
the same port, and the page reloaded itself showing v0.14.0. (That release predates
`VISIONFORGE_NO_BROWSER` and opened a browser tab on its own; releases from this one
on do not.) Also covered by tests: the lookup, the cache, the installer choice, every
blocked case, both routes, the updater with harmless commands in place of the install
(the retry, the restore, the wait for a finished process), and the page's notice and
wait.

**What the second real run broke (2026-10-16).** The button was pressed in the
project's own development environment, a `uv` checkout whose GUI ran as `uvicorn
--app-dir src`, and it wrecked that environment. Four faults, each now fixed and
tested:

- *A checkout was taken for a PyPI install.* With `src` first on `sys.path`, Python
  found a stale `src/visionforge_studio.egg-info` before the editable install's
  metadata, and that egg-info has no `direct_url.json`; `installed_from_source`
  answered no and the button was offered. It now also asks where the imported
  package lives: a `visionforge` package under `src/` next to a `pyproject.toml` that
  names `visionforge-studio` is a checkout, whatever metadata is found first.
- *`uv pip install --upgrade` upgrades every dependency.* It replaced the CUDA build
  of torch (from the PyTorch index) with PyPI's CPU-only torch, among 78 packages. The
  `uv` command now uses `--upgrade-package visionforge-studio`, and the `pip` command
  says `--upgrade-strategy only-if-needed` (pip's default, made explicit): a
  dependency moves only when the new release requires it. The restore uses the same
  command, so it no longer churns the environment either.
- *uv words a locked file differently.* It prints `os error 32` (and `os error 5` on
  the rename after), not pip's `WinError 32`, so the retry never fired. Both
  wordings are matched. The file was held by a second VisionForge server running
  from the same environment, which the updater does not wait for; with only
  VisionForge's own files replaced now, a lock is much less likely, and the retry
  covers a short one. A second server on the same environment remains a known gap.
- *A console window opened.* The updater runs detached, without a console, so Windows
  gave `uv` a new console window: an empty terminal the user saw appear, after which
  nothing seemed to happen. The install now runs with `CREATE_NO_WINDOW`.

## ADR-121 — A model is built from the run's seed, not only trained from it

**Date:** 2026-10-15
**Status:** Accepted
**Corrects:** ADR-098's promise that the same config and seed reproduce the same
numbers, which did not hold for a pretrained network with a new head

**Context:** the seed-variance study's pre-registered sanity check trained resnet18
at seed 0 twice in one process, deterministic mode on, and got 0.774 and 0.681 test
accuracy on USK-Coffee (smaller gaps on the other two datasets), with the training
losses apart from the first epoch. The trainers call `_seed_everything` at the start
of `fit`, but every built-in block creates the model before calling `fit`, and a
pretrained network gets a new final layer for the task's classes, initialised from
the global RNG at creation. That layer came from whatever state the RNG was in: an
earlier run in the same process, or the unseeded state of a fresh one. Measured in
isolation on the RTX 5060 Ti: built without seeding first, the head differed between
builds; seeded first, it was identical; and with identical weights and batch, two
CUDA backward passes gave bit-identical gradients, so the kernels (adaptive average
pooling included) were not a second cause. Researcher-defined tasks were already
correct: their engine seeds before `build_model`.

**Decision.** `core.trainer.seed_before_building(training)` seeds every RNG from the
run's `training.seed` (with its `deterministic` flag) and every block that builds a
model to train calls it first: classification, its K-fold and transfer learning,
regression and its K-fold, segmentation and its K-fold, anomaly. `fit` still seeds
again, so the data order does not depend on how much randomness the build consumed.
Paths that build a model only to load a checkpoint into it (evaluation, Grad-CAM,
ONNX export, batch prediction) are unchanged: the checkpoint overwrites the head.

**Rejected:** *seeding inside the factories*, which do not know the run's seed and
are also used to rebuild models for checkpoints; *`torch.use_deterministic_algorithms`*,
since the measurement showed the kernels were already deterministic here and strict
mode raises on operations some models need.

**Consequences:** two runs of the same config and seed now start from the same
weights; verified by training resnet18 on the 10% USK-Coffee subset twice in one
process (0.816875 both times) and by a regression test that scrambles the RNG
differently before two builds of the classification and regression blocks (it fails
without the fix). Runs made before this change with a pretrained model and a new
head are not reproducible from their seed alone; their `run.json` still records
everything else. The worker count is a separate source a researcher can pin: the
automatic `num_workers` follows the free memory of the moment, and the number of
workers decides which random augmentation each image receives, so a study that must
repeat a seed exactly should fix it (the seed-variance study fixes it at 0).

**Amended 2026-10-16: nothing else in the server draws from the global generator.**
The GUI validation found three runs of one config and one seed diverging from the
first epoch while the Datasets tab's augmentation preview was being used. The job
trains in a thread and its augmentation and shuffling draw from torch's global
generator (`num_workers` 0 on a small dataset); the preview, in another thread, drew
its random flips, rotations and jitter from the same generator, so the job received
a different sequence. The preview now draws from a `random.Random` of its own and
applies the same operations through `torchvision.transforms.functional`, and a test
checks it leaves torch's and Python's global state untouched. No other GUI route
draws from a global generator. With workers above 0 the job's augmentation runs in
worker processes and was not exposed; the DataLoader's shuffle still uses the global
generator of the training thread, which is safe as long as no other thread draws
from it.

## ADR-122 — A broken CUDA context stops the queue, and segmentation masks are checked first

**Date:** 2026-10-16
**Status:** Accepted

**Context:** in the GUI validation of 2026-10-16 a segmentation run whose masks held
a class id above `num_classes - 1` tripped a device-side assert inside the loss. That
poisons the CUDA context of the whole process: every later CUDA call raises. The
trainers seed CUDA even for a CPU job, so from then on every job the GUI accepted
failed within a second with `CUDA error: device-side assert triggered`, CPU ones
included, while the queue kept accepting work and nothing said the server had to be
restarted.

**Decision:** two guards. *Before the GPU:* the segmentation trainer checks every
batch's mask ids on the CPU before moving them to the device (`check_mask_ids`: one
`max` over an integer tensor) and refuses an id outside `0..num_classes-1` that is not
the `ignore_index`, with a message that names the id and what to change; the run fails
as any other, and the context stays healthy. *After the fact:* every job submission
first asks `gpu_health.cuda_context_broken()`, which returns true only when CUDA was
started in this process and a one-element allocation raises a CUDA error; the
submission is then refused with 503 and a message to reopen VisionForge. The probe
never starts CUDA in a server that has not used it and does not wait for a running
job's kernels.

**Alternatives considered:**
- *Restart the server automatically.* It would kill the page's connection and any
  waiting jobs, and hide that something went wrong.
- *Validate mask ids from the dataset statistics before the run.* The statistics
  sample masks; a single mask outside the sample would still poison the context. The
  per-batch check sees every pixel the loss sees.
- *Not seeding CUDA for CPU jobs.* Narrower: it would let CPU jobs run, but GPU jobs
  would still fail silently one after another.

**Consequences:** a bad mask is a clear error in the run instead of a dead server;
any other cause of a device-side assert (a custom task, a detector) now stops the
queue with a reason instead of failing every job. The check costs one reduction per
batch.

## ADR-123 — A K-fold is a History group, like a replicate set

**Date:** 2026-10-17
**Status:** Accepted
**Extends:** ADR-113 (replicate groups in History)

**Context:** the GUI validation of 2026-10-16 found every K-fold's folds listed in
History as loose runs. A classification K-fold also wrote a top-level `run.json` of
its own, in `<name>_cv/run.json`: a second K-fold of the same name overwrote it, its
folds did not point to it, and the History detail showed it as a run with no history.
A regression or segmentation K-fold wrote none: its mean ± std lived only in the
report under `outputs/reports`. The replicate groups of ADR-113 already solve this
for seeds: one `run.json` per job, a unique folder per job, children tagged with
`group_id`, History folding them under one entry with the job's numbers.

**Decision:** a K-fold is a third group kind, `kfold`, written by the same
`core/run_groups.write_run_group`. Its children are the folds (`fold` instead of
`seed`), each with the run folder it trained in; its aggregates are the K-fold's own
mean and standard deviation per metric, copied from the report, never recomputed.
The folder is `<models>/<name>_cv/<group_id>`, one per job. Every K-fold path writes
it: the classification block (which stops writing its old `<name>_cv/run.json`) and
the regression and segmentation runners (whose fold results now record `run_dir`).
History folds the children under the group, and the group detail shows one row per
fold, using the replicate view with "fold" where it says "seed". Each fold scores on its
held-out part of the training split, so the group lists the means under `val_<name>`
(shown with "(val)"), never in the test column the old classification run.json used
(`test_accuracy` held the mean validation accuracy). The old `<name>_cv/run.json` files
already on disk are still listed as before.

**Alternatives considered:**
- *A separate K-fold view.* More code for the same table: a fold row and a seed row
  carry the same fields (status, run, metrics).
- *Sweep and model comparison in the same change.* Their children differ by
  hyperparameters, not by a repetition index; the ranking view they need is the
  replicated comparison's, not this one. They follow in their own ADR.

**Consequences:** a K-fold appears once in History with its mean ± std, and its folds
open from it; two K-folds with the same name stay apart. A fold that could not be
tagged stays a loose run, as a seed does. The detail of a K-fold no longer shows the
classification block's `cv_aggregate` keys; the same numbers are in `group.aggregates`.

## ADR-124 — A sweep or a model comparison is a History group of its trials

**Date:** 2026-10-17
**Status:** Accepted
**Extends:** ADR-113 (replicate groups), ADR-123 (K-fold groups)

**Context:** after ADR-123 the hyperparameter sweeps and the model comparisons were
the last jobs whose trainings History listed as loose runs: a grid of four learning
rates left four unrelated entries, and which one won lived only in the report under
`outputs/reports`. The validation of 2026-10-16 flagged it, and ADR-123 deferred it
because the children differ by hyperparameters, not by a repetition index, so they
need their own view rather than the seed table.

**Decision:** a fourth group kind, `sweep`, written by `core/run_groups.write_run_group`
for every sweep and model comparison: the task sweeps (`core/sweep.py`, grid, random
or Optuna), the task model comparison (`core/comparison.py`), and the classification
grid search, random search and model comparison blocks. Each trial is a child with its
index, the hyperparameters it tried (`overrides`; a model comparison's is
`{"model.name": ...}`), its status, metrics and run folder; the group records the
search mode (`grid`, `random`, `optuna`, `models`), the metric and its direction, and
the index of the trial the report ranked first (`best_index`). The ranking stays the
report's: the group copies who led it and recomputes nothing. The trials now record
their run folder (`SweepTrial.run_dir`, `ComparisonTrial.run_dir`, the classification
trial record), which also makes the trial CSVs list it. History shows the group with
"x/y trials", the search mode and the best trial's hyperparameters; its detail lists
every trial, marking the best.

**Alternatives considered:**
- *Reuse the replicated-comparison view.* Its paired tests need the same seeds across
  variants; a sweep has one run per configuration and no test to show.
- *Recompute the ranking for the group.* A second ranking could drift from the
  report's; the classification sweeps rank by validation loss and the task sweeps by
  the chosen metric, and the group keeps each one's own.

**Consequences:** every multi-run job in VisionForge is now one History entry with its
runs under it. The CSV writer takes the union of the trials' keys, since a trial that
failed early lacks the run folder a finished one records. The classification random
search now trains every trial on the base seed, as the grid does since 2026-10-16
(it gave trial *i* the seed base + *i*, so a gap between trials mixed the seed with
the hyperparameters); its `best_config.yaml` carries that seed.

## ADR-125 — A light theme, following the operating system by default

**Date:** 2026-10-18
**Status:** Accepted

**Context:** the GUI had one palette, dark, fixed in `:root` with `color-scheme: dark`.
The user asked for a light theme. About a thousand style values already read tokens
(`var(--vf-*)`, the shadcn layer, the per-task accents), but some 230 neutral fills,
wells and scrims and about 140 solid status colors (error, success, warning text) were
literals inside inline styles, written for a dark background: white at 2–6 % alpha for
fills, near-black at 30–95 % for wells and overlays, and oklch tones of lightness
0.74–0.95 for text, which all but vanish on white.

**Decision:** three choices, `system` (the default), `light` and `dark`, in a group of
three buttons next to the language switch, each with `aria-pressed`, like the language
pair. The choice lives in `localStorage` (`vf.theme`), a per-browser convenience like
the language (`vf.lang`); `system` follows `prefers-color-scheme` and keeps following
it while the page is open. The resolved theme is set as `data-theme` on `<html>`, by a
small inline script in `index.html` before the first paint (no dark flash on a light
system) and then by `lib/theme.ts`.

The dark theme stays exactly as it was. The light one changes values, not markup:

- `:root[data-theme="light"]` redefines the `--vf-*` and shadcn tokens, the per-task
  accents (darker, so a 0.74-lightness accent does not wash out on white), the
  backdrop, the grain and the scrollbar, and sets `color-scheme: light`.
- The neutral literals become channel tokens with the dark value as the fallback:
  `rgba(255,255,255,a)` → `rgb(var(--lt-ink, 255 255 255) / a)`,
  `rgba(0,0,0,a)` → `rgb(var(--lt-shade, 0 0 0) / a)`, and the near-black surfaces
  (`12,14,18`, `8,10,14`, `4,5,7`…) → `rgb(var(--lt-surface, <same rgb>) / a)`. The
  `--lt-*` variables exist only under the light theme, so the dark one renders the
  original numbers.
- A solid oklch literal of lightness L ≥ 0.74 (a tone meant for a dark background)
  becomes `oklch(calc(L - var(--lt-flip, 0) * (2L - 1.3)) C H)`: unchanged in dark
  (`--lt-flip` unset), mirrored around 0.65 in light (0.85 → 0.45, 0.74 → 0.56), so
  text keeps its hue and gains the contrast it had. Translucent tones (badges' fills)
  stay as they are; they read on both.

The rewrite is one script (`frontend/scripts/theme-codemod.mjs`), kept so a new literal
can be converted the same way.

**Alternatives considered:**
- *Move every literal to a named semantic token* (`--vf-err-fg`, `--vf-well`…). Cleaner
  names, but about 100 distinct values would each need a name and a review, for no
  change in what the user sees; the channel tokens and the lightness mirror get both
  themes from the values already there.
- *Store the theme on the server profile.* The language is per browser too; a theme is
  about the screen in front of the user, not the researcher's data.
- *Two buttons (light/dark) without "system".* Following the OS is what the user asked
  for as the default, and it needs a third state to come back to.

**Consequences:** the light theme reuses the dark layout and accents, so a few
decorations (glows, the grain) are toned down rather than redesigned. The training
story's SVG drawings and the task accents (`training-story/tone.ts`, `lib/task-accent.ts`)
stay literal strings, and the codemod skips both files: the drawings derive translucent
tones from them (`withAlpha`) and their tests compare the markup as text. Instead they
take the resolved theme (`useResolvedTheme`, which watches `<html data-theme>`) and pick a
second set of literals for the light one: the task accents as index.css darkens the tabs'
(`TASK_ACCENT_LIGHT`), the other tones mirrored around 0.65 (validation 0.86 → 0.44), and
`--lt-idle`'s value for what is waiting. The dark set is unchanged. (Chrome 152 does resolve
`var()` and `calc()` in SVG presentation attributes, so `CompareRunsPanel`'s chart labels
with `fill="var(--vf-text-muted)"` follow the theme as they are.) The history, a run's
detail, the leaderboard and the run comparison pick their task colors the same way
(`accentForTask(task, theme)`, `taskAccents(theme)`), so a run is the same color there as
in its drawing in either theme. A new inline color in a
component should be a token or go through the codemod; a literal written for the dark
background will look wrong in the light theme.

## ADR-126 — The default profile keeps the welcome's name, the picker asks each session, and the greeting follows a pronoun

**Date:** 2026-10-20
**Status:** Accepted
**Amends:** ADR-114 (profiles), ADR-090 (the first-run welcome)

**Context:** the user tested the profiles. On a one-person install the welcome
asks a name ("Marcus"), which ADR-090 keeps in the browser only; the runs go to
the default profile (`outputs/models`). After creating a profile "Luana", the
picker listed "Padrão (sem perfil)" and "Luana": "Marcus" had vanished, though
his runs were all there under the default. The user read it as a lost profile.
They also asked to be asked who they are on each visit rather than being dropped
into the last profile, and to choose how the welcome addresses them: Portuguese
"Bem-vindo" is masculine.

**Decision:**

- **The default profile has a name and a pronoun on the server**, in
  `outputs/profiles/default.json` (not a valid slug, so the listing never takes
  it for a profile folder). `GET /api/profiles` reports it as the default's
  `name` ("default" while none is set), and the picker shows "Marcus (padrão)".
  The welcome's name prompt writes it; a browser that already holds a name from
  before this ADR and is on the default writes it once on its next visit, so the
  existing "Marcus" comes back without being asked again. Nothing moves on disk.
- **`PUT /api/profiles/{slug}`** changes a profile's display name and pronoun
  (`default` included); the slug, and so the folder, never changes. The header's
  name chip, which already reopened the name prompt, now edits the current
  profile through it.
- **A pronoun per profile**: `ele`, `ela` or `neutro`, chosen next to the name
  when it is first given (the name prompt and "criar perfil") and stored in
  `profile.json` / `default.json`. Portuguese greets "Bem-vindo", "Bem-vinda" or
  "Boas-vindas"; English says "Welcome" for all three. A profile without one
  (made before this ADR) greets in the neutral form until it is set.
- **"Quem é você?" once per browser session** on a server with named profiles,
  not only when the stored choice is gone: the last profile is preselected, so
  one click or Enter continues. The answer lives in `sessionStorage`, so a
  reload in the same tab does not ask again and a new tab or window does.

**Alternatives considered:**
- *Make the welcome create a real profile.* "Marcus" would be a folder of its
  own and his earlier runs, under `outputs/models`, would sit in a different
  profile from the new ones.
- *Keep the name only in the browser and label the default with it.* Another
  browser, or the same person on another machine, would still see
  "Padrão (sem perfil)".
- *Neutral greeting for everyone, no question.* Simpler, but the user asked for
  the choice.
- *Ask on every page load.* A reload would interrupt work for no new answer.

**Consequences:** on a shared server the default profile's name is whoever named
it, which matches ADR-114: profiles organise, they do not protect. The greeting
chip and the welcome take the pronoun from the profile in use; the stored name
in the browser (`vf.welcome.name`) remains the fallback when the server cannot
be reached.

## ADR-127 — Static files stay within the resolved SPA root

**Date:** 2026-10-20
**Status:** Accepted

**Decision:** The SPA fallback resolves its root and requested file and rejects
paths outside that root before testing or serving files. Legitimate client-side
routes still receive index.html.

**Reason:** Joining a root with an untrusted path does not prevent traversal.
An encoded parent segment exposed a temporary file outside the static directory
in the audit for issue #47. Loopback binding is not a filesystem boundary.

## ADR-128 — Classification splits share the training label vocabulary

**Date:** 2026-10-20
**Status:** Accepted

**Decision:** Classification validates the training class count against the
configured output dimension (two classes for the binary one-logit convention).
Validation and test splits may omit known classes, but their sample labels use
the training class mapping; unknown classes fail before training, and so does
a validation or test split with no images at all.

**Reason:** Independent ImageFolder discovery numbered a missing-class split
differently and silently changed the meaning of its labels (issue #48).

## ADR-129 — Dataset fingerprints version their record encoding

**Date:** 2026-10-20
**Status:** Accepted
**Extends:** ADR-061

**Decision:** Fingerprints include an encoding version and frame path and size
records unambiguously. Comparisons across encoding versions return unknown;
missing versions identify legacy records. Manifest and content remain distinct.

**Reason:** A file named x1 with two bytes and x with twelve bytes both encoded
as x12 in the original manifest (issue #51). Hashing cannot fix ambiguous input.

**Consequences:** version 2 frames each record as a length-prefixed UTF-8 path
plus an 8-byte size, sorts records by the path bytes, and prefixes the preimage
with the version and method, so a manifest digest never equals a content digest.
`DatasetFingerprint.encoding_version` is written to `run.json`; a stored
fingerprint without it reads as version 1. Caches keyed by the digest (the
filtered-dataset copies of ADR-084) miss once after the upgrade. The leaderboard
groups runs by method and digest, so a version-1 run and a version-2 run of the
same folder appear apart rather than as the same dataset; it never merges two
different datasets.

## ADR-130 — Filtered datasets use operating-system leases

**Date:** 2026-10-20
**Status:** Accepted
**Extends:** ADR-084

**Decision:** A stable cache-root lock serializes construction, acquisition and
deletion. Each active caller holds an operating-system lease; cleanup skips
busy leases without waiting. A completion marker is published only after a
successful build. Windows uses msvcrt locks and POSIX uses flock.

**Reason:** A second GUI process cannot infer that another process is dead
from its own empty reference count (issue #50). Operating-system locks release
on process exit and avoid PID reuse assumptions. The root lock file remains
stable to prevent competing processes locking different inodes.

**Consequences:** each owner holds its lease in a `.lease-*` file inside the copy;
the gate lock lives outside the cache tree (`<cache_root parent>/.<name>.materialized.lock`,
under the git-ignored `outputs/`) and is never deleted. A copy is reused only when
its `.in-use` completion marker exists; a half-built one is removed by the next
builder. On Windows a killed process's byte-range locks are freed shortly after
it exits, so `sweep_orphans` retries a busy lease for up to 2 s: a GUI starting
next to a live one spends that long per live copy. Removal when the last user
leaves is best effort: if another process holds the gate at that moment, the copy
stays until the next startup sweep.

## ADR-131 — Sweeps select on validation, never on test metrics

**Date:** 2026-10-20
**Status:** Accepted
**Extends:** ADR-045, ADR-041, ADR-058

**Context:** the generic sweep (`core/sweep.py`, the GUI's sweep for every task
but classification's own grid and random search) ranked trials and steered Optuna
with `runner.metrics(result)`. The regression, segmentation and anomaly runners
take those from the test split, so hyperparameters were tuned on the hold-out the
report then presented as an independent evaluation (issue #49).

**Decision:** `RunResult` carries `selection_metrics` (validation) apart from
`metrics` (final evaluation). The sweep ranks trials and feeds Optuna with
`selection_metrics` only. `SweepTrial.metrics` holds the validation numbers it was
ranked on, `SweepTrial.evaluation_metrics` the final ones, and `selection_split`
is "validation"; the sweep report states `selection_split` too, and the results
view says the ranking is on validation. A trial without a finite validation value
for the metric fails with that reason; test numbers are never a fallback.

Each runner takes `selection_metrics` from the validation record of the checkpoint
its trainer kept (the best-epoch history row), with no extra evaluation pass:
- Regression: r2, rmse, mae, mse.
- Segmentation: miou, dice, pixel_acc.
- Classification: accuracy only; the trainer measures no f1 or AUC on validation,
  so a sweep ranked on those fails each trial. (Classification's grid and random
  search blocks already rank on validation loss.)
- Detection: map50_95 and map50, which Ultralytics and the RT-DETR backend compute
  on the `val` split of `data.yaml`. If a user points `val` at their test images,
  nothing here can tell.
- Researcher tasks that run inside VisionForge's loop (Level 1): the best epoch's
  `val_metrics`.

A runner that cannot measure on a labelled validation split declares
`selection_unavailable`, and the sweep is refused before any training; the GUI
answers 422 with the reason. That covers anomaly detection, which has no
labelled validation split, and Level 2 researcher tasks, whose returned metrics
come from a split VisionForge cannot see. The anomaly panel no longer offers a
sweep, and the self-test has no anomaly sweep case. Replicates and model
comparisons keep reporting evaluation metrics: they do not choose
hyperparameters.

**Alternatives considered:**
- *Fall back to test when validation is missing.* That is the defect itself.
- *Score anomaly on a slice of the test set.* The test set would then both choose
  and judge, under a different name.

**Consequences:** sweep tables show validation numbers, so they are not
comparable with the test numbers History and the replicates show for the same
run. Error messages from the refusal and the per-trial failures are English, like
the rest of `core/sweep.py`.

**Amended 2026-10-21: the anomaly panel compares architectures through
`/api/anomaly/compare`.** Removing the sweep also removed the "architectures"
preset, the panel's only way to set autoencoder against PatchCore. The panel gets
a "Comparar arquiteturas" strategy instead: a card with one checkbox per anomaly
model (all picked, at least two to run) and the ranking metric (AUROC or image
F1), sent by the main Train button to the existing comparison route, whose report
the results view already draws. The card and the results table both say the
numbers are test-set metrics and that the table is a comparison report, not a
model selection; the card adds that a claim of one being better needs replicates
of each. Nothing on the server changes: a comparison trains each model once with
the same config and tunes nothing, so ranking it on test is the evaluation the
runs would report one by one. The card (`CompareModelsCard`) and the
`runCompare` client call are task-agnostic; regression, segmentation and detection
do not offer them yet.

## ADR-132 — routes.py is split one domain at a time, behind an OpenAPI contract test

**Date:** 2026-10-21
**Status:** Accepted

**Context:** `gui/api/routes.py` is about 5,600 lines and holds endpoints, task
dispatch, the state of the running job, report writing and dataset operations
(issue #54). Reviewing a change to one task means reading code that also drives
the queue of every other. It is debt seen by inspection, not an observed defect.

**Decision:** shrink it in steps, one domain per change, keeping every HTTP
contract.

- *Target shape.* Thin routers per domain (profiles, system, datasets,
  runs/history, queue and run state, task dispatch): a router validates input,
  resolves the request's dependencies (language, profile), calls a service and
  maps errors to HTTP status codes. Execution and report services carry no
  FastAPI import and are testable without HTTP. The queue and the running job's
  state (`_current_run`, `_event_queue`, `_active_cancel_token`, the job queue)
  get one explicit owner that routers command through methods.
- *No cycles, no side effects on import.* A router never imports `routes.py`.
  Shared values (`_MODELS_DIR`, `_PROFILES_DIR`) keep one owner and reach the
  routers as a callable read per request until a shared dependencies module
  exists; tests that repoint `routes._MODELS_DIR` keep working. Importing a
  router or service starts no training, opens no connection and creates no folder.
- *The gate.* `tests/gui/test_openapi_contract.py` compares every operation
  (method, path, operationId, parameters, request and response schema names)
  with `tests/gui/fixtures/openapi_paths.json`. Moving an endpoint must leave the
  fixture untouched; an intended API change regenerates it in the same change
  (`VF_UPDATE_OPENAPI_CONTRACT=1`) and the diff is reviewed.
- *Order.* profiles (done: `gui/api/profiles_routes.py`), system, datasets,
  runs/history, queue and run state, task dispatch. A step starts only after the
  previous one is verified.

**Alternatives considered:** a single rewrite into routers and services (one
unreviewable diff over the code that ties UI, data and training together);
separate services or processes (nothing here needs to scale out).

**Consequences:** the contract test sees operations, not the fields inside a
schema; a renamed field in a response model still passes it. Until a shared
dependencies module exists, `routes.py` owns the output folders and the
language dependency, and the extracted router is mounted from it so it inherits
`request_lang`.

**Update (2026-10-21):** system information, device information and model
defaults move to `system_routes.py`; `system_service.py` composes their payloads
without FastAPI. Runtime probes remain injected and are resolved per request.
The profile and system routers inherit the parent language dependency; the
OpenAPI fixture remains unchanged. Update endpoints stay with the execution
state they depend on. Further domains remain incremental work under this ADR.

**Update (2026-10-24):** datasets are done: dataset, picker and credential endpoints (and the helpers only they use) move to `datasets_routes.py`, which owns the dataset-root allow-list; `routes.py` imports from it, never the reverse. The OpenAPI fixture is unchanged. Next: runs/history.

**Update (2026-10-24):** runs/history is done: `GET /runs`, `GET /runs/{run_id}` and `DELETE /runs/{run_id}` move to `runs_routes.py` (`build_runs_router`, lookups and the running job read per request from `routes.py`); the `run.json` parsing (list summary, run span, resume status, metric directions) moves to `run_history.py`, free of FastAPI; the error-response docs shared by routers move to `http_responses.py`. Reveal, resume, export and the per-run actions stay in `routes.py` with the execution state they use. The OpenAPI fixture is unchanged. Next: queue and run state.

## ADR-133 — An end-to-end GUI suite drives the built SPA against a real server on the CPU

**Date:** 2026-10-21
**Status:** Accepted

**Context:** the frontend tests (Vitest) mock the API and the self-test drives
the API without a browser. Nothing checked that what a researcher clicks
(welcome, sample dataset, epochs, device, Train, Stop, History) reaches the
trainer and comes back to the screen (issue #52).

**Decision:** a small Playwright suite, `frontend/e2e/*.e2e.ts` (`npm run e2e`),
drives the SPA served by a real `visionforge gui` and trains the classification
sample dataset for real.

- The server starts through `e2e/serve.mjs` from `frontend/e2e/.work`, recreated
  on every run and git-ignored, so the suite never writes to the user's
  `outputs/` or `datasets/`. It runs with `CUDA_VISIBLE_DEVICES=""` and the tests
  pick CPU in the device selector: it never takes a GPU a real training is using.
- `VF_E2E_EXECUTABLE` selects the server (the project venv locally, the installed
  `visionforge` in CI); the port is 8765 unless `VF_E2E_PORT` says otherwise.
  Locally the installed Chrome is used (no browser download); CI installs
  Playwright's chromium.
- Five serial tests, one worker, no retries (there is one training slot, and a
  retry would meet the runs of the failed attempt): one epoch through to the
  History, stopping a training, a brief status outage, a second browser's queued
  run, and a dropped event stream while status remains available.
- The suite tests the built bundle in `src/visionforge/gui/static`, which is
  committed: rebuild it before running locally. A new CI job, `e2e`, builds it,
  runs the suite and uploads the report and traces when it fails.

**Consequences:** the tests disable pretrained weights in the form and use
random initialization, so no weights download is needed. The connection test
first showed a defect: one failed status
poll made the training overlay declare the run failed and stop watching, although
the run finished on the server. The poll now lives in `lib/status-poll.ts`: a
failure the server answered (HTTP error) still ends the watch, but one it did not
answer is retried after 2, 4, 8, 8 and 8 s, with the overlay saying
"reconectando…" and the run's status left as it was. After those 30 s the overlay
says the connection is lost and keeps trying every 10 s; when the server answers,
the note goes away and the live stream is reattached. The test now requires the
overlay to recover and end as "treino concluído".

**Update (2026-10-21):** every healthy status poll for the page's running job
reattaches a missing event stream, even when the status endpoint never went
offline. An SSE error closes EventSource; waiting for a status-network recovery
alone left progress permanently disconnected. The browser regression blocks
the initial stream while status stays healthy, then requires a new stream and
a real epoch event after recovery. A second browser also verifies a queued job
starts after the active job stops. E2E runs disable pretrained weights in the
form so training needs no model download.

## ADR-134 — Epoch-boundary resume restores randomness and completed bookkeeping

**Date:** 2026-10-21
**Status:** Accepted
**Extends:** ADR-092, ADR-093

**Decision:** Resume format 2 records Python, NumPy and torch CPU/CUDA RNG
states, plus explicit loader generators when present. Trainers restore them
after setup and loader construction, immediately before the next epoch. The
saved best metric, best epoch and patience describe the epoch just completed.
Format 1 remains readable with a warning that RNG continuity is unavailable.
New checkpoints use safe torch loading and atomic publication.

**Reason:** Reseeding at startup restarts the random sequence instead of
continuing it. The old save order also persisted the previous epoch's best
metric and patience. Equivalence is tested on CPU with zero loader workers;
process-local augmentation state with persistent workers and backend-owned
Ultralytics resume are not covered by that guarantee.

**Dataset identity:** before resuming native training, the recorded dataset
fingerprint is compared with the current data using the recorded method.
A confirmed change refuses continuation and preserves the original run. Legacy
or unavailable identity emits a warning; corrupt existing run metadata fails
explicitly. This check inherits manifest hashing's same-size-edit limitation.
When configured outputs lie inside the dataset root, identity is also reported
as unverifiable: the legacy root manifest includes the run's changing artifacts.

**Best-checkpoint isolation:** classification freezes a CPU copy of model state
before submitting its asynchronous writer, including state-dict metadata and
extra state. Waiting for an earlier writer does not freeze tensor storage shared
with the next optimizer step. The snapshot costs CPU memory and a synchronous
device-to-host copy on improvement; disk serialization remains asynchronous.

## ADR-135 — Provenance identifies the source and exports contained artifacts

**Date:** 2026-10-21
**Status:** Accepted
**Extends:** ADR-013, ADR-057

**Decision:** Environment metadata adds versions of training/statistical
dependencies and the actual package checkout's Git revision and dirty flag.
Installed distributions with no verified source checkout report unknown; the
researcher's current directory is never used as a proxy for package source.
Git probes are local, bounded and deferred until metadata is requested.
Native trainers also record the effective loader worker counts in run.json;
the configured automatic value alone does not describe the process layout used.

An experiment bundle includes run.json and explicitly selected config,
history, prediction and model artifacts contained inside that run directory.
It contains a SHA-256/size manifest and records missing or excluded artifacts.
It never walks the dataset or arbitrary output directories. Export refuses an
existing destination and publishes atomically. Verification checks member
names, declared sizes and digests without extracting the archive.

**Reason:** A package version alone does not identify local edits or the
versions of libraries that compute metrics. A portable record needs integrity
checks without silently copying unrelated files or overwriting an earlier
export.

**CLI:** `visionforge bundle RUN_DIR DESTINATION.zip` exports the record and
`visionforge verify-bundle DESTINATION.zip` checks it. Both print JSON; invalid
requests or archives exit with code 2. No training or dataset copying occurs.

## ADR-136 — Data audits compare content and classification CV respects groups

**Date:** 2026-10-21
**Status:** Accepted
**Extends:** ADR-013, ADR-128

**Decision:** A read-only dataset audit hashes image bytes to find exact
duplicates across declared splits, reporting unreadable and unsafe paths with
bounded details and deterministic ordering. It does not infer visual similarity.
Optional CSV metadata uses path,group columns with dataset-relative image paths.
Strict audits report groups crossing split boundaries as invalid.

Classification cross-validation can consume this group metadata and use
GroupKFold or StratifiedGroupKFold, preserving the current split strategy when
no group manifest is provided. Every training sample must have one group;
invalid, ambiguous and missing mappings fail before training. The number of
distinct training groups must support the requested fold count.

**Reason:** Correct class indices and disjoint filenames do not prevent related
samples or identical image bytes from leaking across evaluation boundaries.
Group identity must come from the researcher; the application cannot infer a
patient, plant or acquisition session from image content.

**CLI:** `visionforge audit-data ROOT` prints a JSON report; repeated
`--split NAME=PATH` overrides split locations, `--groups` supplies metadata and
`--strict-groups` requires disjoint groups. `--json` writes a new report without
overwriting an existing file. Exit codes are 0 for a clean audit, 1 for findings
and 2 for invalid requests. Auditing never changes the dataset.

## ADR-137 — Anomaly is amber on screen, and a researcher's task wears its own color

**Date:** 2026-10-21
**Status:** Accepted
**Amends:** ADR-058 (researcher tasks), ADR-125 (light theme)

**Context:** the stage takes its accent tokens (`--accent-vf` and its three
derived tones) from a `.stage[data-task=…]` block in `index.css`. Classification,
detection, regression and segmentation had one; anomaly did not, so its screen
fell back to the default red, the classification color, although its tab, the
history and the training story already call it amber. A researcher's task
registers a `#rrggbb` accent (`@register_task(accent=…)`) that colored only its
tab; the rest of its screen was red too. The user also asked to choose that color
from the interface.

**Decision:**
- `index.css` gains the anomaly block, in both themes, with the amber the rest of
  the interface uses (`oklch(0.80 0.15 75)`, `oklch(0.50 0.15 75)` in light).
- A researcher's task drives the stage from its own color: `lib/task-color.ts`
  derives the four tokens with `color-mix` and the stage sets them inline.
- A dot in the header, shown only while a researcher's task is open, picks its
  color: ten swatches, a free color input and "back to the original color". The
  choice is per browser (`vf.taskColor.<key>`), like the theme and the language;
  only `#rrggbb` values are read back. The task's tab and screen follow it.

**Alternatives considered:**
- *Store the color with the task on the server.* The color is a preference of
  whoever looks at the screen, as the theme is; the registered accent remains
  the task's own default.
- *Let built-in tasks be recolored too.* Their colors identify them across the
  history, the training story and the documentation.

**Consequences:** the history and the training story still show a researcher's
run in the neutral color they used before (`accentForTask` has no color for
`custom:` tasks). In the light theme a chosen color is used as is, without the
darkening the built-in accents get.

## ADR-138 — Explicit resume refuses unusable state before training setup

**Date:** 2026-10-21
**Status:** Accepted
**Extends:** ADR-092, ADR-093, ADR-134

**Decision:** Native trainers require a usable checkpoint whenever resume_dir is
provided. Missing, unreadable, unsupported and invalid state raise a specific
ValueError before model/device preparation, optimizer creation or output writes.
Read-only capability probes retain the nullable load_resume_state contract.
An explicit Ultralytics continuation requires its last.pt; PatchCore rejects
continuation because it has no resumable epoch loop.

**Reason:** A user asking to continue must not unknowingly pay for a new training
trajectory after a checkpoint disappears or becomes unreadable. Probing history
must still tolerate older and damaged runs without breaking the listing.

## ADR-139 — CI validates installed distributions outside the checkout

**Date:** 2026-10-21
**Status:** Accepted
**Extends:** ADR-073, ADR-133

**Decision:** A separate CPU job builds distributions, installs the wheel without
editable source paths and executes a synthetic workflow outside the repository.
It checks package origin, installed static assets, CLI commands, a tiny training
and the resulting experiment bundle. CI actions use supported Node runtimes.

**Reason:** Editable installs can hide missing files and imports that depend on
the checkout. Publishing is too late to discover that the built package cannot
serve its own interface or execute the documented workflow.

**Update (2026-10-25):** The published-package smoke test retries the same spec
up to five times with pip's HTTP cache disabled, waiting 60 seconds only between
attempts. Exhaustion exits with pip's status before any VisionForge command.
The spec passes through a step environment variable to preserve it literally.
After the 0.18.1 release ([issue #90](https://github.com/marcus-vreis/VisionForge/issues/90)),
three runners still listed only 0.18.0 across five
attempts, while three passed; rerunning failed jobs on new VMs passed. Index
propagation or HTTP caching is a plausible explanation, not a confirmed cause.
Each attempt requests the index without reusing pip's local HTTP cache and
preserves installation failures rather than masking them with a missing CLI error.

## ADR-140 — The GUI exposes image audits, classification groups and run bundles

**Date:** 2026-10-21
**Status:** Accepted
**Extends:** ADR-114, ADR-135, ADR-136

**Decision:** Dataset audits run explicitly on local image splits through a
dedicated endpoint off the event loop. Classification CV exposes group-manifest
selection. History can download a contained run bundle; the server resolves runs
inside the active profile and cleans temporary archives after the response.
Audit findings are visible without silently rewriting data or training settings.
An in-process reservation excludes continuation while source files are copied;
request cancellation waits for the writer before removing temporary files.

**Reason:** The local-first interface must expose the safeguards available to
CLI users. Hashing and ZIP creation must not block status polling or copy data
outside the selected run/profile.

## ADR-141 — A restricted CPU baseline checks classification parity before a study

**Date:** 2026-10-21
**Status:** Accepted
**Extends:** ADR-061, ADR-134

**Decision:** The workflow benchmark starts with an independent native PyTorch
classification baseline and a matched VisionForge adapter. Both load identical
initial ResNet18 weights, fixed transforms, splits, seed, optimizer and CPU worker
settings. A paired harness records phase timings and verifies histories, selected
weights and test predictions before any performance interpretation.

**Reason:** Runtime and human effort cannot be attributed to tooling when the
underlying experiments differ. Synthetic runs validate the harness only; they
are not usability observations, scientific benchmarks or evidence of superiority.

## ADR-142 — Shared form controls expose labels and field feedback semantically

**Date:** 2026-10-23
**Status:** Accepted
**Extends:** ADR-140

**Decision:** Shared controls use React `useId` for stable per-instance label
associations and `aria-describedby` for hints, persistent help and validation
errors. Toggles expose switch state; segmented choices expose a named group and
pressed state. Browser tests locate fields by their accessible labels.

**Reason:** Decorative text above an input neither names it for assistive
technology nor focuses it when clicked. Repeated labels need distinct targets,
and closed tooltips must not remove field instructions from the accessibility
tree. Linking existing validation messages identifies the invalid field without
changing form values, task accents or training behavior.

## ADR-143 — Public classification pilots preserve official data provenance

**Date:** 2026-10-23
**Status:** Accepted
**Extends:** ADR-141

**Decision:** The first public-data engineering pilot uses a seeded, balanced
CIFAR-10 subset. Preparation accepts only a locally supplied binary archive
matching the authors' checksum, derives validation from official training records,
and preserves official test membership, source indices and content hashes in a
manifest. It neither downloads data nor overwrites an existing output directory.
The paired CLI defers torch imports until CUDA is hidden and CPU threads are bounded.

**Reason:** A reproducible public corpus complements domain-specific data without
confusing a small CPU parity check with a full benchmark. Resource contention from
an active training job excludes comparative timing claims; larger corpora and
GPU measurements require separately scheduled execution and matched protocols.

## ADR-144 — SonarCloud follows the integration branch

**Date:** 2026-10-23
**Status:** Accepted
**Extends:** ADR-139

**Decision:** VisionForge's main branch in SonarCloud is `development`, matching
the target of contribution PRs. The scanner runs on pull requests and on pushes
to `development`, where it refreshes the base analysis using that branch's own
checkout and coverage. GitHub's release branch remains `main`.

**Reason:** The Free plan only analyzes PRs targeting the Sonar main branch. A
renamed branch with a months-old analysis is not an up-to-date baseline; scanning
the integration branch after each merge keeps the reference tied to real code
without relabeling a PR head as its base or requiring a paid organization plan.

## ADR-145 — Artifact paths pass lexical authorization before resolution

**Date:** 2026-10-23
**Status:** Accepted
**Extends:** ADR-127, ADR-140

**Decision:** Artifact requests must lie inside the absolute outputs root before
filesystem resolution. The resolved path must still remain inside the resolved
root before existence checks or serving. External aliases pointing into outputs
are rejected, as are internal aliases pointing outside it.

**Reason:** Resolving an unauthorized path consults filesystem entries before
the containment check. Lexical authorization rejects those requests first;
canonical containment remains necessary to prevent symlink or junction escapes.

## ADR-146 — Run identifiers and group metadata stay inside the selected model root

**Date:** 2026-10-23
**Status:** Accepted
**Extends:** ADR-002, ADR-113, ADR-123, ADR-124

**Decision:** Built-in and custom task configs validate run names as portable,
single directory components. Group names and IDs use the same validation. A group
writer validates its destination and every child metadata/staging path against
the selected model root before tagging any child. Escaping symlinks or junctions
and multiply linked metadata files are rejected. Explicit model roots outside
outputs remain supported.

**Reason:** A name containing parent segments could create a group outside the
configured root; a child report could redirect tagging to an unrelated run.json.
Preflight validation prevents partial tagging when a later path is invalid and
stops existing staging aliases from redirecting writes. Existing unsafe names
are rejected rather than silently rewritten; ordinary Unicode names remain intact.

## ADR-147 — Sonar analysis versions come from the analyzed checkout

**Date:** 2026-10-23
**Status:** Accepted
**Extends:** ADR-073, ADR-144

**Decision:** The Sonar CI job reads `project.version` from its checkout's
`pyproject.toml` using Python's standard library and passes it as
`sonar.projectVersion`. Missing or unsafe values fail before scanning; neither
installed-package metadata nor build numbers supply a fallback. The properties
file contains no duplicate fixed version.

**Reason:** The fixed `0.0.1` label no longer represents the source version
`0.17.0`. The project's effective New Code definition was verified as Previous
version before this correction. A real version transition can advance that
reference; a greener New Code gate after the transition does not demonstrate
that historical findings were fixed. The definition, quality gate, rules and
exclusions stay unchanged, and overall findings remain subject to triage.

## ADR-148 — Workflow benchmark reports share a phase contract

**Date:** 2026-10-23
**Status:** Accepted
**Extends:** ADR-141, ADR-143

**Decision:** The native and VisionForge classification paths time the same four
machine phases: setup, training, evaluation and evidence. Both run one shared,
untimed preflight before creating their output directory. Setup ends with data,
model and initial weights ready; training ends when the best checkpoint is on
disk; evaluation reloads it and predicts the test split; evidence hashes that
checkpoint and assembles the report. Each report lists the work each path does
inside every phase, the cache conditions, library versions, CPU threads, process
priority and the operator's declaration of concurrent workloads. A failure in
any phase writes a partial report naming the failed phase and keeps completed
artifacts. Reports are created exclusively and never replace an earlier file.
Reports without `phase_contract` keep their original implementation-phase scope;
readers classify them as legacy and do not reinterpret their timings. Phase
times are machine wall-clock time; human effort is never derived from them. The
paired CLI hides CUDA and limits threads before importing torch, then lowers its
process priority.

**Reason:** The earlier timings began and ended at different points and
silently included different artifact writes, so they could not be compared.
Fixed boundaries plus a disclosed work list make differences visible without
pretending the implementations do identical work. Recording failure phases keeps
failed runs in the dataset instead of dropping them. Synthetic fixtures and runs
under concurrent load still cannot support speed or human-effort claims.

**Update (2026-10-24):** The paired entry points also run preflight before
creating the pair's parent output directory, rather than relying only on each
runner's check after that directory exists. Rejected controls or missing weights
therefore create neither a directory nor a partial pair report. JSON is
serialized before opening the exclusive report file, so a non-serializable or
non-finite value cannot reserve an empty file and prevent a valid retry.

## ADR-149 — Segmentation parity uses a restricted CPU recipe and a fixed trimap map

**Date:** 2026-10-23
**Status:** Accepted
**Extends:** ADR-143, ADR-148

**Decision:** The workflow benchmark adds an independent native PyTorch
segmentation path and a paired harness that runs it against the public
`SegmentationDataModule` and `SegmentationTrainer` under the ADR-148 phase
contract. The restricted recipe is binary segmentation (background 0,
foreground 1, ignore 255), no augmentation, bilinear image and nearest-neighbour
mask resize, fixed normalization, cross-entropy with `ignore_index`, SGD without
momentum or scheduler, workers=0 and a small fully convolutional network whose
initial weights both paths load from the same file. The best checkpoint is the
first epoch with the highest validation mean IoU, as in the trainer. Metrics come
from one confusion matrix over non-ignored pixels; mean IoU and Dice average the
classes present in ground truth or prediction. The shared preflight refuses a
mask whose resized pixels are all ignored, because an all-ignored batch makes
cross-entropy undefined; such a mask contributes nothing to the confusion matrix.
Oxford-IIIT Pet trimaps map 1 (pet) to 1, 2 (background) to 0 and 3 (border) to
255; any other value is an error. The harness requires identical losses, best
epoch, per-image confusion matrices, test predictions and best-state tensors,
and allows a 1e-12 absolute difference only on mean IoU, Dice and pixel accuracy.
The report records source revision, dataset fingerprint, weights and prediction
hashes, environment and the failed phase, and is never overwritten.

**Reason:** Both implementations execute the same torch operations in the same
order on one CPU thread, so losses, weights and predictions have no legitimate
source of difference and any tolerance would hide a recipe drift. The metric
reductions differ by library (NumPy versus torch, both float64) over identical
integer confusion matrices, so only rounding can differ; one changed pixel moves
a metric by at least one over the labelled pixel count, far above 1e-12 for any
planned dataset size. The model is a harness input rather than a VisionForge
architecture because the check targets data, loss, selection and metrics, and a
tiny network keeps the fixture tests within seconds on one thread. Synthetic
fixtures prove integration only; the Oxford-IIIT Pet acquisition, its splits,
architecture and full budget remain separate, authorised steps.

## ADR-150 — The group manifest must resolve inside the dataset root

**Date:** 2026-10-23
**Status:** Accepted

**Decision:** `load_group_manifest` resolves the manifest (absolute or relative to
the dataset root, following symlinks) and refuses it with a localized
`ValueError` unless it lies under the resolved root. `POST /api/dataset/audit`
already maps `ValueError` to HTTP 400, so a request naming `/etc/x.csv`,
`../x.csv` or a symlink leaving the root can no longer make the server open
arbitrary CSV files. The CSV rows were already contained; only the manifest
itself was not.

**Consequences:** The CLI (`audit-data --groups`) and grouped cross-validation
(`cross_validation.group_manifest`) share the function and therefore the rule: a
manifest stored outside the dataset root must be copied or linked inside it.
This is a deliberate behaviour change for local users who kept manifests
elsewhere.

## ADR-151 — Typed contexts for native epoch loops and standalone CV

**Date:** 2026-10
**Status:** Accepted

**Decision:** `save_epoch_checkpoint`, `run_epochs` and `fit_training` receive
typed dataclasses for completed epoch state, loaders and lifecycle controls,
checkpoint dependencies and pending writes, task callbacks, metric policy and
fit adapters. `_run_fold` and `run_folds` receive typed fold data/lifecycle and
task factory/metric contexts. Dependency contexts are frozen; completed epoch
state and pending checkpoint writes remain mutable and local to one fit.

**Reason:** Issue #87 identifies five long signatures left by the shared-loop
extraction. Grouping dependencies by lifetime keeps checkpoint bookkeeping and
callbacks coherent without a generic kwargs container or changing task recipes.
Classification retains its AMP path; regression and segmentation retain their
fit adapters. Optimizer and seed defaults, strict best-metric comparison, fold
splits, reporting order, checkpoint formats and `run.json` remain unchanged.
Resume still restores optimizer, scheduler and scaler before restoring RNGs
after loader setup. CPU tests cover history, checkpoint state, stopping and
failure boundaries; paired classification and segmentation harnesses check
engineering parity without making performance or model-quality claims.

## ADR-152 — Model-card rendering is split into pure sections

**Date:** 2026-10
**Status:** Accepted
**Extends:** ADR-132

**Decision:** Model-card rendering lives in `gui/api/model_card.py`, with sections
for identity, configuration, preprocessing, augmentation, metrics, history, plots
and external tests. `routes.py` retains its renderer and stopped-extent aliases
for existing callers. Section functions only format the supplied run.json data
using the current artifact language; they do not mutate it or perform I/O.

**Reason:** `_render_run_markdown` combines independent document sections in one
API hotspot (#86). Extracting each section makes its ordering and optional-field
rules reviewable without changing route dependencies or runtime state. Complete
PT/EN golden files checked against the pre-extraction renderer protect Markdown
equality, including a sanitized run.json from the offline E2E fixture. Synthetic
edge fixtures cover missing fields and stopped runs; none are research evidence.
Queue ownership, dataset inspection and standalone evaluation remain separate
increments.

## ADR-153 — Dataset inspection separates parsing, split scans and layout reporting

**Date:** 2026-10
**Status:** Accepted
**Extends:** ADR-132

**Decision:** The dataset inspection functions in `datasets_routes.py` delegate
YOLO crop parsing and per-image sampling, annotation class counting, segmentation
pairing and mask sampling, regression manifest reading and target aggregation,
and layout role matching and reporting to private helpers in the same domain.
The router, request/response schemas and localization calls keep their contracts.

**Reason:** Issue #86 identifies five S3776 hotspots in dataset inspection.
Separating those responsibilities removes nested traversal and parsing from the
collectors without moving unrelated API state or introducing import side effects.
Characterization tests preserve the distinct YOLO field rules for crops and
counts, the 20-decodable-mask, 500-image-check and 50,000-row limits, and the
complete, partial and unrecognized layout responses. The OpenAPI contract stays
unchanged.

## ADR-154 — Explicit device selection in the workflow benchmark

**Date:** 2026-10
**Status:** Accepted
**Extends:** ADR-148, ADR-149

**Decision:** The restricted paired recipes accept CPU (default) or single-device
CUDA through the existing DeviceConfig.kind. CPU keeps its process guard and
exact comparisons by default. CUDA preserves visibility, threads and priority,
requires an explicit finite nonnegative parity tolerance, enables deterministic
algorithms and cuDNN, and sets CUBLAS_WORKSPACE_CONFIG before importing torch.
Each CUDA phase synchronizes before both timer boundaries and records peak
allocated VRAM bytes. Initial and comparison state dictionaries and best
checkpoints stay on CPU. Reports include device conditions, weight maximum
absolute difference, tolerance and declared concurrent workload.

**Reason:** CPU-only timing cannot measure accelerator runtime or memory. CUDA
launches are asynchronous, and equivalent kernels need not produce bitwise
identical floating-point weights across the two execution paths. Explicit
absolute tolerance (zero relative tolerance, integer buffers exact) makes that
trade-off auditable without silently weakening existing CPU checks. Histories,
selected epochs and predictions retain the existing checks; CUDA tolerance
applies only to floating-point weights. Unavailable CUDA and missing tolerance
fail before any output. Mocked CUDA tests cover instrumentation on CPU; real
GPU measurements require separate authorization and no undeclared load.

## ADR-155 — Receita v2 de classificação e determinismo parcial da segmentação em CUDA

**Date:** 2026-10-28
**Status:** Accepted
**Extends:** ADR-149, ADR-154

**Decision:** Os pares de classificação aceitam `--weight-decay` e
`--scheduler {none,cosine}` (padrão igual à receita anterior), espelhados no
lado nativo e na configuração do Trainer. A segmentação em CUDA ativa
`use_deterministic_algorithms(True, warn_only=True)` nos dois lados, porque
`nll_loss2d` não tem implementação determinística em CUDA, e compara as losses
de época com a tolerância `--parity-atol`; mIoU, Dice, acurácia de pixel, época
escolhida, confusões por imagem e predições continuam com as tolerâncias do
ADR-149. A CPU mantém a comparação exata. `prepare_pet.py` monta os splits do
Oxford-IIIT Pet preservando o `test.txt` distribuído e sorteando a validação só
do `trainval.txt`.

**Reason:** A calibração em GPU mostrou losses divergindo em ~1e-9 entre os
caminhos, com pesos, métricas e predições idênticos. Exigir igualdade exata da
loss reprovaria um par equivalente; relaxar só a loss mantém as demais
verificações estritas e auditáveis. Uma execução em CUDA não é bit-reprodutível
neste caminho; a paridade fica garantida pela tolerância declarada e pelo
relatório.

## ADR-156 — Task configuration in six steps, Treinar unlocked on the review

**Date:** 2026-10-29
**Status:** Accepted
**Amends:** ADR-059 (canonical experiment header)

**Context:** every task configured on one long page: experiment name and
strategy on top, then model, training, dataset, preprocessing and augmentation
stacked in that order, the dataset (the only thing a first run needs) in the
middle. Detection spread its Ultralytics knobs over three top-level sections and
put the image size beside the dataset path; segmentation kept the number of
classes under Model and warned about `ignore_index` in a banner above the
header. Names defaulted to `config` or `detection_001`, so History filled with
runs nobody could tell apart, and a sweep ran every trial under the same name.
The user sketched the flow by steps and approved a standalone demo
(`.claude/plans/2026-10-29-wizard-etapas/`), asking for one Treinar button only,
locked until the last step, and a suggested experiment name on that step.

**Decision:**
- Every task panel renders `StepShell` with six steps: Dados, Estratégia, Modelo,
  Treinamento, Imagem e aumento, Revisão (a researcher's task uses those its
  schema fills). The rail shows each step's state: `padrão`, `n ajustes`,
  `falta`, `n erros`. Any step opens at any time; server errors bring the first
  step with one to the front.
- All steps stay mounted and the inactive ones are `hidden`. The strategy cards
  listen to the bottom bar's run signal and the dataset stats fill other fields
  as they load; unmounting them would break both. The guided tour asks for the
  step that holds its target (`vf:reveal-step`).
- The step per task lives in App, so switching task tabs keeps it.
- Treinar exists only in the bottom bar. It is locked, with the reason as its
  tooltip, until the active step is Revisão and the dataset is set.
- The experiment name moves to Revisão, prefilled with
  `<dataset>_<model>_e<epochs>_lr<lr>[_<strategy>]` (`lib/experiment-name.ts`).
  The suggestion follows the configuration while the name is empty, a task
  default or the previous suggestion; a typed or imported name stays.
- Sweep trials are named after what they vary (`core/trial_name.py`):
  `…_grid-lr` runs `…_grid-lr_lr0.001`, `…_grid-lr_lr0.002`, with the same
  short keys as the suggestion. Each trial's folder follows its name.
- Classification's transfer learning moves from the strategy to Modelo, as in
  segmentation and regression; Estratégia answers only "how many runs and how".
- Detection groups its advanced training knobs (optimizer, LR schedule, loss
  gains, regularization and mechanics) under one Advanced block and its
  augmentation by kind, with the image size on the image step; segmentation's
  number of classes and `ignore_index` sit with the dataset that defines them.

**Alternatives considered:**
- *A wizard that forces the order.* Most runs change one or two values from the
  defaults; making each step a gate would cost clicks on every run.
- *Unmount inactive steps.* Simpler DOM, but the run signal and the dataset
  effects would then depend on which step is on screen.
- *Treinar enabled from step one when the data is set.* The user asked for the
  review to be the gate, so a run is never started from a half-checked screen.

**Update (2026-11-11, amendment, #200):** a run comes back to the form from
History. The run and group details offer *Carregar no formulário* and *Repetir
com 3 seeds*. App keeps one pending import per task, switches to the run's tab
and the panel applies it through the handler its YAML button calls
(`reviewImportedConfig` in classification, the schema check in the standalone
panels, the plain load in a researcher's task), so the warnings and refusals
are the YAML import's, shown on the Revisão name card. Loading lands on
Revisão as a single run; repeating selects Réplicas with explicit seeds after
the largest one the run or its replicate set used, and lands on Estratégia
unless the import reported something. A sweep, K-fold or comparison group only
loads: it has no one config to repeat. A researcher's task that no longer has
a tab keeps both actions on screen, off, with the reason beside them. The
decisions live in `lib/history-rerun.ts` so they are tested without a DOM.

## ADR-157 — Looping animations end while a job runs

**Date:** 2026-11-01
**Status:** Accepted

**Decision:** While the server runs a job, `App` marks the root element and
`index.css` forces `animation-iteration-count: 1` on everything under it. Looping
animations finish their current cycle and the page stops producing frames.

**Why:** Keeping the page on screen made epochs about 17% slower than native on
the benchmark machine, while a closed or minimized browser cost about 1%. Four
loops ran during training (`shimmer`, `spin`, two `pulse-dot`). Cancelling any one
of them saved 3 to 4 s per epoch, and cancelling all four took the 73 s epoch to
62.5 s, the native time. One loop is enough to keep frames coming.

**Consequences:**
- The training spinner and status dots hold still during a run; progress shows
  in the numbers and charts, which update per batch or epoch.
- A loop added later is covered without touching this rule.
- Finite animations such as sheets opening still play.

**Alternatives considered:**
- *Remove the four loops one by one.* The next loop added would bring the cost
  back without anyone noticing.
- *Tell users to minimize the window.* Leaves the cost on the default way of
  watching a run.

---

## ADR-158 — Per-model learning rate in comparisons and grids

**Date:** 2026-11-01
**Status:** Accepted
**Extends:** ADR-099, ADR-100 (suggested rates and collapse evidence), ADR-039 (`schema_version`), ADR-156 (Review)

**Decision:** Classification searches can give each model its own learning rate,
only when the config asks for it.

- `model_comparison.lr_policy`: `fixed` (default) trains every architecture at
  `training.learning_rate`; `suggested` uses
  `suggested_learning_rate(arch, optimizer)` per trial.
- `grid_search.lr_mode`: `absolute` (default) reads the `training.learning_rate`
  axis as rates; `suggested_multiplier` reads it as factors of each model's
  suggested rate. It requires the `model.name` and `training.learning_rate` axes
  and positive numbers.
- The rate a trial trains at is written everywhere under its real name: the
  trial's `config.training.learning_rate` in `run.json`, `effective_learning_rate`
  in `comparison_summary.json`, `ranking.csv`, `grid_search_summary.csv`, the block
  report and the History group row. A factor is kept apart as `lr_multiplier`, and
  `best_config.yaml` holds the effective rate.
- The Review lists each combination's rate from `POST /api/model/learning-rate-plan`,
  which calls the functions the blocks train with. The form computes no rate, and
  the training row stops showing a number the trials will not use.
- The Review warns on `is_collapse_prone`, quoting only `collapse_evidence`. That
  measurement is one model per family at Adam 1e-3, so the sentence names that
  setting instead of implying the flagged combination was run.
- `CURRENT_SCHEMA_VERSION` goes to 2. The step changes no shape: the defaults keep
  a v1 config training as written.

**Why:** A single rate biases the ranking. At Adam 1e-3, vgg16, alexnet and swin_t
collapsed and vit_b_16 did not learn (`core/learning_rate.py`), so a comparison
measured the rate as much as the architecture. A Cartesian grid does not help,
since the same absolute rates are far from the working point of half the models.

**Consequences:**
- Sweeps still select on validation, never on the test set (ADR-131). The suggested
  rate comes from the architecture and optimizer, not from this dataset.
- The suggested rates were measured on one 4-class setup, for one model per family.
  `suggested` removes a known failure; it does not tune each model, and a ranking
  under it should be reported as such. The factors exist to check around the
  suggestion.
- An older build refuses a v2 config instead of ignoring `lr_policy` or `lr_mode`.
  Ignoring them would train at other rates without saying so, and would read 3 as
  a learning rate. This is why an additive field got a version bump, against the
  "breaking changes only" wording of ADR-039.
- Random search has no list in the Review: its rates are sampled at run time.
- The task-agnostic comparison (`core/comparison.py`, `POST /{task}/compare`) is
  unchanged: the suggestions and evidence were measured on classification.

**Alternatives considered:**
- *Silently apply the suggestion in every comparison.* Changes old results and is
  the failure this module exists to avoid.
- *A per-model map of rates in the config.* More flexible, but the researcher would
  transcribe the suggestions by hand, and the grid already covers explicit choices.
- *Compute the rates in the form.* Two copies of the table, which would drift.

## ADR-159 — Queueing and watching a run from the same tab

**Date:** 2026-11-03
**Status:** Accepted
**Extends:** ADR-075 (run queue), ADR-088 (stopping a running job)

**Context:** the queue panel says to submit as many runs as you like, and the
server takes them, but one tab could not: with a run in flight the main button
only reopened its training sheet. Changing the label would not have been enough.
The hook followed one run and a new submission erased its state; the sheet read
its task and strategy from the open form; and the event stream handed each event
to a single reader, so a reloaded page had no log and a second tab took epochs
from the first.

**Decision:**

- While anything runs or waits, the main button reads **Add to queue**, still
  behind the Review lock. The submission leaves the followed run, its log and its
  result alone, and a short notice confirms it.
- **View training** on the running row of the queue panel opens the sheet, and
  replaces the bar's "open training", which only knew this tab's own run. It also
  works after a reload or from another tab, from the job's queue entry.
- What the sheet shows about a run is fixed at submission, per run.
- `gui/api/run_events.py` keeps the current run's events and every reader walks
  them. `GET /api/experiment/events` gains `after`, so a dropped stream asks only
  for the rest, and `run_id`, so a client never reads another job's events as its
  own. Events are per epoch and dropped when the next run starts.

**Consequence:** the sheet does not jump to the next job, and the result of a run
nobody was watching does not open by itself, since it would interrupt someone
configuring another run. It is in the History. The tab's title and notification
follow the watched run only.

**Alternatives considered:**

- *Rename the button only.* Each of the problems above would have shown up as a
  wrong log or a lost result.
- *Follow several runs in the hook.* More state for no gain: one GPU runs one job,
  and only that job has a log to read.

## ADR-160 — A live "Results" tab on the training sheet

**Date:** 2026-11-04
**Status:** Accepted
**Extends:** ADR-118 (the training sheet's tabs), ADR-119 (overfitting sign)

**Context:** during a training the sheet showed the log and the illustrative
story, so the only way to read the curves or know which epoch the saved model
came from was to wait for the run to end and open its result. The numbers were
already on the page: every `epoch_end` carries the losses and the task's metric.
This is phase A of `docs/dev/design-resultados-ao-vivo-e-colab.md`.

**Decision:**

- A third tab, **Results**, between the log and "How it works". It draws the loss
  curves and the task's headline metric of the training in flight, from the
  `epoch_end` events the sheet already receives. Nothing changes on the server
  and the training pays nothing.
- A card names the **best epoch** with the rule the trainer applies when it saves
  `best_model.pth`, so it is the epoch whose weights are kept: lowest `val_loss`
  for classification and regression, highest `val_miou` for segmentation,
  highest mAP@50-95 for Ultralytics and mAP@50 for torchvision (it measures no
  mAP@50-95), lowest training loss for the anomaly autoencoder. A tie keeps the
  earlier epoch, as the trainers save only on a strict gain. The rules live in
  `lib/live-results.ts` beside the trainer each one comes from.
- A researcher's own task gets no card: its metric's direction is declared in
  its task info and is not in the stream, and a guessed direction would name the
  wrong checkpoint. PatchCore has no epochs and says so.
- In a multi-unit job (grid, comparison, cross-validation, replicates) the curves
  are the running unit's, and a table lists the closed ones with what each
  changed and its best epoch.
- The overfitting sign of ADR-119 is repeated here, from the same function.

**Consequence:** the card and the trainer's rule must change together. A new
trainer, or a new checkpoint metric, needs its line in `bestRule`.

**Alternatives considered:**

- *Have the server send the best epoch in each `epoch_end`.* Exact by
  construction, but it touches every trainer for a number the client can already
  compute; left for phase B, which adds server fields anyway.
- *Reuse the History comparison chart.* It plots finished runs read from
  `run.json` with one colour per run; the live tab plots two series of one run
  and grows by the epoch.

## ADR-161 — Learning rate and validation confusion in the epoch event

**Date:** 2026-11-04
**Status:** Accepted
**Extends:** ADR-160 (the live "Results" tab)

**Context:** the live tab drew only what `epoch_end` already carried. Two more
figures cost the training almost nothing: the learning rate, which tells a
schedule apart from a stuck run, and, for classification, the validation
confusion, whose predictions `_eval_epoch` already computes. This is phase B of
`docs/dev/design-resultados-ao-vivo-e-colab.md`.

**Decision:**

- Every trainer adds `lr` to `epoch_end`: the rate the epoch trained with, read
  before the scheduler steps (`training_loop.EpochLr`), from the parameter group
  the config's `learning_rate` sets (the head when `backbone_lr_multiplier`
  scales the backbone). Ultralytics reports its own `trainer.lr["lr/pg0"]`,
  written at the end of the epoch before its scheduler steps.
- Classification adds `confusion` (`core/live_confusion.py`): the full matrix up
  to 30 classes, `matrix[true][predicted]`; above that, the 10 most frequent
  mistakes as `[true, predicted, count]`, ties to the lower class pair. It is
  `null` rather than wrong when a label falls outside the class list.
- Neither goes into `run.json`. The history keeps its fields, and the test-set
  confusion matrix in `metrics` stays the one a paper reports.
- The key is `confusion`, not `val_confusion`: the client types `val_*` as the
  numeric metrics of researcher-defined tasks (ADR-058).

**Consequence:** a client that does not know the fields ignores them, so older
GUIs keep working. The confusion of an epoch is gone once the run ends; the
optional `live/epoch_metrics.jsonl` of the design was left out until something
needs it.

**Alternatives considered:**

- *Add `lr` to each task's `EpochResult`.* It would enter `run.json` history and
  every reader of it, for a number the schedule config already determines.
- *Read the rate when the epoch is reported.* The scheduler has stepped by then,
  so it is the next epoch's rate: a StepLR run would show every drop one epoch
  early.

## ADR-162 — Live Grad-CAM of fixed validation images

**Date:** 2026-11-04
**Status:** Accepted
**Extends:** ADR-160 (the live "Results" tab), ADR-161 (live epoch fields)

**Context:** the curves say how much a classifier learns, not what it looks at.
`core/gradcam.py` already explains a finished checkpoint, but only after the run,
and its `GradCAM` writes the parameters' `.grad` and leaves the model in `eval`,
which is harmless after training and not during it. This is phase C of
`docs/dev/design-resultados-ao-vivo-e-colab.md`.

**Decision:**

- `training.live_gradcam_every` (classification), **0 by default**: every N
  epochs, on the last epoch and on each new best (lowest `val_loss`, the
  checkpoint rule), `core/live_insights.py` explains 6 fixed validation images
  and the stream sends `{"event": "insight", "kind": "gradcam", "epoch", "best",
  "images": [{"path", "true_class", "predicted_class"}]}`. The Results tab shows
  the unit's latest set.
- The training must not notice it, and a test holds it to that: the same seed on
  CPU with `deterministic` gives the same history and the same weights with the
  option at 1 and at 0, with a shuffled training loader that would reorder its
  batches if any generator moved. A mutant that draws one random number inside
  the pass fails that test. The rules:
  1. the samples are chosen once with a `torch.Generator` of their own, taking
     the classes in turn, and read by index from `val_loader.dataset`, never by
     iterating the loader; every pass runs inside a fork of the Python, NumPy and
     torch generators (`preserved_rng`);
  2. the gradient is `torch.autograd.grad(score, activation)`: no parameter's
     `.grad` is written. The input requires grad, so a frozen backbone still
     puts the activation in the graph;
  3. the model runs in `eval` and goes back to the mode it was in;
  4. a failure logs one warning and turns the insights off for the run.
- Files: `insights/latest/` is rewritten each time and copied to
  `insights/best/` on a new best. Nothing per epoch, nothing in `run.json`.
  The images are served by the existing `/api/artifacts` route, so a run whose
  `models_dir` is outside `outputs/` writes them but the tab cannot show them.
- Overlays undo the run's own `normalize_mean`/`normalize_std`, not ImageNet's.

**Consequence:** a run that turns it on spends K forward and backward passes per
explained epoch (K = 6) and writes 6 small PNGs. Off by default, a run and its
`run.json` are unchanged. Detection, segmentation, regression and anomaly need
other pictures (boxes, masks, a scatter, the anomaly map) and are left for later.

**Alternatives considered:**

- *Reuse `GradCAM` from `core/gradcam.py`.* Its `.backward()` accumulates into
  the parameters' `.grad`; the optimizer zeroes them before the next step, but a
  run with gradient accumulation or a custom loop would not, and the rule is
  simpler kept absolute.
- *On by default with N = 5 on GPU.* The cost is small, but a default run would
  start writing files and the setting would differ between machines.
- *A "generate now" button.* Left out: it needs a channel from the GUI to the
  running job, as the stop has (ADR-088), for a convenience.

## ADR-163 — Training on Google Colab through the official CLI

**Date:** 2026-11-05
**Status:** Accepted — phases 1 (connection and login), 2 (remote training, classification), 3 (importing the result) and 4 (parallel queues) implemented
**Extends:** ADR-075 (the run queue), ADR-114 (profiles), ADR-160 (the live Results tab)

**Context:** many users have no local GPU, and Colab has one. The 2026-11-04
plan in `docs/dev/design-resultados-ao-vivo-e-colab.md` was a hand-run notebook
(F1), because Colab had no API. Google has since published `google-colab-cli`,
which allocates a runtime, runs code with live output, moves files and
releases the VM, all from the user's own account. A spike on 2026-11-05 (one
run, one T4, a paid account) measured it on this machine:
- the T4 was granted in 4.1 s;
- 180 of 180 one-second ticks arrived, with gaps between 0.95 and 1.06 s and an
  accumulated delay under 0.1 s;
- `colab stop` left no session behind.

The numbers and the code reading behind this ADR are in the design doc's
2026-11-05 sections.

**Decision:**

- **Backend.** `google-colab-cli`, pinned to an exact version (0.7.5 today).
  It is installed by VisionForge into a venv it manages, in
  `~/.visionforge/colab-cli/venv`, and never into the project environment. A
  CLI upgrade is a deliberate change: bump the pin, rerun the spike.
- **Windows runs it natively.** No WSL is needed. The README says "Linux and
  macOS only", but the only POSIX-only code is `colab console`, which
  VisionForge does not use.
- **Login per profile.** Every CLI call runs with `USERPROFILE` and `HOME`
  pointing at `~/.visionforge/colab/<profile>/`, so the token, the session
  list and the log stay in that folder.
  - The login is Google's copy-and-paste OAuth flow. The GUI shows the URL and
    passes the pasted code to the CLI's stdin.
  - VisionForge only checks that `token.json` exists. It never reads the
    token, logs it or sends it anywhere.
  - The account is identified by the e-mail that `colab whoami` prints.
- **Queues.** The local queue stays as ADR-075 made it: one job at a time on
  this machine, on the GPU or the CPU, never both.
  - Colab gets one queue per Google account, not per profile, because the
    session quota and the compute units belong to the account.
  - A local job and a Colab job may run at the same time.
- **Events come from the `exec` stdout.** The remote runner prints the usual
  training events, one JSON line each. The backend feeds them into the same SSE
  stream the Results tab already reads.
  - The job ends with a final event of its own (status `ok` or `error`),
    distinct from the per-training `end`.
  - The exit code of `colab exec` is not trusted: in 0.7.5 a failing script
    exits 0, and a healthy one exits 1 once it outlives `--timeout`.
  - `--timeout` is therefore set above the longest job the account allows.
  - The remote runner never calls `sys.exit`, because that hangs `exec`.
- **Balance.** The GUI shows the hours left: the balance divided by the chosen
  accelerator's rate, both read from `colab usage`. It shows this only when the
  account reports compute units.

**Phases, one PR each:**

1. **Connection and login.**
   - The managed venv and its pinned install.
   - The per-profile login from the GUI, with the account shown and a switch
     to change it.
   - The "Train on Colab" button stays disabled without a login.
   - `colab usage` shown as hours left.
2. **Remote training.**
   - A remote runner that installs `visionforge-studio` at the local version
     and gets the dataset from where it already is (URL, Drive, Hugging Face),
     not through `colab upload`.
   - It runs the job and prints the events.
   - The backend relays them to the Results tab and stops the VM in a
     `finally`.
   - At startup, VisionForge lists leftover sessions and offers to stop them.
3. **Importing the result.** `colab download` of the zipped run directory at
   the end, imported as a local run with `device_used` and the remote
   environment kept. This is the bundle import the F1 plan already needed.
   - As built: the VM writes the run with `export_experiment_bundle`, the same
     format a bundle exported by hand has. The bundle's VM path and size go out
     in `job_end`, and the job downloads it before `colab stop`.
   - The file is the proof that the download worked, as `job_end` is for
     `exec`: the job checks that it exists and has the announced size.
   - `import_experiment_bundle` verifies the manifest, refuses unsafe names
     and anything over 2 GB, and builds the run in a staging folder outside
     `models/`. One rename then puts it in `models/<experiment>/<run>`.
   - run.json gets `origin: {kind: "colab", ...}`, and History marks the run.
     The paths recorded on the VM are rewritten to the files that arrived.
   - Plots are not part of the bundle format, so an imported run has its
     run.json and checkpoint, but no PNGs.
4. **Parallel queues.** One Colab queue per account next to the local queue.
   This means decoupling the SSE stream and the result records from the single
   active job of `RunQueue`.
   - As built (backend): `RunLanes` holds the local `RunQueue` and one per
     account, keyed `colab:<e-mail>` from `colab whoami`. A login with no e-mail
     reported gets a lane of its own, `colab:profile:<slug>`, rather than one
     shared by guess.
   - Each started job has a `JobState` (event log, cancellation token, status)
     by `run_id`, which its executor looks up; the three single-run globals of
     `routes.py` are gone. The status and event endpoints take a `run_id` and
     default to the job started last.
   - `GET /api/queue` keeps the local lane's `active` and `pending` and adds
     `lanes`, and every job carries its `lane`.
   - As built (GUI): the queue sheet has one section per lane, this machine's
     and the connected account's Colab one (shown empty before its first job),
     and each row says where its job runs. Stop and watch act on their row's
     job. The bottom bar's badge counts every lane, the training sheet reads its
     run's stop point in whichever lane runs it, and a Colab job still waiting
     says it waits for its account rather than for the GPU.

**Consequence:**
- A user without a GPU trains from the same GUI, with the same live tab, on
  their own Google account. Nothing listens on the user's machine and no tunnel
  is opened.
- What VisionForge stores is a token file in a folder the user owns, like the
  personal credentials in `~/.visionforge/`.
- The cost is a dependency on a young CLI that talks to Colab's internal
  endpoints (`/tun/m/...`) and changes often.

**Risks:**
- A CLI release can break the contract. The pin and the spike are the guard,
  and `colab update` only announces releases; it never installs one.
- `colab whoami` is hidden ("developer-only") and may change. Plan B is the
  OpenID `userinfo` endpoint, which would mean reading the token.
- Free accounts were not tested: whether they get a T4 at all, and what
  `colab usage` shows for them. The GUI must say "no GPU on this account"
  instead of failing quietly.
- If Google restricts headless use, the hand-run notebook (F1) is still
  there.

**Alternatives considered:**

- *A notebook the user runs by hand (F1).* No account access is needed, but
  there is no live tab and no queue, and every run takes manual steps. It
  stays as the fallback.
- *Our own OAuth client against the Drive API, plus a tunnel.* VisionForge
  would own an OAuth client, with Google's app verification and the support
  that comes with it, and would expose a port on the user's machine. The
  CLI's flow avoids both.
- *Running the CLI in WSL.* It works, but it is an installation hurdle on
  Windows that native support makes unnecessary.
- *`colab run` instead of `new` + `exec` + `stop`.* Its exit code is
  reliable, but it creates and releases a VM per call, which breaks a job
  that uploads, trains and downloads on the same machine.

## ADR-164 — Colab datasets from the user's Google Drive, with the consent in the GUI

**Date:** 2026-11-06
**Status:** Accepted — implemented (classification, like ADR-163 phase 2)
**Extends:** ADR-163 (training on Colab through the official CLI)

**Context:** a Colab run takes its dataset from a public zip or a public
dataset. Many users keep theirs in Google Drive, synced to the PC by Google
Drive for desktop, so VisionForge already reads it as a local folder. Two
spikes on 2026-11-05 (CPU runtimes, `google-colab-cli` 0.7.5, design doc)
measured `colab drivemount`:
- it asks for a browser consent and then waits for Enter on stdin. With the
  consent, the mount sees the files that were already in the Drive (50 of 50
  top-level entries older than 7 days);
- the consent is not kept: a new runtime asks again. The credential is
  ephemeral, per VM (`authorize-for-drive-credentials-ephem`);
- an Enter sent before the consent gives `Error propagating: 400`, and the CLI
  then hangs.

Marcus decided that the user must never touch a terminal: the GUI only, and
opening another tab to allow the Drive is fine.

**Decision:**
- **The source.** A `drive` source means "the form's own dataset folder". The
  server translates `data.base_dir` to `/content/drive/MyDrive/...`
  (`gui/api/colab_drive.py`). It finds "My Drive" in this order:
  - the folder the user typed;
  - the `DefaultMountPoint` of Drive for desktop in the registry;
  - a "My Drive"/"Meu Drive" component right under a drive letter or under
    the home folder.
  It sends the local file count along. A folder outside the Drive, or an empty
  one, is refused before a VM is paid for.
- **The mount.** It runs after the kernel warm-up and before the runner,
  whenever the job reads from the Drive or exports to it.
  - The consent URL becomes a `drive_consent_required` event, and never a log
    line, because its `state` carries a token.
  - The job thread waits in a `DriveConsent` until
    `POST /api/colab/drive/continue` arrives, then writes the Enter itself.
  - A 400 kills the hung CLI and the mount is tried again with a new URL, up
    to 3 times.
  - The person's wait (600 s per attempt) and the CLI's silence (120 s) have
    deadlines of their own. Neither counts against the runner's startup
    watchdog.
- **The copy.** The runner copies the folder to the VM's disk (reading
  through the FUSE mount is slow) and fails when it finds fewer files than
  this machine has, which is what a folder still uploading looks like.
- **The GUI.** The training sheet shows "Allow Drive access", which opens the
  page in a new tab. The return of the window answers the server (focus, or
  the tab visible again, only after the window lost it). "I already allowed"
  is the fallback when the focus never fires.
- **Export (optional).** The run's bundle is also copied to
  `MyDrive/VisionForge/runs/`, on top of the phase 3 download. A failure there
  does not fail the run.

**Consequences:**
- No upload and no Drive API of our own: the CLI's credential propagation is
  the only path to the Drive, and the token stays on the server.
- Every Colab run that uses the Drive needs one click from the person, per
  runtime. A queued job that reaches the mount while nobody is at the screen
  waits up to 3 × 10 min, and then fails and releases the VM.
- "My Drive" in a language other than English or Portuguese, and the mirror
  mode outside the home folder, need the folder typed by hand.

**Risks:**
- The flow reads the CLI's English output ("Please visit", "Error
  propagating", "Mounted at"). A CLI release that rewords it breaks the mount.
  The pin and the fake-CLI tests are the guard.
- The detection was never run on a machine with Drive for desktop installed.
- Answering on focus can send the Enter before the person has finished on
  Google's page. That costs one attempt (a 400 and a new URL), not the run.

**Alternatives considered:**
- *Uploading the dataset with `colab upload`.* It works for any folder, but it
  costs the upload time on every run, which is what the Drive avoids.
- *Our own OAuth client and the Drive API.* Rejected in ADR-163 for the same
  reasons: app verification, support, and a token of our own.
- *Showing the CLI in a terminal for the Enter.* It was the spike's setup, and
  Marcus ruled it out for users.

## ADR-165 — A read-only dataset explorer in the Datasets surface

**Date:** 2026-11-05
**Status:** Accepted — phase 1 of issue #117 (classification and detection, read only) implemented
**Extends:** ADR-132 (routes split by area), ADR-153 (dataset inspection)

**Context:** the Datasets button of the bottom bar only downloaded data and kept credentials. Before a training run the researcher still had no way to look at a dataset: the stats probes (`/dataset/stats`, `/<task>/dataset/stats`) return counts and a few sample crops, and `audit_dataset` (ADR-153) is a CLI-shaped report over a whole tree with no progress and no way to stop it. Issue #117 proposes a data-tools surface in phases; this is phase 1, the explorer. Editing, merging and re-splitting are not part of it, and so the dataset fingerprint (ADR-129), the filtered datasets (ADR-130) and the group manifest (ADR-150) are not touched here: they matter once a tool creates a new dataset version, which is where the issue recommends a new folder or a manifest so that old `run.json` fingerprints stay valid.

**Decision:**

- **Where.** A second tab, "Explore", in the existing `DatasetsOverlay`, next to the download tab. It is not a new tab beside the tasks: a dataset is not owned by a task (ADR-132's reason for the overlay).
- **Scope.** Classification (ImageFolder, with the same configurable `train`/`val`/`test` folder names the task panels use) and detection (YOLO, with the split resolution the trainer uses, `resolve_yolo_split`). Class names for YOLO come from `classes.txt`/`names.txt` and, new here, from the `names` of `data.yaml`. The explorer never writes inside the dataset.
- **Two passes with different costs.**
  - `core/dataset_explorer.build_index` only lists directories and reads label files. It feeds the per-class and per-split counts, the boxes per class and the gallery, and is cheap enough to rerun on every open or refresh.
  - `scan_integrity` opens every image: it verifies it (Pillow `verify`), hashes the bytes (SHA-256, the same as `audit_dataset`, by reusing its `_verified_image_digest`), and reads the size. From that it reports images that do not open, exact copies, the same content in more than one split (leakage) and the size and aspect-ratio distributions. A copy whose copies carry different class labels is flagged as a label conflict.
- **The scan cannot hang the server.** It runs in a worker thread (`gui/api/explorer_jobs.ScanJobs`) with progress and the existing `CancellationToken` (`core/cancellation.py`). One scan at a time (409 otherwise): it is disk-bound. A hard limit (`max_images`, 20,000 by default, 500,000 at most): above it the scan reads an evenly spaced sample, and the report says `sampled` and tells the user that a copy or leak found is real while finding none proves nothing. Cancelling returns the partial report, marked `cancelled`.
- **No full-resolution image reaches the page.** `GET /api/explorer/thumb` resizes on the server (32–512 px, JPEG), and draws the YOLO boxes for detection. It takes the dataset root and a path relative to it; the root must be one the explorer already opened (the allow-list of `/dataset/file`), and the relative path may not contain `..` or leave the root through a link.
- **The gallery is paginated** (48 per page, 200 at most) and filtered by split and class on the server from an in-memory index (last four datasets). A detection class filter matches the images that have a box of that class.
- **Where the code lives.** Logic in `core/dataset_explorer.py` (no `gui` import); the thread registry in `gui/api/explorer_jobs.py`; the routes in `gui/api/explorer_routes.py` (`/api/explorer/{summary,images,thumb,scan}`); on the page, the pure helpers in `lib/dataset-explorer.ts` and the view in `components/DatasetExplorer.tsx`. All texts are in the dictionaries (pt-BR and English).

**Consequences:**

- The OpenAPI snapshot gains the six explorer operations; no existing endpoint changed.
- The scan's hash is the same SHA-256 as the audit's, so the two reports agree on what an exact copy is. Visual near-duplicates are not detected (as in the audit).
- The index of a big dataset is held in memory per opened dataset (a few hundred bytes per image). The cache keeps four.
- `datasets_routes._read_yolo_class_names` still reads only the `.txt` files. The explorer's reader also understands `data.yaml`; folding the two into one is left for later so this change does not touch the stats probes.

**Risks:**

- A sampled scan can give the false impression of a clean dataset. The report and the page both say that a sample proves nothing about absence, and the default limit is high enough for the usual datasets to be scanned whole.
- Thumbnails of very large images still decode them (JPEG is decoded at reduced scale, other formats are not). Pillow's decompression-bomb guard applies and a file that fails answers 422, shown as a broken tile.

**Alternatives considered:**

- *Reuse `audit_dataset` for the scan.* It walks the tree recursively with its own rules, has no progress, no cancel and no per-class view, and returns one big report at the end. Its digest function is reused; its walk is not.
- *A scan inside the request.* Simple, but a dataset with tens of thousands of images would hold a worker for minutes and give the page nothing to show.
- *Serving the original files for the gallery (`/dataset/file`).* No resizing, so a page of 48 photographs could move hundreds of megabytes.
- *A new tab beside the tasks.* Rejected in issue #117's discussion: the explorer is not a task.

## ADR-166 — Incorporating new images writes a new dataset, never an edit

**Date:** 2026-11-06
**Status:** Accepted — phase 2 of issue #117 (classification and detection) implemented
**Extends:** ADR-165 (explorer), ADR-129 (fingerprint), ADR-130 (materialized copies), ADR-136 (content audit), ADR-153 (dataset inspection)

**Context:** a researcher collects more images after a first training and wants them in the dataset. Writing them into the folder would change its fingerprint (ADR-129), and every earlier `run.json` remembers the fingerprint of the folder as it was: the runs would then point at data they did not train on. Issue #117 recommends a new folder for any tool that creates a dataset version; this is that tool, for the two layouts of the explorer.

**Decision:**

- **Never in place.** The result is a sibling folder (`<name>-v2` by default, or the name the user types, validated by `validate_path_component`) with the same structure. The original is not opened for writing at any point. A test hashes the whole tree (paths and bytes) before and after.
- **How it is written.** Under `.<name>.tmp-<id>` in the same parent, then one `os.rename`. Cancellation, an error, or a destination that appeared meanwhile removes the temporary folder, so nothing is left at the destination. Files of the original are hardlinked when the volume allows it and copied when it does not (`link_or_copy`: any `OSError` falls back to a copy; symbolic links are copied). `materialized_dataset.py` has no such mechanism (it re-encodes every image), so this is new, and small. New images are always copied: the folder of new images is the user's scratch space and must not share bytes with the dataset. The copy is hashed while it is written and must match the digest of the preview, so an image replaced between the preview and the write is refused.
- **Existing images do not move.** The mirror is made first; only then new files are placed.
- **Splits of the new images.** Stratified by class, by largest remainder, seeded (`random.Random(f"{seed}|{class}")` over images sorted by content hash, so the file order does not matter). The proportion is the dataset's own (images per split) or one the user types; a split the dataset does not have gets zero and a positive ratio for it is refused, so `test` is never created. The seed (default 0) is recorded. Detection stratifies by the most frequent class of the image; an image without boxes is its own stratum.
- **Checks before applying** (all listed in a preview the user confirms). Classification: `<class>/<image>`, anything loose is out. Detection: the image with a `.txt` next to it or in a sibling `labels` tree, five numeric fields per row, integer id >= 0. Images that do not open (Pillow `verify`, the digest function of the audit) are out. Exact copies by SHA-256 against the dataset and among the new images (first kept) are out. Every exclusion keeps its reason and detail.
- **New classes.** A folder name the dataset lacks (classification) or an id beyond its list (detection) is a new class and its images stay out unless the user ticks it. When the new folder has its own `classes.txt`/`data.yaml`, its labels are read through that list and matched by name, so a `0` there need not be a `0` here; the label files are rewritten only if an id moves. Without a list, ids are the dataset's own and an unknown id `N` is the class `class_N`. Accepted classes are appended; existing ids never change. Where a bare id leaves a gap, the gap is filled with `class_<i>` placeholders. The result's `classes.txt`/`names.txt` and `data.yaml` (`names`, `nc`, and an absolute `path` that pointed at the original) are written anew, never through a hardlink.
- **Record.** No convention existed (the group manifest of ADR-150 is another thing), so the new dataset gets `visionforge_dataset.json` (`schema_version` 1): the origin path and manifest fingerprint of the base, the folder incorporated, seed, requested and effective ratios, split folder names, final classes, accepted new classes, counts (base, found, added, per split and class, excluded by reason) and how many files were linked or copied. The explorer opens the result unchanged; the extra file is just a file of the dataset.
- **Jobs.** `gui/api/incorporate_jobs.py`, one at a time, thread plus `CancellationToken`, progress counter. The preview (read, verify, hash both sides) keeps its analysis in memory so changing a checkbox, the ratios or the seed re-plans through `POST .../plan` without hashing again, and the apply writes exactly what was shown. If the original changed (manifest fingerprint) or a new image changed since the preview, the apply is refused (409) and the user redoes the preview.
- **API.** `POST /api/explorer/incorporate/preview` (the dataset must be one the explorer opened: 403 otherwise, the allow-list of thumbnails), `GET .../{job_id}`, `POST .../{job_id}/plan`, `.../apply`, `.../cancel`. The source folder may not be inside the dataset nor contain it. Errors are documented with `responses=` from `http_responses.py` (which gains `FORBIDDEN`) and job errors are translated when the job is read.
- **Page.** A button "Incorporar imagens" inside the Explore tab, once a dataset is open: source folder (the existing picker), preview (counts per class and split, exclusions with reasons, ticks for new classes, ratio, seed), destination name, apply with progress and cancel, then "open in the explorer". Texts in both dictionaries; pure logic in `lib/dataset-incorporate.ts`.

**Consequences:**

- The OpenAPI snapshot gains five operations; no existing endpoint changed. `DatasetExplorer` shares its inline styles through `datasetExplorerStyles.ts`.
- A hardlinked file is the same file: editing an image of the new dataset in place, with a tool that writes through the existing inode, also changes the original. Tools that save a new file and rename (most editors) are safe. The record says how many files were linked.
- The preview hashes the whole dataset (SHA-256 of every existing image) to find copies; that is the cost of comparing by content, as in ADR-136.
- Near-duplicates are not detected (as in the audit). Symbolic links to folders inside the original are not followed by the mirror, as the explorer does not follow them.

**Risks:**

- A crash (process killed) between the first write and the rename leaves a hidden `.<name>.tmp-*` folder next to the dataset; nothing sweeps it yet. It never has the real name, so no run can pick it up by mistake.
- The ratio is respected per class, but not corrected across classes: many classes with few new images each can drift a little from the overall proportion.

**Alternatives considered:**

- *Edit in place with a backup.* Breaks the fingerprint of every old run before the user decides anything.
- *A manifest layer over the original (no copy).* Cheap, but no trainer reads it and every consumer would have to learn it.
- *Reusing `materialized_dataset`.* It re-encodes images and owns a cache with leases; copying a dataset byte for byte is a different job.
- *Moving existing images to rebalance.* Out of scope (phase 4, re-splitting) and it would make results incomparable with old runs.


## ADR-167 — Merging two datasets writes a new one, with a class mapping

**Date:** 2026-11-06
**Status:** Accepted — phase 3 of issue #117 (classification and detection) implemented
**Extends:** ADR-166 (incorporate), ADR-165 (explorer), ADR-129 (fingerprint), ADR-136 (content audit)

**Context:** two datasets of the same problem (two collection campaigns, a public set and the lab's own) have to become one before training, and their classes rarely agree: `Cat` and `cat`, `car` and `vehicle`, a class one side never needed. As with ADR-166, writing into either folder would change a fingerprint that old `run.json` files remember, so the result is a new dataset. Phase 4 (re-splitting) stays separate: a merge that also reshuffles would make results incomparable with both sources.

**Decision:**

- **Never in place, shared mechanics.** The result is a sibling of dataset A (`A.parent/<name>`), built under `.<name>.tmp-<id>` and published with one `os.rename`; cancellation, an error or a destination that appeared meanwhile removes the temporary folder. B is only read. What ADR-166 had inside `dataset_incorporate.py` moved to `core/dataset_derived.py` and both features use it: `link_or_copy`, `free_name`/`claim_name`, the temporary-folder-plus-rename wrapper (`prepare_destination`, `publish`), the fingerprint re-check, the progress counter, the registry writer and the class-file writers (`classes.txt`, `names.txt`, `data.yaml`). The jobs got the same treatment (`gui/api/derived_jobs.py`). Incorporate behaves as before; its tests ran unchanged. `IncorporateError` now derives from `DerivedError`, and `link_or_copy` is still importable from `dataset_incorporate`.
- **Tasks.** Classification (ImageFolder) and detection (YOLO); both datasets are read as the task of the open one. A source that has no images, that is not a folder, that equals the other or contains it is refused (400).
- **Class mapping.** One row per class of A and of B (origin, image count), each with a target name or "discard". A class the mapping does not mention keeps its own name, so classes that share a name merge without asking, and two rows with the same target merge. Targets are stripped and must be non-empty; for classification they must be a valid folder name (`validate_path_component`), and two targets that differ only by case are refused (`Cat`/`cat` would be one folder on Windows and two on Linux, so VisionForge does not guess); for detection a target may not contain a line break. Names that differ only by case, spacing, `_` or `-` are reported as a hint and never merged by themselves.
- **Final order.** A's surviving targets in A's order, then the targets B adds in B's order; ids are renumbered from 0 and the registry records the whole mapping. A label whose ids all stay and whose boxes all survive is linked byte for byte; any other label (an id moved, or a box of a discarded class removed) is written fresh, never through a hardlink, as are `classes.txt`/`names.txt` and `data.yaml` (`names`, `nc`, an absolute `path` that pointed at A, and the folder of a split only B has when A's file did not name it).
- **Splits.** Each image keeps its own split. The result has a split if either source has it; the preview lists the splits that come from one source only. Detection puts B's images in A's layout for that split (`resolve_yolo_split`: `images/<s>` or `<s>/images`), and in the standard `images/<s>` + `labels/<s>` when A lacks it. Classification uses A's split folder names; the nested folders an image had inside its class are kept.
- **Duplicates.** SHA-256 of every image of both sources. If the bytes are in both, A's copy stays and B's is dropped and listed; it is flagged when the two copies are in different splits (leakage the merge avoids). If A's copy does not survive the mapping (its class was discarded) B's copy is kept, so nothing is lost to a copy of something the user threw away. Duplicates inside one source are kept and only counted: removing them is the audit's job, not this tool's.
- **Detection labels.** Boxes of discarded classes are removed and ids remapped. An image that had boxes and lost all of them is dropped and counted; an image that was already empty (background) is kept, with its label. Only valid rows (five numeric fields, integer id >= 0) count, as in the explorer.
- **What is carried.** Images, YOLO labels, the class list and the registry. Other files of the sources are not copied: in a renamed class folder they would leave a stray folder that looks like a class.
- **Names.** A file whose name is taken in its target folder becomes `stem-<sha8>` (with a counter if needed); for YOLO the `.txt` stem must be free too and the label follows the image.
- **Record.** `visionforge_dataset.json` (`schema_version` 1, `kind: "merged"`): both sources (path and manifest fingerprint), the mapping as resolved, the final classes, counts (found per source, images, per split, per class and split with boxes, dropped by reason, duplicates dropped and across splits, repeated inside one source) and files linked or copied. The apply is refused (409) if either fingerprint changed since the preview, and again inside the write; an image whose size or mtime changed is refused too.
- **API.** `POST /api/explorer/merge/preview` (A must be a dataset the explorer opened, 403 otherwise; B is the user's own pick in the local GUI and is only read), `GET .../{job_id}`, `POST .../{job_id}/plan` (instant: counts per final class and split, dropped images, no hashing), `.../apply`, `.../cancel`. One job at a time. The suggested name is `<A>-<B>`, then `-2`, `-3`.
- **Page.** A button "Juntar com outro dataset" beside "Incorporar imagens" in the Explore tab: pick the other folder, preview, the mapping table (target box with a datalist of existing names, a discard checkbox, a "merged" mark), live counts through `/plan`, what is dropped and why, destination name, apply with progress and cancel, "open in the explorer". Logic in `lib/dataset-merge.ts`; the progress bar and the polling hook are shared with the incorporate panel.

**Consequences:**

- The OpenAPI snapshot gains five operations; the request models are named `Merge*` so the schemas of the incorporate endpoints keep their names.
- The preview hashes both datasets completely, the cost of comparing by content (as in ADR-136 and ADR-166).
- Hardlinks mean the result and the sources share bytes; a tool that rewrites an image in place through the existing inode changes both. The registry says how many files were linked.
- Near-duplicates are not detected.

**Risks:**

- A crash between the first write and the rename leaves a hidden `.<name>.tmp-*` next to dataset A (same as ADR-166).
- Class identity is by name. Two datasets that use the same name for different things merge them unless the user renames one; the preview shows the counts so this can be seen, but nothing can check meaning.

**Alternatives considered:**

- *Merge into A in place.* Breaks the fingerprint of every old run.
- *Automatic merge of names that differ by case or spacing.* A wrong guess is silent and irreversible in the result; a hint costs one click.
- *Dropping B's copy and A's copy alike when they are in different splits.* Loses data to avoid a problem that keeping A's copy already avoids.
- *Re-splitting while merging.* That is phase 4 and would make results incomparable with both sources.


## ADR-168 — Re-splitting a dataset writes a new one, with the test split locked by default

**Date:** 2026-11-06
**Status:** Accepted — phase 4a of issue #117 (classification and detection) implemented
**Extends:** ADR-167 (merge), ADR-166 (incorporate), ADR-165 (explorer), ADR-150 (group manifests), ADR-129 (fingerprint), ADR-136 (content audit)

**Context:** the split a dataset came with is often wrong for the experiment (a validation set of 3 images, a class missing from the test set, copies of one photo on both sides). Fixing it by moving files inside the folder would change a fingerprint that old `run.json` files remember, and re-splitting everything would make a new run incomparable with the old ones, which were evaluated on the old test set. So the result is a new dataset, and the test split is kept as it is unless the user says otherwise.

**Decision:**

- **Never in place, shared mechanics.** The result is a sibling of the opened dataset, built under `.<name>.tmp-<id>` and published with one `os.rename` (`core/dataset_derived.py`: `link_or_copy`, `claim_name`, `publish`, `ensure_unchanged`, registry). Images and labels are hardlinked (copied when linking fails); cancellation, an error or a destination that appeared meanwhile leaves nothing. Class files (`classes.txt`, `names.txt`) are linked unchanged: no class changes here. The YOLO `data.yaml` is re-written (never linked) so its absolute `path` points at the new folder and its split keys follow the splits that exist; `names` are not touched.
- **What is re-split.** By default the train and val images are pooled and redistributed and test stays exactly as it is (same paths, same bytes). The option "keep the test as it is" is on by default and only counts when the dataset has a test split; with it off, every split is pooled. A positive ratio for the test while it is locked is refused (400).
- **Ratios.** Per split, relative weights, normalised. Without a request: the current proportions of the pooled splits. A ratio of 0 means the split does not exist in the result (its folder is not created and, for YOLO, its key leaves the yaml). A split the dataset lacked is created only when the user gives it a positive ratio (`images/<split>` + `labels/<split>` for YOLO); the preview says it is new. The ratios must sum to more than zero.
- **Stratified and seeded.** The stratum is the class (classification: the folder; detection: the most frequent class of the image, ties by name, and images without boxes form their own background stratum, as in ADR-166; the dominant-class rule and the largest-remainder allocation moved to `dataset_derived` and both features use them). The quota of every stratum per split comes from the largest-remainder method; leftovers and equal candidates are ordered with `random.Random(f"{seed}|{stratum}…")` over items sorted by SHA-256, so the result depends on the seed and the content, never on the order of the files. The seed is exposed and recorded.
- **Units that cannot be split.** Images with the same SHA-256 are always in the same split. With a group file, the images of a group are too. Both relations are joined (a connected set of "same bytes" and "same group" is one unit), so a re-split never creates leakage. Units are placed greedily, largest first, equal sizes in seeded order, each into the split with the largest remaining deficit for its dominant stratum and, when that class is full everywhere, for the images overall. Without groups and without copies every unit is one image and the greedy fill gives exactly the quotas.
- **Leakage with the locked test.** A pooled image that is an exact copy of a locked image, or that is in a unit that touches the locked split (same group, or a copy of a copy), is dropped from the result and listed with its reason (`duplicate_of_test`, `group_in_test`): moving it would put test content into training. The preview reports how many copy sets were kept together and how many of them were spread over several splits before.
- **Groups.** A `path,group` CSV (ADR-150 format, UTF-8, paths relative to the dataset root) picked by the user, who may be anywhere on disk because it is only read. Unlike the audit it need not cover every image: an image the file does not list is its own group (counted), and rows naming no image of the dataset are ignored (counted). A repeated path or an empty group is an error. The file's SHA-256 is part of the analysis; the group choice (`by_group`) is a plan option, so switching it needs no new analysis, but a different file does (a new preview).
- **Preview.** Counts per class and split before and after, images that change split, achieved against requested share per split, dropped images with the reason, copies kept together, and warnings: a new split, an empty split, a split that disappears, a class absent from a split. Changing the ratios, the seed, the lock or the group switch re-plans through `/plan` in milliseconds without hashing again.
- **Names.** A moved file whose name is taken in its new folder becomes `stem-<sha8>` (with a counter if needed); the YOLO `.txt` follows the image and its stem must be free too. Classification moves go to `<split folder>/<class>/`, so a nested subfolder an image had inside its class is not kept for moved images.
- **Record.** `visionforge_dataset.json` (`schema_version` 1, `kind: "resplit"`): source path and fingerprint, seed, requested and achieved ratios, the lock, the group file (path, SHA-256, groups, images without a group, ignored rows), counts before and after (per split and per class and split), images moved, dropped by reason with the first 200 listed, copy sets, and files linked or copied. The apply is refused (409) if the dataset fingerprint or the group file changed since the preview, and again inside the write.
- **API.** `POST /api/explorer/resplit/preview` (the dataset must have been opened in the explorer, 403 otherwise; optional `group_csv`), `GET .../{job_id}`, `POST .../{job_id}/plan` (`ratios`, `seed`, `lock_test`, `by_group`), `.../apply` (+ `dest_name`), `.../cancel`. One job at a time (409 busy). The suggested name is `<name>-resplit`, then `-2`, `-3`.
- **Page.** A button "Refazer divisão" beside "Incorporar imagens" and "Juntar com outro dataset": optional group file, analysis with progress, lock and group switches, current proportion or custom percentages, seed, before/after table, requested against achieved share, warnings, what was dropped and why, destination name, apply with progress and cancel, "open in the explorer". Logic in `lib/dataset-resplit.ts`; progress bar and polling hook are shared.
- **Route documentation.** The three derived-dataset routers (incorporate, merge, re-split) now spell their `responses=` out with literal status keys (`{400: {"description": BAD_REQUEST_TEXT}}`) instead of spreading the dicts of `http_responses.py`: Sonar S8415 reads the keys of a literal and does not follow a spread of an imported dict. The OpenAPI snapshot only records paths, so it gains the re-split operations and nothing else.

**Consequences:**

- The OpenAPI snapshot gains five operations; request models are named `Resplit*`.
- The preview hashes every image once (the cost of comparing by content, as in ADR-136, 166 and 167); everything after it is in memory.
- A re-split with the test unlocked is a different experiment: its numbers are not comparable with runs that used the old test set. The page says so next to the option.
- Hardlinks mean the result and the source share bytes (as in ADR-166 and 167).
- Near-duplicates are not detected; only identical bytes are kept together.
- With large groups the ratios are approximate: a group is never cut, and the preview shows how far the achieved share is from the requested one.

**Risks:**

- A crash between the first write and the rename leaves a hidden `.<name>.tmp-*` next to the dataset (as in ADR-166 and 167).
- A class that ends up absent from a split (few images, big groups) breaks per-split class indexing in ImageFolder training; the preview warns, nothing prevents it.

**Alternatives considered:**

- *Re-split in place.* Breaks the fingerprint of every old run and the comparison with them.
- *Re-split everything by default.* Makes a new run incomparable with the old ones; it stays one click away.
- *Moving the copies of a locked-test image to the test too.* It would grow the test set the user asked to keep untouched.
- *Reading the group file with the audit's reader.* It demands one row per image and a file inside the dataset; a re-split is more useful when a partial, external file is enough.


## ADR-169 — Detection datasets convert between YOLO and COCO by writing a new dataset

**Date:** 2026-11-06
**Status:** Accepted — phase 4b of issue #117 (detection only) implemented
**Extends:** ADR-168 (re-split), ADR-167 (merge), ADR-166 (incorporate), ADR-165 (explorer), ADR-129 (fingerprint)

**Context:** detection data arrives in two formats. VisionForge trains and explores YOLO; most public datasets and annotation tools speak COCO. Converting by hand is where boxes go wrong (centre against corner, normalized against pixels, ids that start at 1, crowd boxes, boxes that leave the image). As with the other derived-dataset tools, the conversion must not touch its source: a changed folder would change a fingerprint that old `run.json` files remember.

**Decision:**

- **Never in place, shared mechanics.** Both directions write a sibling folder under `.<name>.tmp-<id>` and publish it with one rename (`core/dataset_derived.py`: `link_or_copy`, `claim_name`, `publish`, `ensure_unchanged`, registry). Images are hardlinked (copied when linking fails); cancellation, an error or a destination that appeared meanwhile leaves nothing. The apply is refused (409) if the fingerprint of the source folder changed since the preview, and again inside the write.
- **Two named conversions.** `yolo_to_coco_bbox` (centre and size normalized to `[0, 1]` → top-left corner and size in absolute pixels, rounded to 6 decimals) and `coco_bbox_to_yolo` (its inverse, after clipping the box to the image). The convention is in the docstring; both are pure and tested on hand-computed values.
- **Export YOLO → COCO** (`<name>-coco`). `images/<split>/` plus `annotations/instances_<split>.json` per existing split. `images` carry the real width and height (read with PIL), `categories` use `id = yolo id + 1` and `supercategory ""`, `annotations` use `iscrowd 0`, `segmentation []`, `area = w*h`; image and annotation ids are unique across the splits. `info` records the VisionForge version, the date and the source fingerprint. An image that does not open, or whose label has a malformed line (not five fields, a non-numeric or non-finite value, a negative or fractional class id, a width or height ≤ 0) is **excluded and reported**: an image is never half-converted. A background image (no label, or an empty one) is kept with no annotations.
- **Import COCO → YOLO** (`<name>-yolo`). The folder the user picks is only read. Annotation files are `*.json` at the top and in `annotations/`; a JSON without the `images`, `annotations` and `categories` lists is listed as ignored. The split of each file comes from its name (`train`, `val`/`valid`/`validation`, `test`, with digits such as `val2017`); a name that says none is *unassigned* and the page asks for a split or to skip the file. Two files on one split are refused in the plan (400). The images folder of a file is the first of `images/<token>`, `<token>` (`val2017` as written, then the canonical split), the same with the file stem, folders named like a split (only for files with no detected split), `images`, the folder itself, that holds any of the first 20 `file_name`s; if none does the images are reported missing. A `file_name` that is absolute or contains `..` is refused.
- **What the import does to boxes and ids.** Boxes only; segmentation is ignored and the preview says so. `iscrowd` boxes are skipped; boxes are clipped to the image, and one with width or height ≤ 0 afterwards is dropped; annotations of an unknown image or category and malformed `bbox`es are skipped; the real image size replaces the JSON one when they differ. Everything is counted by reason and the first 200 are listed. YOLO values are written with 6 decimals; images without annotations get an empty label file. Category ids become `0..N-1` in ascending COCO id; two categories with the same name are both kept and the later one becomes `name_<coco id>` (reported); one id with two different names in two files is refused (400). Blank or multi-line names are normalized so `classes.txt` keeps every id in place. The result has `classes.txt` and a `data.yaml` (`path`, the splits, `names`, `nc`). A file name already taken in a split (subfolders are flattened) becomes `stem-<sha8>`.
- **Preview does the conversion.** The import preview opens each image (header only) and converts every box, so what the page shows is exactly what the apply writes; the plan (split per file, destination name) is instant and only decides which files feed which split and numbers the classes. The apply links and writes labels.
- **Record.** `visionforge_dataset.json` (`schema_version` 1, `kind: "coco_export"` or `"coco_import"`): source path and fingerprint, classes, the id mapping (import), counts per split, everything left out or changed by reason with the first 200 listed, and files linked or copied.
- **API.** `POST /api/explorer/coco/export/preview` (the dataset must have been opened in the explorer, 403 otherwise), `GET .../{job_id}`, `POST .../{job_id}/plan` (checks `dest_name`), `.../apply`, `.../cancel`; the same under `/import/` with `source_dir` instead (no allow-list: the folder is the user's own pick in the local GUI) and `splits` in plan and apply. Export and import share one job registry: one conversion at a time (409 busy). Suggested names are `<name>-coco` and `<name>-yolo`, then `-2`, `-3`. Every raised status is documented with literal keys in `responses=`.
- **Page.** In the Explorar tab, "Exportar para COCO" (with a detection dataset open: counts per split, what is left out, destination, progress and cancel, the result path; no "open in the explorer", which does not read COCO) and "Importar COCO" (always visible: folder picker, a card per JSON with its split select, images folder, counts and reasons, the COCO id → YOLO id table, destination, apply, then "Abrir no explorador"). Logic in `lib/dataset-coco.ts`; progress bar and polling hook are shared.

**Consequences:**

- The OpenAPI snapshot gains the COCO operations; request models are named `Coco*`.
- A round trip YOLO → COCO → YOLO keeps class names and boxes within 1e-5 (normalized); it is a test.
- The import preview keeps the converted boxes of the whole folder in memory and opens every image once; for a very large COCO this is the cost of showing exact numbers.
- The source fingerprint (manifest method, ADR-129) is "unavailable" above 200 000 files, in which case the "folder changed" check cannot detect anything.

**Risks:**

- A file that says no split is searched for its images only by the folders named like a split, its own stem, `images` and the root; an unusual layout shows every image as missing in the preview.
- A crash between the first write and the rename leaves a hidden `.<name>.tmp-*` next to the source (as in ADR-166 to 168).

**Alternatives considered:**

- *`pycocotools`.* A new dependency for a format that is plain JSON, and the tool's own readers do more than this needs.
- *Converting at the apply.* The preview could not then show what is left out and why.
- *Importing crowd boxes and polygons.* YOLO detection has no crowd flag, and a polygon's bounding box is a different annotation from the one the author drew.

## ADR-170 — Classification runs can export to local MLflow by profile

**Date:** 2026-11
**Status:** Accepted
**Extends:** ADR-013, ADR-054, ADR-114, ADR-135

**Decision:** A profile can enable MLflow for locally submitted classification
training. Submission captures that choice and the profile slug. The optional
MLflow client writes epoch metrics to a local SQLite store under
`outputs/mlflow`, groups runs by profile and exports the final `run.json`
after evaluation. History exposes export failures and lets the operator retry.

**Reason:** Researchers can compare their runs in the MLflow UI while the
training record and TensorBoard remain available independently. Retrying a
metric with a fresh timestamp can duplicate its history in SQLite. Re-export
therefore reconciles the stored epoch values before writing missing metrics.

## ADR-171 — Explicit private Hugging Face Hub export from History

**Date:** 2026-11
**Status:** Accepted
**Extends:** ADR-114, ADR-135, ADR-170

**Context:** A researcher sharing a VisionForge server may select any profile.
The profile picker scopes files (ADR-114); it does not authenticate anyone.

**Decision:** Store the Hub write credential separately for each workspace and
profile under the server user's VisionForge configuration directory. Never use
the dataset-download credential, ambient SDK login or a browser-stored token.
Return only whether a credential exists. Warn when the bind address is external
or unknown: anyone who can select that profile can use or replace its token.
History previews a limited package; explicit submission identifies its digest
and a private model repository. Training never publishes a run.

**Consequences:** The server account can read the secret. Use a narrowly scoped
Hub token. A change to a previewed file requires a fresh preview. Visibility
can still change remotely after the final repository check.

**Alternatives considered:** A process-wide Hub login would let profiles use
one another's credentials. Sending the entire run directory would disclose
configuration, dataset paths and unrelated outputs.

## ADR-172 — Private Weights & Biases export of classification runs

**Date:** 2026-11-08
**Status:** Accepted
**Extends:** ADR-170 (MLflow export), ADR-171 (profile tokens)

**Context:** phase 3 of #118. MLflow (ADR-170) keeps a run on this machine; a
research group that already follows its runs on Weights & Biases wants them
there too, without the export deciding anything about the training and without
anything private leaving the computer. `wandb.init` creates a missing project
with the account's default visibility, which may be public, and the SDK sends
the git remote, the code, the console, the package list, the host name and the
command line unless told not to.

**Decision:**

- A profile setting `wandb: {enabled, project, entity}`, off by default; an
  empty `entity` is the key owner's user. The API key is saved like the Hub
  token (ADR-171), in `<config>/wandb-tokens/<hash of profile>`, and the page
  only learns whether one exists. `gui/api/profile_tokens.py` now holds both.
- Classification training only, as for MLflow. The trainer and the block take a
  list of trackers (`core/run_trackers.py`) instead of the MLflow-only fields, so
  both exporters ride on the same calls: start when the run folder exists, one
  call per epoch, finish on every exit path. Every call is a best-effort
  boundary; a failure is written to `wandb.json` beside the run.
- **Private before the first upload.** Through W&B's GraphQL API, with only
  the profile's key, the project is created with `access: "PRIVATE"`, or an
  existing one is read. Any other visibility is refused and left as it is; the
  run is not sent. The call goes over HTTPS with Basic auth and never through a
  saved `wandb login`.
- **What leaves:** the public parameters (the same projection as MLflow, now
  shared in `core/tracking_privacy.py`), the epoch metrics at `step=epoch`, and
  the final public metrics (the Hub's list) as the run summary. Not `run.json`,
  which carries the dataset path. The settings turn off git, code, console,
  system stats, host metadata and `requirements.txt`, and the mode is forced to
  `online`, so an ambient `WANDB_MODE` cannot silently drop the export.
- The W&B run id is the first 32 hex digits of the hash of profile and run
  folder, with `resume="allow"`: the History's retry reopens the same run, and
  W&B drops the steps it already has, so only missing epochs are added.
- Optional extra `visionforge-studio[wandb]`, pinned like `mlflow`.

**Consequences:** checked against the real `wandb` 0.31.0 in offline mode with
a real one-epoch run: the record holds the chart keys, `test_*` metrics and
`model.name`; no file of the run holds the user's home folder, the dataset path
or `base_dir`; `files/` is empty; and `WANDB_MODE=disabled` did not stop it. The
online path (the GraphQL calls, the private project and the upload) is covered
only by fakes: it needs a real key. The W&B log `logs/debug.log` stays local and
names the user's folder. A project made private here can still be made public
on W&B later. A self-hosted W&B server is not supported: the address is the
SDK's default.

**Alternatives considered:**

- *Let `wandb.init` create the project.* Its visibility would be the account's
  default, which this code cannot see beforehand.
- *Set an existing project to private.* It would change a project the user
  configured on purpose; refusing says what is wrong without touching it.
- *Read the visibility through the SDK.* Its public `Project` has no access
  field in 0.31; the private GraphQL helpers it uses can change in any release.
- *Upload `run.json` as a file, as MLflow does.* MLflow keeps it on this
  machine; on W&B it would publish the dataset path and the config's folders.

## ADR-173 — The "Code" tab: the real training code, annotated

**Date:** 2026-11-08
**Status:** Accepted
**Extends:** ADR-118 (the training sheet's tabs and "it must never teach
wrong"), ADR-160 (the Results tab)

**Context:** the user wants people who do not program yet to see the code
behind a training: the data pipeline, the model, the loop, validation and the
test. "How it works" (ADR-118) draws the idea; it does not show code. A
hand-written, simplified copy would drift from what trains and could teach
something the tool does not do.

**Decision:**

- A fourth tab on the training sheet, **Code**, after "How it works". It shows
  six functions of a classification training as they are in the installed
  package: `_build_transforms` (data), `replace_final_layer` (model),
  `Trainer._build_criterion` (loss), `Trainer._train_epoch`,
  `Trainer._eval_epoch` and `Evaluator.evaluate`.
- The server reads them with `inspect` on each request
  (`GET /api/code-tour/{task}`, `core/code_tour.py`) and sends the source, the
  first line number and the file path inside the package, never the absolute
  path of the install.
- **Notes point at lines by text, never by number.** Each note names the text
  of its first line (and of the first line after it). A test resolves every
  note against the real source, so changing the trainer in a way that loses a
  commented line fails CI instead of pointing at the wrong place. The texts are
  in the message catalog, in Portuguese and English.
- **This run's path.** A note can depend on a setting: binary or multiclass,
  mixed precision, augmentation, the model family. The page sends what the
  submitted form decides and the server marks each such note as this run's
  path, a branch it does not take, or undecided. A setting the form does not
  decide is left out rather than guessed: mixed precision is marked on only
  where the trainer would turn it on (`mixed_precision` and a real CUDA
  device), so from the form it is only ever "off" (turned off, or CPU) or
  undecided.
- No line-by-line live tracking: the stream reports per epoch, so the tab says
  which path the code takes, not which line runs now.
- Classification only; other tasks say so. Detection runs inside Ultralytics,
  whose loop is not this project's code.

**Consequences:** renaming or rewriting a commented line means updating its
note in the same change. The functions shown are private ones
(`_build_transforms`, `_train_epoch`); the tab is a reason to keep them
readable. A run reattached after a page reload has no form settings, so its
branches show as undecided.

**Alternatives considered:**

- *A simplified, standalone PyTorch script.* Easier to read, but a second copy
  of the training that can drift, and that never ran.
- *Line numbers in the notes.* Any edit above a note would move it silently.
- *Notes in the frontend.* The anchors could not be checked against the Python
  source in one test.

## ADR-174 — A run's bundle goes to Zenodo as an unpublished draft

**Date:** 2026-11-08
**Status:** Accepted
**Extends:** ADR-135 (experiment bundle), ADR-171 (profile tokens)

**Context:** phase 4 of #118. A paper cites an experiment by DOI, and Zenodo
mints one when a record is published. Publishing is permanent: a published
record and its files cannot be deleted, only versioned. The experiment bundle
(ADR-135) already holds what a reader needs to check a run (run.json, the
checkpoint, conventional records, a SHA-256 manifest), but its run.json names
this machine's folders by absolute path, and a path under the home folder
carries the account name.

**Decision:**

- History gets a "Create a Zenodo draft" panel beside the Hub one. The server
  builds the bundle, the page shows its file name, size, members and what was
  left out, and the upload must match the previewed manifest digest, as with the
  Hub.
- **Draft only.** The export creates an empty deposition
  (`POST /api/deposit/depositions`) and puts the bundle in its file bucket. It
  never sends metadata and never calls `actions/publish`. Title, authors,
  license and publishing happen on Zenodo. The page recommends a token with
  `deposit:write` only: without `deposit:actions` the token cannot publish.
- **A public run.json.** `export_experiment_bundle(..., public=True)` writes the
  record with `run_dir` as `.`, the artifacts named as bundle members, and every
  other local path reduced to its final name (`core/public_run_record.py`):
  whole-value paths, and drive, UNC and home-like POSIX paths inside text. The
  manifest notes the rewrite; the run on disk is untouched, and the bundle still
  verifies and imports (the model comes back).
- **Two sites, two tokens.** zenodo.org and sandbox.zenodo.org have separate
  accounts. The profile keeps one token for each (`zenodo`, `zenodo-sandbox`,
  ADR-171's storage), and the page picks the site, so a test token never reaches
  the real Zenodo.
- The token travels only as a Bearer header over HTTPS, and only to links on the
  chosen site's host: a bucket link on another host is refused before the
  upload. Zenodo's answer body is never shown; a 401 or 403 becomes "Zenodo
  refused the token" and anything else a generic failure.

**Consequences:** an export creates a new draft every time; a researcher who
sends twice has two drafts to keep or discard on Zenodo. The bundle carries what
ADR-135 allows (graphics are not members), and a checkpoint goes as it was saved.
A record that names a local path inside some other kind of text (not a drive,
UNC or home-like path) keeps it; the preview is where it shows.

**Alternatives considered:**

- *Publishing from the GUI.* One click short of a permanent DOI, with nothing
  to undo a wrong file or author list.
- *Filling the draft metadata from the profile.* The profile name is a label,
  not an author list (ADR-114), and a license is a decision the tool cannot
  infer.
- *The raw bundle.* It would put the account name into a permanent archive.

## ADR-175 — A run's record, without weights, committed to a private GitHub repository

**Date:** 2026-11-08
**Status:** Accepted
**Extends:** ADR-171 (profile tokens, private destinations), ADR-174 (public run.json)

**Context:** the last destination of #118. Code and configuration already live
on GitHub; a group that keeps one repository of experiments wants each run there
too, as text it can diff and review, next to its code. Git is a poor place for
weights (Git LFS has a small free quota), and the Hub (ADR-171) and Zenodo
(ADR-174) already take them.

**Decision:**

- History gets "Send the report to GitHub". A run becomes one folder,
  `runs/<experiment>/<run>/`, holding a README report generated from run.json
  (summary, final metrics, a table per epoch capped at 200 rows, the charts
  embedded), the public run.json of ADR-174 (`artifacts.model` is `null`) and
  the run's charts under `figures/` (PNG and JPEG, 5 MB each, at most 20).
  No weights, ever.
- **Preview, then one commit.** The page shows the folder, each file with its
  size and the README; the commit must match the previewed digest. All files go
  in one commit on the default branch through the Git Data API (blobs, a tree on
  top of the branch's, a commit, a fast-forward of the ref, never forced). An
  empty repository first gets the README through the contents API, which is the
  only way to make a first commit.
- **An existing, private repository only.** The export reads the repository
  first and refuses one that is not private before writing anything. It never
  creates a repository: that needs a token allowed to administer the account.
  The page asks for a fine-grained token limited to the experiments repository
  with Contents read and write.
- The token is saved per profile like the others (ADR-171, service `github`)
  and goes only to `https://api.github.com`, as a Bearer header. GitHub's answer
  is never shown: 401, 403 and 404 each have their own message (404 also means
  the token cannot see a private repository), the rest is a generic failure.

**Consequences:** sending a run again replaces its folder's files in a new
commit; files removed from a run stay in the repository until someone deletes
them. A history longer than 200 epochs keeps every value in run.json but not in
the README table. Only github.com: GitHub Enterprise hosts are not supported.

**Alternatives considered:**

- *One commit per file through the contents API.* Works on any repository, but
  a run of six charts would leave eight commits.
- *Creating the repository when missing.* Saves a step, but needs a token with
  account administration, much more than writing one repository.
- *Weights through Git LFS.* Small free quota, and the weights already have two
  better homes.


## ADR-176 — A run versioned with DVC in a local repository of experiments

**Date:** 2026-11-08
**Status:** Accepted
**Extends:** ADR-174 (public run.json), ADR-175 (one folder per run, README report)

**Context:** the DVC item of #118, deferred until the other destinations were
done. DVC keeps data and weights next to git: git holds small pointer files,
DVC keeps the content in its cache and in a remote (S3, Google Drive, a shared
folder). A group that already versions its code with git wants each run's
weights versioned the same way, and wants to compare runs with
`dvc metrics show`, `dvc params diff` and `dvc plots show`.

**Decision:**

- History gets "Version with DVC". The researcher names a folder on this
  computer that is a git repository with DVC initialized at its root. "Prepare
  folder" makes one: it creates the folder when its parent exists, runs
  `git init` when needed and `dvc init`, and commits only the three files
  `dvc init` writes.
- A run becomes `runs/<experiment>/<run>/`, the same folder as ADR-175. Git
  gets the README report, the public run.json (`artifacts.model` names the
  weights), the charts, and the files DVC compares: `params.yaml` (the run's
  config, local paths reduced), `metrics.json` (final numeric metrics),
  `history.csv` (every epoch, every numeric column) and a `dvc.yaml` that
  declares them at the top level, with no stages. DVC gets the weights,
  `model.<ext>`, through `dvc add`; git gets `model.<ext>.dvc`.
- **Preview, then one commit.** The page lists every file and whether git or
  DVC keeps it, plus the repository's state and default remote. The export must
  match the previewed digest, which covers the repository path and the weights'
  SHA-256. The commit is `git commit -- <folder>`: whatever the researcher had
  staged elsewhere stays staged. A run exported again with no change makes no
  commit.
- `dvc push` of that run's pointer runs when the repository has a default
  remote. A failed push keeps the commit and the page says so; exporting again
  retries the push. `git push` never runs: publishing the repository stays with
  the researcher.
- The real tools run, not a reimplementation of DVC's formats: DVC from the
  `[dvc]` extra (`python -m dvc`, pinned to 3.67.1) or from the PATH, git from
  the PATH. They run without a terminal or prompt, with `DVC_NO_ANALYTICS=1`.
  Their output goes to the server log only; the page names the tool and the
  step that failed.
- **This machine only.** Every route takes a server folder from the request and
  runs git and DVC there, including the repository's own hooks, so it answers
  only a loopback client and refuses a page of another site, like "open
  folder". The folder is remembered in the browser, not in the profile.
- The dataset does not go: run.json already carries its fingerprint and origin.

**Consequences:** the weights are copied into the repository's folder and into
DVC's cache, so a run takes its checkpoint's size twice on disk until DVC's
cache link type is changed. A remote is configured with DVC itself (`dvc remote
add`); the page only uses the default one. Files removed from a run stay in the
repository until someone deletes them, as in ADR-175.

**Alternatives considered:**

- *Writing `.dvc` files and the cache without DVC.* No dependency, but it
  copies a format DVC owns and can change, and a push still needs DVC.
- *A `dvc.yaml` stage per run.* `dvc repro` would need the dataset and the
  training command on the machine, and a run already happened: the record is
  the result, not a pipeline to rerun.
- *Configuring the remote from the page.* Each storage has its own credentials
  and login; DVC already does that well.

---

## ADR-177 — Four places where a result did not match its config

**Date:** 2026-11-09
**Status:** Accepted
**Fixes:** ADR-083 (`transforms.augment`), ADR-101 (the head is the final Linear), ADR-061 (paired tests and their power floor)

**Context:** the 2026-11-09 audit (`docs/dev/AUDIT-2026-11-09.md`, R1–R5) reran
the statistics and the data pipelines against real tensors. The formulas were
right; four places around them made a run differ from what its `run.json`
records, which is the one thing the project promises not to do.

**Decision:**

- **Segmentation honours `augment`.** `SegmentationDataset` read the flip,
  rotation and jitter fields and never the flag, so the baseline arm of an
  augmentation ablation trained augmented. The flag now gates all three, the
  same rule as `core/data._build_transforms`.
- **A fold trains with the single-run pipeline.** `_FoldDataModule` rebuilt the
  transforms by hand, without `augment` and without `data.preprocessing`. It now
  calls `_build_transforms` with the fold's own normalisation. The fold
  statistics are measured on `_statistics_view`: preprocessing, resize and
  centre crop, as the model is fed. Reading the pool with `ToTensor` alone
  measured pixels the model never sees and could not batch images of different
  sizes.
- **Fine-tuning's full-rate group is the final Linear.** ADR-101 moved feature
  extraction to `final_linear`, but fine-tuning still took the last named child:
  119M parameters on VGG and 54M on AlexNet trained at the head's rate. The
  same loop over named children also missed parameters the model holds
  directly, so ViT's class token was never optimised (its position embedding
  belongs to the `encoder` child and was).
  The groups are now the final Linear and every other trainable parameter.
- **The Wilcoxon floor counts the pairs the test uses.** scipy drops zero
  differences (`zero_method="wilcox"`), so six seeds with three ties is a
  three-pair test whose smallest p is 0.25; the floor said 0.031 and the result
  was not flagged as underpowered.

**Consequences:** runs made before this change keep their numbers; a
segmentation run with `augment: false`, a K-fold with augmentation off or with
preprocessing, or a fine-tuning on VGG, AlexNet, ViT or ConvNeXt trains
differently now. Each fix has a test that failed before it
(`TestAugmentFlag`, `TestFoldTransformsFollowTheConfig`,
`TestFoldStatisticsOnMixedSizes`, `test_the_full_rate_group_is_exactly_the_final_linear`,
`TestFloorWithTiedPairs`).

**Amended 2026-11-25: the power note says the same thing as the floor.** `_power_note` quoted `n_pairs` (ties included) and always asked for "6+ for alpha=0.05": eight seeds with four ties read "with 8 paired seeds ... increase to 6+", and seven untied seeds at alpha=0.01 were told 6 suffice. `PairedComparison` now records `n_tested` (the pairs the test used); the note, through `tr("replicated.power_note")` in pt/en, quotes it, the tied count and the minimum `ceil(1 - log2(alpha))` untied pairs for the configured alpha (6 at 0.05, 8 at 0.01).

---

## ADR-178 — The GUI answers only to its own names (Host and Origin guard)

**Date:** 2026-11-09
**Status:** Accepted
**Extends:** ADR-114 (lab server via `--host`), ADR-120 (open folder / update refuse a foreign Origin)

**Context:** the server answered any `Host`. Its only middleware was CORS for the Vite dev server, which hides an answer from another site's page but does not stop the request. Two attacks follow from a page the user merely visits. *DNS rebinding:* the attacker's name resolves first to their machine, then to 127.0.0.1, so the browser treats the local server as the attacker's site and reads every answer (reproduced: `Host: evil.example` got 200 on `/api/health`). *Cross-site requests:* a page posts to localhost. Only four routes checked `Origin` (reveal, update/apply, DVC); the other ~62 state-changing routes did not.

**Decision:**

- An ASGI middleware, `gui/security.py` (`HostOriginGuard`), runs before CORS and every route.
- **Host:** a request whose `Host` (port removed, brackets around IPv6 removed, case and trailing dot ignored) is not an accepted name is refused with 400 and a pt/en message. A missing or malformed `Host` is refused too.
- **Origin:** a request whose method is not GET/HEAD/OPTIONS/TRACE, and every WebSocket handshake, is refused (403, or close code 1008) when it carries an `Origin` that is neither loopback nor an accepted name. `null` and unparsable origins are refused. A missing `Origin` stays allowed: that is a same-origin navigation or a client that is not a browser.
- **Accepted names.** Bound to loopback (the default): `localhost` and the loopback addresses (`127.0.0.1`, `[::1]`, reusing `is_loopback_host`). Bound to any other address (lab server, ADR-114): also the machine's hostname, its FQDN, the addresses they resolve to, and the bind address unless it is a wildcard. Behind a DNS alias or a reverse proxy's public name, the person starting the server adds names with `--allowed-host NAME` (repeatable), which works in both modes.
- With `--host 0.0.0.0` the browser tab opens `http://127.0.0.1:<port>`. The wildcard is neither reachable nor an accepted Host.
- The route-level checks in reveal, update/apply and DVC stay. They are stricter (loopback only) and still apply on a lab server, where the page's own origin is the machine's name.
- The policy lives on `app.state.host_policy`, set by `start_server`. Without it (`uvicorn ...:app`, tests), the default is loopback only.

**Consequences:** reaching a lab server by a name the machine does not resolve for itself (a DNS alias, a proxy) needs `--allowed-host`. Until then the page gets a 400 that names the flag. The updater's relaunch (ADR-120) passes only `--host/--port`, taken from the socket that served the request, so `--allowed-host` values are not carried across an update.

**Alternatives considered:**

- *Starlette's TrustedHostMiddleware.* It checks Host only, not Origin, and has no notion of "this machine's names when bound to 0.0.0.0".
- *Accepting any Host in lab mode.* That leaves every lab user exposed to rebinding through their own browser.
- *Requiring `--allowed-host` for every non-loopback bind.* Safer, but it breaks every lab already reaching the server by hostname or IP.
- *A per-route Origin dependency.* Each new route would have to remember it. The middleware covers routes that do not exist yet.
## ADR-179 — Security headers on every response, API docs only on request

**Date:** 2026-11-09
**Status:** Accepted
**Builds on:** ADR-125 (inline theme script), the Host/Origin guard of SEC-1

**Context:** the 2026-11-09 audit (`docs/dev/AUDIT-2026-11-09.md`, finding 6, R8) found that no response set a Content-Security-Policy, `X-Frame-Options`, `X-Content-Type-Options` or `Referrer-Policy`. A script that reached the page could load and send anything, and another site could frame the GUI. `index.html` also preconnected to `fonts.googleapis.com` and `fonts.gstatic.com`, which it never used because the fonts come from `@fontsource`, so every visit told Google the GUI was opened. `/docs` and `/redoc` were public, and their pages load JavaScript from `cdn.jsdelivr.net`.

**Decision:**

- **One middleware, outermost.** `gui/security.py:SecurityHeaders` adds the headers to every HTTP response, including the 400/403 refusals of `HostOriginGuard`.
- **SPA policy:** `default-src 'self'; script-src 'self' 'sha256-…'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'`.
  - The hash is that of the inline theme script. It is computed from the `index.html` on disk, with line endings normalised as the HTML parser does. It is computed again when the file's size or mtime changes, so a rebuild under a running server keeps the theme script allowed.
  - `style-src` has no `'unsafe-inline'`. Evidence: React sets the `style` prop through the CSSOM (`el.style[x] = …`, `style.setProperty`), which CSP does not govern. The bundle's only `createElement("style")` is React's `<style precedence>` hoisting, which the app does not use. The e2e test sees no violation.
  - `data:` images are needed because dataset crops arrive as data URIs, and the CSS noise texture is one. `blob:` covers downloads.
- **Fixed headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: same-origin`. `same-origin` was chosen over `no-referrer`. Both send nothing to other sites. Under `no-referrer`, though, the Fetch standard has a POST carry `Origin: null`, which the guard refuses. Chromium was checked and sends the real origin on same-origin fetches anyway. Other engines were not checked, and `same-origin` does not depend on that behaviour.
- **No preconnect** in `frontend/index.html`. A test fails if either `index.html` names `googleapis`, `gstatic` or any absolute `src`/`href`.
- **API docs off by default.** `FastAPI(docs_url=None, redoc_url=None)`. `/docs`, `/docs/oauth2-redirect` and `/redoc` answer 404, not the SPA. `VISIONFORGE_DOCS=1` (or `true`/`yes`/`on`) brings them back with a policy of their own, sent only on those paths: `'unsafe-inline'` and `https://cdn.jsdelivr.net` for scripts and styles, Google Fonts for ReDoc, `blob:` workers, and `frame-ancestors 'none'`. `/openapi.json` stays: it is the API contract that tests and clients read.

**Consequences:** a third-party script or style cannot be added to the SPA without changing this policy, and the e2e test `frontend/e2e/csp.e2e.ts` fails on any violation or on any request to another host. A change to the inline theme script needs no manual hash update. Anyone who used `/docs` now sets `VISIONFORGE_DOCS=1`. Tests: `tests/gui/test_security_headers.py`, `frontend/e2e/csp.e2e.ts`.
## ADR-180 — Third-party telemetry is switched off in one module

**Date:** 2026-11-09
**Status:** Accepted
**Extends:** ADR-120, ADR-170

**Context:** the README promises that nothing leaves this computer unless the user asks for it. The audit of 2026-11-09 (finding 9) found that only the DVC export disabled analytics (`DVC_NO_ANALYTICS` in `dvc_export.py`). ultralytics ships with `settings['sync']=True` and reports train/val/export events. huggingface_hub reports download telemetry. MLflow has collected usage data since 3.2.

**Decision:** `visionforge/utils/third_party_privacy.py` exposes an idempotent `apply_opt_outs()`. It sets `HF_HUB_DISABLE_TELEMETRY`, `DO_NOT_TRACK`, `MLFLOW_DISABLE_TELEMETRY` and `DVC_NO_ANALYTICS` with `os.environ.setdefault`, so a value the user already set is kept. It also sets `enabled = False` on ultralytics' process-wide `events` instance (`ultralytics.utils.events.events`), inside try/except so a missing or broken SDK never stops the app. `settings.update({'sync': False})` was rejected: it writes the user's global ultralytics settings file and would silence every other project on the machine (it did, during development, and was restored). It runs in `main()` before any SDK is imported, on import of `detection_trainer`, and before the MLflow import in `MlflowTracker._start`. `YOLO_OFFLINE` is not used because it would also block the weights download the user asks for. `NETWORK_SDKS` documents how each network-capable dependency is covered, and a guard test fails when `pyproject.toml` gains a known network SDK with no entry there.

**Consequences:** a new network SDK has to be added to `NETWORK_SDKS` (and opted out if it reports). The environment variables are process-wide, so they also reach subprocesses. The guard only knows the SDKs on its own list of known network names, so a new SDK outside that list is not caught.

**Amended 2026-11-10: `visionforge gui` sets only the environment switches up front (#258).** `apply_opt_outs()` imports `ultralytics.utils.events` (~1.7 s with torch) just to switch the events off, and the GUI does not need Ultralytics until a run does. `main()` now calls `apply_env_opt_outs()` before parsing and `apply_opt_outs()` after it for every command except `gui`. In the GUI the events are switched off where Ultralytics is first imported: on import of `detection_trainer` (every `YOLO` goes through it) and in `routes._has_detection_image` before `ultralytics.data.utils`. Both run before any train/val/export event can be sent. `tests/utils/test_third_party_privacy.py` parses `src/` and fails when an `ultralytics` import has no `apply_opt_outs()` call before it in the same function or module.
## ADR-181 — The "Code" tab for every task (amends ADR-173)

**Date:** 2026-11-09
**Status:** Accepted
**Amends:** ADR-173 (the "Code" tab: the real training code, annotated)

**Context:** ADR-173 showed the annotated training code for classification only: one global set of settings, a route with fixed query parameters (`kind`, `amp`, `augment`, `model`) and a page that decided support with `taskKey === "classification"`. The user asked for the real code of what runs in the other tasks too. Regression, segmentation, anomaly detection and torchvision detection all train with this project's code; only detection on the Ultralytics backend runs inside the library.

**Decision:**

- **One tour per task, same contract.** `core/code_tour.py` keeps `Section` and `Note` (anchors by text, never by line number; source read with `inspect` on each request) and adds `TOURS[task]` with `TOUR_SETTINGS[task]`: the settings each task's notes depend on and the values they take. Tours now exist for classification, regression (loss mse/mae/huber), segmentation (loss cross_entropy/dice/combined; architecture unet/torchvision), anomaly (method autoencoder/patchcore) and detection (backend; detector faster_rcnn/ssd/retinanet).
- **Generic, validated settings.** `GET /api/code-tour/{task}` reads its query through `tour_settings(task, query)`: a key the task does not know, or a value it does not take, is dropped and its notes stay undecided. `true`/`false` read as `on`/`off`; `model` is a model name turned into the setting it decides (classification family, segmentation architecture, detector). A section may be shown whole while its notes are all off this run's path (the PatchCore coreset in an autoencoder run): the page fades them instead of hiding code.
- **The list comes from the server.** `GET /api/code-tour` lists the tasks with a tour and each one's settings; the page decides support from it, so a new tour needs no frontend change beyond its section titles. `codeSettingsFor(taskKey, payload)` reads each task's settings from the submitted form.
- **Unavailable tour.** A task without a tour, or a run whose training is not this project's code, is a 404 whose `detail` says why, in the request's language; the page shows that text rather than "failed". Detection has a tour only when the run says `backend=torchvision`; a run without a backend gets the reason too, because Ultralytics is the default backend.
- Shared code is shown once per tour where it runs: regression and segmentation show `training_loop.train_epoch`, with notes that say which hooks (`validate_targets`, `outputs_transform`) each task passes.

**Consequences:** each task's commented lines are now under the same test (every note resolves against the installed source, every key has pt and en text, every `when` value is declared in `TOUR_SETTINGS`), so renaming one in the regression, segmentation, anomaly or detection trainers means updating its note in the same change. The route's OpenAPI contract changed: the explicit query parameters were replaced by the generic query. Notes teach what each task does differently (best checkpoint by val mIoU in segmentation, by train reconstruction loss for the autoencoder, by val mAP@0.5 in torchvision detection), so a change to those rules must update the note.

**Alternatives considered:**

- *Explicit query parameters per setting on the route.* Typed in OpenAPI, but the route would grow with every task and could not reject a value that belongs to another task.
- *Sections chosen by setting (show only the active builder).* Less noise, but undecided runs would need a fallback and a run's page would hide code it could take; faded notes keep one rule for every tour.
- *A tour for Ultralytics detection.* Its loop is the library's code, not this project's; tracked separately (GUI-8).

## ADR-182 — A non-finite validation metric stops the run

**Date:** 2026-11-09
**Status:** Accepted
**Amends:** ADR-099 (training health warnings); extends the `run.json` contract of ADR-013 with `status: "failed"` and an `error` field

**Context:** `save_epoch_checkpoint` compared `metric < best`. NaN never compares as an improvement, so a run whose validation loss became NaN saved no `best_model.pth`, counted every epoch against patience, trained to the end, and then the block failed with a `FileNotFoundError` that named neither the epoch nor the metric. Segmentation was worse: it ranks by mIoU, which stays a finite number for NaN logits (argmax picks class 0), so epoch 1 was saved as the "best" with NaN weights.

**Scope:** classification, regression, segmentation, torchvision detection, the anomaly autoencoder and `GenericTaskEngine` (custom tasks, Level 1 loop). Each of the last three checks through `training_loop.require_finite` and writes its own failed `run.json` through `record_non_finite_failure`: detection checks `val_loss` and `map50`, the autoencoder `train_loss` (its only loss), the engine `train_loss` and the task's primary metric. They check before the epoch is added to the history, so the failed epoch is not counted in `total_epochs`. Ultralytics detection is out: the library owns that loop. PatchCore has no epoch loop.

**Decision:** `save_epoch_checkpoint` raises `NonFiniteMetricError(name, epoch, value)` when the ranking metric is not finite, and `run_epochs` raises it for a non-finite `val_loss` when the ranking metric is another one. `run_epochs` first lets a pending best-checkpoint write finish. The fit of each task (`Trainer.fit` and `fit_training`, which serves regression and segmentation) writes the run's `run.json` with the epochs done and a `non_finite_metric` health warning (pt/en), then `record_non_finite_failure` sets `status: "failed"` and `error`, and replaces NaN/inf by null so the file stays valid JSON. The exception message always contains "non-finite", the metric and the epoch, and says whether a checkpoint exists: when an earlier epoch improved, `best_model.pth` of that epoch (`best_epoch`) is kept and the message and warning say so (`health.non_finite_metric_kept`); otherwise no checkpoint was saved. `ensure_checkpoint` is never reached, so it writes no weights. Both fits close the TensorBoard writer in a `finally`, even if writing `run.json` fails.

**Consequences:** a diverged run stops in the epoch it diverged instead of spending the remaining epochs. The last resume state is the previous epoch's. The block needs no change: the exception reaches it with its own message.

**Alternatives considered:**

- *Keep training and warn at the end.* Wastes the remaining epochs and still leaves no checkpoint.
- *Treat NaN as the worst possible score.* Keeps the run alive but would pick whatever finite epoch came before, hiding that the later epochs diverged.

## ADR-183 — Figures in PDF/SVG and PNG at 300 dpi

**Date:** 2026-11-10
**Status:** Accepted
**Amends:** ADR-116 (figure language); extends `OutputConfig`

**Context:** `core/plotter.py` saved every figure as a 150 dpi PNG, which a journal rejects for line art, and replicate sets drew no figure at all although they already wrote a `.tex` table.

**Decision:** `OutputConfig` gains `figure_formats` (subset of `png`, `pdf`, `svg`, deduplicated, never empty, default `["png"]`) and `figure_dpi` (72 to 1200, default 300). Every `MetricsPlotter` method takes keyword-only `formats` and `dpi` and saves through `save_figure`, which writes one file per format next to the given path (suffix replaced); dpi only affects PNG. The classification, regression, segmentation and anomaly blocks pass the config values; the `graphics` list of `run.json` names one file per figure (the PNG, else the first format asked). `MetricsPlotter.replicates_strip` draws one point per finished seed with the mean, and `write_replicates_strip` writes it beside the replicates `.tex` and from `run_replicates(figure_dir=...)`. No `schema_version` bump: both fields have defaults and a config without them loads as before.

**Consequences:** the default PNG is now 300 dpi (larger files). Detection, the generic engine and the GUI test endpoint use the plotter defaults (PNG, 300 dpi) and do not read the new fields yet; the GUI has no selector or download for PDF/SVG. The heatmap of adjusted p-values for the paired matrix is left out.

**Alternatives considered:**

- *Bump `schema_version`.* No shape changes for old files, so a migration step would be empty.
- *Keep 150 dpi by default.* Preserves file size but misses the paper-ready goal of the issue.
## ADR-184 — `visionforge run` runs the GUI's multi-run strategies

**Date:** 2026-11-10
**Status:** Accepted
**Relates to:** ADR-056 (replicates), ADR-061 (replicated comparison), ADR-045/052 (sweeps), ADR-050 (task K-fold), ADR-113/123/124 (run groups)

**Context:** `visionforge run` trained one config. Replicates, the paired comparison, the standalone sweeps and the regression/segmentation K-fold existed only behind the GUI's HTTP routes, so the number a paper cites (mean ± CI, Holm-corrected paired tests) and its `.tex` table could not be produced from a script. The steps after the orchestrator (stopped/failed decision, report shape, summary files, History group) lived in private functions of `gui/api/routes.py`.

**Decision:** a YAML whose top level is `strategy:` + `config:` is a strategy envelope. `strategy.kind` is `replicates`, `replicated_comparison`, `sweep` or `cv`; its other keys are validated by the same request model the matching route uses (`ReplicatesRequest`, `ReplicatedComparisonRequest`, `SweepRequest`, `TaskCvRequest`), and `config` is the task config unchanged. `cli_strategies.py` picks the task's `TaskRunner` (or K-fold function) and calls the orchestrator in `core`. The post-orchestrator step moved from the routes to `core/run_groups.py` (`finish_replicates`, `finish_replicated_comparison`, `finish_sweep`, `finish_kfold`, `resolve_seeds`, `write_group_summary`); the routes and the CLI both call it, and the routes keep their behaviour. A YAML without `strategy` takes the old path. Sweeps cover the tasks the GUI sweeps through a runner (detection, regression, segmentation, anomaly) and K-fold covers regression and segmentation; any other task fails before training with a message that names the alternative (the classification blocks). An invalid envelope or option exits with status 2, a job with no result with 1.

**Consequences:** the CLI has no statistics of its own, so it cannot drift from the GUI. The CLI has no stop button and no profile: the token is `None` and the config's own `output` paths are used, with `outputs/models` for the group when the config names none. Custom tasks are not covered yet.

**Update (#264):** a variant of the paired comparison or a sweep space that overrides `output.*` is refused by the CLI with status 2 and the GUI's message, through `core.run_groups.refuse_output_paths`, the check the routes now call too (ADR-114).

**Alternatives considered:**

- *A `strategy` key inside the task config.* Every task config loader would have to ignore it, and the options would mix with the task's fields.
- *New subcommands (`visionforge replicates ...`).* Options on the command line instead of in the versioned YAML, and four parsers to keep in sync with the request models.

## ADR-185 — The run queue and the started jobs' state have one owner: `gui/api/execution.py`

**Date:** 2026-11-11
**Status:** Accepted
**Relates to:** ADR-132 (splitting `routes.py`), ADR-075 (run queue), ADR-163 phase 4 (lanes and per-job state), ADR-159 (event log)

**Context:** ADR-132 asked for the queue and the running job's state to get "one explicit owner that routers command through methods". They still lived in `gui/api/routes.py`, next to 150 endpoints: `_JOB_STATES`, `_latest_run_id`, the lanes (`_RUN_QUEUE`, `_LANES`), the queue callbacks, the stop points, the SSE reader and the classification executor. Any handler in the file could assign them, and tests replaced them on `routes` (#204).

**Decision:** `gui/api/execution.py` owns that state and has no FastAPI import. Only the queue adds or drops a job's state, through `_begin_job` and `_finish_job` (the lanes' `on_start`/`on_finish`); an executor writes only its own job's `run` and log, reached through `_job_state(run_id)`; everything else reads through `_run_state`, `_sse_generator` and `_LANES`, and commands the lanes through their methods. `_execute_experiment` moves with it and is split into `_block_types`, `_wire_block`, `_run_dir_of` and `_failed_run` (Sonar S3776), with the block imports still before the `try`. The routes that only read that state or command the lanes (`GET /api/experiment/{status,events,result/{id}}`, `GET /api/queue`, `DELETE /api/queue/{id}`) move to `gui/api/execution_routes.py`, included by `routes.py` where they used to be defined, so the OpenAPI document and the route order are unchanged. `routes.py` keeps submission (`_submit_job`, which reads `_MODELS_DIR`) and the other executors, which import `_job_state` from `execution.py`.

**Consequences:** `routes.py` no longer defines or assigns the queue or a job's state; `routes.py` drops from 3612 to 3202 lines. A test that replaces `_JOB_STATES`, `_latest_run_id`, `_MAX_FINISHED_STATES` or `_execute_experiment` patches `execution`, and the `tests/gui/conftest.py` helpers take that module. `run_experiment` and `resume_run` look `execution._execute_experiment` up when the job starts, so a patch there reaches them. The other executors (detection, regression, segmentation, anomaly, groups, Colab, custom tasks) stay in `routes.py` for a later step.

**Alternatives considered:**

- *Keep re-exports in `routes.py`.* `routes._JOB_STATES.clear()` would still work, but `monkeypatch.setattr(routes, "_JOB_STATES", {})` would silently patch a name nobody reads.
- *A class holding the state.* A cleaner interface, but a behaviour-bearing rewrite inside what should be a move; the module boundary already gives the single owner.

## ADR-195 — Ultralytics (AGPL-3.0) is the optional `detection` extra

**Date:** 2026-11-11
**Status:** Amended by ADR-243 (`ultralytics` is a required dependency again; `detection` is an empty extra)
**Supersedes in part:** ADR-106 (for `ultralytics` only; the other six stay in `dependencies`)

**Context:** ADR-106 made `ultralytics` a base dependency of an MIT package. It
and `ultralytics-thop` are AGPL-3.0, so every `pip install visionforge-studio`
installed AGPL code, and an organisation that does not accept the AGPL could
not adopt VisionForge at all, even for classification, without buying
Ultralytics' Enterprise License (#223). The code already tolerated its absence:
`detection_trainer` binds `YOLO = None` when the import fails, and the
`torchvision` detection backend (Faster R-CNN, RetinaNet, SSD, SSDlite) uses no
AGPL code. The maintainer chose option B+C of #223.

**Decision:**

- `ultralytics>=8.3` leaves `dependencies` and becomes the only content of the
  `detection` extra: `pip install "visionforge-studio[detection]"` (from a
  checkout, `pip install -e ".[detection]"`).
- Every path that needs it fails with one message (`detection.ultralytics_missing`,
  pt/en) naming that install line and the `torchvision` alternative: training,
  testing and ONNX export of an Ultralytics run, and the submit of a detection
  config with `backend: ultralytics`, which is refused with 422 before it enters
  the queue. The dataset check at submit no longer needs `ultralytics` for its
  list of image formats.
- `GET /api/system/info` reports `ultralytics_installed` and
  `detection_install_command`. The detection form shows the install line under
  the backend choice when Ultralytics is selected and absent, and the
  `torchvision` option carries a "sem AGPL" / "no AGPL" badge.
- The install smoke installs both variants (plain and `[detection]`) and checks
  that only the second brings `ultralytics`; the wheel smoke checks the plain
  wheel does not. The CI jobs that run the test suite install `[detection]`, as
  they had `ultralytics` before.

**Consequences:** the default backend in the detection form stays `ultralytics`
(its configs and run records are unchanged), so a plain install sees the install
line until the user picks `torchvision` or installs the extra. A plain install
still brings torch, through `timm`, so `doctor --fix` keeps swapping an existing
torch as ADR-107 describes.

**Alternatives considered:**

- *A: keep it a base dependency and only document the AGPL.* Leaves the
  adoption problem in place.
- *B without the badge.* The GUI would not tell which detection backend is free
  of the AGPL; the badge costs one label.

## ADR-194 — The "Code" tab for Ultralytics detection: our layer, then the library's loop (amends ADR-173)

**Date:** 2026-11-11
**Status:** Accepted
**Amends:** ADR-173 (and ADR-181, which left detection on Ultralytics without a tour)

**Context:** YOLO / RT-DETR is the default detection backend, and its "Code" tab
said only that the loop is the library's. VisionForge still writes real code
around it: the form becomes `YOLO.train` arguments, augmentation is switched
off with neutral values, a callback records each epoch and honours the stop
button. The library's own loop (`BaseTrainer._do_train`) is readable with
`inspect`, but it is AGPL-3.0 code (ADR-195) and Ultralytics has no upper
version bound here, so a release can move any line a note points at (#224,
option b).

**Decision:**

- `GET /api/code-tour/detection` with `backend=ultralytics`, or no backend
  (Ultralytics is the default), returns the tour `ULTRALYTICS_DETECTION`:
  four sections of VisionForge's layer (`_ultralytics_train_kwargs`,
  `_augmentation_kwargs`, `_fit_ultralytics`, `_record_ultralytics_epoch`)
  whose notes are tested like every other tour, then `BaseTrainer._do_train`
  of the installed Ultralytics.
- A `Section` can name a `library`. Such a section carries
  `library: {name, version, license}` (version from `ultralytics.__version__`,
  license as its metadata declares it), and its notes are tolerant: a note
  whose line this version lacks is dropped instead of failing. Every section
  reports `notes_resolved` and `notes_total`. Without the `detection` extra the
  library section is left out and the layer is still shown.
- The page shows, above a library section, whose code it is, the version, the
  license and how many notes found their line. The "only torchvision" reason
  (`code.detection.only_torchvision`) is gone.

**Consequences:** a test pins that every library note resolves on the installed
version, so a bump that moves a line fails CI and names the note to fix, while
users on another version still get the section with fewer notes. The source
shown is read from the user's own install, so nothing of Ultralytics is
redistributed.

**Alternatives considered:**

- *(a) Only VisionForge's layer.* Hides the loop where the epoch actually
  happens, which is what the tab exists to show.
- *(c) (b) plus an upper version bound on `ultralytics`.* Ties every user to
  one release for the sake of the notes; the tolerant notes already degrade
  gracefully.
## ADR-191 — Unexpected 500s and validation 422s carry the page's language; what stays English on purpose

**Date:** 2026-11-11
**Status:** Accepted
**Amends:** ADR-116 ("Still as written, on purpose or not yet")

**Context:** after the four slices of ADR-116, two kinds of error still reached the page in a language nobody chose: the `500` of an action that failed unexpectedly (testing a run, Grad-CAM, batch prediction, ONNX export, a dataset download) was the bare `RuntimeError: x`, and every `422` carried Pydantic's English sentence. ADR-116 also left a list whose items were a mix of "on purpose" and "not yet".

**Decision:** `gui/api/error_messages.py` owns both frames. `unexpected_error(exc, detail=None)` words a `500` as `errors.unexpected` (a sentence in the page's language ending in `Type: message`, or in the caller's already-redacted detail); the two remaining `HTTPException(500, f"...")` sites use it. An app-level handler for `RequestValidationError` reads `X-VF-Lang` from the request itself (a body that fails validation may never reach the router dependency that binds the language) and, for Portuguese, replaces the `msg` of 14 common Pydantic error types with a catalog sentence (`validation.<type>`) filled from the error's `ctx`; `type`, `loc` and `input` are unchanged, and a type not listed, or a `ctx` missing a value the sentence needs, keeps Pydantic's text. English keeps Pydantic's text as is. The response shape is FastAPI's, so the OpenAPI document does not change.

**Closes the ADR-116 list.** Stay in English on purpose, with no further slice planned: the trainers' `logger.*` lines, the epoch and trial lines of the sheet, `STOPPED_NOTE`, the `description` of the schema fields, and the text of an exception raised by a library (it is the technical part of the `500` frame and of a failed run's `error`). Pydantic's messages outside the 14 types stay English too. The LaTeX captions and the Markdown export already follow `tr_artifact` (fourth slice).

**Consequences:** a test that compared a `500` detail to `Type: message` now compares it to the framed sentence. A script without the header reads the Portuguese frame and Portuguese 422 messages; one that parses `msg` should send `X-VF-Lang: en` or read `type`.

**Alternatives considered:**

- *Translate every Pydantic error type.* Around a hundred, most of which no form in the page can produce; an untranslated `msg` is still correct English and its `type` is stable.
- *Let the page word the 422 from `type` and `ctx`.* Every client of the API would need its own table; the catalog already exists on the server.

## ADR-192 — Apple's GPU (MPS) is an experimental device

**Date:** 2026-11-11
**Status:** Accepted (experimental, not announced)
**Relates to:** ADR-098 (determinism cost), ADR-099 (training-health warnings), ADR-107 (doctor never swaps a working torch)

**Context:** on a Mac with Apple Silicon, PyPI's torch drives the GPU through Metal (MPS), but `DeviceConfig.kind` only knew `cpu`, `cuda` and `multi_cuda`: every Mac run trained on the CPU, and `visionforge doctor` recommended the CPU build of torch. Nobody on the project has a Mac to validate a run end to end.

**Decision:** `DeviceConfig.kind` gains `"mps"`, probed by `utils.cuda.mps_available()` (never raises; only a real `True` counts, so a mocked torch never reads as a Mac). `resolve_device` honours it (`mps (Apple GPU)`, or `cpu (fallback: MPS unavailable)`), and `cuda`/`multi_cuda` on a machine without CUDA now fall back to MPS before the CPU, with the label `mps (fallback: CUDA unavailable)`. Detection uses the same order through one `DetectionTrainer._effective_kind()` for both backends; this also fixes `kind: cuda` on a CPU-only machine reaching Ultralytics as GPU `0`. `deterministic: true` on MPS adds the `mps_not_deterministic` warning to `run.json`, because the switch configures cuDNN and MPS kernels have no deterministic mode. `/api/device/info` reports `mps_available` (a flag, not an entry in `gpus`: no index or memory to show), the device selector offers "Apple GPU (MPS)" marked experimental, and the doctor counts MPS as a working GPU; its `--fix` no longer offers to replace a torch that already runs on the GPU (CUDA or MPS).

Not done, on purpose: AMP stays CUDA-only; `pin_memory` stays as configured (MPS ignores it with a warning); `PYTORCH_ENABLE_MPS_FALLBACK` is not set, so an operator without an MPS kernel raises instead of running silently on the CPU; no MPS cache release between sweep runs. Regression, segmentation and anomaly runs pick the device through `resolve_device` but do not carry the determinism warning yet.

**Consequences:** verified only with mocks (`torch.backends.mps.is_available`, `mps_available`, `torch.cuda.is_available`) on Windows; CI has no Mac. The feature is experimental and stays out of the README and the site until a run on a real Apple Silicon Mac is recorded here. A config that asked for `cuda` on a Mac now trains on the GPU instead of the CPU, and its numbers may differ between runs.

**Alternatives considered:**

- *Map `cuda` to `mps` silently.* The label would claim a device the run did not get; the `fallback` wording keeps `run.json` honest.
- *Wait for a tester before merging.* The code paths are testable without a Mac, and keeping the feature unannounced costs nothing to people without one.
## ADR-193 — The replicated comparison in the GUI: one axis, the form as baseline

**Date:** 2026-11-11
**Status:** Accepted
**Relates to:** ADR-061 (replicated comparison), ADR-056 (replicates), ADR-044 (single-seed model comparison), ADR-114 (output paths), ADR-156 (strategy step), ADR-184 (CLI strategies)

**Context:** every task had `POST /api/{task}/replicated-comparison` (and `/api/custom/{key}/replicated-comparison`), and `visionforge run` reached it from YAML, but no panel had a form for it and `frontend/src/api/client.ts` had no function for the route. The Holm-corrected paired matrix could only be started over HTTP or from the CLI. The panels' "compare" strategy is the single-seed model comparison of ADR-044.

**Decision:** the strategy step of the six task panels (classification, detection, regression, segmentation, anomaly and custom tasks) offers **Comparação replicada** (`PanelStrategy` `"replicatedComparison"`), drawn by the reusable `ReplicatedComparisonCard`. The variants lie on one axis: one dot-path and 1 to 3 values besides the form's own, so 2 to 4 variants. The form as it stands is the variant `baseline`, with no overrides; each other variant is named `<last path segment>=<value>`, with characters outside `[A-Za-z0-9._-]` turned into `-`, because the backend suffixes the label to the run name and it becomes a folder. A value that parses as a number is sent as one, otherwise as text (the sweep editor's rule). Seeds are a count (2 to 20) or an explicit list (2 to 20, no repeats); 20 is the backend's `n_replicates` cap, held for the explicit list too since the cost is variants × seeds. The card shows that cost and, when Wilcoxon's floor `2 / 2**n` is above alpha 0.05 (fewer than 6 seeds), says no difference could come out significant: the condition `significance.min_achievable_p` and the report's `underpowered` use. Alpha stays the backend default and the metric list is the one Réplicas offers (accuracy for classification). `buildReplicatedComparisonPayload` validates and shapes the request, mirroring the backend's 422s (fewer than 2 variants, repeated seeds, an `output.*` path); `runReplicatedComparison` and `runCustomReplicatedComparison` post it, and App queues it with `blockKind: "replicated-comparison"`, the strategy name the queue reports. Classification sends `block: "classification"`, as Réplicas does, and Colab stays off with the reason while the comparison is picked. The backend is unchanged.

**Consequences:** when the job ends, the results view reads the report as the History group it becomes (`comparisonGroupFromReport`, a port of `_copy_comparison_group`) and draws it with `RunGroupDetail`, the component of the group's detail in History: variants with mean ± CI95, the paired tests with the Holm verdict and the power flag, and the seeds. Its button opens the group in History (the group's id is the job's run id), and a seed's button opens that seed. A value equal to the form's own is not caught: it trains a copy of the baseline. A path the config lacks is caught only by the backend's 422. Variants that change several fields at once are left to the YAML of `visionforge run` and to HTTP.

**Alternatives considered:**

- *Free variants, several fields each.* More general, but a table of variants × fields is a second editor to validate, and one axis covers the paper's ablations one factor at a time.
- *Expose alpha.* The report records the alpha used, but a GUI knob invites tuning it after seeing the p-values; 0.05 stays the default of the route.
## ADR-190 — The user-task and user-model extension API is stable within a minor release, and a test pins its signatures

**Date:** 2026-11-11
**Status:** Accepted
**Relates to:** ADR-058 (custom tasks), ADR-048/049 (custom models), ADR-039 (config schema versions)

**Context:** `TaskSpec`, `register_task`, `register_model` and the Level 2 `ctx` are the extension points a researcher writes code against (ADR-058, ADR-048). The built-in tests exercise them only through the toy specs in `tests/tasks`, so renaming `compute_loss`'s `model` parameter, or dropping a hook, would keep every built-in test green and break every custom task in the wild. The guides in `docs/custom/` said nothing about how long these names hold.

**Decision:** Before 1.0 the listed names, parameters and defaults may change only in a minor release, with a "Changed" entry in the CHANGELOG, never in a patch. `docs/custom/TASKS.md` and `MODELS.md` get a Stability section that lists them. `tests/tasks/test_task_api_contract.py` locks them with `inspect.signature`, `TaskSpec.__abstractmethods__`, the dataclass fields of `TaskInfo` and `TaskRunContext`, and `BaseTaskConfig.model_fields`; annotations are not locked. Changing the API means editing that test and the guides in the same commit. No `TASK_API_VERSION` constant is added: with one maintainer and a CHANGELOG entry per change, a number would be a second place to forget; it can be added with 1.0 if third-party task packages appear.

**Consequences:** A rename fails CI with the old and new signature side by side. Adding an optional parameter with a default or a new optional hook is still a change to the test, so it is a visible decision. The engine's internals and any name not listed stay free to change.

**Alternatives considered:**

- *A `TASK_API_VERSION` integer checked at load time.* Useful once packages are installed by others; today it only adds a number to bump.
- *Only the prose promise.* Nothing would fail when the prose and the code drift.

## ADR-186 — The pytest hook runs on `git push`, not on every commit

**Date:** 2026-11-11
**Status:** Accepted
**Relates to:** ADR-060 (the `slow` marker and the fast suite kept "seconds-fast" for the pre-commit hook)

**Context:** The local `pytest` hook in `.pre-commit-config.yaml` has `always_run: true` and no `stages`, so every `git commit`, including a docs-only one, ran the whole fast suite: 329 s in the 2026-11-09 audit, 5 to 8 minutes in practice. ADR-060 assumed that suite stayed seconds-fast. The cost pushes contributors to `--no-verify`, which skips ruff, codespell and mypy as well. CI already skips the hook (`SKIP: pytest`) and runs the suite in its own job.

**Decision:** The `pytest` hook gets `stages: [pre-push]`. ruff, ruff-format, codespell and mypy stay on `pre-commit`. This supersedes in part ADR-060: the fast suite is still the one the hook runs, but it runs on push, not on commit. CONTRIBUTING documents `pre-commit install --hook-type pre-commit --hook-type pre-push`, to be run in every clone and worktree, because hooks are installed per checkout.

**Consequences:** A commit takes seconds and a push takes the suite's time. A clone that ran only `pre-commit install` never runs the tests locally; CI still does. The release bump (`bump-my-version`) commits through the commit stage only, so it no longer waits for the suite.

**Alternatives considered:**

- *Keep the hook on commit but filter by changed files.* A change to any `src/` file still runs everything, and the filter has to track which paths affect which tests.
- *Remove the hook and rely on CI.* The suite would then run first on GitHub, after a push, which is a slower loop than a local one.
## ADR-187 — The interval of a paired difference inverts the test beside it

**Date:** 2026-11-11
**Status:** Accepted
**Amends:** ADR-061 (item 1: "a bootstrap CI of the difference")

**Context:** `paired_comparison` reported a percentile bootstrap interval of the paired differences next to a paired t or Wilcoxon p-value. With few seeds the percentile bootstrap under-covers (it can only resample the values it has), so the two disagreed: in `research/seed_variance/results/summary.json` four cells have an interval excluding zero beside p between 0.06 and 0.09 (n = 10). The GUI report, `run.json` and the LaTeX comparison table showed both (#162).

**Decision:** the interval is the dual of the test that produced the p-value, at 95% (the comparison's alpha = 0.05). `paired_t`: the Student-t interval of the mean difference, `mean ± t(0.975, n-1)·sd/√n`. `wilcoxon`: the Hodges-Lehmann interval, `[W(k), W(M+1-k)]` over the `M = n(n+1)/2` sorted Walsh averages, with `k` the largest integer such that `P(T+ ≤ k-1) ≤ 0.025` under the exact null distribution of the signed-rank statistic (built by convolution, no ties). `PairedComparison` gains `ci_method` (`"t"` or `"hodges_lehmann"`), `ci_estimate` (the mean, or the Hodges-Lehmann pseudo-median the interval brackets) and `ci_confidence` (the coverage actually attained: the exact distribution is discrete, so it is at least 95%). Below 6 pairs no Hodges-Lehmann interval reaches 95% (`2/2^n > 0.05`), so `ci_low`/`ci_high` are `null`, the same reason the test is flagged underpowered. The LaTeX caption names the method of each interval; a report written before this ADR has no `ci_method` and is labelled percentile bootstrap. `bootstrap_ci` stays for the replicate aggregates (ADR-061 item 2), and the seed-variance study keeps its pre-registered percentile interval.

**Consequences:** the interval excludes zero exactly when p < 0.05 for the paired t, and for the Wilcoxon when there are no ties or zero differences (scipy's p is then exact). With ties or zero differences scipy's p uses the normal approximation while the interval keeps the exact critical value; the test suite accepts disagreement in at most 5% of tied samples (measured: 1.7% with differences rounded to one decimal). Numbers already shown for a comparison change when it is recomputed. `paired_comparison` and `comparison_matrix` no longer take a `seed`.

**Alternatives considered:**

- *BCa or bootstrap-t for every test.* Better coverage than the percentile bootstrap, but still not the dual of the test: the disagreement shrinks without disappearing, and it remains stochastic.
- *Keep the bootstrap with an under-coverage warning below 20 pairs.* Leaves the contradiction on screen for every realistic seed count.

**Amended 2026-11-25: the interval follows the comparison's alpha.** "At 95% (the comparison's alpha = 0.05)" was implemented as a fixed 95%, so with `alpha=0.01` a paired t of `[2, 3, 4]` vs zeros gave p = 0.035 (not significant) beside `[0.52, 5.48]`, which excludes zero. `paired_comparison` (and so `comparison_matrix` and the replicated comparison) now builds the interval at `1 - alpha` (`[-2.73, 8.73]` there) and `ci_confidence` records that coverage (at least `1 - alpha` for Hodges-Lehmann). `underpowered` already compared the floor with the same alpha; with six pairs at alpha = 0.01 both the test and the 99% Hodges-Lehmann interval are out of reach, so the ends are `null`.
## ADR-188 — A public bundle leaves out `weights/*.pt` and rewrites `config.*`

**Date:** 2026-11-11
**Status:** Accepted
**Amends:** ADR-174 (public run.json), ADR-175 (no local path leaves the computer)

**Context:** ADR-174 cleaned only run.json in a public bundle. Ultralytics pickles its `train_args` (`data`, `project`, absolute paths) into `weights/best.pt` and `weights/last.pt`, and `config.json`/`config.yaml` went in as written, so a Zenodo draft could still carry the account name (#164).

**Decision:** `export_experiment_bundle(..., public=True)` leaves out every `.pt` under a `weights/` folder and notes each one in the manifest ("checkpoint left out of a public bundle: it may embed local paths"); run.json then names the model as `null`. A member named `config.json`, `config.yaml` or `config.yml` is parsed and written back through `without_local_paths`, with a manifest note; one that does not parse or exceeds 16 MB is left out with its own note. The files on disk are untouched, the private bundle is unchanged, and the manifest keeps `format_version` 1, so earlier bundles still verify. A test exports a run whose checkpoints and configs name the home folder and greps every member of the zip for it.

**Consequences:** a public bundle of a detection run carries no checkpoint; whoever needs the weights shares the private bundle or the Hub export. State-dict checkpoints (`best_model.pth`, the other tasks) still go: they hold tensors only. History, predictions and other members are not rewritten.

**Alternatives considered:**

- *Rewriting `train_args` and keeping the checkpoint.* Needs unpickling a checkpoint to edit it, and any other path ultralytics adds later would leak again.

## ADR-189 — Offline mode refuses every non-loopback socket

**Date:** 2026-11-11
**Status:** Accepted
**Extends:** ADR-120 (update check), ADR-180 (third-party telemetry in one module)

**Context:** the README promises that nothing leaves the computer unless the researcher asks, but that held only by reading the code: no switch made it a guarantee and no test trained a model with the network cut off (#226).

**Decision:** `visionforge gui --offline`, `visionforge run --offline` or `VISIONFORGE_OFFLINE=1` (any value but empty or `0`) call `third_party_privacy.enable_offline_mode()` right after parsing, before `apply_opt_outs` imports ultralytics. It sets `HF_HUB_OFFLINE`, `YOLO_OFFLINE` and `VISIONFORGE_NO_UPDATE_CHECK` to `1`, overriding the user (a guarantee, not a default), sets `VISIONFORGE_OFFLINE=1` for child processes, and installs a socket guard: `socket.socket.connect`, `connect_ex` and `sendto` refuse an `AF_INET`/`AF_INET6` destination that is not a loopback address or `localhost`, and `getaddrinfo`, `gethostbyname` and `gethostbyname_ex` refuse a name other than an IP literal, `localhost` or this machine's own name, so no DNS query leaves either. A refusal raises `NetworkBlockedError`, a `ConnectionRefusedError`, so network clients handle it as any failed connection. `tests/integration/test_offline_guard.py` trains one epoch of classification on CPU (resnet18 without pretrained weights, a synthetic dataset) under the guard, checks that external connections fail and loopback works, and starts `visionforge gui --offline` and reads `/api/health`.

**Consequences:** with offline mode on, downloads of weights or datasets fail at once with a connection error instead of hanging. The guard covers this Python process; a child process inherits the environment switches and, if it is VisionForge's CLI, turns the guard on itself, but a native library that opens its own sockets, or asyncio's Windows proactor connecting to an IP literal, is not intercepted.

**Alternatives considered:**

- *Only the environment switches.* Each SDK honours its own, and a new dependency would reach the network unnoticed; the guard makes the promise checkable.
- *A firewall rule.* Stronger, but needs administrator rights and differs per OS.

## ADR-197 — MLflow and W&B export every task from its finished run.json

**Date:** 2026-11-11
**Status:** Accepted
**Amends:** ADR-170 (MLflow), ADR-172 (Weights & Biases)

**Context:** #286. Both exporters took classification training only. Regression, segmentation, anomaly, detection and custom tasks run their own trainers, which have no tracker hooks, but all of them write the ADR-013 `run.json` with `config`, `metrics` and per-epoch `history`.

**Decision:**

- The profile switches are unchanged. A single training of any other task (`/regression/run`, `/segmentation/run`, `/anomaly/run`, `/detection/run`, `/custom/{key}/run`) is sent once its executor ends (`gui/api/tracker_exports.exported`): `core/run_trackers.export_run_record` opens each tracker on the run folder, replays every `history` row as one epoch with the `run.json` names (`train_loss`/`val_loss` keep the chart keys `loss/train`/`loss/val` of ADR-170) and closes it as `FINISHED`, or `KILLED` when the stop cut it. Classification keeps its live per-epoch export.
- MLflow attaches the `run.json` as before. The W&B summary is the final `metrics` under their `run.json` names: the shared public list, plus `best_val_miou` and `box_loss`, which only the private trackers show (the Hub card keeps its list), and, for a custom task, every finite number under a plain `[A-Za-z0-9_]` name, since the task's own code names them and the history already carries them.
- The History retry no longer refuses a non-classification run: the exporter's sidecar already proves it was exported.

**Consequences:** the other tasks reach the trackers after the run, not live; a run that raises is not exported (no sidecar, so nothing to retry); sweeps, K-fold, comparisons and resumed runs of the other tasks are not exported, as for classification. The live events end before the upload, but the next queued job waits for it (a stuck W&B call, up to its timeout).

**Alternatives considered:**

- *Tracker hooks in each task trainer.* Live charts, but five trainers and the custom-task engine change, for the same final record.
- *Widening the shared public list.* The Hub card would publish `box_loss`, which it leaves out on purpose.
## ADR-196 — Starting a run leaves `routes.py`: submission, the single-run endpoints and the Colab objects

**Date:** 2026-11-11
**Status:** Accepted
**Relates to:** ADR-132 (splitting `routes.py`), ADR-185 (queue and job state in `execution.py`), ADR-163 (Colab), ADR-164 (Drive consent), ADR-058 (custom tasks)

**Context:** After ADR-185, `gui/api/routes.py` still held everything that starts one run: `_submit_job` and the training-stack preload, `POST /api/experiment/run`, the standalone tasks' schema and `run` endpoints with their submit-time dataset checks and executors, the Colab run with its executor, and the custom-task endpoints. It also created and owned the Colab objects (`_COLAB_CLI`, `_COLAB_LOGINS`, `_COLAB_JOBS`, `_DRIVE_CONSENTS`). The file was 3202 lines (#204).

**Decision:** `gui/api/submission.py` holds `_submit_job`, `_task_label_from_config`, the training stack (`_TRAINING_STACK`, `_DETECTION_STACK`, `_TASK_STACK`, `load_training_stack`) and `profile_dependency`, which turns the resolver `routes.py` owns into the `X-VF-Profile` dependency of a split-out router; it holds no state. A job submitted without a profile is labelled with the default profile's slug, which is all `default_profile(_MODELS_DIR)` gave it, so `_submit_job` no longer reads `_MODELS_DIR`. `gui/api/training_routes.py` holds `POST /api/experiment/run`, `/api/{detection,regression,segmentation,anomaly}/{schema,run}`, the `_check_*` dataset checks and the four executors, which now share `_execute_block`. `gui/api/colab_run_routes.py` holds `POST /api/colab/experiment/run`, `/api/colab/drive/continue`, `_execute_colab` and becomes the owner of the Colab objects; `routes.py` passes three of them to `build_colab_router`. `gui/api/custom_task_routes.py` holds `GET /api/tasks`, `/api/custom/{key}` (hide, unhide, delete, schema, run), `GenericTaskEngine` and `_execute_custom_task`. `routes.py` includes these routers where the routes used to be defined, so the OpenAPI document and the route order are unchanged; it keeps the profile resolution and `_tracker_factories` (both read `_PROFILES_DIR`/`_MODELS_DIR`) and passes them in.

**Consequences:** `routes.py` drops from 3202 to 2066 lines and keeps the strategies (sweep, compare, K-fold, replicates, replicated comparison) and the per-run actions. Every caller, `routes.py` included, looks `submission._submit_job` up at call time, and `resume_run` looks the executors up in `training_routes`, so a test patches the module that defines a name: `submission` for `_submit_job`, `cuda_context_broken` and the stack; `training_routes` for the executors; `colab_run_routes` for the Colab objects, `_execute_colab`, `_drive_source` and `_colab_lane_of`; `custom_task_routes` for `GenericTaskEngine`, `_execute_custom_task` and `_get_custom_task_or_404`. `routes.py` re-exports none of them. The check that no route imports training code before `load_training_stack` now reads the three new route modules too.

**Alternatives considered:**

- *Keep the Colab run in `routes.py`.* No state would change owner, but `run_experiment` and the task runs sit on both sides of it, so the split would need one more include and the Colab executor would stay behind; the Colab objects belong with the code that drives them.
- *Keep re-exports in `routes.py`.* As in ADR-185: `monkeypatch.setattr(routes, "_submit_job", ...)` would silently patch a name nobody reads.

## ADR-198 — Torch floors per hardware extra, a cu130 extra, and one lock fork per extra

**Date:** 2026-11-12
**Status:** Accepted
**Amends:** ADR-005 (hardware extras)

**Context:** #302. The lock held a single torch, 2.7.1+cu118, for every extra: the extras had no declared conflict, so uv resolved them together and kept the first index. Dependabot flags torch below 2.10 (remote code execution through an untrusted checkpoint, GHSA-63cw-57p8-fm3p), below 2.9.1 and up to 2.12.1 (memory corruption, low). The PyTorch indexes stop at different versions: cu118 at 2.7.1, cu121 at 2.5.1, cu124 at 2.6.0, cu128 at 2.11.0, cu126 and cu130 at 2.14.1, cpu at 2.15.0.

**Decision:**

- `[tool.uv] conflicts` declares the seven hardware extras mutually exclusive, and `[tool.uv.sources]` maps each extra to its index (`extra = "cu130"`), so the lock resolves one torch per extra. CUDA sources apply on Linux and Windows only; a Mac with a CUDA extra, or any install with no extra, takes torch from PyPI.
- Floors: `cpu`, `cu126` and the new `cu130` require torch>=2.13 (no open advisory); `cu128` requires torch>=2.10, the highest floor its index can meet, leaving only the low `torch.jit.script` advisory. `cu118`, `cu121` and `cu124` keep torch>=2.3: no fixed wheel exists on their indexes, so they stay as legacy extras for drivers too old for CUDA 12.6, and the README says what they miss.
- `constraint-dependencies = ["setuptools>=84.1"]`: the old torch forks otherwise pull setuptools 81, which has its own advisory.
- `visionforge doctor` recommends `cu130` for a driver that reports CUDA 13.0 or newer; 12.8 and 12.9 keep `cu128`.

**Consequences:** verified on the RTX 5060 Ti (driver 621.01, sm_120) with torch 2.14.1+cu130: every task trained one epoch on cuda (6/6) and `visionforge selftest` passed every case on CPU. The lock records old torch versions for the legacy extras, so Dependabot keeps those alerts open; they describe what those extras install. Two GPU runs of the same segmentation config differ in the fourth decimal of mIoU even with `deterministic=True`: the project sets `cudnn.deterministic`, not `torch.use_deterministic_algorithms`, and the bilinear upsampling backward is not deterministic on CUDA (tracked separately; REPRODUCIBILITY.md already makes no GPU bit-identity claim).

**Alternatives considered:**

- *Dropping cu118/cu121/cu124.* Clears every alert, but leaves machines whose driver predates CUDA 12.6 with CPU only.
- *A torch>=2.10 floor for every extra.* Unsatisfiable on the legacy indexes, and it leaves the 2.12.1 advisory open on the indexes that can clear it.

## ADR-199 — `deterministic=True` also turns on PyTorch's deterministic kernels

**Date:** 2026-11-12
**Status:** Accepted
**Amends:** ADR-098 (the determinism switch), ADR-121 (rejected strict mode)

**Context:** #305. On the RTX 5060 Ti with torch 2.14.1+cu130, two runs of one segmentation config with seed 0 and `deterministic=True` gave mIoU 0.4779 and 0.4780. `_seed_everything` set only `cudnn.deterministic`; the CUDA backward of bilinear upsampling (the segmentation decoder) accumulates with `atomicAdd`, and five backward passes of the same input gave different gradients. Under `torch.use_deterministic_algorithms(True)` they were identical. ADR-121 rejected strict mode because it raises on operations without a deterministic kernel; it had measured classification only.

**Decision:** with `deterministic=True`, `_seed_everything` also calls `torch.use_deterministic_algorithms(True, warn_only=True)` and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` unless the user set it; with `False` it turns strict mode off. `warn_only` keeps an op that has no deterministic kernel running, with a PyTorch warning, instead of stopping the run.

**Consequences:** the six tasks trained one epoch on cuda twice gave the same metrics both times (they did not for segmentation before). Two ops still warn: `nll_loss2d_forward` (segmentation's cross-entropy) and torchvision's `roi_align` backward (Faster R-CNN detection); their runs repeated here, but nothing guarantees it, and REPRODUCIBILITY.md says so. Strict mode is process-wide, so the GUI server sets it again at the start of every run. A test fails if the switch stops turning strict mode on or off, and, on a CUDA machine, if bilinear gradients stop repeating.

**Alternatives considered:**

- *Strict mode without `warn_only`.* Detection would stop at the first `roi_align` backward.
- *Replacing bilinear upsampling with a deterministic layer.* Changes the model, and every past segmentation result with it.

## ADR-200 — Private Comet export, on the W&B contract

**Date:** 2026-11-12
**Status:** Accepted
**Extends:** ADR-172 (Weights & Biases export), ADR-197 (every task from run.json)

**Context:** #287. Groups that follow their runs on Comet want the same opt-in, private export W&B has. `comet_ml` 3.58.12 collects far more by default than `wandb`: importing it hooks the import of torch and other frameworks, starts Sentry error reports (with `sys.executable` and the installed packages) for half of the processes, and replaces `sys.excepthook`; a new `Experiment` logs code, git metadata and patch, console, CLI arguments, packages, host, user, IP, CPU, GPU, disk and network; its `end()` posts a summary notification titled with `getpass.getuser()`; on comet.com its internal reporting posts SDK tracebacks. A missing project is created with the server's default visibility.

**Decision:**

- A profile setting `comet: {enabled, project, workspace}`, off by default; an empty `workspace` is the key's default one. Names are lower-case `[a-z0-9][a-z0-9._-]{0,99}`, since Comet keeps project names in lower case. The API key is saved like the W&B key (ADR-171/172), in `<config>/comet-tokens/<hash of profile>`; the page only learns whether one exists.
- Same contract as W&B: classification live per epoch through `core/run_trackers.py`, the other tasks from `run.json` when the job ends (ADR-197), every call a best-effort boundary, failures in `comet.json` beside the run, and the History's retry at `POST /api/comet/runs/{run_id}/retry`.
- **Private before the first upload.** With `comet_ml.API(api_key=<profile key>, cache=False)`, the project is read with `get_project`, or created with `create_project(..., public=False)` and read again; anything but `isPublic: false` is refused and left as it is.
- **Nothing ambient.** Before the import the process gets `COMET_DISABLE_AUTO_LOGGING=1`, `COMET_ERROR_TRACKING_ENABLE=false` and `COMET_INTERNAL_REPORTING=false`. `comet_ml.start(api_key=..., workspace=..., project_name=..., experiment_key=..., mode="get_or_create", online=True, experiment_config=ExperimentConfig(...))` gets every collector as an explicit `False` (`log_code`, `log_graph`, `parse_args`, `log_git_metadata`, `log_git_patch`, `log_env_details`, `log_env_gpu/host/cpu/network/disk`, `auto_output_logging`, `auto_param_logging`, `auto_metric_logging`, `auto_log_co2`, the histograms), `display_summary_level=0` and `disabled=False`: an explicit argument wins over `COMET_*` variables and `.comet.config` in the SDK's own precedence. The experiment's `send_notification` is replaced with a no-op before `end()`.
- **What leaves:** the public parameters, the epoch metrics at `step=epoch`, the final public metrics (the W&B list) and `visionforge_status`. The SDK adds `curr_step`, `curr_epoch` and `process duration`.
- The experiment key is the first 32 hex digits of the hash of profile and run folder. Comet keeps a repeated step, unlike W&B, so `comet.json` records `synced_epoch`, the last epoch Comet confirmed with `flush()` at the end; a retry sends only later epochs.
- The SDK keeps one live experiment per process and ends the previous one when another starts, so a second Comet export while one is open fails with `CometBusyError` and waits for the retry.
- Optional extra `visionforge-studio[comet]`, pinned to `comet-ml==3.58.12`; like the other exporter extras it stays out of the license gate.

**Consequences:** checked with the real `comet_ml` 3.58.12 offline, with the same `ExperimentConfig`: the archive held only the parameters, metrics, name and status, without user, host or path; Sentry was not initialised and no import hook was installed. The online path (account calls, private project, upload, retry) is covered only by an in-memory fake: it needs a real key. If the server dies between an epoch and the final flush, a retry sends those epochs again and Comet shows them twice. The three environment switches stay set for the whole process. The server address is the SDK's (`COMET_URL_OVERRIDE` moves it); a self-hosted Comet was not tried. `comet_ml` pulls `sentry-sdk`, `dulwich` and `everett` (MPL-2.0) into the environment of whoever installs the extra.

**Alternatives considered:**

- *Let `comet_ml.start` create the project.* Its visibility would be the workspace default, which this code cannot see beforehand.
- *Set an existing public project to private.* It would change a project the user configured on purpose, as in ADR-172.
- *Ask Comet which steps it already has before a retry.* More of the private REST surface for each retry; the sidecar is local and enough.
- *`Experiment`/`ExistingExperiment` directly.* `start(mode="get_or_create")` already picks between them by key and passes the same explicit settings to both.

## ADR-201 — A run's History entry is its results page

**Date:** 2026-11-12
**Status:** Accepted
**Amends:** ADR-193 (the results view sends to History), ADR-113 (run groups in History)

**Context:** a finished training had two places that showed its numbers: the results view the training sheet opened ("Ver resultados") and the run's detail in History. The first had the numbers and nothing to do with them; the second had every action (rerun, bundle, publish, ONNX, tests) under a wall of paths and library versions, with the metrics at the bottom. Charts were drawn by two separate hand-written SVGs with only the min and max labelled, and no way to read a value.

**Decision:**

- "Ver resultados" opens History on the finished job. `lib/history-target.ts` names the entry from the result the server sent: the folder of `report.train` / `report.detection` / a custom task's `run_dir`; the job id for a strategies job (its report carries `report_dir`); and for classification's K-fold, search and model comparison, whose group id carries a stamp the result does not have, the group of one of its units. History retries the lookup four times, 750 ms apart, since the entry is written as the job closes.
- A job History cannot name (no folder in the report, a group that was not written, a search run on Colab whose units stayed on the VM) keeps the results view. The view is not removed.
- The run detail reads as a results page (`lib/run-detail-layout.ts`): header, metrics with the task's headline number first, charts, actions, tests, then the reference blocks folded. Dataset and training start open; location, environment and pipeline wait for a click.
- Actions are grouped by intent (reuse, download, publish, use the model) and every one has an icon; the publish targets carry their brand marks (`components/ui/BrandIcons.tsx`, paths from simple-icons 16.37.0, CC0; Comet has no mark there and uses a generic one).
- One line chart (`components/ui/LineChart.tsx`, pure SVG, no dependency) for the live tab, the comparison and the run detail. It joins the measured points as they are, with no smoothing or interpolation, and breaks where a value is missing. Hover, touch or the arrow keys snap to the nearest epoch and show each series' exact value. The run detail charts train against validation for the quantities its task leads with (`lib/run-curves.ts`); the matplotlib figures stay below, as the files a paper uses.
- A value under 5e-4 is written with an exponent, so a learning rate is not shown as `0.000`.
- Text in History is at least 13 px, mono labels at least 11 px.

**Consequences:** one place to read a run and act on it. The results view remains for the jobs above and for a server whose History write failed.

**Alternatives considered:**

- *Keep the results view and add the actions to it.* Two copies of the same page to maintain, and History would still open on the old layout.
- *A charting library.* The charts need lines, a grid and a tooltip; a dependency of several hundred kB for that is not justified in a bundle that is committed.
- *Smooth the lines.* A smoothed curve shows values the run never measured.

## ADR-202 — Profiles: no "default" in the interface, the picker on every load, integrations inside the profile

**Date:** 2026-11-12
**Status:** Accepted
**Amends:** ADR-126 (the picker asks once per session), ADR-114 (profiles are output folders)

**Context:** the welcome showed the legacy-layout profile as "Padrão" / "<name> (padrão)", which read as a profile somebody else had created. The picker asked once per browser tab, so a shared machine reloaded into the last person's profile. Each profile row in the picker carried the MLflow, W&B and Comet checkboxes with their paragraphs, which buried the one question the screen asks. A fresh install creates nothing on disk and reports one unnamed profile; that part was already right.

**Decision:**

- The storage of ADR-114 does not change: the first person's profile is still the legacy layout (`outputs/` itself), which is what lets runs made with the CLI show up in History. `is_default` stays in the API as a storage detail.
- The interface never says "default". That profile is shown by its name like any other. One with no name beside named profiles (older data) is listed as "Sem nome", and choosing it asks for the name.
- Entry rules (`entryStep` in `lib/profile.ts`): no profile has a name → name and pronoun, nothing else on the screen; exactly one profile → straight in; more than one → the picker on every page load, with the last one used preselected and focused. The per-tab memory is removed.
- The picker lists names only. A profile is created from the picker, or from the profile menu when there is only one.
- The header has one profile chip, present on a single-profile install too. Its menu holds the integrations as switches (MLflow, Weights & Biases, Comet, each with its fields when on), "Trocar nome", and "Trocar de perfil" or "Novo perfil". The settings calls and their validation are the ones of ADR-170, ADR-172 and ADR-200.

**Consequences:** reloading on a shared server costs one click or Enter. A browser that had a name saved and picks an unnamed profile is now asked to confirm the name instead of entering silently.

**Alternatives considered:**

- *Give the first profile its own folder under `outputs/profiles/`.* It would hide every CLI run from the GUI's History and needs a data migration; the request was about what the screen says.
- *Hide the unnamed profile when others exist.* Its runs would become unreachable from the interface.

## ADR-203 — Colab: the Drive dataset folder is chosen in the panel

**Date:** 2026-11-12
**Status:** Accepted
**Extends:** ADR-164 (Colab datasets from the user's Google Drive)

**Context:** in Drive mode the dataset path came from the form's `data.base_dir` and the user had to know how a local Drive for desktop path maps to `/content/drive/MyDrive/...`. The panel explained it in paragraphs. The accelerator was a native `<select>`.

**Decision:**

- `GET /api/colab/drive/folders?rel=&my_drive_root=` lists the sub-folders of a folder inside the local "Meu Drive" root: directory names only, sorted, at most 500 with a `truncated` flag; files, links, junctions, dot-folders and Windows hidden or system folders are left out. The answer carries the local path and the Colab path it translates to (`remote_path`). With no root found it answers `root: null` and an empty list.
- `rel` is refused when it has `..`, `.`, `:` or is absolute, and the path is resolved before the containment check. Both routes take requests from this machine only, with the Origin check the folder opener uses (403 otherwise). 400 outside the Drive, 404 missing, 409 unreadable.
- `POST /api/colab/drive/open` opens that folder in the file manager through the existing opener.
- `ColabDriveSource` gains an optional `path`; empty keeps `data.base_dir`, so older requests behave as before.
- The panel has three titled sections (account and usage, train on Colab, runtimes), one line per control and the longer explanations behind a closed disclosure. The folder browser is in the panel: breadcrumb from "Meu Drive", click to enter, "Usar esta pasta". With no Drive found it shows three steps and the typed-root fallback of ADR-164.
- No Google Drive API and no new OAuth scope: browsing Drive on the web would need access the project does not ask for (ADR-164).

**Consequences:** the root search probes `X:\My Drive` and `X:\Meu Drive` on every drive letter, so a disconnected network drive can delay the answer. Detection on a machine with Drive for desktop installed is still not exercised: the browser was checked with a typed root pointing at a local folder. The typed root of ADR-164 is whatever folder the person names, so the listing is confined to that folder, not to a verified Drive; what keeps another site or machine out is the loopback and Origin check, the same trust the dataset picker and explorer already have.

**Alternatives considered:**

- *The native folder dialog (tkinter), as the dataset picker uses.* It opens anywhere on the disk and says nothing about the Colab path; the browser is confined to the Drive and shows the translation.

## ADR-204 — One "live code" tab: the run as steps, with the real code of the step it is in

**Date:** 2026-11-12
**Status:** Accepted
**Amends:** ADR-118 (the training story tab), ADR-173 and ADR-181 (the "Code" tab)

**Context:** the training sheet had two tabs for the same question. "Como funciona" drew an illustration with the stage and the live numbers; "Código" listed every function of the training with all its notes open, in a fixed order, with no relation to what the run was doing. A newcomer had to connect the two alone, and a specialist had to scroll to the function that mattered.

**Decision:**

- The two tabs become one ("Código ao vivo"; the tab id stays `story`, so anything that stored it keeps working). Left: the run as steps (data, model, loss, train epoch, validation, test and report), one plain sentence each, with the live figures on the running step. Right: the real code of the selected step, read from the installed package as before (ADR-173).
- Every tour section declares its `step` and a one-sentence `summary` in both languages (`core/code_tour.py`, `STEPS`); the API sends both, and a test requires every section of every task to have them.
- **Following.** While the run goes, the running step is selected and its code is marked "trecho em execução". Choosing another step or function pauses following; "voltar ao que está rodando" resumes it. After the run ends nothing is marked.
- **The position is what the event stream proves** (`lib/run-steps.ts`), as in ADR-118, and never a line: a loop passes its lines thousands of times a second and no tracer runs. Queued, finished, interrupted or failed marks nothing. Before `start`, or between two units of a job, the group data + model + loss is marked, since the stream cannot tell them apart. After `start` or a non-final `epoch_end`, the group train + validation; while `batch` is the newest event, train alone. The last epoch, `end`, or the last unit closed marks test. PatchCore's phases map to train (features, memory bank) and validation (scoring).
- Notes are closed by default: a quiet marker in the gutter opens the note in place. Branches the run does not take stay dimmed, and the library notice of ADR-194 stays.
- The illustration of ADR-118 remains as a compact element of the left column on wide screens. The stage strip, the caption and the replay of the old tab are removed; the overfitting hint stays on "Resultados".

**Consequences:** the tab answers "what is running now and what does that code do" on one screen. The granularity is the step: `BatchProgressReporter.batch_done` does not emit the last batch, so after it the tab still says "train" for the rest of the epoch including its validation, and an epoch shorter than the batch reporter's interval (about 2 s) never emits `batch` and shows the train + validation group. Emitting the final batch would let validation be marked on its own; it is not done here because it changes what trainers emit. During PatchCore's feature extraction the code shown is the coreset function, since extraction has no tour section. Checked in the browser with classification only; PatchCore, YOLO, segmentation and detection were not watched live.

**Alternatives considered:**

- *A line pointer driven by a tracer (`sys.settrace` / `sys.monitoring`).* It slows the training it describes, and at loop speed the pointer would be noise.
- *Animate the pointer through the loop on a timer.* It would look like tracing while being invented, which ADR-118 rules out.
- *Keep two tabs and link them.* The link is the thing the user had to make; one screen removes it.

## ADR-205 — The last batch of every epoch is always reported

**Date:** 2026-11-14
**Status:** Accepted
**Amends:** ADR-204

**Context:** ADR-204 left a known gap: `BatchProgressReporter.batch_done` skipped an epoch's last batch, leaving it to `epoch_end`. Validation runs between the two, so the "Código ao vivo" tab kept saying "train" through it, and an epoch shorter than the reporter's interval (about 2 s) sent no `batch` at all and showed the train + validation group for the whole run.

**Decision:**

- `batch_done` reports the epoch's last batch (`index >= total`) always, whatever the interval; the batches before it are still limited by the interval. This holds for the loops VisionForge owns (`BatchProgressLoader`) and for Ultralytics' (its `on_train_batch_end` fires for the last batch too).
- `lib/run-steps.ts`: when the newest event is a `batch` with `batch >= total_batches`, the position is validation alone, with no pinned function; a smaller `batch` is still train alone in the loop function. Everything else of ADR-204 stays, PatchCore's phases included.

**Consequences:** every epoch now adds one `batch` event to the stream, also in short runs whose streams used to hold only `start`, `epoch_end` and `end`; tests that pinned the old sequences were updated to the new contract. The progress bar holds an epoch's last batch half a batch short of the epoch's end (`lib/training-progress.ts`), because the validation still runs: it neither steps back nor says 100% before the last `epoch_end`. Every tour has a validation step. For Ultralytics it is VisionForge's record of the epoch (`yolo_epoch`); the library's own `validate()` call sits in `yolo_loop`, which belongs to the train step, so YOLO's validation is shown through the code that reads its metrics. A run stopped mid-epoch does not report a last batch and keeps showing training until the stop.

**Alternatives considered:**

- *A dedicated `validation_start` event.* Clearer, but every trainer (and Ultralytics' callback) would need a new emission point and the frontend a new event type; the last `batch` already marks the same instant.
- *Infer validation from time since the last `batch`.* It would guess, which ADR-118 rules out.

## ADR-206 — The sample dataset that already exists answers 200, and three small layers of in-app help

**Date:** 2026-11-14
**Status:** Accepted
**Amends:** ADR-115 and ADR-117 (the sample dataset), ADR-156 (the strategy step), ADR-201 (the History entry as the results page)

**Context:** two things left over from the interface review. `POST /api/sample-dataset` answered 409 when `datasets/exemplo-<tarefa>/` already held files, and the client turned that 409 into success with `existed: true`. Every caller treated it as success, yet the browser logged a failed request each time, which reads as a bug to anyone with the console open. And the interface said what each thing does, in the terms of someone who already knows it, but not when to pick it or what to do after: a strategy selector listed six strategies with a one-line recipe each, a finished run showed its metrics and stopped, and the empty History said only that it was empty.

**Decision:**

- `POST /api/sample-dataset` answers **200** with `existed: true`, the path and empty `classes` and `counts` when the folder is already there, and **201** with `existed: false` when it wrote it. The folder is still never read or touched (it may hold the researcher's own files). 409 leaves the route's contract; the client reads `existed` from the body and treats any error status as a failure.
- **When to use.** Under every strategy selector, one sentence says when that strategy is the right choice (`components/WhenToUse.tsx`, `whenToUse` in the dictionaries): the classification block selector and the strategy card of the other tasks. The selectors' existing hints stay: they say what the strategy does.
- **What next.** The run detail shows, under the top metrics, at most two suggestions with one sentence and at most one action each (`lib/next-steps.ts`, `components/NextSteps.tsx`). The rules read only what the run recorded; where none applies the block is absent. No rule diagnoses from a new heuristic.
- **The empty History** says what will appear there and offers the "first training" guide.
- Each of the three is one sentence where the decision is made. No panel, tour or modal was added.

**Consequences:** a second click on "usar dataset de exemplo" logs nothing in the console. A client written against the 409 (none is known outside this repository) would now read a 200 it may not expect; `existed` tells the two cases apart. The "when to use" sentences are advice, not rules: they are in the dictionaries so a reviewer can correct one without touching code.

**Alternatives considered:**

- *Keep the 409 and silence it in the client.* A browser logs a failed request whatever the page does with it.
- *A help panel or a longer tour.* The complaint that started the interface work was too much text on the screen; a sentence where the choice happens costs one line.

## ADR-207 — Guides by reader, and a full-screen page for someone who has never trained a model

**Date:** 2026-11-15
**Status:** Accepted
**Amends:** ADR-115 (the guide menu and the first-visit invitation), ADR-206 (in-app help); the capture script of issue #221

**Context:** the guides in `docs/gui/` answer "I want to do X" and assume the reader knows what an epoch is. Three readers were not served: someone new to the field, someone who trains models already and only needs to learn this tool, and someone who wants to see what the tool does before installing it. For the first one a tour card is the wrong size: the tours point at the screen, and this reader needs the field explained, with pictures, before the screen means anything.

**Decision:**

- **Three entry pages in `docs/guias/`**, one per reader, linked from the top of `docs/gui/README.md`: `comecando-do-zero.md`, `ja-treino-modelos.md` and `o-que-o-visionforge-faz.md`. They point at the task pages instead of repeating them.
- **"Começando do zero" is also a page inside the interface** (`components/learn/`, `lib/learn.ts`, `learn` in the dictionaries): full screen, eight chapters read top to bottom, a rail that follows the reader, the new terms of each chapter in one line each. It opens from the guide menu, where it is the first entry, and from a link on the first-visit invitation. It is not in the guide registry, which holds scripts of steps; nothing else opens it.
- **Three kinds of figure, each labelled for what it is.** The network drawing is the component the training sheet already has (ADR-118), running on its own. The split, batches, loss curves and confusion matrix are drawn from closed formulas and fixed numbers in `lib/learn.ts`, and the page says they are illustrative. The screens are captured from the interface: two stills and the film of a real run.
- **The film is made by the capture script.** `capture_docs_images.py` trains a second run of eight epochs on the sample dataset and saves the screen as frames in a smaller window; Pillow assembles `docs/images/vf-training-film.gif` for the pages on GitHub and an animated WebP for the interface (`frontend/src/assets/learn/`, with the two stills). `--check` requires the GIF and its provenance line. The three WebP files add about 200 kB to the package.
- **The page links out**: to the documentation of the libraries the project is built on and to seven free courses and books, each checked to answer on the day this was written. Every link opens in a new tab with `rel="noopener noreferrer"`; the page says they need a connection. The links to the project's own pages point at `main`, and a test checks that each file exists in the repository.

**Consequences:** the guide menu has three entries and the invitation one more line. A release that ships this page before `docs/guias/` reaches `main` would link to pages that are not there yet: both land together in the next release. The screens inside the page are in Portuguese and in the dark theme whatever the reader's language and theme are, and the English captions say so. The stills and the film go stale when the interface changes, like every image in `docs/images`; the same command refreshes them. The external links can rot: nothing checks them after today.

**Alternatives considered:**

- *A longer guided tour.* A card beside a control cannot hold a figure or three paragraphs, and it would make the reader click through an explanation that is better scrolled.
- *Only a page in `docs/`.* The reader who needs it most is the one least likely to look for documentation on GitHub; the interface is where they already are.
- *A recorded video file.* Playwright records WebM, which GitHub does not show inline in a Markdown page, and a video in the wheel is heavier than a few dozen frames. Frames assembled by Pillow need no encoder beyond what the project already depends on.
- *Figures as images.* A drawn figure follows the theme and the language, can be tried (the learning rate switch), and costs no bytes.

## ADR-208 — The screens name what a run reports and what it trains

**Date:** 2026-11-15
**Status:** Amended by ADR-241 (the backend's `val_auroc` is renamed after all)
**Amends:** ADR-048 (custom models), ADR-051 (timm), ADR-038 (anomaly), ADR-201 (the History entry as the results page)

**Context:** after 0.22.0 the first training of every task other than classification was walked through the interface, and a custom model was used the way `docs/custom/MODELS.md` describes. Every run trained. Three things on screen were wrong:

- the detail of a regression, segmentation or anomaly run showed its metrics under the raw `run.json` keys (`TEST_R2`, `BEST_VAL_MIOU`), because its label table only knew classification and detection, while the comparison between runs had labels for all five tasks in another table;
- anomaly's per-epoch scores were labelled "(val)". The task has no validation split: the trainer scores every epoch on the labelled test split, and the backend's `val_` prefix is a field name, not a description;
- a run built from `model.custom_model` or `model.timm_model` was listed in the History, and shown in the Modelo and Revisão steps after a YAML import, under `model.name`, the builtin default those two fields override.

**Decision:**

- The run detail reads its labels from `lib/metric-labels.ts`: its own short table first, then `compareRuns.metrics`, which gains the four names only the detail shows. One dictionary names a metric in both places.
- No label calls an anomaly score a validation one: `auroc`, `image_f1` and `threshold` read "(melhor época)", and the live series read "(teste)". The `run.json` keys do not change.
- The History's architecture is `custom_model`, then `timm_model`, then `model.name` (`run_history._model_arch`). The classification panel shows a note in the Modelo step and a row in the Revisão step when the form carries one of the two, with a button that clears it (`lib/model-source.ts`).
- Two browser tests keep this: `e2e/tasks.e2e.ts` trains the four other tasks on their sample datasets and reads the labels, and `e2e/custom-model.e2e.ts` imports a YAML with `custom_model`.
- Dependabot groups only minor and patch updates; a major one arrives alone.

**Consequences:** anomaly still selects and reports on the same split, which the labels now say instead of hiding; a validation split for anomaly would be a change to the task, not to its words. The form still has no field to type a custom model: it arrives by YAML, as before. The end-to-end job trains five more tiny runs, about a minute on a CPU.

**Alternatives considered:**

- *Rename the backend's `val_auroc`.* It is in every anomaly `run.json` already written and in the event stream; the name is wrong, and changing it breaks reading old runs for a label the interface can fix.
- *A field for `custom_model` in the Modelo step.* Useful, and a larger change: it needs the list of registered builders from the server. The note covers the case that exists today, a YAML that names one.

## ADR-209 — The cu118, cu121 and cu124 extras are removed

**Date:** 2026-11-17
**Status:** Accepted
**Amends:** ADR-198 (torch floors per hardware extra), ADR-005 (hardware extras)

**Context:** ADR-198 kept `cu118`, `cu121` and `cu124` as legacy extras for drivers too old for CUDA 12.6. Their PyTorch indexes stop at torch 2.7.1, 2.5.1 and 2.6.0, so the lock kept three forks that carry nine of the repository's ten torch advisories, one of them critical (`torch.load` with `weights_only=True` leads to remote code execution, GHSA-53q9-r3pm-6pq6) and one high (code execution through an untrusted checkpoint, GHSA-63cw-57p8-fm3p). VisionForge loads checkpoints, and the Hub and bundle import bring checkpoints from elsewhere. ADR-198's reason to keep them was that a driver predating 12.6 would be left with the CPU; but CUDA's minor version compatibility runs any CUDA 12.x runtime on any CUDA 12 driver (525.60+, 527.41+ on Windows), which is the same premise the Docker default already rests on (ADR-072: the cu128 image "works on any CUDA 12 driver").

**Decision:**

- `pyproject.toml` loses the three extras, their indexes, sources and lock conflicts; `uv.lock` was regenerated and only lost packages (the three torch/torchvision forks, the CUDA 11, 12.1 and 12.4 runtime libraries and three old tritons).
- `visionforge doctor` maps any CUDA 12 driver below 12.6 to `cu126`. A driver that only reaches CUDA 11 gets `cpu`, with a warning that names the driver version that brings the GPU back.
- README, QUICKSTART and CONTRIBUTING list `cu126`, `cu128`, `cu130` and `cpu`, and say that `cu126` runs on any CUDA 12 driver.

**Consequences:** a machine whose NVIDIA driver only reaches CUDA 11 (older than 525.60, from before late 2022) trains on the CPU until the driver is updated; before, it got a torch with a critical advisory. The one torch advisory left is a low one on `cu128` (GHSA-rrmf-rvhw-rf47, `torch.jit.script`), whose index stops at 2.11 while the fix is 2.13; RTX 50-series cards that want it gone use `cu130`.

**Alternatives considered:**

- *Keep the extras and dismiss the alerts as accepted risk.* Nothing changes for anyone, and a researcher on an old driver keeps a torch whose checkpoint loading is exploitable.
- *Pin the legacy extras to their last torch and document the risk.* That is what ADR-198 did; the advisories stayed open and the extras kept being offered first to old drivers by `doctor`.

## ADR-210 — `visionforge reproduce` re-trains one seed and judges it by a prediction interval

**Date:** 2026-11-17
**Status:** Accepted
**Extends:** ADR-135 (experiment bundle), ADR-188 (public bundle paths)

**Context:** a verified bundle proves that a record is intact, not that its result can be obtained again. The roadmap asked for a command that re-trains a seed from a bundle and compares the result with the record. Two things shaped it: the bundle's config has its paths reduced (ADR-188), so the dataset is not where the record says; and a replicate group records the Student-t `ci95` of the mean of its seeds, whose half-width shrinks with √n, so one new seed falls outside it often without anything being wrong.

**Decision:**

- `visionforge reproduce BUNDLE.zip [--data DIR] [--seed N] [--output DIR]` verifies the bundle, reads `run.json` and the config member by member from the ZIP (no extraction), and accepts a single run or a `replicates` group; any other group kind, a researcher-defined task or a classification search/transfer block exits 2, "not supported yet".
- The new run trains through `cli_strategies.runner_for`, the dispatcher `visionforge run` uses, under the name `<name>_repro`, with the seed (`--seed`, which must be one of the group's; default: the first seed that finished), the dataset (`--data`, else the recorded path if it exists; never a download) and the output folders replaced. The original run is never written to. `core/reproduce.py` receives the dispatcher as an argument, so `core` does not import the CLI.
- Compared metrics: the runner's final test metrics (validation ones when the task has no test split) plus every metric the group aggregates. The recorded value is the same seed's own value; the delta is reported.
- The verdict uses the 95% prediction interval for one new observation, `mean ± t(0.975, n-1) · std · √(1 + 1/n)`, computed from the recorded `n`, `mean` and `std` and clipped to the metric's domain as the `ci95` is (`replicates.metric_domain`). The recorded `ci95` is reported next to it, unchanged.
- One JSON on stdout (logs go to stderr): values, delta, `ci95`, `pi95`, `inside`, and the environment differences (python, torch, torchvision, visionforge, device). Exit 0 inside the interval or with no interval to judge (a single run), 1 outside it or when the training fails, 2 for a bundle that cannot be reproduced.

**Consequences:** a published replicate set can be checked by a third party with one command, on their own copy of the data. A single-run bundle gets a delta and no verdict: one run says nothing about the spread. With two seeds, `t(1) ≈ 12.7` makes the prediction interval wide; that is the honest width for that evidence. `reproduce.py` imports two private helpers of `core.replicates` (`_t_critical_95`, `_clip_to_domain`), so the interval follows any change to how the group's own interval is built.

**Alternatives considered:**

- *Judge against the recorded `ci95`.* It bounds the mean, not a seed; it would call ordinary seed-to-seed variation a failed reproduction.
- *Extract with `import_experiment_bundle`.* It writes into the profile's models folder, more than a check needs.
- *Train through `build_task_block`.* It returns no run result, so the metrics would have to be parsed back from files.

**Amended 2026-11-18: four gaps found in review.** A metric the reproduction reports as NaN or infinite, where the group has a `pi95`, counts as outside (exit 1); before, it left the verdict at `no_interval` and exit 0. A replicate group whose reproduction has no test split exits 2: the group's aggregates are test scores and cannot judge validation ones. The recorded value and the delta are reported only for the seed's own finished run: a single run re-trained with `--seed` other than its recorded seed, or a replicate that stopped or failed, gets `recorded: null`. A `--output` that is a file, or a config whose `training` or `output` is not a mapping, exits 2 instead of a traceback.

## ADR-211 — Colab is chosen where the GPU is chosen, and its dataset is asked on the Dados step

**Date:** 2026-11-18
**Status:** Accepted
**Amends:** ADR-163 (training on Colab through the official CLI), ADR-164 (dataset in Google Drive)

**Context:** ADR-163 put everything about Colab in one panel of the header: installing the CLI, signing in, and also a second "train" button with its own dataset field and GPU dropdown. The GUI then had two ways to start a training, and the one for Colab bypassed the Revisão step and the Treinar button every other run goes through. The first use with a real account showed the cost: the person looked for Colab where the GPU is picked, and pointed Drive for desktop at the local dataset folder as a backup, which lands under "Computers" and not "My Drive", so the VM cannot reach it; nothing on screen said so.

**Decision:**

- The device selector of the bottom bar offers the account's Colab under this machine's devices: one button per GPU (`T4`, `L4`, `A100`, `H100`, `G4`). With it picked, the usual Treinar button reads "Treinar no Colab" and sends the classification to `/api/colab/experiment/run`. Picking a local device goes back to this machine. Nothing changes on the server.
- While Colab is picked, the selector's button shows what only Colab has: the GPU asked for, the balance in compute units and the runtimes alive (`/api/colab/usage`, `/api/colab/sessions`). The status is asked when the selector mounts and each time its menu opens; the balance and the runtimes only once connected and on screen.
- The Dados step of the classification shows a card that asks where the VM takes the dataset from: a folder of "My Drive" (the browser of ADR-203) or a link (zip URL, `torchvision:`, `hf:`), and whether the run is also saved to the Drive. The card says that a backup-only folder is not in "My Drive". The step counts as filled when that origin can be sent, and Treinar stays locked, with the reason, until then.
- The options are off, with the reason, while the account is not connected, on another task, and for Réplicas and the replicated comparison (ADR-056, ADR-193).
- The header panel keeps the install, the login, the account and the runtimes, and points to the selector for training.

**Consequences:** one way to start a training. A Colab run now passes the Revisão step like any other. The three origins stay: the Drive is the first option, not the only one, because a public dataset by link costs no upload. A Drive folder picked in the card is also a folder of this computer, so it fills the form's dataset folder and the form reads its splits and classes as for a local dataset; the Revisão step adds a row with where the VM takes the dataset from. A link's dataset is not read here before the run, so its task type and number of classes stay the form's, and the card says so.

**Alternatives considered:**

- *Make the Drive the only origin.* Simpler to explain, and it would take away the zip URL and the public catalogs that work today.
- *Lift the header panel's state into the page and share it with the selector.* One request less per menu opening, at the price of threading the panel's eight handlers through the header; the selector asking for itself keeps the two apart.
- *A fifth `kind` in the device selection.* The selection is saved in the config as this machine's device; Colab is where the run goes, not a device of this machine, so it is a separate choice.

## ADR-212 — A multi-user server mode in this repository: threat model, route inventory and phases (proposal)

**Date:** 2026-11-18
**Status:** Proposed. Nothing here is built; this ADR fixes what has to be true before the first line of server code, and each phase gets its own ADR when it lands.
**Extends:** ADR-075 (one run queue per server), ADR-114 (profiles are output folders, not access control), ADR-179 (security headers)

**Context:** VisionForge is a local tool: the browser and the Python process are on the same machine and belong to the same person. ADR-114 let a lab share one `visionforge gui --host` and said in its title what it was not: access control. Issue 352 asks for the other thing: one server for many people, each signing in, each seeing only their own runs and datasets, a queue that says how many jobs are ahead and not whose, datasets that arrive by upload and are deleted after use, and sign-in with GitHub and other accounts to publish weights and runs.

The code as it is on 2026-11-18 (`development`, b95c6919), read through the FastAPI app's own route table:

- **Identity is a header the client writes.** 79 of the 163 routes read `X-VF-Profile`; the server believes it. Any client can send any slug.
- **84 routes do not read the profile at all:**
  - 16 are read-only and hold nothing personal: the schemas, `/api/tasks`, the code tour, `/api/system/info`, `/api/device/info`, the model defaults, `/api/health`, `GET /api/update` and the SPA.
  - 18 read a dataset from a path of the server's disk that the client names: the `dataset/stats`, `samples`, `detect`, `preview_*` and `audit` routes of every task, `/api/dataset/file`, and the explorer's `summary`, `images`, `thumb` and `scan`. `/api/dataset/file` is confined to an allow-list, and that list is one per process: a folder one person probed is readable by all.
  - 28 write to a path of the server's disk that the client names: the explorer's merge, resplit, incorporate and COCO import/export jobs (25 routes, with job ids nobody owns), `/api/dataset/download`, `/api/sample-dataset` and `/api/dvc/prepare`.
  - 6 act on the desktop of the machine the server runs on: the tkinter pickers (`/api/dataset/pick`, `/api/checkpoint/pick`, `/api/detection/dataset/pick_yaml`) and the Drive for desktop routes (`/api/colab/drive/folders`, `open`, `continue`).
  - 16 hold state every user shares: `/api/credentials` (the dataset providers' keys), `/api/profiles` (anyone lists, creates and renames), `/api/queue` and `DELETE /api/queue/{run_id}` (every job with its profile and name, cancellable by anyone), `/api/experiment/events`, `status` and `result/{run_id}`, `/api/artifacts/{path}` (everything under `outputs/`, every profile's), the custom tasks' delete, hide and unhide, and `/api/update/apply`.
- **Code a user brings runs inside the server process.** A training is `asyncio.to_thread(block.run)`: a thread of the server, with its memory, its files and its tokens. A custom task is a Python file loaded with `exec_module` (`tasks/registry.py`). The 22 `torch.load` calls of the package all pass `weights_only=True`, but a detection checkpoint goes to `YOLO(weights_path)`, and Ultralytics 8.4.60 sets `weights_only=False` when the caller does not say (`ultralytics/utils/patches.py`): a `.pt` a user supplies for detection is a pickle the server executes.
- **Tokens are plain files.** The Hugging Face, GitHub, Zenodo, W&B and Comet tokens of a profile are text files under the config folder, found from the profile's slug. Whoever sends that slug publishes with that token.
- **What already exists.** `HostOriginGuard` refuses a foreign `Host` or `Origin`; the CSP and the fixed headers are on every response; `reveal`, `dvc/prepare`, the Drive `open`/`continue` and `update/apply` already answer only a loopback client.

**Threat model.** What is protected: each person's datasets, runs, checkpoints and tokens; the server's disk and GPU; the other tenants' time in the queue. Who is assumed hostile in phase 1: any signed-in user, and anyone on the network who is not signed in. Who is trusted: the operator, and in phase 0 the members of the group.

| Threat | Today | Needed |
| --- | --- | --- |
| Pass for someone else | send their slug in `X-VF-Profile` | identity from an authenticated session only; the header is ignored |
| Read or delete another's runs | `/api/artifacts`, `/api/experiment/*`, `/api/queue` are shared | every route that names a run checks its owner; the queue answers a count |
| Read the server's disk | dataset routes take any path | every path resolves inside the user's own root; no native dialogs |
| Write over the server's disk | explorer jobs and downloads take any output path | same confinement; jobs owned by who started them |
| Run code as the server | custom task, detection `.pt`, the training thread | phase 0: accepted, said in the docs; phase 1: one process per job with no access to the server's files, or the feature is off |
| Use another's publishing token | plain file found from the slug | read only for the session's user; encrypted at rest in phase 1 |
| Starve the others | one queue, no limits | per-user limits on queued jobs, disk and upload size |
| Fill the disk with an archive | zip downloads are extracted as they come | size and entry limits before extraction |
| Take over a browser session | not applicable (no session) | HttpOnly, Secure, SameSite cookie; the existing Origin check covers cross-site posts |
| Leftover data | datasets stay where the user put them | uploaded datasets deleted at the end of the run, the deletion logged and written to `run.json` |

**Decision:**

- **One repository, one flag.** Server mode is `visionforge gui --server`, off by default. Without it nothing changes: the local tool keeps its pickers, its folder buttons and its header. A fork is rejected: every fix would be made twice.
- **Phase 0, a group that trusts its members.** VisionForge writes no login. An authenticating reverse proxy (oauth2-proxy or Authelia, with GitHub or Google) sits in front; VisionForge listens on loopback, accepts the user name only in a header from the proxy's address (`--trusted-proxy`), and refuses to start in server mode without one. That name is the profile; `X-VF-Profile` from the client is ignored and `/api/profiles` stops creating and renaming.
- **In server mode every route is in one of four classes,** declared where the route is declared and checked by a test that walks the app's route table, so a new route without a class fails the suite:
  - *open*: the 16 read-only routes;
  - *owner*: the 79 that read the profile today, plus the queue, the events, the status, the result, the artifacts and the explorer jobs, each checking that the run or job is the session user's;
  - *confined*: every route that takes a dataset path, which must resolve inside that user's dataset root;
  - *off*: the native pickers, `reveal`, the Drive for desktop routes, `update/apply`, the shared `/api/credentials`, and the custom tasks. An *off* route answers 404 and the GUI hides its button, with the reason where a person would look for it.
- **The queue answers a number.** A user sees their own jobs in full and, of the rest, only how many are ahead.
- **Phase 0 does not isolate code.** A member can supply a detection checkpoint, and with it run code as the server. The install guide says so in its first paragraph, and that is why phase 0 is for a group that already shares the machine.
- **Phase 1, open multi-user,** needs, in this order: each job in its own process with its own working folder and no read access to the config folder (a container where the host has one); upload of datasets in resumable parts with a per-user quota and archive limits; deletion at the end of the run, logged and recorded in `run.json`; tokens encrypted at rest and OAuth with GitHub for publishing; per-user limits in the queue. Custom tasks and user-supplied checkpoints stay off until the first item is done.
- **Colab in server mode** is the Drive through Google's API with the user's consent, not Drive for desktop, and waits for phase 1.

**Consequences:** the local tool pays nothing: the flag is off and the classes only bite in server mode. Phase 0 is small (a header reader, the class of each route, owner checks, path confinement) and is useful by itself to the lab ADR-114 was written for, which today has organisation and no privacy. The route-class test turns "did someone remember to protect the new route" into a failing check. The hard part is phase 1's process isolation; until it exists, VisionForge must not be offered to people who do not trust each other, and the docs say so.

**Not verified:** the inventory comes from the route table and each route's dependencies, not from reading the 163 handlers; a route that reads the profile may still take a path outside it, and phase 0 starts by reading each one. Nobody has tried to break the current `--host` mode on purpose. The Ultralytics behaviour was read in the installed 8.4.60 only.

**Alternatives considered:**

- *A fork for the server.* Two code bases for one trainer; rejected in the issue.
- *VisionForge's own passwords.* Password storage, reset, lockout and 2FA are a product of their own, and a proxy that does them well exists. Revisit only if someone needs a server with no identity provider.
- *One VisionForge process per user behind a spawner (the JupyterHub way).* Real isolation with little new code, and it fits phase 1's "one process per job" well. It loses the shared queue, which is the point of sharing one GPU; kept as the fallback if confinement inside one process proves leaky.
- *Keep the header and add a shared password.* It protects the server from the network and nobody from each other.

## ADR-213 — Server mode starts as a gate: only the login proxy is answered, it names the user, and a route answers only once it has a class and was read

**Date:** 2026-11-18
**Status:** Accepted
**Extends:** ADR-212 (the server-mode proposal), ADR-114 (profiles are output folders), ADR-178 (the Host and Origin guard)

**Context:** ADR-212 proposed phase 0 of a multi-user server: a group that trusts its members, behind a reverse proxy that signs them in. Its inventory found 84 routes that ignore the profile and 79 that believe a header the browser writes. Phase 0 cannot land in one change: each of the 163 routes has to be read before it may answer several people. What can land first is the part every later change stands on: who is asking, and a rule that keeps a route closed until someone has read it.

**Decision:**

- `visionforge gui --server --trusted-proxy <address>` turns the mode on. It is off by default and experimental. Without `--trusted-proxy` the command stops with the reason; `--trusted-proxy` and `--user-header` without `--server` stop too.
- `ServerModeGate` (`gui/server_mode.py`) is an ASGI middleware inside the Host and Origin guard. Off, it passes every request through. On:
  - a client whose address is not one of the trusted proxies gets 403: someone who reaches the port directly could write the user header themselves;
  - the account comes from the proxy's header (`X-Forwarded-User`, or `--user-header`), trimmed and case-folded; missing or unusable is 401. `GET /api/health` alone needs no user, for the proxy's probe;
  - the account becomes a profile: the folder is `<readable part>-<10 hex of the SHA-256 of the whole name>`, made on the first request. The digest is what keeps `a.b@lab.org` and `a-b@lab.org` apart, since both fold to the same readable part;
  - the gate overwrites `X-VF-Profile` on the request with that folder. The 79 routes that resolve a profile keep their code and now work in the signed-in person's folder, whatever the browser sent.
- Every route has a class in one table, `ROUTE_CLASSES`: `open`, `owner`, `confined` or `off` (ADR-212). A test compares the table with the app's OpenAPI document in both directions, so a new route without a class, or a class of a route that is gone, fails the suite. The gate finds the route of a request by matching the table's own path templates, not through the app's router: how a router keeps the routes of an included one changed between FastAPI 0.135 and 0.165, and the first version of the gate, which walked `app.routes`, found no API route on the newer one.
- **The gate fails closed.** In server mode a route answers only when it is `open`, or `owner` and listed in `OWNER_READY`, the routes whose handler was read. Anything else is 404 with a message that says the feature is not available on a server: `off`, `confined`, the `owner` routes not read yet, a route the table does not know, and every WebSocket.
- This change reads and opens the History (`GET /api/runs`, `GET` and `DELETE /api/runs/{run_id}`, which look only inside the profile's models folder) and the 18 token routes (a file found from the profile). `GET /api/session` is new and `open`: whether the server is in server mode, and the account and profile of who asks.

**Consequences:** server mode cannot train yet: the dataset routes, the training routes, the queue and the events are closed, and the guide says so in its first paragraph. Each later change opens a group of routes after reading it (confinement of dataset paths to a per-user folder, owner checks on the queue and events, then training). The GUI does not read `/api/session` yet and shows buttons that answer 404. The routes that check for a loopback client today (`reveal`, `dvc/prepare`, the Drive routes, `update/apply`) would be fooled by a proxy on the same machine, whose address is loopback; they are `off`, so the gate refuses them before that check runs.

**Amends ADR-212:** the class of a route is declared in one table and not where the route is declared. The guarantee is the same (the test fails for a route without a class), the change does not touch 163 decorators, and the whole policy can be read in one place. `--trusted-proxy` does not force a loopback bind: the proxy may run in another container, and the address check is what protects the port.

**Not verified:** no install behind a real oauth2-proxy or Authelia; the gate was exercised with simulated requests from a trusted and an untrusted address. The 21 `owner` routes opened here were read; the other 50 were not.

**Alternatives considered:**

- *A FastAPI dependency per route instead of a middleware.* It would see the matched route for free, and it would run beside the profile dependency with no guaranteed order; the header has to be overwritten before any handler reads it.
- *Open every `owner` route now, since each already resolves a profile.* The training routes take dataset paths in their body, and a route that resolves a profile can still read outside it. Reading first is the point.
- *The readable slug alone as the folder.* Two accounts that fold to the same slug would share runs and tokens.

**Addendum (2026-11-24): `POST /api/model/defaults` is `confined`.** It was `open` because it suggests settings for an architecture, but its optional `base_dir` is walked and up to 200 images of it are opened for the suggested size: on a server that read any folder the server account could. `ModelDefaultsRequest.base_dir` is now `DatasetDir | None` and the route is listed in `READY`; without `base_dir` it answers as before. The walk runs in a worker thread (`asyncio.to_thread`), and an `OSError` from it leaves the size unsuggested instead of answering 500.

**Addendum (2026-11-25): the proxy is the peer, never a forwarded address.** uvicorn's default `proxy_headers=True` trusts `X-Forwarded-For` from 127.0.0.1 and puts that address in the request's `client`. The gate reads `client` to recognise the login proxy, so two things went wrong: a proxy on the same machine (`--trusted-proxy 127.0.0.1`, the example in `docs/gui/servidor.md`) sends the browser's address and every request was refused with 403; and any local process could send `X-Forwarded-For: <proxy address>` and be taken for a remote proxy, naming any account. In server mode `start_server` now passes `proxy_headers=False`; a local install keeps uvicorn's default. Tests: `tests/gui/test_server_mode_proxy_headers.py` builds the app exactly as uvicorn serves it.

## ADR-214 — In server mode a request reads datasets only from the person's own folder and the folders the operator shares

**Date:** 2026-11-18
**Status:** Accepted
**Extends:** ADR-212 (the server-mode proposal), ADR-213 (the gate)

**Context:** ADR-213 left the 45 `confined` routes closed: each takes a dataset path the client writes, and on a server that path could be anyone's folder or the server's own files. The routes that only read a dataset to show it (splits, counts per class, samples, thumbnails) are what the Dados step needs first. They share one shape: an absolute `base_dir` and names inside it (`train`, `images`, `train.csv`). `/api/dataset/file` serves a thumbnail from an allow-list of folders "the user probed this session", and that list is one per process: on a server it is everybody's.

**Decision:**

- A person has a dataset folder, `outputs/profiles/<slug>/datasets/`, made by the gate on their first request. The operator shares folders with everyone, read-only, with `--datasets-dir <folder>` (repeatable; it must exist, and it needs `--server`).
- The gate binds a `DatasetScope` for each request, in a context variable: the person's folder first, then the shared ones, all resolved. `GET /api/session` lists them in `dataset_dirs`. A local install binds none.
- Two field types carry the rule (`gui/api/dataset_scope.py`), and the request models of the dataset routes use them in place of `str`:
  - `DatasetDir`, for `base_dir`: with a scope bound it must be absolute, not start with `~`, and resolve inside the scope; the model keeps the resolved path, so the handler works on the path that was checked.
  - `DatasetPart`, for a name inside the dataset: with a scope bound it may only go down (no `..`, no root, no drive, no `~`), read both as a Windows and as a POSIX path.
  - With no scope bound both accept what `str` accepts: the local tool is unchanged, and so is the OpenAPI document.
- A path outside the scope fails the validation of the body (422 with the reason), before any handler runs.
- `_is_inside_allowed_root`, behind `/api/dataset/file`, asks the request's scope when one is bound and ignores the process-wide list.
- Nine `confined` routes are read and open in server mode: `/api/dataset/detect`, `stats`, `samples` and `file`, the detection `stats` and `samples`, and the segmentation, anomaly and regression `stats`. A test finds the request model of every open `confined` route in the OpenAPI document and fails when one has a plain `str` field that is not on a short list of fields that are not paths.

**Consequences:** the Dados step can inspect a dataset on a server. Nobody can add a dataset through the GUI yet: the download, the sample dataset and the Laboratório de datasets write to the disk and stay closed until a write scope (the person's own folder only) exists; until then the operator places the files. The two previews stay closed because they serve their images through `/api/artifacts`, which is still everyone's. `OWNER_READY` is now `READY`, since it lists open `confined` routes too.

**Known limits:** a regression manifest may name an image by an absolute path; the handler does not open it, but the count of missing images tells whether the file exists. A symlink inside a dataset folder that points outside is followed for the names inside (`DatasetPart` checks the text, not the target); in phase 0 only the operator can create one.

**Not verified:** the validators run in the request's context because FastAPI validates the body in the task the middleware calls; that was exercised through the real app with a test client, not behind a real proxy. The 36 other `confined` routes were not read.

**Alternatives considered:**

- *Check the path in each handler.* Nine handlers today and 45 in the end, each a place to forget; the type is checked by a test.
- *Inspect the JSON body in the gate.* The gate would have to know which strings are paths, and it would check a path the handler then reads again in another spelling.
- *A fixed list of dataset names instead of paths.* Cleaner for a server, and it would fork the request models and the forms from the local tool, which names folders.

**Addendum (2026-11-24): outside the scope, one answer whether the path exists or not.** `/api/dataset/file` and `/api/explorer/thumb` resolved the path strictly (404 when missing), and `/api/dataset/file` checked the extension (400) before asking the scope (403), so the status told whether a file of another profile existed. Both now resolve without `strict`, ask the scope first, and only then check the extension and that the file or folder exists. `DatasetAuditRequest.splits` became `dict[str, DatasetPart]`: with `layout: yolo` a climbing split was looked up on the disk and the error named the folder found or not found.

## ADR-215 — In server mode a run, its progress and the queue belong to who started it, and a config names only what its owner may reach

**Date:** 2026-11-18
**Status:** Accepted
**Extends:** ADR-075 (one run queue per server), ADR-212, ADR-213 (the gate), ADR-214 (the dataset scope)

**Context:** after ADR-214 a person could inspect a dataset on a server and do nothing with it. Three groups of routes stood between that and a training. The queue and the run routes (`/api/queue`, `/api/experiment/events`, `status`, `result`) showed every job to everyone and let anyone stop any of them. `/api/artifacts` served everything under `outputs/`. And the five run routes take a whole config: a dataset folder, weights, names inside the dataset, and a model name that Ultralytics and timm resolve as a file or a remote address when it looks like one (`YOLO("../x.pt")` unpickles that file).

**Decision:**

- The request scope (ADR-214) also names its owner, the signed-in person's profile. A started job keeps the profile it was submitted under, and the record a lane keeps after the job's state is gone carries it too.
- With an owner bound:
  - `/api/experiment/status`, `events` and `result/{run_id}` and `DELETE /api/queue/{run_id}` treat a run of another profile as a run that does not exist: the same 404, idle status or empty stream. With no `run_id`, "the latest run" is the person's own latest, not the server's.
  - `GET /api/queue` lists the person's jobs and adds `others_ahead` to each lane and to the snapshot: how many jobs of other people run or wait before the person's first job, or all of them when the person has none. No label, id or profile of another person's job leaves the server. It is 0 on a local install.
  - `/api/artifacts/{path}` serves only what sits under the person's profile folder.
- The five run routes (`/api/experiment/run` and the detection, segmentation, regression and anomaly `run`) call `confine_config` on the validated config, after the profile has rewritten its output folders. With a scope bound it walks every field and answers 422 naming the field when:
  - a path is outside the person's profile folder and the shared dataset folders, or a path the run writes (`output.*`, `output_csv`, `output_onnx`) is outside the person's folder;
  - a name inside the dataset (`train_dir`, `images_subdir`, `train_csv`...) climbs out of it;
  - `model.name` or `model.timm_model` has a path separator, a colon, a leading dot or `~`, or `..`;
  - `model.custom_model` is set.
- The configs live in `utils/` and are shared with the command line, so the rule is a walk over the validated model and not a field type. A test lists every text field of the five configs and fails when one is in none of the four sets (dataset part, model name, forbidden, read and free): a new text field has to be read before the suite passes.
- These routes are open in server mode. The strategies (23 routes), resume, the evaluation routes and the exporters stay closed until each is read.

**Consequences:** a person can train one run at a time on a server with a shared dataset and follow it; nobody sees what the others train, only how long the line is. Weights must sit in the person's folder or a shared one, and nobody can put a file there through the GUI yet, so a detection checkpoint, which Ultralytics unpickles, can only be one the operator placed or one the person's own run wrote. The training still runs in a thread of the server: one person's job can exhaust the memory or the GPU of all, and phase 0 remains for a group that trusts its members.

**Known limits:** the config is validated before it is confined, and its own validators say whether `base_dir` or `weights_path` exists, so the 422 of a path outside the scope can differ from that of a path that does not exist. `status.queued` still counts every waiting job on the server. A data YAML or a CSV inside a shared dataset may name files elsewhere; the operator controls those.

**Not verified:** nothing ran behind a real proxy; no GPU run, and only classification was trained (see the amendment).

**Amended 2026-11-18: a search could leave the scope.** `grid_search.hyperparameters` and `random_search.search_space` were listed here as text that names nothing on the disk. They name fields of the config, which the block sets per trial after the walk. ADR-216 closes every classification `block` but the plain run on a server.

**Amended 2026-11-18: a real run.** The real server, in server mode in a temporary folder, trained a one-epoch classification on the CPU for one account while a second account submitted 40 ms later: the second job queued and ran next, each run landed in its owner's folder and History, the second account got an idle status, 404 and 404 for the first one's status, result and stop, and a plot of the first run answered 200 to its owner and 403 to the other. The first attempt found a bug that is not server mode's: the second submission answered 500, because `cuda_context_broken` reads `sys.modules["torch"]`, which holds the module from the first line of its import, while the first job's thread was still importing it and `torch.cuda` did not exist yet. A torch without `cuda.is_initialized` now reads as no context to break. A local install had the same window on a cold server: two submissions within the three seconds of the first import.

**Alternatives considered:**

- *One queue per person.* Nobody would wait for anybody, on one GPU that cannot be true; the shared queue with a count is what the request asked for.
- *Hide `others_ahead` too.* The person would see their job waiting with no reason.
- *Field types in the configs, as in ADR-214.* `utils/` imports nothing of the GUI, and the command line must keep accepting any folder.

## ADR-216 — In server mode the screen asks the server what it offers and hides the rest

**Date:** 2026-11-18
**Status:** Accepted
**Extends:** ADR-213 (the gate), ADR-214, ADR-215

**Context:** after ADR-215 a person could train on a server, and the page did not know it was on one. It offered the folder dialog (which would open on the server's screen), four strategies per task, the datasets sheet, Colab and a dozen actions in History; each answered "not available". On opening it also called `/api/profiles` and `/api/colab/status`, both closed. Reading how the page submits a strategy found a hole in ADR-215: classification sends cross-validation and the searches through `/api/experiment/run`, as the `block` field, so the gate let them through; and a search rewrites fields of the config for each trial (`{"data.base_dir": [...]}`) after `confine_config` has walked it. A grid axis could name a dataset, weights or an output folder outside the person's reach. `grid_search.hyperparameters` had been listed as text that names nothing on the disk.

**Decision:**

- `GET /api/session` also returns `closed`, the keys of `ROUTE_CLASSES` that do not answer (`"METHOD /api/template"`), and `closed_blocks`, the values of classification's `block` the server refuses. Both are empty on a local install. The list of what a server offers stays in one place, the table of ADR-213: the page holds no copy of it.
- With a scope bound, `confine_config` refuses every `block` but `classification` with 422. The two fields that rewrite others are listed apart (`REWRITING`), not as free text.
- The page reads the session before it is first drawn (`bootSession`, at most 3 s of wait; a slower answer is applied when it comes) and keeps it in a store outside React (`lib/session.ts`). An older server, a failed read and a local install all read as "nothing closed": the server is what refuses, the page only stops offering.
- A component asks by the key of the route it would call (`ROUTE.pickFolder`, `strategyRoute(task, strategy)`). What is closed is absent, not disabled: a greyed button with "not on a server" on every card would be most of the page. A test on the server side reads `session.ts` and fails if a key it asks about is not in `ROUTE_CLASSES`.
- Where the folder dialog is gone, the dataset field lists the folders the server accepts (`dataset_dirs`), the person's own first. The profile menu names the account and the profile folder and offers no other profile. The first-run invitation offers the tour alone, and the tour skips the datasets button.

**Consequences:** a person on a server sees a smaller page in which everything works, and each route opened later lights up its button with no change to the page, as long as the button asks by that route's key. A classification search on a server is closed until its axes are confined; the local tool is unchanged. The greeting still asks a name on the first visit and keeps it in the browser, since the profile cannot be renamed there yet.

**Known limits:** the page asks about 32 routes and a route it does not ask about is still offered; opening a closed route by hand still answers 404, which is the rule that counts. The integrations (MLflow, W&B, Comet) are not configurable on a server, because they are saved through `PUT /api/profiles/{slug}`. A session that arrives after the 3 s wait redraws the page with fewer buttons.

**Not verified:** nothing ran behind a real login proxy: the page was opened behind a stand-in that only adds the account header. Not checked at phone width or with a screen reader in this mode. No Playwright test covers server mode.

**Alternatives considered:**

- *A list of features in the page ("on a server, hide these").* Two lists to keep equal; the one in the page would be wrong the day a route opened.
- *Disable instead of hide.* Honest about what exists, but most of the page would be grey, and the reason would be the same sentence everywhere.
- *Confine the axes of a search now.* It is the right end state, and it belongs with the strategies, which are read together (23 routes); until then the block is closed.

**Amended 2026-11-19:** ADR-217 reads the axes and opens the strategies; `closed_blocks` now lists only the two blocks that do not train.

## ADR-217 — In server mode a strategy runs, and what it rewrites per trial is read like the config

**Date:** 2026-11-19
**Status:** Accepted
**Extends:** ADR-215 (the config), ADR-216 (which closed the searches)

**Context:** ADR-216 closed every classification `block` but the plain run, because a search sets fields of the config for each trial after `confine_config` has walked it. The other strategies have the same shape and were still closed by the gate: a sweep (`search_space`), a paired comparison (`variants`) and a model comparison (`model_names`) each apply dotted names to the base config once per trial. Twenty routes and five values of `block` were waiting on one question: what may be rewritten.

**Decision:**

- `confine_axes(config_type, axes)` reads the dotted names a strategy sets. An axis may name only a field whose annotation holds numbers, text, switches or choices (`int`, `float`, `bool`, `str`, `Literal`, an enum, and lists, tuples and unions of those). A path, a `dict`, a config section, `Any` and a name the config does not have are refused with 422. So are a name inside the dataset (`DATASET_PARTS`), a custom model, `block` and the search spaces themselves, which are text or choices by type. The rule is by annotation, not by value, so a field that is `None` in the base config (`model.weights_path`) is still seen as a path, and a field added later is closed unless it is plain.
- A model name an axis offers (`model.name`, `model.timm_model`) is checked like the field itself: every text inside the offered value (a list, a `choice` spec's options, one variant's value) must be a plain name. `YOLO("../x.pt")` unpickles that file.
- The value an axis offers for any other open field is not read: the trial validates it against the config.
- Every strategy start confines its base config like a plain run, after the profile scopes its output: comparison, sweep, K-fold, replicates and paired comparison. `confine_config` calls `confine_axes` on `grid_search.hyperparameters` and `random_search.search_space`, so the classification blocks follow the same rule through `/api/experiment/run`.
- The twenty strategy routes of the built-in tasks join `READY`. `OPEN_BLOCKS` becomes every block that trains; `batch_prediction` and `export_onnx` stay closed and are read with the evaluation routes. The custom-task strategies stay `off` with the custom tasks.
- `cross_validation.group_manifest` is a path the loader resolves against the dataset folder and refuses outside it: a relative one only has to go down (`DATASET_FILES`); an absolute one is checked like any path.
- The page changes nothing: it asks by route key and by `closed_blocks` (ADR-216), so the strategies appear when the server stops listing them.

**Consequences:** on a server a person can cross-validate, search, compare and replicate, and no trial reads or writes outside what the base config was allowed. A search cannot vary the dataset or the weights there: that takes one job per dataset. The local tool is unchanged, since neither function does anything with no scope bound.

**Known limits:** a strategy is one job in the queue from its first trial to its last, and nothing caps the number of trials: a long search holds the queue for everyone until it ends or its owner stops it. The rule trusts the annotations: a text field that named a file would have to be added to one of the sets of ADR-215, which the suite enforces for every text field of the five configs.

**Not verified:** nothing ran behind a real login proxy. Only a classification grid search trained for real in this mode (two trials, CPU); for the other tasks and strategies the requests were tested and the training was not. Optuna and stopping a strategy midway were not tried in this mode.

**Alternatives considered:**

- *Check each offered value against the field's rule (allow a search over datasets inside the scope).* It needs the type of the field and the validators that resolve it, per trial combination; a config validator that joins or expands a path would be one more thing to mirror. Varying the dataset inside one search is rare, and a job per dataset does it.
- *Confine inside the trial, when the config is built.* The scope is bound to the request, and the trial runs later from the queue in a worker thread; and a refusal would come after GPU time was spent, as a failed trial instead of a 422.
- *A list of the fields a search may vary.* Shorter to read, and wrong the day a config gains a field; the annotation rule fails closed instead.

## ADR-218 — In server mode a finished run can be resumed, tested, explained and exported, inside its owner's reach

**Date:** 2026-11-19
**Status:** Accepted
**Extends:** ADR-214 (the dataset types), ADR-215 (the config)

**Context:** after ADR-217 a person could train anything on a server and do nothing with the result but look at it: resuming, testing on another dataset, Grad-CAM, batch prediction and ONNX export still answered 404. The four evaluation routes take a run id and a body; they load the checkpoint `run.json` records (`artifacts.model`), read images from a folder the body names, and write into the run folder or, for the CSV and the ONNX file, wherever the body says. Resuming rebuilds the config from `run.json` and queues it. Loading a detection checkpoint unpickles it.

**Decision:**

- The folder a body names is typed `DatasetDir`, like every dataset path since ADR-214: `RunTestRequest.data_dir`, `GradCamRequest.input_dir`, `BatchPredictRequest.input_dir`. The three routes move from `owner` to `confined`, and the test that walks the bodies of the confined routes covers them.
- An output path is typed `RunFolderFile`: on a server only an empty one passes (`ExportOnnxRequest.output_onnx`, `BatchPredictRequest.output_csv`), so the file is written into the run's own folder. A person has no other folder on the server worth naming, and one rule (the run folder) is easier to state than a second scope for writes.
- Before any of the four actions runs, `_reject_foreign_checkpoint` reads `artifacts.model` and answers 409 when it resolves outside the owner's profile folder. A run trained on the server always records a file of its own; the check is there so that what gets unpickled never depends on a `run.json` being as the server left it.
- Resuming calls `confine_config` on the stored config after the profile scopes its output: the folders a server shares may have changed since the run started, and a stored config is read like a new one.
- The five routes join `READY`. The classification blocks `batch_prediction` and `export_onnx` stay closed for good on a server: they take a checkpoint path and an output path in the config, and the run routes do the same work from a run id.
- The page changes little: the actions appear because the server stops listing them. The "browse" button of their folder fields is absent where the folder dialog is closed, and the note of ADR-216 lists the folders the server accepts under each.

**Consequences:** a run trained on a server can be used there. Nothing an action reads comes from outside the person's folders and the shared ones, and nothing it writes leaves the run folder. The local tool is unchanged: the types accept what `str` accepts and the checkpoint check does nothing with no scope bound.

**Known limits:** the four evaluation actions run at once, in a worker thread, outside the queue: they share the CPU or the GPU with whatever job is training, another person's included. `run.json` itself is trusted for everything but the checkpoint path (the model section the action rebuilds, the names of the dataset parts): it sits in the person's own folder, which only the server writes today. Downloading the run (ZIP, Markdown), the publishers and the tracker retries are still closed.

**Not verified:** nothing ran behind a real login proxy. Test, Grad-CAM, batch prediction and ONNX export ran for real on one classification run (CPU); the other four tasks were not run in this mode. Resuming was tested as a request, not by continuing a stopped run.

**Alternatives considered:**

- *Let the output path be anywhere in the person's folder.* A second rule and a second type for a need nobody has on a server: the file is downloaded from the run, not picked up from a path.
- *Open the two classification blocks instead.* They name a checkpoint and an output file as config paths; confining them would repeat what the run routes already guarantee by taking a run id.
- *Put the evaluation actions in the queue.* It would settle the shared-GPU limit, and it changes how the page waits for their result (a request today, a job then); it belongs with a decision about fairness, not with opening the routes.

**Update (2026-11-24):** a regression test reads images named by the rows of the chosen CSV, not only the CSV itself. On a server each row's image must resolve inside the scope before the checkpoint is loaded (`_confine_manifest_images`); a row with an absolute path or `..` that leaves it answers 400 naming the line, never the path, so the answer does not tell whether that file exists.
**Addendum (2026-11-24): an anomaly test calibrates inside the scope.** Testing an anomaly run on a folder sets `base_dir` to that folder's parent and keeps the run's `train_dir/normal_dir`, the split the threshold is calibrated on. Choosing a shared folder itself put that parent, and the split, one level above it. `_evaluate_anomaly_checkpoint` now refuses (400, `testrun.anomaly_train_outside`) a training split the scope does not hold, before checking whether it exists, so the answer says nothing about the disk outside.

## ADR-219 — In server mode a person writes a dataset into their own folder only, with their own key

**Date:** 2026-11-19
**Status:** Accepted
**Extends:** ADR-214 (the dataset scope), ADR-055 (downloads), ADR-115 (the sample dataset)

**Context:** on a server only the operator could put data there, by sharing a folder. The two routes that write a dataset were closed: `POST /api/dataset/download` writes wherever `out_dir` says, and `POST /api/sample-dataset` writes into `datasets/` under the server's working folder, shared by everyone. A download also falls back to the keys saved on the machine and to the process environment (`resolve_credential`, `load_dataset(token=None)`), which on a server are the operator's; and the Kaggle client takes its key only from `KAGGLE_API_TOKEN`, which `download_kaggle` sets in the process.

**Decision:**

- `DatasetScope` gains `own`, the signed-in person's dataset folder, which the gate already made and listed first in `readable`. It is the only place a request may write a dataset.
- `DatasetDownloadRequest.out_dir` is typed `OwnDatasetDir`: on a server an absolute path that resolves to a sub-folder of `own`. The folder itself is refused, so two downloads cannot mix their split folders at its root; a shared folder, another person's folder and a scope with no `own` are refused.
- The sample dataset is written under `own` on a server (`_sample_root`), so each person has their own copy and the 200 "already there" answer speaks of their folder, not of someone else's.
- On a server a download uses only the key its request carries (`_credential`). With none, Roboflow answers that a key is needed, and the Hub client is given `token=False`, which sends no token, where `None` would let it read one from the environment or from the server account's saved login.
- Kaggle is refused on a server with a message that says why and what to do. Putting a person's key in the process environment would hand it to every other download running in that process.
- The download's 422 hides its body, because the body may hold a key (`quiet_validation_route`). It now names the rule when `out_dir`'s own validator is what refused the request, still without repeating anything the request carried.
- The page asks for a folder name, not a path, on a server, shows where the dataset will land, leaves Kaggle out of the providers, and takes the Roboflow key and the Hub token as fields of this one download (the saved-key controls call the credential routes, which stay off). The datasets sheet shows only the tabs whose routes answer.

**Consequences:** a person can bring a public dataset, or a private one with their own key, to a server without the operator, and nothing they download lands where another person reads or writes. The first-training guide is offered again there, since it needs only the sample dataset. The local tool is unchanged: the type accepts what `str` accepts, the sample goes to `datasets/`, and the saved keys are used as before.

**Known limits:** no disk quota per person and no cap on concurrent downloads; a download runs at once, outside the queue. The Hub's cache sits in the home of the account the server runs as, so a second download of the same dataset is served from it. A key typed for a download travels in the request body, like on a local install, and the proxy in front must be TLS for that to be private. The Laboratory (explore, merge, re-split, incorporate, COCO), the previews and the audit are still closed.

**Not verified:** nothing ran behind a real login proxy. A torchvision download and the sample dataset ran for real in this mode; Roboflow and Hugging Face were exercised only through recorded clients. The first-training guide was not walked through on a server.

**Alternatives considered:**

- *Let a download go anywhere in the scope.* The shared folders are the operator's; a person writing there changes what everyone trains on.
- *Per-person saved keys.* The right end state for convenience, and it needs the credential store moved under the profile (it is one file per machine today); typing the key per download is safe now.
- *Set and unset `KAGGLE_API_TOKEN` around a Kaggle download.* The client authenticates at import, once per process, and two downloads at once would race on the variable; a lock would serialize every Kaggle download and still leave the first key authenticated.

**Addendum (2026-11-24): split and class names stay inside the target.** `DatasetDownloadRequest.splits` was `list[str]`, so a split with `..` or an absolute path wrote outside `out_dir`, into another person's folder among others. It is now `list[DatasetPart]` (422 before anything is fetched), and `dataset_download._child` refuses, in any mode, a split or class folder that does not resolve strictly below its parent (message `download.name_outside`): a Hugging Face dataset names its own splits and labels, and a `..` or, on Windows, a `\` in one of them climbed the same way. The Hub's split names are all checked before the first image is written. The test that every path of a confined route is typed now also fails on a `list` or `dict` of plain `str`.

## ADR-220 — In server mode the dataset Laboratory reads the scope, its jobs have an owner, and what it writes lands in the person's folder

**Date:** 2026-11-19
**Status:** Accepted
**Extends:** ADR-214 (the dataset scope), ADR-219 (the own folder), ADR-166, ADR-167 (the derived datasets)

**Context:** the Laboratory is 31 routes under `/api/explorer`: open and page a dataset, an integrity scan, and five features that read a dataset and write a new one (re-split, incorporate, merge, COCO export, COCO import). Three things kept it closed on a server. Its request models took paths as plain text. Its background jobs sat in registries shared by the process: a job id was all it took to read a preview, apply it or cancel it, one running job blocked everyone, and the registries kept only the last few jobs of anyone. And every feature writes its result next to its source (`root.parent / dest_name`), which for a shared dataset is the operator's folder, building it out of hard links to the source files.

**Decision:**

- Every path of a Laboratory request is typed: `DatasetDir` for the dataset and for the second folder of a feature (the other dataset of a merge, the new images, the COCO folder), `DatasetPart` for the split names and the group file. The thumbnail route already asked the scope (`_is_inside_allowed_root`).
- A job records `owner`, the profile of the request that registered it (`job_owner`, `""` on a local install), and a registry answers a job only to its owner: `get` returns `None` for anybody else, so status, plan, apply and cancel answer 404. "One job at a time" becomes one per person, and a registry drops the oldest jobs of the person registering, so another person's work never evicts a preview about to be applied.
- The five build functions take `parent`, the folder the result is created in; `None` keeps "next to the source". `derived_jobs.placement(source_root)` decides it per request: nothing on a local install; nothing on a server when the source's parent is inside the person's own folder; otherwise `parent=own` and `link=copy_only`, a link function that always fails, so `link_or_copy` copies every file. The routes check the destination name against that same folder (`destination_parent`).
- The 31 routes join `READY`.
- The page hides the folder dialog's button in the Laboratory where the dialog is closed and lists the accepted folders in its place; the hint under a destination name says, on a server, that the result goes into the person's folder and is a copy when the original is shared.

**Consequences:** on a server a person can inspect a shared dataset and derive their own version of it without the operator and without touching what everyone else reads. Their derived dataset shares no inode with the shared one, so nothing done to either can reach the other. The local tool is unchanged: no owner, one job at a time as before, and the result next to its source with hard links.

**Known limits:** a copy costs the dataset's size in the person's folder, and there is no disk quota. Laboratory jobs run at once, outside the training queue: one per person and kind. The suggested destination name is computed next to the source; for a shared source it can already exist in the person's folder, and the apply then answers 409. The index cache and the jobs live in the process: a restart forgets them, as on a local install.

**Not verified:** nothing ran behind a real login proxy. Exploring, the scan and a re-split of a shared dataset ran for real in this mode; incorporate, merge and the COCO conversions were exercised through their requests and the shared placement rule, not run end to end there. Two people using the Laboratory at the same time was tested with the registries and the test client, not with two browsers.

**Alternatives considered:**

- *Refuse to derive from a shared dataset.* One line, and it removes the main use on a server: the data is the operator's, the split is the person's.
- *Hard-link into the person's folder.* Free on disk, and it ties a person's dataset to files the operator may replace or remove in place; a copy is theirs.
- *Keep one job at a time for the whole server.* It protects the disk, and one person's long analysis would answer "busy" to everyone else with no way to see why.
- *Per-owner registries.* A dictionary of registries keyed by owner; the owner on the job does the same with one lock and one place to look a job up.

**Update (2026-11-24):** `placement(source_root, *other_roots)` takes every source of the result. A merge passes both datasets: when dataset A is the person's own and B is shared, the result is still written next to A, and with `link=copy_only`, so it shares no inode with B. Incorporating images needed no change: it hard-links only the base dataset's files and always copies the new images.

## ADR-221 — In server mode a run leaves the server through its owner: download, and publish with the owner's token

**Date:** 2026-11-19
**Status:** Accepted
**Extends:** ADR-213 (the gate), ADR-140 (the bundle), ADR-171, ADR-174, ADR-175 (the publishers), ADR-176 (DVC)

**Context:** a run trained on a server could be used there (ADR-218) and could not be taken out: the ZIP, the Markdown card and the publishers still answered 404. Reading them found nothing to confine. Each takes a run id and looks the run up in the caller's profile (`find(run_id, profile)`), which on a server is the signed-in person's. Each packager takes its files from inside the run folder and refuses a path outside it and a linked file (`_contained_file`, `_safe_file`). A publisher sends with `read_token(service, profile)`, a file per profile that the token routes, open since ADR-213, already write; no publisher falls back to a key of the machine. DVC is the exception: it runs the server's own `git` and `dvc` on a folder of the server's disk, commits with the identity of whoever operates the machine, and its router already refuses a client that is not the machine itself.

**Decision:**

- Eight routes join `READY`: `GET /api/runs/{run_id}/bundle`, `GET /api/runs/{run_id}/export_md`, and `preview` and `export` of `/api/hub`, `/api/zenodo` and `/api/github`. No handler changes.
- The two DVC routes that were waiting (`preview`, `export`) become `off`, like `prepare`: the feature is about a repository on the machine the page is opened on, which on a server is nobody's.
- The tracker retries (MLflow, W&B, Comet) stay closed: they need the integrations a profile turns on, which a server cannot configure yet.
- The page changes nothing: the buttons appear because the server stops listing the routes, and the DVC panel stays absent.

**Consequences:** a person can download their run and publish it under their own account, and nobody else can do either: the run does not exist for them and their token is never read for another profile. The local tool is unchanged.

**Known limits:** a bundle is built in the server's temporary folder and streamed; several large downloads at once use that much disk for as long as they last. The Markdown card and the `run.json` inside the plain ZIP carry the server's absolute paths (the run folder, the dataset), which the person then holds; the Hub, Zenodo and GitHub packages carry a public record without them (ADR-171, ADR-174, ADR-175). Tokens are stored on the server, per profile, readable by whoever operates it.

**Not verified:** nothing ran behind a real login proxy. The downloads and the three previews ran for real in this mode; no real upload to the Hub, Zenodo or GitHub was made there (the tests record the upload and check whose token it carries).

**Alternatives considered:**

- *Add a second containment check in the routes.* The packagers already refuse what is outside the run folder, for the local tool too; a copy of that rule in the routes would be a second place to keep right.
- *Open DVC against a folder inside the person's profile.* It would still commit as the server's git user and push with the server's credentials and remotes; a person's repository belongs on their machine, fed by the ZIP.

## ADR-222 — In server mode a preview's images are written inside the person's profile

**Date:** 2026-11-19
**Status:** Accepted
**Extends:** ADR-214 (the dataset scope), ADR-215 (the artifact route)

**Context:** three dataset routes were still closed on a server: the augmentation preview, the preprocessing preview and the audit. All three read a dataset; the two previews also write a few PNGs and return their paths, which the page loads through `/api/artifacts/{path}`. Two things were in the way. Their `base_dir` was plain text (a `Path` for the audit). And the previews write into `outputs/preview_cache`, one folder for the process: each render first deletes the PNGs already there, so on a server one person's preview would erase another's, and the artifact route, which in server mode serves a person only what sits under their profile folder (ADR-215), would refuse the images anyway.

**Decision:**

- `base_dir` of the three requests is typed `DatasetDir`. The audit's split paths and group manifest were already confined to the dataset root by `audit_dataset` and `load_group_manifest`, so they stay as they are.
- `dataset_scope.preview_cache(local)` says where a preview writes: `local` (`outputs/preview_cache`) on a local install, and on a server `preview_cache/` beside the person's dataset folder, inside their profile. Both previews use it. The images land where the artifact route already serves their owner from, and a render clears only its owner's earlier ones.
- The three routes join `READY`. The page changes nothing: its preview and audit panels hide by route key (ADR-216) and appear when the server stops listing them.

**Consequences:** on a server a person sees what their augmentation and preprocessing do to an image, and audits a dataset for duplicates across splits, before training. Nothing one person previews is visible to, or erased by, another. The local tool is unchanged.

**Known limits:** the previews and the audit run at once, in a worker thread, outside the queue; an audit hashes every image of the dataset. A preview folder holds one render per kind and class and is never larger than a handful of PNGs, but it is not removed when a profile stops being used.

**Not verified:** nothing ran behind a real login proxy. The three routes ran for real in this mode against a shared dataset, and the images were fetched through the artifact route; the panels themselves were not opened in the page there.

**Alternatives considered:**

- *A cache folder named after the person under `outputs/preview_cache`.* It keeps the previews together, and the artifact route would need a second place it serves a person from; the profile folder is already that place.
- *Return the images inline (base64) on a server.* No file to place, and a different response shape for the page in one mode only.

## ADR-223 — In server mode a person edits their own profile; W&B and Comet use their key, local MLflow is off

**Date:** 2026-11-19
**Status:** Accepted
**Extends:** ADR-213 (the account is the profile), ADR-216 (the page hides closed routes)
**Amends:** ADR-170 (local MLflow), on a server only

**Context:** the trackers a training feeds (MLflow, ADR-170; Weights & Biases, ADR-172; Comet, ADR-200) are switched on in the profile, through `PUT /api/profiles/{slug}`. That route was `off` on a server and `GET /api/profiles` closed: the first takes any slug, the second lists everybody. So nobody could turn an integration on, the three History retries had nothing to resend, and the page worked on a placeholder profile: the name a person typed stayed in the browser. The token routes of W&B and Comet were already open, each bound to the signed-in person's profile.

**Decision:**

- `GET /api/profiles` answers on a server with one profile: the signed-in person's. `PUT /api/profiles/{slug}` answers for that slug only; any other, `default` included, is 404, the answer for a profile that does not exist. `POST /api/profiles` stays `off`: the profile is the account.
- W&B and Comet work as on a local install. The settings are the person's, each export is built in the request that starts the training with the key read from their profile, and the two retries already look the run up inside the caller's models folder. W&B's own local files go to `outputs/wandb/<profile>`, one folder per person.
- Local MLflow is off on a server. Its store is one SQLite file on the server's disk; no route reads it, so the person who turned it on could not open it, and an operator who served it with `mlflow ui` would show every person's parameters to the others. Turning it on answers 422 with the reason, its retry is `off`, and `profile_trackers` leaves it out there even when a profile carried over from a local install has it on.
- The page: on a server the welcome takes the listed profile as the one in use. While its name is still the login (how `ensure_profile` creates it) it asks the name and the greeting and saves them in the profile; later visits go straight in. The profile menu shows the integrations again, without the MLflow switch, which hides by its route key like everything else (ADR-216).

**Consequences:** a person on a shared server follows their runs in their own W&B or Comet project and resends one that failed. Their name and greeting follow them across browsers. Of the routes that waited for a review since ADR-213, none is left closed: what does not answer on a server is `off` by decision.

**Known limits:** an export runs inside the training job, and a retry at once in a worker thread, outside the queue; a stuck W&B call holds the next queued job up to its timeout, as on a local install (ADR-197), and here that job may be another person's. A person who already typed a name in the browser is asked once more, with that name filled in. The key hint in the menu still says the key is saved "on this computer"; on a server that is the server.

**Not verified:** nothing ran behind a real login proxy. No real export to W&B or Comet ran in this mode: the tests use the fake W&B SDK and check whose key a retry sends; Comet's retry is only checked for who reaches it.

**Alternatives considered:**

- *Keep MLflow, with one store per person inside their profile.* Isolated, and still a file nobody but the operator can open; serving `mlflow ui` per person is a second web application to sign people into.
- *A separate `/api/me` route for the settings.* A cleaner name, and a second code path for the page's profile menu, which already speaks `/api/profiles`.

## ADR-224 — The e2e suite runs a second server in server mode, with the browser as the login proxy

**Date:** 2026-11-19
**Status:** Accepted
**Extends:** ADR-133 (the Playwright suite), ADR-212 (server mode)

**Context:** eleven slices opened server mode route by route (ADR-212 to ADR-223). Each was tested with the app object under pytest and checked by hand in a browser behind a stand-in proxy; no browser test ran in this mode, so a change to the welcome, the profile menu or the route keys the page hides could break a shared server and pass CI.

**Decision:**

- `playwright.config.ts` starts a second `visionforge gui` through the same `e2e/serve.mjs`, with `VF_E2E_SERVER_MODE=1`: `--server --trusted-proxy 127.0.0.1 --datasets-dir <work>/shared`, in its own work folder (`e2e/.work-server`) and port (8766). A second project, `server-mode`, points its `baseURL` there and runs `e2e/server-mode.e2e.ts`; the `chromium` project ignores that file. `npm run e2e` runs both, so the CI job does not change.
- The browser plays the login proxy. The server trusts this machine's address, and a context sets `X-Forwarded-User` through `extraHTTPHeaders`: the file's default page and `request` are one account, and a second `browser.newContext` is another. A client with no header checks the 401.
- The file covers the path a person walks, with two accounts on the one server: the refusal without an account; the first visit's name, saved in the profile; the profile menu (W&B and Comet, no MLflow, no other profile); one epoch on the sample dataset written into the person's own folder, found in the History; the other account's empty History and 404 for that run; and a few refusals of the running process (a dataset path outside the folders, another profile, MLflow, the folder picker).

**Consequences:** the server-mode start-up through the real CLI flags, the gate and the page are exercised together on every pull request, about 35 s more on the e2e job. The suite now needs two free ports.

**Known limits:** one training, of classification; the strategies, the evaluation actions, the downloads, the Laboratory, the publishers and the previews are covered by pytest only. The servers start for every run, also when one project is selected.

**Not verified:** a real login proxy. The header comes from the test itself, so this proves what the server does with an account, not that a proxy sets it or strips a forged one.

**Alternatives considered:**

- *A small proxy process in front, as in the manual checks.* Closer to the deployment, and a third process and port for what `extraHTTPHeaders` already does; it would still not be oauth2-proxy.
- *oauth2-proxy with a mock identity provider in CI.* The only way to test the real hand-off; a container stack for the e2e job, left for the deployment test of phase 0.

## ADR-225 — In server mode `--quota-gb` caps each person's folder, checked when an action that writes is asked for

**Date:** 2026-11-20
**Status:** Accepted
**Extends:** ADR-213 (the account is the profile), ADR-214 (a person's own dataset folder)

**Context:** on a shared server every person writes into their own profile folder: runs with their checkpoints, plots, reports, the datasets they download or derive (ADR-219, ADR-220). Nothing limited it, so one person could fill the disk and stop everybody's training. Every earlier server-mode decision listed "no disk quota" among its known limits.

**Decision:**

- `visionforge gui --server --quota-gb N` gives every person the same ceiling, in GiB (shown as GB), on the size of their profile folder, datasets included. Without the option nothing is measured. The option needs `--server` and a value above zero.
- `gui/disk_quota.py` measures a folder (`folder_bytes`: regular files, links not followed) and remembers the figure for 15 s per folder (`DiskUsage`), since the walk is slow on a large dataset. `grows(route)` says which routes write new data, by their last path segment: a training or strategy start, `resume`, `download`, `sample-dataset`, a Laboratory `apply`, and `test`, `gradcam`, `batch_predict`, `export_onnx` of a run. A test pins how many answering routes that is (37), so a new one is a decision.
- `ServerModeGate` checks it before the handler: for a route that grows, when the folder is at or over the ceiling, it answers 507 with the used size and the ceiling, and says to delete runs in the History. Everything else answers as before: reading, opening, downloading, publishing and deleting. After any `DELETE` request the person's figure is forgotten, so freed space counts on the next request.
- `GET /api/session` gains `quota` (`used_bytes`, `limit_bytes`, or null). The profile menu shows it, with a notice when the folder is full.

**Consequences:** an operator bounds what each account can take, and a person who hits the ceiling is told why and what to do, and can do it. The local tool is unchanged.

**Known limits:** it is a check before an action, not a limit on writing: a job that already runs finishes past the ceiling, and several started while under it may pass it together. A figure up to 15 s old can let one more action start. The ceilings are not reserved: a disk smaller than their sum can still fill. A person cannot delete a dataset of their own from the page; only runs. The previews and the tracker stores are not gated (a few files each). One ceiling for everybody; there is none per account.

**Not verified:** nothing ran behind a real login proxy. The walk was not timed on a folder with tens of thousands of files. The symlink test is skipped on a Windows account that cannot create links; it runs on the Linux CI.

**Alternatives considered:**

- *Check inside each handler.* The handler knows what it will write, and thirty-seven of them would each need the call; the gate has one table and one test.
- *Filesystem quotas (XFS project quotas, ZFS datasets).* A hard limit enforced while writing, the only real one; it depends on the operator's filesystem and privileges, and a write that fails mid-training is a worse message than a refusal before it starts. The two can be used together.
- *Refuse by free disk space instead of per person.* Protects the disk, and lets one person take all of it.

## ADR-226 — In server mode at most `--max-actions` heavy actions run at once outside the queue

**Date:** 2026-11-20
**Status:** Accepted
**Extends:** ADR-218 (the actions of a finished run), ADR-219 (downloads), ADR-221 (publishers), ADR-223 (tracker retries)

**Context:** a training waits in the queue (ADR-185). The evaluation actions, a dataset download or audit, the publishers and the tracker retries do their work inside the request, in a worker thread. Each server-mode decision that opened one listed it as a known limit: nothing bounded how many ran together, so a few people pressing Grad-CAM or downloading at once could exhaust memory or the GPU for everybody. Two designs were on the table: turn these actions into queued jobs, or bound how many run at once. The owner chose the bound: queued jobs mean every action's screen follows a job and may wait hours behind a long training.

**Decision:**

- `gui/action_limit.py` names the routes that count (`HEAVY_ROUTES`, eleven): `test`, `gradcam`, `batch_predict` and `export_onnx` of a run; `dataset/download` and `dataset/audit`; the Hub, Zenodo and GitHub exports; the W&B and Comet retries. `ActionSlots` counts the ones running.
- `ServerModeGate` takes a slot before the handler of such a route and gives it back when the response is done, whatever the outcome. With `--max-actions` (default 2) taken, the next one is answered 429 with a message saying the server is busy and to try again. Nothing waits in line. The bound is across everybody, not per person.
- The option needs `--server` and a value of 1 or more. Without `--server` nothing is counted.

**Consequences:** the memory and the threads these actions take are bounded by a number the operator chooses. A person who is refused loses nothing and tries again. The page needed no change: it already shows the server's message where the action was asked.

**Known limits:** the bound does not wait for a training: an action still shares the CPU or the GPU with whatever job is running. One person can hold every slot. The Laboratory jobs and the dataset scan run after their request ends and are not counted; they keep their own bound, one per person and kind (ADR-220). The previews, the sample dataset, the run's ZIP and the reads are not counted. A refused request is not retried by the page.

**Not verified:** nothing ran behind a real login proxy. No measurement of how many of these actions a given machine holds: 2 is a guess for one GPU.

**Alternatives considered:**

- *Queue these actions as jobs.* It would also keep them off a running training; set aside by the owner for its cost in every action's screen.
- *Wait for a free slot instead of refusing.* Friendlier for a few seconds, and a request that hangs behind a ten-minute download with no word to the person.
- *A bound per person.* Fair between people, and four people still run four at once.

## ADR-227 — In server mode a person lists and deletes the datasets of their own folder

**Date:** 2026-11-20
**Status:** Accepted
**Extends:** ADR-219 (a person writes datasets only into their own folder), ADR-225 (the disk quota)

**Context:** the quota of ADR-225 counts a person's datasets, and the page could delete runs only. A folder filled by downloaded or derived datasets left its owner locked out of every action that writes, with the operator as the only way out; ADR-225 listed it as a known limit.

**Decision:**

- `gui/api/own_datasets.py` adds two routes, both `owner` and open on a server. `GET /api/datasets/own` lists the folders directly inside the signed-in person's dataset folder, each with its name, path and size (`folder_bytes`). `DELETE /api/datasets/own/{name}` removes one of them with everything in it and answers with its name.
- What may be deleted is decided by name, not by path: one path segment, a real folder (not a link) whose parent is the person's own folder. Anything else is 404, the answer for a dataset that is not there. The shared folders and other people's are out of reach by construction.
- While a job of that person runs or waits in the queue the delete answers 409: the job may be reading the dataset, and which one it reads is not worth parsing out of every task's config.
- On a local install nobody has an own folder: the list is empty, the delete is 404, and the page does not offer the tab.
- The page: the Datasets sheet gains a third tab, "Meus datasets", on a server. Each row has the size and a delete button that asks in the row itself before deleting. After a delete the list and the session are read again, so the profile menu's disk figure follows. The gate already forgets the person's measured size after any `DELETE` (ADR-225).

**Consequences:** a person at the quota can always get under it on their own. Deleting a dataset does not touch the runs trained on it; actions that read it again (resuming, the default folder of an evaluation) fail for those runs as they would for any dataset that moved.

**Known limits:** there is no trash and no undo. Only whole top-level folders are listed and deleted, not parts of one. A Laboratory job or a preview reading the dataset is not checked, only queued and running trainings. The size of each folder is walked on every listing.

**Not verified:** nothing ran behind a real login proxy. Deleting a dataset of several gigabytes was not timed: the request waits for it.

**Alternatives considered:**

- *Delete by path, typed as `OwnDatasetDir`.* One more place where a path from the browser reaches `rmtree`; a name under a fixed folder cannot point anywhere else.
- *Refuse only when a job reads this dataset.* Kinder, and it needs the dataset path of every task's config, strategies included; the queue is short-lived and the message says what to do.
- *A trash folder emptied later.* An undo, and the space is not freed, which is the reason the person is deleting.

## ADR-228 — The image audit can look for visual near-duplicates across splits

**Date:** 2026-11-21
**Status:** Accepted
**Extends:** ADR-136 (content audit), ADR-140 (the GUI exposes the audit), ADR-153 (dataset inspection)

**Context:** the audit compares SHA-256, so it finds only the same bytes in two splits. The leak that matters more in practice is the same photo saved twice, once as PNG and once as JPEG, resized, or a little brighter: the bytes differ and the test set still sees the training images. The report has carried `visual_similarity_checked: false` since ADR-136, and the explorer, the incorporation, the merge and the re-split say "near-duplicates are not detected" for the same reason. Issue 351 asks for it as the first slice of a dataset analysis.

**Decision:**

- `core/near_duplicates.py` hashes each image with a 64-bit difference hash (dHash): grayscale at 9x8 pixels, one bit per pair of horizontal neighbours, set when the left one is brighter. Two images are near when the hashes differ in at most `MAX_DISTANCE = 6` bits.
- Candidates are found without comparing every pair: the hash is cut into 8 bands of 8 bits, and only hashes that share a band are compared. With 7 or fewer differing bits some band is equal, so no pair within the distance is missed. Images with the same hash are joined directly; the near pairs are joined into groups (union-find), so A near B near C is one group.
- A group is reported when it spans more than one split and holds more than one SHA-256: pairs within a split are not a leak, and byte copies are already in `duplicate_groups`.
- A flat image (one colour) hashes to 0 and is left out: a red and a blue square would otherwise match.
- `audit_dataset(..., near_duplicates=True)` decodes every readable image after the exact pass. An image that verifies but fails to decode (a truncated file) is counted unreadable. The report always carries `near_duplicate_groups` (empty when not looked for), `near_duplicate_max_distance` and `visual_similarity_checked`, which is now true when the search ran. Near groups make `ok` false.
- The option is off by default everywhere: `--near-duplicates` on `visionforge audit-data`, `near_duplicates` on `POST /api/dataset/audit`, and a toggle "Procurar quase-duplicatas entre splits" in the audit panel of the data step. The screen states the distance before the first audit with a constant that a test holds equal to the server's.

**Consequences:** the audit finds re-encoded, resized and re-exposed copies across splits. With the option on it decodes the whole dataset: on 12 000 PNGs of 32x32 pixels (CIFAR-sized, warm disk cache, Ryzen 7 9800X3D) the audit took 6.0 s against 2.9 s for the exact pass alone; large photographs take much longer. The request still runs in one call without progress, as the exact audit did.

**Known limits:** a crop, a rotation, a mirror or another photo of the same object is not near under dHash. Very similar distinct images (frames of a video, a fixed camera) can be joined, so a group is something to look at, not a verdict. The threshold is fixed. The search lives in the data step's audit, not yet in the Laboratory, and has no progress or cache; issue 351 keeps those for the next slices.

**Not verified:** the time on a dataset of full-size photographs and the false-positive rate on a real dataset were not measured; the timing above used synthetic random images, which form no groups.

**Alternatives considered:**

- *Perceptual hash (pHash, DCT) or CNN embeddings.* More robust to crops and colour changes; pHash needs a DCT per image and embeddings need a model and a GPU. dHash needs Pillow only and is enough for the re-encoded and resized copies that cause most leaks.
- *Compare every pair of hashes.* Exact but quadratic: 60 000 images are 1.8 billion pairs in Python. The band index makes it near linear for real data.
- *Turn it on by default.* Decoding every image slows down an audit that today only reads bytes; the researcher asks for it.

## ADR-229 — Live Grad-CAM can show N images of each class, up to 48

**Date:** 2026-11-21
**Status:** Accepted
**Extends:** ADR-162 (live Grad-CAM of fixed validation images)

**Context:** the live gallery explains 6 validation images in all (`LIVE_GRADCAM_SAMPLES`), taking the classes in turn. With more than 6 classes some never show; with two or three, each has two or three images and one look says little about what the model attends to in a class. Issue 349 asks for a number of images per class.

**Decision:**

- `training.live_gradcam_per_class` (0 to 16, default 0). 0 keeps the 6 images of ADR-162, so a saved config and a default run are unchanged. A positive value takes that many images of each class (all of a smaller class), still drawn once with the run's seed and a generator of its own.
- The gallery is capped at `LIVE_GRADCAM_MAX = 48` images: an epoch that writes it costs at most 48 forward and backward passes up to the target layer. The classes are still taken in turn, so past the cap every class shows once before any shows twice.
- The images go through the model in chunks of `LIVE_GRADCAM_BATCH = 16`, so a large gallery of a large model needs no more memory than a small training batch. In eval mode each image's map depends on that image alone, and a test holds the chunked maps equal to one batch.
- As before, only the latest gallery and the best epoch's are kept on disk (`insights/latest`, `insights/best`); no gallery per epoch.
- The Results tab groups the gallery by true class, the classes in name order, each under its name and count.
- The four rules of ADR-162 hold; the test that compares history and weights with and without the gallery runs with 3 images per class too.

**Consequences:** with many classes the researcher sees each one; with few, several images of each. The cost of an epoch that writes the gallery grows with it, up to 8 times the default's.

**Known limits:** the cap and the per-class bound are fixed. The images are chosen once, at random within each class; there is no choosing which ones. Keeping every epoch's gallery is still not offered.

**Not verified:** the gallery of a real training with 48 images, on screen and on a GPU.

**Alternatives considered:**

- *A total instead of a per-class count.* The issue asks per class, and a total already exists in the default.
- *Keep each epoch's gallery.* 48 images per written epoch is disk the run keeps forever, for a comparison the latest and the best already give.
- *No cap.* 16 images of each of 100 classes would be 1 600 backward passes on every written epoch.

## ADR-230 — The image audit reads the layout of every task

**Date:** 2026-11-21
**Status:** Accepted
**Extends:** ADR-136 (content audit), ADR-140 (the GUI exposes the audit), ADR-228 (near-duplicates)

**Context:** the audit walks one folder per split and was offered only in the classification panel. The other tasks keep their images elsewhere: a YOLO root in `images/<split>` or `<split>/images` beside the labels, segmentation in `<split>/<images_subdir>` beside the masks, anomaly in `train/<normal_dir>` and `test/<defect>`, and regression in one `images_dir` that CSVs split. Walking a segmentation split whole would report every all-background mask as a leak; a regression dataset has no split folder at all. Issue 351 asks for the same checks in every task as its second slice.

**Decision:**

- `audit_dataset` takes `split_images`: for each split a list of image paths, or None when the list is missing. A path listed twice in one split is one image. A listed path is resolved like a walked one: outside the root it is reported as escaping, absent it counts as unreadable (`FileNotFoundError`).
- `core/dataset_audit_layouts.py` turns a task's layout into the audit's input: `"folders"` passes the folders through, `"yolo"` resolves each split with `resolve_yolo_split` (the trainer's own resolver; a split neither layout has is reported missing), `"csv"` reads each CSV's `image_column` under `images_dir` (a CSV outside the root or without the column is a 400; an absent CSV is a missing split).
- `optional_splits` names splits training runs without. One that does not exist is listed in `absent_optional_splits`, leaves `image_counts` and is no finding; one that exists is audited like any other. The panels pass `test` for detection, segmentation and regression, whose trainers run without a test set. Classification and anomaly pass none, as before.
- `POST /api/dataset/audit` gains `layout`, `images_dir`, `image_column` and `optional_splits`. The defaults keep every earlier request unchanged.
- Each task panel shows the audit beside its dataset stats, with one line on what it reads: detection the YOLO image folders (not with a `data.yaml` source, whose paths the server does not resolve), segmentation `<split>/<images_subdir>` only, regression the CSVs (a blank optional CSV is left out), anomaly `train_dir` and the whole `test_dir`.

**Consequences:** every task can find byte copies, near copies, unreadable images and missing or empty splits before training. In regression an image listed in two CSVs is reported as a cross-split duplicate whose two paths are the same file.

**Known limits:** a detection dataset given as a `data.yaml` is not audited. The CLI `audit-data` still walks folders only. As in ADR-228, the request runs in one call, without progress or cache; moving the analysis into the Laboratory stays in issue 351.

**Alternatives considered:**

- *Let the browser name the folders for YOLO too.* It does not know which of the two layouts a root uses; the server already resolves it for training.
- *Audit the whole split folder in segmentation and skip mask-named folders.* The mask folder name is configurable; naming the images folder is exact.

## ADR-231 — The Laboratory scan looks for near-duplicates on request and keeps its last report

**Date:** 2026-11-21
**Status:** Accepted
**Extends:** ADR-165 (explorer and its integrity scan), ADR-220 (the Laboratory on a server), ADR-222 (previews inside the profile), ADR-228 (near-duplicates)

**Context:** the Laboratory's integrity scan runs in the background with progress and cancellation, and finds unreadable files, exact copies and cross-split leaks. The near-duplicate search of ADR-228 lives only in the data step's audit. And a finished scan lives only in the page: reopening the dataset, or reloading, meant opening every image again. Issue 351 asks that each analysis run in the background with progress and keep its result.

**Decision:**

- `scan_integrity(..., near_duplicates=True)` adds a second pass over the images the first could read: dHash each one and group the near ones across splits with `near_duplicate_groups` (same threshold and rules as ADR-228). Progress counts both passes (`total` doubles). An image that verifies but fails to decode is reported as corrupted. The report gains `near_checked`, `near_max_distance`, `near_group_count` and `near_groups` (capped at `MAX_REPORTED`); cancelling during the second pass keeps the first pass's findings and leaves `near_checked` false.
- `core/scan_store.py` keeps the last finished report of each dataset as JSON, named by a hash of the dataset root, task and split folders (`scan_key`), next to a signature: a hash of every listed image's split, path, labels, size and modification time (`scan_signature`). A cancelled scan is not kept. A sampled scan is, and says so as before.
- `POST /api/explorer/scan/saved` rebuilds the index, recomputes the signature and returns `{report, saved_at, stale}`; `report` is null when nothing was kept, or when the file is unreadable or of another format.
- The store is `outputs/explorer_scans/` on a local install. On a server it is `explorer_scans/` inside the person's profile, next to their dataset folder, resolved in the request (the worker thread does not see the scope), so a shared dataset scanned by two people keeps two reports and neither sees the other's. The route is `confined`.
- The Overview shows a toggle "Procurar quase-duplicatas entre splits" beside the limit. On opening, it asks for the kept scan and shows it under "Última varredura guardada: <data>", with a warning when stale. A scan started in the page replaces it.

**Consequences:** reopening a scanned dataset shows its last findings at once, and says when they may be out of date. The near search costs a second decode of every image, about twice the scan time. Checking staleness reads the size and time of every listed image on each open of the Overview.

**Known limits:** one report per dataset, the last one: a scan with the near search on is replaced by a later one without it. A change that keeps an image's size and modification time is not seen. The kept reports are never pruned. Detection and classification only, as the Laboratory itself.

**Alternatives considered:**

- *Keep the report inside the dataset folder.* The explorer is read-only by design (ADR-165), and a shared dataset is not the person's to write into.
- *A content hash as the signature.* Exact, but it costs what the scan costs; size and time are what `dataset_fingerprint`'s manifest mode already trusts.
- *Keep every scan.* A history nobody asked for; the last one answers "what did I find here".

## ADR-232 — The Laboratory says whether the splits share the class mix

**Date:** 2026-11-21
**Status:** Accepted
**Extends:** ADR-165 (explorer summary), ADR-168 (stratified re-split)

**Context:** the Overview draws images per class, split by split, but leaves the reading to the eye: with twenty classes nobody sees that one is 3% of training and 12% of test, or absent from validation. A validation or test split with another class mix measures the model on another problem. Issue 351 lists "class distribution per split and whether the division is stratified" as the first classification analysis.

**Decision:**

- `core/split_balance.py` reads the per-class counts the summary already has; no image is opened. Splits with no labeled image are left out.
- Per class: its share of each split's labels (percent), the gap between its largest and smallest share in percentage points, and the splits where it is missing. Classes are listed largest gap first.
- Per split: the imbalance ratio, largest class count over smallest among the classes it holds (None with one class).
- Cramér's V of the split-by-class table, computed by hand (no scipy at import): 0 for the same mix everywhere, 1 when the split decides the class. It is reported as an effect size; no p-value, because with thousands of images every difference is significant.
- A verdict for the headline: `one_split` (nothing to compare), `missing_class` (some class absent from some split), `stratified` (largest gap ≤ 2 points), `close` (≤ 5 points), `uneven`. The thresholds travel in the report (`thresholds_pp`) so the screen states them.
- Shares are over a split's labels, so in detection (an image per class it contains) they sum to 100 as in classification.
- `POST /api/explorer/summary` gains `balance`. The Overview shows a card "As divisões têm a mesma mistura de classes?" under the per-class bars, with the verdict, the missing classes, the five largest gaps, V, the imbalance per split and the full table behind a toggle; an uneven division points at Re-split, which stratifies by class (ADR-168).

**Consequences:** a non-stratified division, a class missing from validation and a heavily imbalanced split are stated in words on opening the dataset, at the cost of a few arithmetic operations per class.

**Known limits:** the thresholds are fixed rules of thumb, not derived from the dataset's size: a class of 10 images moves several points by chance. Classes absent from every split are not listed. Classification and detection only, like the Laboratory.

**Alternatives considered:**

- *A chi-square test with its p-value.* It answers "is there any difference", which on a real dataset is always yes; the gap in points and V say how large it is.
- *Compute it in the browser.* The numbers belong with the other dataset analyses, tested by hand-computed cases on the server.

## ADR-233 — The integrity scan describes the boxes of a detection dataset

**Date:** 2026-11-21
**Status:** Accepted
**Extends:** ADR-165 (explorer and integrity scan), ADR-231 (kept scan)

**Context:** issue 351 lists the detection analyses done before training: boxes per image, box size (small, medium, large), aspect ratio, a heatmap of positions, classes seen together, invalid boxes and images with no box. The Overview counted boxes per class and nothing else; a dataset of tiny objects, or one whose objects always sit in the middle, or one with rows the trainer silently drops, looked the same as any other.

**Decision:**

- `core/box_stats.py` tallies one image at a time (`BoxTally.add(split, rel, label, size)`) and reports JSON. It reads the label file; it never writes.
- It runs inside the integrity scan, for detection only: the scan already opens every image to learn its size, and COCO's size buckets need the area in pixels, which a normalized YOLO row cannot give alone. The report gains `boxes` (None for classification). The scan's sampling, cancellation, progress and kept result (ADR-231) apply unchanged.
- Box size by COCO's rule on the pixel area: small below 32², medium below 96², large above; also per class. These are the buckets of `mAP_small/medium/large`, so the counts say which of those numbers rests on enough boxes.
- Boxes per image in bins 0, 1, 2–3, 4–7, 8–15, 16–31, 32+; mean and maximum. Images with no box are split into no label file and an empty label (or one with only invalid rows).
- Aspect ratio of the box in pixels; a 10×10 grid of box centers; the ten most frequent pairs of classes in the same image.
- A row is a box when it is five finite numbers with an integer class (the trainer's five-field rule), the class is below the count `data.yaml`/`classes.txt` declares (none declared: no range check), width and height are positive, and the box stays inside the image with a 1% tolerance for the rounding of labeling tools. Otherwise it is counted by reason (`malformed`, `class_out_of_range`, `zero_area`, `outside_image`) and the first 200 are listed with file and line.
- A box of an image that does not open is counted and placed on the heatmap but has no pixel size (`unsized`).
- The screen shows a card "Caixas" under the scan's results, with bars for size, boxes per image and aspect, the heatmap, the pairs, a per-class size table and the invalid rows behind toggles.

**Consequences:** the scan of a detection dataset reads one more small text file per image. Box statistics appear only after a scan; the summary stays free of image I/O.

**Known limits:** in a sampled scan the statistics describe the sample. The edge tolerance and the bins are fixed. Pairs count images, not boxes, and only the top ten are kept. Classification and detection only, like the Laboratory.

**Alternatives considered:**

- *Compute from the labels alone in the summary.* Fast, but sizes would be in fractions of the image, not COCO's pixel buckets, and the aspect ratio would be wrong on non-square images.
- *A separate job.* It would open every image a second time for the size the scan already reads.

## ADR-234 — The masks of a segmentation dataset are analyzed before training

**Date:** 2026-11-22
**Status:** Accepted
**Extends:** ADR-037 (segmentation), ADR-165 (explorer jobs), ADR-231 (kept results)

**Context:** issue 351 lists for segmentation the share of pixels per class, the coverage of each mask, tiny regions, masks whose size differs from the image and class ids outside the expected range. The Data step paired images and masks by stem and sampled class ids from 20 train masks; nothing measured the masks themselves. A dataset whose masks hold an id past `num_classes`, or that is 99% background, trained without a word.

**Decision:**

- `core/mask_stats.py` pairs images and masks by stem per split (the trainer's rule and extension lists, copied so it does not import torch), opens each pair and tallies: images without a mask and masks without an image; color (RGB/RGBA) masks, which the trainer refuses; masks whose size differs from the image; the pixel count of each id per split; ids that are neither `0..num_classes-1` nor `ignore_index`; the coverage of each mask (share of pixels that are neither class 0 nor void; class 0 as background is the usual convention and the screen says so); and connected regions of one class (`scipy.ndimage.label`, 4-connectivity) below 0.01% of their image.
- Past 5 000 pairs it reads an evenly spaced sample and says so, like the integrity scan. Cancelling keeps what was read.
- It runs as a background job in the explorer's registry: `ScanJobs.start_work` takes any work with a token and a progress callback, the integrity scan becomes one such work, and status and cancel go through the existing `/explorer/scan/{job_id}` routes. One job per person at a time, as before.
- `POST /api/explorer/masks` starts it; `POST /api/explorer/masks/saved` returns the last finished result. Results go to the scan store (ADR-231) under a key of the folders, the class count and `ignore_index` (another class count is another analysis), with a signature of every pair's paths, sizes and modification times; a change marks the result stale.
- On a shared server both routes are `confined`, like the scan.
- The Segmentation panel shows "O que as máscaras contêm" under the image audit, with the per-class pixel table per split, coverage bars, the tiny-region count and the files found behind a toggle.

**Consequences:** each pair's mask is decoded once and labelled once per class it holds; on large masks with many classes that is the slow part. Results survive a restart of the GUI.

**Known limits:** the tiny threshold is a fixed share of the original image, not of the training size. Class 0 is taken as background for coverage. A color mask is reported, not converted. Regression and anomaly have their own analyses still to come.

**Alternatives considered:**

- *Extend the image audit.* It is synchronous and its report is shared by every task; mask statistics would grow it for one task.
- *Open segmentation in the Laboratory.* The Laboratory's index, gallery and tools are built around classes per image; a mask analysis fits the Data step, where the class count and `ignore_index` are known.

## ADR-235 — The targets of a regression dataset are analyzed before training

**Date:** 2026-11-22
**Status:** Accepted
**Extends:** ADR-036 (regression), ADR-234 (analysis jobs)

**Context:** issue 351 lists for regression a histogram of each target per split, outliers, missing targets and the correlation between targets. The Data step reported count, mean, min and max of the first 50 000 rows and checked the images of the first 500. A validation split whose targets sit in another range than training, a column with a few "n/a" cells, or two targets that are the same quantity in two units went unnoticed until training failed or reported a misleading score.

**Decision:**

- `core/target_stats.py` reads every split's CSV in full and, per target and split, reports the numeric count, blank or non-numeric cells, mean, sample standard deviation, inclusive quartiles, minimum and maximum, and a 20-bin histogram over the target's range across all splits, so the splits share bins and can be drawn on one axis.
- Outliers by Tukey's rule with k = 1.5, on fences from the training split (the split the model learns from), with the CSV line and the image of each.
- The shift of a split: (split mean − train mean) / train standard deviation. The screen warns from 0.5 standard deviations; it is a size, not a test.
- Pearson's r for every pair of targets, over the rows where both are numbers, all splits pooled; None with fewer than three pairs or a constant side.
- Every listed image is checked for existence; the first 50 missing images and the first 50 bad cells are listed with their line.
- It runs as an explorer job through `ScanJobs.start_work` (ADR-234): `POST /api/explorer/targets` and `POST /api/explorer/targets/saved`, `confined` on a shared server. The kept result's signature is the CSVs' size and modification time; a CSV path or `images_dir` outside the dataset is refused with 400 before the job starts.
- The frontend's job handling (start, poll, cancel, kept result) moved into `useAnalysisJob` and `AnalysisJobCard`, shared by the mask analysis and this one.

**Consequences:** a CSV of a million rows is held in memory as floats per target; at the sizes of image datasets that is small. Checking every image is one stat per row.

**Known limits:** the kept result does not notice an image added or removed without a CSV change. Fences come from training only; with no training CSV there are no outliers. The 0.5 sd threshold is a rule of thumb.

**Alternatives considered:**

- *Extend the existing `/regression/dataset/stats`.* It answers synchronously with row and image caps that keep the Data step fast; the full read belongs in a job.
- *A Kolmogorov–Smirnov test per split.* With thousands of rows every difference is significant; the shift in standard deviations says how large it is.

## ADR-236 — An anomaly dataset is analyzed before training

**Date:** 2026-11-22
**Status:** Accepted
**Extends:** ADR-038 (anomaly), ADR-234 (analysis jobs), ADR-228 (near-duplicates)

**Context:** issue 351 lists for anomaly detection the count of normal and anomalous images, an anomalous image in the training folder and the consistency of resolution. The Data step counted images per folder. A defect photo filed under `train/good` teaches the model that the defect is normal, and nothing pointed at it; a test split in grayscale against an RGB training split, or a defect kind with three images, went unnoticed.

**Decision:**

- `core/anomaly_stats.py` lists the MVTec layout (`train/<normal>`, `test/<normal>`, `test/<defect>`, optional `ground_truth/<defect>`), opens every image (an evenly spaced sample past 20 000) and reports images per group, extra folders under train (training reads only the normal one), the most common sizes and color modes per group and the groups whose most common size or mode differs from training's.
- An anomalous training image cannot be detected without labels; the closest observable sign is a training image that looks like a test defect image. Each image gets the 64-bit dHash of the near-duplicate search (ADR-228) and every training image is compared with every defect image (vectorized XOR and popcount); a distance within `near_duplicates.MAX_DISTANCE` is reported with the pair. The same comparison finds normal test images that look like a training image, where the test measures memory. Flat images (hash 0) never match.
- A defect kind with fewer than 10 test images is pointed at: one image moves its AUROC by several points.
- With `ground_truth/`, defect images without `<stem>_mask.*` (or `<stem>.*`) are counted.
- It runs as an explorer job (ADR-234): `POST /api/explorer/anomaly` and `POST /api/explorer/anomaly/saved`, `confined` on a shared server; the kept result's signature covers every image's path, size and modification time.
- The three per-task analyses now share one route helper for the dataset folder.

**Consequences:** each image is decoded twice (size and dHash) — at MVTec sizes, seconds. The all-pairs comparison is train × defects; at tens of thousands each it is the slow part.

**Known limits:** a defect that does not resemble any test defect photo is not found; this looks for the same photo, not the same kind of defect. The 10-image threshold is a rule of thumb.

**Alternatives considered:**

- *Score the training images with a trained model.* That needs a run; it belongs with the suspicious-label analysis that depends on one.
- *Reuse the image audit's near-duplicate pass.* It groups across splits without knowing which side is a defect; here the direction (train against defects) is the finding.

## ADR-237 — Suspicious labels after a classification run

**Date:** 2026-11-22
**Status:** Accepted
**Extends:** ADR-041 (batch prediction), ADR-218 (run actions on a server)

**Context:** issue 351 asks, after training, for the images whose label the model disputes with high confidence, and for the classes the model takes for each other. A mislabeled image costs twice: it teaches the wrong class and it counts as a model error in evaluation. Finding it by hand means looking through every misclassification.

**Decision:**

- `core/label_suspects.py` implements confident learning (Northcutt, Jiang and Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels", JAIR 70, 2021) in its simplest form: the threshold of class j is the mean probability the model gives j on the images labeled j; an image counts toward the most probable class among those over their threshold; it is a suspect when that class is not its label. Suspects are ranked by the probability left to the given label. The confident joint (given × confident counts) gives the pairs of classes the model confuses.
- Deviation from the paper: each threshold is capped at 0.95. A network trained to saturation gives its classes a mean self-confidence near 0.999; found on the first real run, an image labeled `class_b` that the model called `class_a` with 0.9988 never passed `class_a`'s 0.9994 threshold. Below the cap, calibrated models behave as in the paper.
- The probabilities come from the run's checkpoint through `BatchPredictionBlock`, the inference `/runs/{id}/batch_predict` already uses (binary runs: the single sigmoid becomes two columns). The label of an image is its class folder; images outside a known class folder are skipped. The class order is `run.json`'s `class_names`.
- `POST /api/runs/{id}/label_suspects` (body: which splits, validation and test by default) predicts each split, writes `<run_dir>/label_suspects/<split>.csv` and `report.json`, and returns the report; `GET` returns the kept one. Both are `owner` on a shared server: they read the run's own dataset from its `run.json`. Refused with 400 for any other task, an unknown split, or a run without class names or checkpoint.
- The run detail shows a "Rótulos suspeitos" tool for classification runs: split toggles (training marked in-sample), the suspects with both probabilities, and the confused pairs.

**Consequences:** the action runs inference over every image of the chosen splits in the request, like batch prediction; on a large dataset it takes minutes on CPU. Probabilities on the training split are in-sample and the screen says so; out-of-fold probabilities (the paper's setting) would need a K-fold run, which this does not require.

**Known limits:** single-label classification only. The cap is a fixed rule of thumb. The request is synchronous (no progress bar), as `batch_predict`.

**Alternatives considered:**

- *Depend on cleanlab.* A new dependency for twenty lines of arithmetic that are tested here by hand.
- *List every misclassification.* That is the confusion matrix already in the run; the thresholds keep only the images the model is confidently sure about.

**Addendum (2026-11-24): on a server.** `POST /api/runs/{id}/label_suspects` reads the run's stored dataset like `/resume` (ADR-218): `confine_stored_config` validates the stored config, scopes its outputs and calls `confine_config`, so a dataset that left the shared folders is refused with 422 before any prediction. The action writes CSVs and `report.json` into the run and runs inference in the request, so it counts for the quota (`disk_quota._GROWING`, 507) and for the action limit (`HEAVY_ROUTES`, 429). A run without `data.base_dir` and a damaged `report.json` answer 400 with a message (`suspects.no_base_dir`, `suspects.report_unreadable`) instead of 500.

## ADR-238 — The integrity scan measures brightness, contrast and sharpness

**Date:** 2026-11-22
**Status:** Accepted
**Extends:** ADR-165 (explorer and integrity scan), ADR-231 (kept scan), ADR-233 (box statistics in the scan)

**Context:** issue 351 asks for images out of pattern in brightness, contrast or sharpness: an underexposed or blown-out frame, a blank image, a photo out of focus or upscaled from a thumbnail. They rarely break training, but they are usually capture errors or images from another source, and one in a test split moves the metric.

**Decision:**

- `core/image_quality.py` decodes each image as 8-bit grayscale (PIL's `L`), shrunk to at most 256 px on its longest side so sizes compare and a large dataset stays cheap (JPEG decodes at a reduced scale through `draft`). Brightness is the mean gray level (0 to 255); contrast, the standard deviation; sharpness, the variance of the 4-neighbour Laplacian, the usual focus measure (Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy: a comparative study", ICPR 2000).
- An outlier is outside Tukey's fences (1.5 × IQR) over all scanned images, so a dataset that is dark by nature is not flagged as a whole. Four kinds: too dark and too bright (brightness), flat (low contrast), blurry (low sharpness). High contrast and sharpness are not defects and are not flagged. Sharpness spans orders of magnitude, so its fences are taken on `log1p`. The IQR is floored per metric (8 gray levels, 4 gray levels, 0.5 in `log1p`), so a dataset of near-identical images does not flag every small difference.
- It runs inside the integrity scan, for classification and detection, on the first pass, which already opens every image for its size (ADR-233). The report gains `quality`: per metric the quartiles, the fences and the median per split; per kind the count and up to 50 images, most extreme first.
- An image that passes PIL's `verify()` but fails to decode is now reported as corrupted in the first pass, and left out of duplicates, sizes and boxes. Before, it was counted as readable there and only the optional near-duplicate pass (ADR-231) reported it.
- The Overview shows a card with the table and, for each kind found, thumbnails of the images with their value.

**Consequences:** the first pass now decodes every image instead of reading its header. Measured on 1000 random-noise 640×480 JPEGs (the worst case for decoding), warm disk cache, CPU: 0.54 s without the measure, 3.47 s with it, about 3 ms more per image, so roughly a minute more at the default limit of 20 000 images. The scan runs in the background with progress, as before. The scan's sampling, cancellation, progress and kept result apply unchanged; a scan kept before this has no `quality` and shows no card until scanned again. A dark or blown-out image often also shows as flat and blurry, since it loses its tones; the screen says so.

**Known limits:** grayscale only, so a color cast is not seen. The fences are relative to the dataset: a dataset where most images are blurry flags the sharp ones as normal and nothing as blurry. Segmentation, regression and anomaly datasets have their own analyses (ADR-234 to ADR-236), which do not measure this yet.

**Alternatives considered:**

- *Absolute thresholds (e.g. brightness below 30).* What is dark depends on the domain; X-rays and night footage would be flagged whole.
- *A separate analysis job.* It would open every image a second time; the scan already does.

## ADR-239 — Similar classes are compared before training

**Date:** 2026-11-22
**Status:** Accepted
**Extends:** ADR-165 (explorer), ADR-231 (kept scan), ADR-234 (analyses as explorer jobs)
**Related:** ADR-237 (suspicious labels after a run)

**Context:** issue 351 asks which classes look alike before any training. ADR-237 answers it after a run, from the run's own model; before training there is no model of the dataset, and pixel statistics (color, size) say nothing about what an image shows.

**Decision:**

- `core/class_similarity.py` describes each image with torchvision's resnet18 pretrained on ImageNet, last layer removed (512 features, the weights' own preprocessing: resize 256, center crop 224, ImageNet normalization), on the CPU so a training running on the GPU is not disturbed. At most `per_class` images per class (200 by default, 2000 at most), evenly spaced from the first to the last over all splits.
- For each described image, its nearest other image by cosine similarity, in any split. Per class, the share whose nearest image is of the same class. Per pair of classes A, B: the share of A's images whose nearest image is a B and the reverse (what a 1-nearest-neighbour classifier on these features would confuse), and the cosine between the classes' mean features. A pair is flagged when either share reaches 10%. The images where two classes meet are listed with their nearest image of the other class, most similar first, for every listed pair and not only the flagged ones: a single misfiled image is a small share (1 of 17 on the first check) and would otherwise stay hidden.
- The similarity matrix is computed in blocks of 1024 rows, so memory stays at 1024 × N.
- It runs as an explorer job (`POST /api/explorer/classes`, status and cancel through `/scan/{job_id}`) and the last finished result is kept under the integrity scan's signature (`POST /api/explorer/classes/saved`). Classification only; `confined` on a shared server like the other analyses.
- The Laboratory's Overview shows a "Classes parecidas" card for classification datasets.

**Consequences:** the first run loads the ImageNet weights from torchvision's cache, downloading them (about 45 MB) when they are not there; offline without them the job fails with torchvision's error. Tests use random weights in place of the download (ADR-010). Measured on a Ryzen 7 9800X3D CPU: 800 JPEGs of 640×480 (4 classes, 200 each) in 10.4 s, about 13 ms per image, so a 10-class dataset at the default 200 per class takes under half a minute.

**Known limits:** the features come from ImageNet, not from the dataset: classes that differ by a detail ImageNet ignores (two defect types on the same part, two similar species) can look alike here and still be learned apart, and the screen says so. The 10% threshold is a fixed rule of thumb. Exact copies across classes are the integrity scan's label conflict (ADR-165), not repeated here.

**Alternatives considered:**

- *Color and size statistics per class.* Cheap, but two classes photographed under the same light would always look alike.
- *A short training run.* Closer to the real task, but minutes of GPU before the user has decided to train; ADR-237 covers the after-training case.
- *CLIP or DINOv2 features.* Better general descriptions, but a new dependency or a larger download; resnet18's ImageNet weights are already the default backbone here.

## ADR-240 — Segmentation, regression and anomaly datasets are measured for brightness, contrast and sharpness

**Date:** 2026-11-22
**Status:** Accepted
**Extends:** ADR-238 (the measure in the integrity scan), ADR-234 (masks), ADR-235 (targets), ADR-236 (anomaly)

**Context:** ADR-238 measures brightness, contrast and sharpness in the Laboratory's integrity scan, which only opens classification and detection datasets. Its known limits named the other three tasks; issue 396 asks for the same measure there.

**Decision:**

- The same `QualityTally` (ADR-238) runs inside each task's own analysis, so no image is opened by a second job. `QualityTally.measure` measures one file and says whether it decoded.
- Segmentation: the mask analysis measures each pair's image (not its mask), per split. An image that does not decode counts as unreadable, as a mask that does not open already did.
- Anomaly: the analysis measures each image it already hashes, per group (training normals, test normals, each defect kind). The fences are taken over all groups together, so a defect kind photographed darker than the normals shows as dark: that is a difference between the groups, which the screen's medians per group make visible.
- Regression: the target analysis read only the CSVs. It now also measures the listed images that exist, once per split even when several rows name the same one, on an evenly spaced sample past 20 000. Progress counts the CSVs, then the CSVs plus the images. The report gains `images_listed`, `images_sampled` and `images_unreadable`.
- Each task's data step shows the same "Brilho, contraste e nitidez" card next to its analysis, with the thumbnails. On anomaly the groups are named as the rest of that screen names them, training first and defect kinds last.

**Consequences:** the mask and anomaly analyses decode each image once more at a reduced size (about 3 ms per image, ADR-238). The target analysis, which read only text, now opens every listed image and takes longer by that cost. Results kept before this have no `quality` and show no card until run again.

**Known limits:** Tukey's fences come from the quartiles, so when a quarter or more of the images share a defect the normal range widens around them and they stop being flagged; found while writing the tests, where three planted images out of eleven left the flat one unflagged. The regression result is marked stale by its CSVs only (ADR-235): replacing an image file without touching the CSVs does not mark it stale.

## ADR-241 — Anomaly's per-epoch AUROC is named `test_auroc`

**Date:** 2026-11-23
**Status:** Accepted
**Amends:** ADR-208 (anomaly labels), ADR-038 (anomaly)
**Relates to:** ADR-013 (`run.json`), ADR-092 (resume), ADR-197 (tracker export)

**Context:** the anomaly trainer scores every epoch on the labelled test split (`data_module.test_loader()` in `core/anomaly_trainer.py`) and wrote that AUROC as `val_auroc`. There is no validation split in the task. Nothing leaks into the selection: the checkpoint is the epoch with the lowest `train_loss`, and the AUROC is only reported. ADR-208 fixed the labels on screen and kept the key, because renaming it would break reading the runs already saved. The key is still read outside the interface, though: in the `run.json` a researcher opens, in the columns a tracker shows, in the event stream. A reader of any of those is told the curve is a validation curve.

**Decision:**

- Writers use `test_auroc`: the `AnomalyEpochResult` field, the `run.json` `history` records, the `epoch_end` event, the resume snapshot and the trackers that receive the history (MLflow, W&B, Comet). The TensorBoard tag goes from `auroc/val` to `auroc/test`. The final `metrics` keep their names (`auroc` is the best epoch's, `test_auroc` the test of the saved checkpoint).
- Compatibility rule: a history saved before this ADR is read under the new name, and the file on disk is not rewritten. Each side has one normalizer that knows the old name. On the server, `core/history_keys.py` (`upgrade_run_history`, `upgrade_anomaly_epoch`) is applied by the run detail (`runs_routes._build_run_detail`, which also feeds the comparison), by the tracker export and its retry (`run_trackers.export_run_record`, `tracker_exports.retry_record`) and by resume (an old `resume.pt`). In the page, `lib/history-keys.ts` (`upgradeRunHistory`) is applied in `fetchRunDetail`. When a record carries both names, `test_auroc` wins.
- Only anomaly runs are renamed. A researcher's own task (ADR-058) streams `val_<metric>` for a metric it measured on its validation split, so its `val_auroc` keeps its name.
- The curve keeps its label, "AUROC (teste)" / "AUROC (test)", now under the key `test_auroc`.

**Consequences:** a new anomaly `run.json` and an old one plot, compare and export the same series. `run.json` has no version of its own (ADR-039 versions the config), so the rule lives in the two readers instead of a migration. The exports that copy the record as it is (the experiment bundle, Zenodo, DVC, the Hub card, the Markdown report) keep the name each run was saved with. An old run already sent to MLflow or W&B keeps `val_auroc` there until it is sent again. The other two per-epoch anomaly fields, `val_image_f1` and `val_threshold`, are also measured on the test split (and the threshold on the training normals) and still carry `val_`; the screens label them, as ADR-208 does.

**Alternatives considered:**

- *Keep the key and fix only the labels (ADR-208).* The labels were right, and the file and the trackers still said validation.
- *Rewrite the saved `run.json` files on read.* It edits a research record behind the researcher's back, and a run folder may be read-only or copied elsewhere.
- *Rename `val_image_f1` and `val_threshold` in the same change.* Same reasoning, a larger surface; left for a separate decision.

## ADR-242 — A K-fold can score each fold at its last epoch

**Date:** 2026-11-24
**Status:** Accepted
**Relates to:** ADR-039 (`schema_version`), ADR-158 (an ignored field bumps the version), ADR-111 (stopped folds), ADR-123 (K-fold `run.json`)

**Context:** the classification K-fold reloads each fold's `best_model.pth`, the epoch with the lowest `val_loss` on the fold's validation split, and scores it on that same split. The choice and the score read the same data, so the fold mean is optimistic. Issue #212 made the LaTeX caption, the README and `docs/dev/VALIDATION.md` say so. The maintainer (STAT-7) asked for a way to get a fold score with no selection on the fold.

**Decision:**

- `cross_validation.select_on: Literal["best", "last"] = "best"`. `best` is the behaviour before this ADR. `last` scores the weights the fold's model holds after its final epoch and does not load `best_model.pth`. Classification only: the regression and segmentation K-folds (`_cv_common`) take no `CrossValidationConfig` and keep scoring the best epoch.
- Every fold record carries `select_on` and `selected_epoch` (the epoch scored: `best_epoch` for `best`, the last epoch run for `last`; `null` for a fold that scored nothing), in `metrics.fold_results` of the K-fold `run.json` and in `cv_summary.json`. The group report and `cv_summary.json` also carry `select_on` at the top.
- The LaTeX K-fold caption ends with the sentence of the rule: the #212 sentence for `best`, "Each fold is scored at its last trained epoch, with no epoch selected on its own fold's validation split" for `last`. A report without `select_on` gets the `best` sentence, which is what it did.
- `CURRENT_SCHEMA_VERSION` goes to 3. The v2 → v3 step changes no shape: the default keeps a v2 K-fold scoring as written. An older build refuses a v3 config instead of reading `select_on: last` and scoring the best epoch without saying so, the reasoning of ADR-158.
- The GUI K-fold card gains the field (pt "Época por fold", en "Epoch per fold") with its explanation.

**Consequences:**

- A v2 config or a saved `run.json` loads as before and scores the best epoch. Configs and `run.json` files written from now on carry `schema_version: 3`, and builds before this one refuse them, as with ADR-158. Every task config shares the number.
- `last` removes the epoch selection, not every use of the fold's validation split. With `training.early_stopping_patience` above 0 (the default is 0), the epoch where training stops is still decided by that split, and a `plateau` scheduler still reads it. The caption says "last trained epoch", which stays true in both cases.
- The `best_model.pth` in each fold folder is still the best epoch. The weights `last` scores are not saved apart; rerunning the same config reproduces them.
- `best_val_loss` in each fold record, and its mean in the K-fold `run.json`, remain the best validation loss of the fold, under either rule.
- The classification K-fold writes no LaTeX table today (only the task K-folds go through `write_group_summary`); the caption rule applies wherever a K-fold report is turned into one.
- Tests that mock a fold's `TrainResult` now set `best_epoch`, as the real one does.

**Alternatives considered:**

- *Nested cross-validation.* Removes the bias and keeps the selection, at K times the training. Larger than the problem asked for; still open.
- *Save `last_model.pth` and reload it.* Same score, one more file per fold, and a second checkpoint path to keep in step with the trainer. The weights are already in memory when the fold is scored.
- *Turn off early stopping under `last`.* It would change a training setting the config states; documenting it leaves the choice to the researcher.
- *Add the field without a version bump.* An older build would ignore it and score the best epoch while the config says `last`.

**Addendum (2026-11-24): regression and segmentation.** The first Decision bullet kept these two K-folds on the best epoch; that no longer holds. `TaskCvRequest.select_on: Literal["best", "last"] = "best"` is a field of the request, not of the task config, because these K-folds take their options from the request (API body, CLI `strategy` block, GUI card) and never had a `cross_validation` section. The runners (`run_regression_cross_validation`, `run_segmentation_cross_validation`) take `select_on` and hand it to `_cv_common.run_folds`; `last` scores the model as `fit` left it and skips the `best_model.pth` reload, and each `FoldResult` carries `select_on` and `selected_epoch` (`best_epoch` or `total_epochs`; `null` for a fold that scored nothing). The group report and `cv_summary.json` carry `select_on` at the top, so the LaTeX caption rule above applies to them. No `schema_version` bump: no config or `run.json` shape the older builds read changes, and an older server ignores an unknown request field, so a client that sends `select_on: last` to one gets a best-epoch score without a warning; the report of that server has no `select_on`, which the caption reads as `best`. The Consequences above hold for these tasks too.

**Addendum (2026-11-24): `last` with early stopping is refused.** The maintainer decided that `select_on: last` with `early_stopping_patience > 0` is an error, not a warning: early stopping picks the final epoch from the fold's own validation split, so the fold would still be scored at an epoch that split chose. `utils.config.check_select_on_early_stopping` raises it (message `config.select_on_last_early_stopping`, PT and EN) from three places: the `ExperimentConfig` validator (classification K-fold), the `TaskCvRequest` validator (regression and segmentation through the API and the CLI, a 422 before anything is queued), and `_cv_common.run_folds` (a direct Python call). `best` with early stopping and `last` without it are unchanged. The `plateau` scheduler also reads the fold's validation loss to lower the learning rate; it was not part of the decision and is still accepted with `last`, which the caption does not mention.

**Addendum (2026-11-24): `last` with the plateau scheduler is refused too.** The maintainer extended the refusal to `scheduler.kind: plateau`: `ReduceLROnPlateau` lowers the learning rate from the fold's own validation loss, so the last epoch's weights still depend on that split. The helper became `utils.config.check_select_on_last(select_on, *, patience, scheduler_kind)` (new message `config.select_on_last_plateau`, PT and EN), called from the same three places. `cosine`, `step` and `none` stay accepted with `last`: they follow the epoch count only.
## ADR-243 — Ultralytics is a required dependency again (amends ADR-195)

**Date:** 2026-11-24
**Status:** Accepted
**Amends:** ADR-195 (reverts its `detection` extra), ADR-106 (`ultralytics` is back in its list)

**Context:** ADR-195 made `ultralytics` (AGPL-3.0) the optional `detection`
extra (SEC-11, #223), so a plain `pip install visionforge-studio` trained
detection only with the `torchvision` backend, and YOLO / RT-DETR needed a
second install line. On 2026-11-24 the maintainer decided to revert it, after
being reminded of ADR-195: the default detection backend should work from the
one install, and the install stays simpler without a second line to know
about. ADR-243 skips 242 on purpose: another change in flight reserved it.

**Decision:**

- `ultralytics>=8.3` is back in `[project].dependencies`. The `detection` extra
  stays, empty, like the six ADR-106 kept: an install line saved with
  `[detection]` keeps resolving instead of failing.
- A missing `ultralytics` now means a damaged install, as it did before
  ADR-195. Training, testing, ONNX export and the submit of a
  `backend: ultralytics` config still fail with `detection.ultralytics_missing`
  (pt/en), which now names the reinstall line of `missing_package_hint`
  instead of the extra.
- `GET /api/system/info` no longer reports `ultralytics_installed` or
  `detection_install_command`, and the detection form drops the install line
  that used them. The "sem AGPL" / "no AGPL" badge on the `torchvision` backend
  stays: it is still true, and still the backend to pick to avoid AGPL code.
- The install smoke installs one variant again, the CI jobs install `[dev]`
  instead of `[dev,detection]`, the wheel smoke no longer checks that
  `ultralytics` is absent, and the license gate walks `[project].dependencies`
  without extras.
- `THIRD_PARTY_NOTICES.md`, the README and `docs/INSTALL.md` say that
  `pip install visionforge-studio` installs `ultralytics`, `ultralytics-thop`
  and `ultralytics-platform`; their description of what the AGPL-3.0 asks is
  the text ADR-195 left there, unchanged.

**Consequences:** every install carries AGPL-3.0 code again, the situation
ADR-195 described: an organisation that does not accept the AGPL cannot adopt
VisionForge, even for classification, without Ultralytics' Enterprise License
(#223). The packages already published (0.22.x) keep the extra; the change
ships with the next release.

**Alternatives considered:**

- *Keep ADR-195.* The maintainer's call was that the simpler install is worth
  more than a plain install without AGPL code.
- *Drop the `detection` extra.* The empty names exist so an install line saved
  in a requirements file or a README elsewhere keeps resolving (ADR-106); the
  same holds for `[detection]`.

## ADR-244 — Third-party run exporters through the `visionforge.exporters` entry-point

**Date:** 2026-11-24
**Status:** Accepted
**Relates to:** ADR-005 (trust boundary is the user's filesystem), ADR-058 (researcher tasks), ADR-170 (MLflow export)

**Context:** the AUDIT-2026-11-09 section 6 listed an entry-point for third-party exporters as a larger bet without a task. The seven built-in exporters (MLflow, W&B, Comet, Zenodo, GitHub, DVC, Hugging Face Hub) each have a GUI route and a page in `docs/gui/`; a lab that wants an internal registry or a format of its own has to fork the package. The review `product-competitive` also warned that each integration inside the package adds maintenance against a third party's API, so the way out for the next ones is to live outside it.

**Decision:**

- A package declares an object under `[project.entry-points."visionforge.exporters"]`: a class (instantiated with no arguments) or an instance with `name: str`, `description: str` and `export(run_dir, options) -> ExportResult`. `ExportResult` is a frozen dataclass with `destination` (URL or path) and `notes`. Both live in `core/exporter_plugins.py`, with `discover_exporters()` and `run_exporter()`.
- An entry-point that fails to load or instantiate, or does not follow the protocol (empty `name`, no `description`, no callable `export`), is logged as a warning and skipped. On a duplicate `name` the first one found stays. Discovery never raises, so a broken plugin cannot stop the CLI.
- `run_exporter` refuses a `run_dir` without `run.json` with a `ValueError` before calling the plugin.
- `visionforge export --list` prints name and description; `visionforge export RUN_DIR --to NAME [--option key=value ...]` prints the destination on stdout and the notes on stderr. An unknown name, a bad option or an error the plugin raises as `ValueError`/`OSError` exits with code 2. The handler is in `cli_export.py`, not in `__main__.py`.
- The built-in exporters are not migrated and keep their routes. The GUI does not list plugins. Messages go through `tr()` (`exporters.*`, pt/en). `docs/custom/EXPORTERS.md` is the guide.

**Consequences:**

- Installing a package that declares the group is enough to run its code under `visionforge export`; the trust is the one of any installed package (ADR-005). VisionForge passes no credential and the guide says to read tokens from the environment, not from `--option`.
- `RunExporter`, `ExportResult` and the group name become public API: they change only in a minor release below 1.0, with a CHANGELOG note. The shape of `run.json` is not promised to plugins.
- The options are strings only, so a plugin that wants a typed option parses it itself.
- A plugin raising any other exception from `export` shows a traceback; only `ValueError` and `OSError` are treated as errors the user can fix.
- Discovery runs on every `export` call and loads every declared entry-point, including ones for other names than `--to`. Acceptable for a CLI that runs once per export.

**Alternatives considered:**

- *Migrate the built-in exporters to the protocol.* Out of scope: their routes carry GUI state (credentials, progress) the protocol does not model.
- *A GUI listing for plugins.* Would need a form schema for the options; the CLI is enough for the first version.
- *Load plugins from an `user_exporters/` folder, like models and tasks.* The entry-point lets a plugin be versioned and installed with `pip`; the folder remains possible later.

## ADR-245 — In server mode each person has at most `--max-queued` jobs in the queue, and the queue takes turns between people

**Date:** 2026-11-25
**Status:** Accepted
**Extends:** ADR-075 (one run queue per server), ADR-212 (phase 1: per-user limits in the queue), ADR-215 (the queue belongs to who started each job), ADR-226 (a bound and a 429 for the actions outside the queue)

**Context:** in server mode every training of every person goes into one queue, drained in submission order. ADR-212 listed "starve the others" among the threats, with "one queue, no limits" as what existed. One account could line up fifty jobs, and the next person waited behind all fifty; ADR-215 only let them see how many were ahead.

**Decision:**

- `visionforge gui --server --max-queued N` (default 3, `DEFAULT_MAX_QUEUED` in `run_queue.py`) is how many jobs one person may have waiting or running, over every lane. The option needs `--server` and a value of 1 or more.
- The gate binds the limit in the request scope (`DatasetScope.max_queued`, beside its `owner`). `_submit_job`, the one place every run route, strategy and resume goes through to queue a job, counts the person's jobs (`RunLanes.count_of`) and answers 429 with `server.queue_full` (pt/en) when they already have N. The count and the submit have no `await` between them, so two requests of the same person cannot both pass the last free place. Nothing is queued and nothing is written by the refusal itself.
- A submission made in server mode marks its lane fair (`RunQueue.make_fair`). In a fair lane, when the GPU frees, the next job is the oldest one of the profile whose job started least recently; a profile that never started a job in this process comes first, and between those the oldest job wins. A1, A2, A3 then B1 run as A1, B1, A2, A3. The pending list stays in arrival order; `run_order()` simulates the turns from the starts so far, and the snapshot behind `GET /api/queue` lists the jobs in that order, so `others_ahead` and a person's own order match what will run.
- A local install never binds an owner: no count, no 429, FIFO as before, also between local profiles (ADR-114).
- `GET /api/session` gains `max_queued` (null on a local install). The page shows nothing new: a refused submission shows the server's message where the training was asked, as for the other 429.

**Consequences:** one person's batch no longer holds everybody behind it, and nobody can take more than N places in the line. Someone who wants to line up a long night of experiments on a server sends N at a time, or the operator raises N. The order a person sees can change after they look: a job of someone who has not run yet goes before the jobs of someone who has.

**Known limits:** the turns are per process: a restart forgets who ran last, as it forgets the queue. Turns count jobs, not GPU time: a person whose jobs take hours has the same turn as one whose jobs take minutes. A strategy (a sweep, replicates, a K-fold) is one job however many trainings it runs. The limit is the same for everybody; there is none per account. `status.queued` still counts every waiting job on the server.

**Not verified:** nothing ran behind a real login proxy, and no real training was queued: the tests submit through the real app and routes with the classification executor replaced by one that holds until released. No GPU.

**Alternatives considered:**

- *Check the limit in the gate, like `--max-actions`.* The gate would need its own list of the routes that queue (the run routes, the strategies, resume), and it would refuse before the body is read; `_submit_job` is already the one door every job goes through.
- *Weighted fair queueing by GPU time.* Fairer for unequal jobs, and it needs a job's cost before it runs, which nobody knows. Turns by job are simple to explain in one sentence.
- *Strict FIFO with only the limit.* With N = 3 and five people, the last one still waits behind fourteen jobs; the turns put them second.
- *One queue per person, served in turn.* The same order with more state; one list with a computed order keeps cancel, `is_writing` and the snapshot as they were.
