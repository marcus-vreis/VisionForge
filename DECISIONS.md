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

---

## ADR-014 — User-configurable model weights

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
