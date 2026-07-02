# Phase 10 — Custom Tasks ("Blank" task SDK) — Design Plan

> Status: **Proposed** (ADR-058). No code yet — this is the reviewed design.
> Kickoff target: after the ReplicatesCard GUI brick.
> Author: research-grade review session, 2026-07-01.

The researcher's request: *"temos classificação, detecção… e aí o **Blank**, onde
o pesquisador coloca tudo — nome, cor, as variáveis, o pipeline — bem facilitado,
modificando nos .py, tudo documentado e funcional."*

A sixth tab where a researcher defines their own task family by writing **one
well-documented Python file** in `user_tasks/` — and gets, for free, everything
the built-in tasks have: schema-driven GUI form, live training monitor, run
history, `run.json` provenance, and (via the `TaskRunner` handle) replicates,
sweeps and model comparison.

## Why this is feasible now (the seams already exist)

This is not speculative — every load-bearing piece already shipped:

| Needed | Already exists |
|---|---|
| Drop-in user code discovery | `user_models/` + `@register_model` (ADR-048) — same registry pattern, proven |
| Auto-generated config form | `ParamPanel`'s schema-driven renderer (`SchemaFieldVF` + `field-renderer.ts`) builds the whole classification form from `model_json_schema()` |
| Declarative task descriptor in the GUI | `TASKS: TaskDefinition[]` (`types/tasks.ts`) — key/label/accent/params/defaults already drive `TabBar`/`TaskHero` |
| Uniform training contract | SSE `start`/`epoch_end`/`end` + ADR-013 `run.json` + environment block — every panel component consumes these, not task internals |
| Orchestration for free | `TaskRunner` Protocol (ADR-041) — comparison (ADR-044), sweep (ADR-045/052), replicates (ADR-056) all run over the handle |
| Reusable config blocks | `OutputConfig`/`DeviceConfig`/`TransformConfig`/`PreprocessingConfig`/`SchedulerConfig` reused by all five tasks |
| Reusable trainer plumbing | `resolve_device`, `_seed_everything`, early stopping, checkpoint-best, `MetricsPlotter.metric_curve` |

The gap: each built-in trainer re-implements its epoch loop (~500 lines each).
The SDK's job is to package that loop once, with hooks, so the researcher writes
~80 lines instead of ~500.

## What the researcher writes (Level 1 — declarative, target: 90% of cases)

`user_tasks/cell_counting/task.py` (created by `visionforge new-task cell_counting`):

```python
from visionforge.tasks import BaseTaskConfig, TaskSpec, register_task
from pydantic import Field
import torch, torch.nn as nn

class CellCountingConfig(BaseTaskConfig):
    # BaseTaskConfig already provides: name, training (epochs/lr/batch/seed/
    # early stopping/scheduler/AMP/deterministic), data (base_dir/workers),
    # output, device, transforms. Add ONLY what is task-specific:
    density_sigma: float = Field(default=2.0, gt=0, description="Gaussian σ do mapa de densidade")

@register_task(
    key="cell_counting",
    label="Contagem de células",
    accent="#2dd4bf",                    # the tab color the user asked for
    description="Conte objetos via mapa de densidade",
    metrics={"mae": "lower", "rmse": "lower"},   # name → direction
    primary_metric="mae",
)
class CellCountingTask(TaskSpec):
    Config = CellCountingConfig

    def build_model(self, cfg) -> nn.Module: ...          # any nn.Module
    def build_loaders(self, cfg) -> tuple[DataLoader, DataLoader, DataLoader | None]: ...
    def compute_loss(self, model, batch, cfg) -> torch.Tensor: ...
    def compute_metrics(self, model, loader, cfg) -> dict[str, float]: ...
```

The **GenericTaskEngine** (VisionForge code, not user code) then drives: seeding
(+ determinism flag), device resolution, the epoch loop, early stopping,
best-checkpoint save, SSE events, `run.json` + environment, TensorBoard, and the
loss-curve plot. The researcher never touches React, FastAPI, or the loop.

**Level 2 (full control):** implement `run(cfg, ctx) -> dict[str, float]`
instead of the four hooks; `ctx` provides `run_dir`, `emit(event)` (SSE),
`write_run_json(...)`, `save_checkpoint(...)`. For researchers whose training
isn't epoch-shaped (GANs, two-stage, EM-style loops).

## What the GUI does (no user-provided JS — this is the standardization)

- `GET /api/tasks` returns built-ins + registered customs: `{key, label, accent,
  description, schema_url, metrics, primary_metric}`.
- `App.tsx` merges customs into `TASKS`; `TabBar` renders them (it is already
  data-driven; accent comes from the descriptor). Custom accent is applied via
  inline CSS var (the `[data-task]` tokens stay for built-ins).
- The custom tab's panel is the **generic schema-driven form** (the ParamPanel
  renderer pointed at `/api/custom/{key}/schema`), with the shared cards
  (dataset picker where `data.base_dir` exists, device, YAML import/export).
- `TrainingOverlay`/`ResultsView`/`HistoryOverlay`/`RunDetailPanel` consume the
  metrics dict + declared metric metadata (name, direction, format) — generic
  fallback rendering for any task key they don't specially know.
- Custom visual identity = name, color, description, icon (optional emoji).
  **Deliberately not** user-supplied React/JS: that is what keeps custom tasks
  functional, upgrade-safe, and reviewable.

## API surface

- `GET /api/tasks` — list all task descriptors (built-in + custom).
- `GET /api/custom/{key}/schema` — the task Config's `model_json_schema()`.
- `POST /api/custom/{key}/run` — validate + dispatch through the shared
  single-run state (409/422 semantics identical to the built-in endpoints).
- `CustomTaskRunner(key)` adapter → `/api/custom/{key}/{compare,sweep,replicates}`
  come for free from `_start_comparison/_start_sweep/_start_replicates`.

## Bricks (each lands green on CPU-only CI, ADR-010)

- [ ] brick 1 — `visionforge/tasks/` package: `BaseTaskConfig` (composes the
  shared config blocks), `TaskSpec` ABC (Level 1 hooks + optional `run`),
  `@register_task` + `user_tasks/` discovery (mirror `models/registry.py`,
  reject collisions with built-in keys). Tests with a toy task fixture.
- [ ] brick 2 — `GenericTaskEngine`: the hook-driven loop reusing
  `_seed_everything(deterministic=…)`/`resolve_device`/early-stop/checkpoint/
  SSE/`run.json`/TensorBoard/`metric_curve`. Tests: toy task trains on synthetic
  tensors, run.json shape, early-stop, best-checkpoint, SSE callback capture.
- [ ] brick 3 — API: `GET /api/tasks`, `GET /api/custom/{key}/schema`,
  `POST /api/custom/{key}/run` + dispatch. Route tests (schema, dispatch+SSE,
  409, 422, unknown key 404).
- [ ] brick 4 — `CustomTaskRunner` adapter + compare/sweep/replicates endpoints
  for custom keys. Tests mirror the existing runner tests.
- [ ] brick 5 — `visionforge new-task <key>` CLI scaffolder → generates the
  commented template (+ `--split` variant with config/data/model files),
  ships `user_tasks/example_counting/` as the working example (like
  `user_models/example_custom_model.py`). `user_tasks/README.md` walkthrough
  (PT + EN): "your task in the GUI in 30 minutes".
- [ ] brick 6 — GUI: fetch `/api/tasks`, merge dynamic tabs, generic schema
  panel + generic results/history fallback, SPA rebuild. Vitest for the
  merge + payload builder.

## Risks / honest hard parts

- **Windows spawn pickling** (ADR-030): user datasets must be top-level classes.
  The template says so loudly and the engine raises a diagnostic that names the
  fix when a loader worker fails to pickle.
- **User-code failures**: surface as the existing failure panel with the full
  traceback (routes already do this for built-in blocks); never crash the server.
- **History for unknown metrics**: brick 6's generic fallback must land with the
  feature, or custom runs look broken in History (learned from Phase 5.5's
  backend-ahead-of-GUI debt).
- **Schema drift**: custom configs get `schema_version` like everyone (ADR-039);
  the registry stamps `task: "custom:<key>"` into run.json.
- **Trust boundary**: same stance as ADR-048 — it runs the user's own Python on
  the user's own machine; no sandboxing, no network fetch of task code.
