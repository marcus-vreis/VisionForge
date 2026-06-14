# Cross-task feature parity — analysis & plan

> Status: **in progress** (ADR-041). Slice 1 (the `TaskRunner` handle +
> classification adapter) and slice 2a (generic comparison + regression/
> segmentation adapters, ADR-044) have shipped; the remaining slices below are
> still to do. This doc is the design implemented in shippable slices.

## Problem

Classification has eight strategy/utility blocks (grid search, random search,
cross-validation, transfer learning, model comparison, batch prediction, ONNX
export, Grad-CAM). The four standalone tasks (detection, regression,
segmentation, anomaly) have **none** of them. Researchers reasonably expect to
sweep hyperparameters on a regression model, compare segmentation backbones, or
batch-infer a detection checkpoint — today they can't from the GUI.

The question is not "copy all eight to all four" — it's *which features make
sense per task*, and *how to extend them without re-introducing the coupling the
task-standalone pattern (ADR-033/036/037/038) deliberately avoids*.

## Current coupling (why it isn't already general)

The orchestration blocks are `ExperimentBlock` subclasses hard-wired to the
classification path. Concretely (from `blocks/model_comparison.py`,
`blocks/batch_prediction.py`, `blocks/grid_search.py`):

- they take `ExperimentConfig` and instantiate `ClassificationBlock` per trial;
- they read **classification metric names** (`accuracy`, `f1`, `auc_roc`);
- batch prediction calls the classification `ModelFactory` and assumes
  softmax/sigmoid heads.

So they can't be pointed at a `RegressionConfig` or a `SegmentationBlock` as-is.

## Verdict per feature

| Feature | Detection | Regression | Segmentation | Anomaly | Verdict |
|---|:--:|:--:|:--:|:--:|---|
| **Model comparison** (rank N archs) | ✓ | ✓ | ✓ | ✓ | **Generic runner — high value, low risk** |
| **Batch prediction** (infer a folder) | ✓ | ✓ | ✓ | ✓ | **Generic runner — high value** |
| **Grid / Random search** | ✓ | ✓ | ✓ | ✓ | **Generic runner — high value, needs the handle** |
| **Transfer learning** (freeze/fine-tune) | ~ (Ultralytics owns it) | ✓ | ✓ | ~ | Per-task knob where backbones are shared |
| **Cross-validation** (K-Fold) | ~ | ✓ | ✓ | ~ | Supervised+splittable only; defer detection/anomaly |
| **ONNX export** | ~ (Ultralytics export) | ✓ | ✓ | ✓ | Medium; per-task export fn |
| **Grad-CAM / explainability** | — | saliency | per-class CAM | — | Research; lowest priority |

`✓` clearly makes sense · `~` partial / library already provides it · `—` not a
natural fit.

## Design: a generic task-run handle

The orchestration features (comparison, search, batch-predict) are all the same
shape: *run the task's pipeline N times and aggregate*, or *load a checkpoint and
predict*. They don't need to know task internals — only a thin, uniform handle.

Propose a `TaskRunner` protocol (in `core/`, task-agnostic) that each task
exposes:

```python
class TaskRunner(Protocol):
    config_type: type            # the task's Pydantic config
    def run(self, cfg) -> RunResult: ...          # one training run → run.json
    def metrics(self, result) -> dict[str, float]: ...   # task metric names → values
    def primary_metric(self) -> str: ...          # e.g. "map50", "r2", "miou", "auroc"
    def load_checkpoint(self, cfg, path): ...      # for batch predict
    def predict(self, model, inputs) -> Any: ...   # task-shaped outputs
```

Then the orchestrators become **task-agnostic**:
- `GenericComparisonRunner` ranks by `runner.primary_metric()` — no hard-coded
  `accuracy`/`f1`.
- `GenericSweepRunner` wraps any task's `run()` for grid/random search, reusing
  the existing trial-progress SSE plumbing (`blocks/_search_utils.py`, ADR-032).
- `GenericBatchPredictor` uses `load_checkpoint` + `predict`.

This honours the project's stance (ADR-033): **reuse the genuinely-shared
orchestration; keep task-specific logic in the task.** It does *not* fold tasks
into `ExperimentConfig` — each task keeps its own config and exposes a handle.

Per-task features (transfer-learning knobs, CV folds, ONNX export) stay in each
task's own modules, because they touch task internals (head freezing, fold
splitting, export graph) and don't generalise cleanly.

## Implementation order (shippable slices, CPU-CI-testable)

1. ✅ **`TaskRunner` protocol + classification adapter.** Make classification
   implement the handle; refactor `ModelComparisonBlock` to consume it. No new
   user-facing feature yet — pure de-coupling, fully covered by existing tests.
2. **Model comparison for regression + segmentation** (the two closest to
   classification) via the generic runner.
   - ✅ slice 2a (backend, ADR-044): `core/comparison.run_model_comparison` +
     `RegressionRunner`/`SegmentationRunner` adapters + tests.
   - ✅ slice 2b API: `POST /api/{regression,segmentation}/compare` (background
     run over the shared single-run state, ranked report via `/experiment/result`).
   - ✅ slice 2c GUI: `ComparisonCard` in the regression/segmentation tabs
     (pick archs + ranking metric → `/compare`), `TaskComparisonReport` renders
     the ranked table in `ResultsView`.
   - ✅ persistence: every comparison/sweep writes `<kind>_summary.json` +
     `<kind>_ranking.csv` to `outputs/reports/<name>/<ts>/` (mirrors the
     classification ModelComparisonBlock), and the report carries `report_dir`.
3. **Batch prediction generic** — regression/segmentation/detection/anomaly.
4. **Generic sweep** (grid/random) over the handle.
   - ✅ backend (ADR-045): `core/sweep.run_sweep` + `POST /api/{regression,
     segmentation}/sweep` + tests. Same search-space grammar as classification.
   - ✅ GUI: `SweepCard` (grid/random search-space editor, `lib/sweep-space.ts`)
     in the regression/segmentation tabs → `/sweep`; `TaskSweepReport` renders
     the best trial + ranked table in `ResultsView`.
   - ✅ **Detection**: `DetectionRunner` adapter (`blocks/detection_runner.py`,
     ranks by `map50_95`, also exposes `map50`) + `POST /api/detection/{compare,
     sweep}` + the comparison/sweep cards in the detection tab. Comparison model
     options are backend-aware (Ultralytics vs torchvision).
   - ✅ **Anomaly**: `AnomalyRunner` (`blocks/anomaly_runner.py`, ranks by `auroc`,
     also `image_f1`) + `POST /api/anomaly/{compare,sweep}` + cards in the anomaly
     tab. **Comparison + sweep now cover all four standalone tasks, backend + GUI.**
5. **Per-task**: transfer-learning knobs (regression/segmentation), then CV,
   then ONNX export. Each its own ADR if it changes a config surface.
   - ✅ ONNX export for regression + segmentation: shared `core/onnx_export.py`
     (classification's `ExportONNXBlock` refactored to reuse it), per-task export
     in `gui/api/torch_onnx_export.py` (segmentation wrapped to a logits tensor),
     wired into the existing `/api/runs/{id}/export_onnx`. Detection = Ultralytics;
     anomaly excluded (PatchCore's memory-bank scoring has no forward graph).
6. Grad-CAM/explainability last (research-grade, per-task).

Each slice: new modules + reuse, an ADR if it changes behavior, the full
`vf-verify` gauntlet, mocked tests (no weight downloads, ADR-010).

## Open questions for the agent team
- Where does `TaskRunner` live so it doesn't violate layer boundaries? Likely a
  new `core/task_runner.py` (protocol only, imports nothing internal) with
  adapters next to each trainer.
- Do the standalone tasks need a uniform `RunResult` first? They already share
  the `run.json` contract — formalise it as a typed object the handle returns.
- Sweep over Ultralytics detection: tune via its own hyperparameters, or skip
  and let Ultralytics' tuner own it?
