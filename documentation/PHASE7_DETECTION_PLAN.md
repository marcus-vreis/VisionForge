# Phase 7 — Object Detection (Ultralytics) — Design Plan

> Branch: `feat/object-detection-ultralytics`
> Status: **Complete** — Ultralytics + torchvision backends both end-to-end & tested (mAP@50). Only follow-up: opt-in real-data smoke test.
> Kickoff 2026-06-01 · hybrid backend confirmed by user 2026-06-01
> Author: tech-leader iteration
>
> Progress (see TASKS Phase 7 for the live checklist):
> - [x] brick 1 — `DetectionConfig` (`utils/detection_config.py`)
> - [x] brick 2 — `DetectionDataModule` (`core/detection_data.py`) + `[detection]` extra
> - [x] brick 3 — `DetectionTrainer` (`core/detection_trainer.py`)
> - [x] brick 4 — `DetectionBlock` (`blocks/detection.py`) + ADR-033/034
> - [x] brick 5a — `/api/detection/{schema,run}` run path (`gui/api/routes.py`)
> - [x] brick 5b — GUI Detecção tab (`DetectionPanel`, `lib/detection-models.ts`, `App.tsx`)

Object detection is the first task that does **not** reuse the classification
engine. This document records the architecture before code lands, so the build
stays clean and reviewable.

## 1. Why detection breaks the existing engine

The classification stack (`core/data.DataModule`, `core/trainer.Trainer`,
`core/evaluator.Evaluator`, `models/factory.ModelFactory`) assumes:

- **ImageFolder** datasets (one subdir per class).
- A torchvision classifier with a replaceable FC head.
- A hand-written PyTorch epoch loop with CE loss + accuracy/F1.

Detection violates all three:

- Datasets are **images + per-image box annotations** (YOLO `data.yaml` + `.txt`
  labels, or COCO json). There is no `ImageFolder`.
- Metrics are **mAP@50 / mAP@50-95, precision, recall** per class — not accuracy.
- **Ultralytics owns its training loop** (`YOLO(...).train(...)`). We do not write
  the epoch loop; we drive and observe theirs.

Forcing detection through `Trainer` would corrupt both. Instead detection gets
its own `DetectionTrainer` + `DetectionDataModule` behind the **same
`ExperimentBlock` contract** (`setup/run/report`), so the GUI/dispatch layer is
unchanged. This honors ADR-003 (plugin blocks) and ADR-006 (task abstraction).

## 2. Model sources — the one real decision

The user asked for "ultralytics … YOLO, Faster e afins". Ultralytics is
**YOLO + RT-DETR only**; Faster R-CNN / SSD / RetinaNet come from
`torchvision.models.detection`. Decision:

- **Primary backend = Ultralytics** (`ultralytics` package): `yolo11{n,s,m,l,x}`,
  `yolov8*`, and `rtdetr-*`. This is the clean, batteries-included path the user
  asked for (auto-download weights, training loop, val mAP, export).
- **Secondary backend = torchvision.detection** for `faster_rcnn`, `ssd`,
  `retinanet` — for users who explicitly want those families.
- A `DetectionModelConfig.backend` Literal (`"ultralytics" | "torchvision"`)
  selects the path; `model.name` is validated against the chosen backend's set.
  v1 ships the **Ultralytics path end-to-end**; the torchvision path is scaffolded
  with a clear `NotImplementedError` and filled in a follow-up so the first PR
  stays shippable. (→ to be ratified as ADR-033.)

## 3. New modules (mirrors the classification layout)

```
src/visionforge/
├── config/  (or utils/config.py extension)
│   └── DetectionConfig, DetectionModelConfig, DetectionDataConfig
├── core/
│   ├── detection_data.py    # DetectionDataModule — resolves/writes data.yaml
│   └── detection_trainer.py  # wraps ultralytics YOLO.train(), streams progress
└── blocks/
    └── detection.py          # DetectionBlock(ExperimentBlock)
```

- `task` Literal gains `"detection"`; `block` Literal gains `"detection"`.
- `DetectionBlock` emits the **same SSE events** as classification
  (`start`/`epoch_end`/`end`) via a progress callback fed from Ultralytics'
  `on_train_epoch_end` callback — so the live monitor and ADR-032 streaming work
  for detection with zero frontend changes to the event contract.
- `run.json` stays the contract (ADR-013): metrics become `map50`, `map50_95`,
  per-class AP; artifacts point at Ultralytics' `results.png`, `confusion_matrix.png`,
  PR/F1 curves (it already renders these), plus `best.pt`.

## 4. Dependency & CI

- Add `ultralytics` under an optional extra `[detection]` in `pyproject.toml`
  (keeps the base install lean; ADR-005 keeps torch user-managed). Ultralytics
  pulls torch/torchvision which are already declared.
- CI: detection tests must run on CPU without downloading weights — mock
  `ultralytics.YOLO` in unit tests (same pattern as the `ClassificationBlock.run`
  patching in `tests/blocks/`). A tiny opt-in integration test (1 epoch on a
  2-image synthetic set) guarded by an env flag, skipped in CI.

## 5. Build sequence (each a small, tested, shippable brick)

1. **`DetectionConfig` Pydantic models + tests** — no heavy deps; validates
   backend/model coherence, dataset paths, image size, epochs. ← start here.
2. **`DetectionDataModule`** — resolve an existing `data.yaml`, or synthesize one
   from a YOLO-layout folder; tests on a synthetic fixture.
3. **`DetectionTrainer`** — wrap `YOLO(model).train(...)`, translate args, hook
   the epoch callback to our progress callback, write `run.json`. Unit-tested
   with a mocked `YOLO`.
4. **`DetectionBlock`** — `setup/run/report`, register in `BlockRegistry`,
   dispatch in `routes._execute_experiment`, wire `_progress_callback`.
5. **GUI**: activate the Detecção tab (currently "em breve"); schema-driven form
   for `DetectionConfig`; results view reads mAP + Ultralytics plots.
6. **ADR-033** (model-source split) + **ADR-034** (Ultralytics owns the loop) +
   TASKS Phase 7 checklist update.

## 6. Resolved decisions

- **Hybrid backends (§2): confirmed** by the user on 2026-06-01 — Ultralytics
  primary (v1 end-to-end), torchvision seam scaffolded (`NotImplementedError`)
  for Faster R-CNN/SSD/RetinaNet in a follow-up. Ratified as ADR-033 (standalone
  detection path) and ADR-034 (Ultralytics owns the loop).
- **Standalone config/block/run path (§1, §3): confirmed** — detection does not
  reuse `ExperimentConfig`/`ExperimentBlock`; it has its own tree and a dedicated
  `/api/detection/*` run path (ADR-033).

## 7. Phase 7.1 — parity with the classification surface

Phase 7 shipped the detection *training* path. The classification task, however,
exposes a much wider post-training surface (run history, run detail, per-model
test history, dataset stats, results view). This phase brings detection to parity
with those, **reusing** the existing endpoints/components and dispatching by task
rather than duplicating them (clean-code: table-driven projections + small
task-strategy classes, no forked copies).

Gap analysis (what is classification-only today) and the brick plan:

- [x] **brick A — run-history parity** (`routes._parse_run_summary`,
  `HistoryOverlay`): detection runs were mislabeled `block="classification"` and
  showed no mAP (the metric projection only knew `accuracy/f1/val_loss`). Added a
  table-driven `_SUMMARY_METRIC_KEYS` per-task projection + `_summary_metrics`,
  inferred `block="detection"` from `task`, and made the history card's metric
  keys task-aware (`map50`, `map50_95` → "mAP@50 / mAP@50-95"). Tested.
- [x] **brick B — run-detail parity** (`RunDetailPanel`): the panel already reads
  `artifacts.graphics`/`artifacts.model` and `metrics` generically, so detection
  runs render. Added detection metric labels (`map50`, `map50_95`, `box_loss`) and
  plot labels (`results.png`, `BoxPR_curve.png`, `BoxF1_curve.png`) so they show
  human names instead of raw keys/filenames.
- [x] **brick C — results-view parity** (`ResultsView`): already shipped in
  brick 5b — `ResultsView` carries detection `GRAPH_LABELS` + `METRIC_LABELS` and
  renders metrics/plots generically, so a completed detection run shows mAP + the
  Ultralytics plots. No change needed.
- [x] **brick D1 — guard classification-only actions for detection** (done):
  the per-model test, batch CSV inference, and ONNX export are classification-only
  (ModelFactory + Evaluator). `RunDetailPanel` now hides all three for a detection
  run (`config.task === "detection"`), and the backend rejects them in
  `_require_classification_run` (HTTP 400) as defense in depth. Prevents the
  previous guaranteed-500-on-click. Tested.
- [x] **brick D2 — detection-native per-model test** (done): new
  `gui/api/detection_testing.evaluate_detection_run`; `_execute_run_test`
  dispatches to it for `task=="detection"`. The torchvision path rebuilds the
  detector + DataLoader and computes mAP@50 via `detection_metrics` (CPU,
  dependency-light); the Ultralytics path drives `YOLO(...).val` (patchable
  module global). Records mAP into `tests[]`. `RunDetailPanel` re-enables the
  "+ testar" form for detection (ONNX/batch stay hidden). Tested (torchvision
  real eval, Ultralytics mocked, error paths; batch/ONNX still 400 for detection).
- [x] **brick E1 — YOLO dataset-stats backend** (done): new
  `POST /api/detection/dataset/stats` + `_collect_detection_dataset_stats`.
  Counts images and annotation *instances* per class across the `.txt` label
  files per split, flags unlabeled images and class imbalance (max/min > 2). Reads
  class names from `classes.txt`/`names.txt` (else generates them). Extracted a
  shared `resolve_yolo_split` into `core/detection_data.py` (the trainer now
  delegates to it — dedup). Tested (8 cases).
- [x] **brick E2 — YOLO dataset-stats GUI** (done): new `DetectionDatasetStats`
  component (+ `fetchDetectionDatasetStats` client) rendered below the dataset
  picker in `DetectionPanel`. Shows the class id→name map, per-split image/instance
  counts, per-class annotation bars, unlabeled-image count, and the imbalance
  warning; auto-applies the detected class count to `model.num_classes`. Mirrors
  the classification `DatasetStats` look.
- [ ] **brick F — export / batch inference parity** (optional, lower priority):
  Ultralytics-native ONNX export + folder inference, surfaced like the
  classification ONNX/batch-predict run actions.
