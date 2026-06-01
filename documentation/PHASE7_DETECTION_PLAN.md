# Phase 7 — Object Detection (Ultralytics) — Design Plan

> Branch: `feat/object-detection-ultralytics`
> Status: **Backend complete** (bricks 1–5a done & tested) · **GUI tab remaining** (brick 5b)
> Kickoff 2026-06-01 · hybrid backend confirmed by user 2026-06-01
> Author: tech-leader iteration
>
> Progress (see TASKS Phase 7 for the live checklist):
> - [x] brick 1 — `DetectionConfig` (`utils/detection_config.py`)
> - [x] brick 2 — `DetectionDataModule` (`core/detection_data.py`) + `[detection]` extra
> - [x] brick 3 — `DetectionTrainer` (`core/detection_trainer.py`)
> - [x] brick 4 — `DetectionBlock` (`blocks/detection.py`) + ADR-033/034
> - [x] brick 5a — `/api/detection/{schema,run}` run path (`gui/api/routes.py`)
> - [ ] brick 5b — GUI Detecção tab (schema-driven form + results)

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
