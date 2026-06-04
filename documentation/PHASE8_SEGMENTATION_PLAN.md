# Phase 8 — Semantic Segmentation — Design Plan

> Branch: `feat/segmentation` (stacked on `feat/image-regression`; rebases onto
> `development` once regression merges — segmentation shares no code with
> regression beyond the reused `OutputConfig`/`DeviceConfig`/`TransformConfig`).
> Status: **In progress** — brick 1 (`SegmentationConfig`) landing first.
> Kickoff 2026-06-04 · author: tech-leader iteration
>
> Progress (see TASKS Phase 8 for the live checklist):
> - [x] brick 1 — `SegmentationConfig` (`utils/segmentation_config.py`) + tests
> - [x] brick 2 — `SegmentationDataModule` (`core/segmentation_data.py`) + tests
> - [x] brick 3 — `SegmentationModelFactory` (torchvision DeepLab/FCN/LR-ASPP + U-Net)
> - [ ] brick 4 — `SegmentationTrainer` (CE/Dice loss, IoU/Dice/pixel-acc)
> - [ ] brick 5 — `SegmentationBlock` (`setup/run/report`) + ADR-037
> - [ ] brick 6 — `/api/segmentation/{schema,run}` run path
> - [ ] brick 7 — GUI Segmentação tab (SegmentationPanel, wired end-to-end)

Semantic segmentation is the fourth task (after classification, detection,
regression). Like detection and regression it does **not** reuse the
classification `ExperimentConfig` wholesale: the target is a **per-pixel class
map** (a mask image, not a class index or a continuous value), which cascades
into the dataset format (paired image+mask directories), the model head (a dense
`num_classes`-channel output at input resolution, not a pooled vector), the loss
(pixel-wise cross-entropy / Dice), and the metrics (mean IoU, Dice, pixel
accuracy — not accuracy/F1/AUC/mAP/R²).

It still **reuses** the shared CNN-era infrastructure: `OutputConfig`,
`DeviceConfig`, `TransformConfig`, `PreprocessingConfig`, `SchedulerConfig`, the
`outputs/` run layout, the `run.json` contract (ADR-013), and the SSE
`start`/`epoch_end`/`end` event shape so `TrainingOverlay` and `ResultsView`
work with zero contract changes.

## 1. Why segmentation needs its own config/block (not an ExperimentConfig block)

The classification stack assumes:

- **ImageFolder** datasets (one subdir per class) — segmentation labels are
  *images* (masks), not folder names.
- A pooled classifier head sized to `num_classes`, trained with cross-entropy on
  a single label per image.
- Metrics are accuracy / F1 / AUC — segmentation needs **per-pixel** IoU/Dice.

Forcing this through `ExperimentConfig`/`ExperimentBlock` would break the
classification block ABC's `setup(self, config: ExperimentConfig)` contract — the
same Liskov argument that produced ADR-033 (detection) and ADR-036 (regression).
Segmentation therefore gets its own `SegmentationConfig` tree and a standalone
`SegmentationBlock` dispatched through a dedicated `/api/segmentation/*` run path.
(→ to be ratified as **ADR-037** when brick 5 lands.)

## 2. Dataset format — the one real decision

Semantic segmentation needs a per-pixel label, supplied as a **mask image** the
same H×W as the input, where each pixel holds its class index (0..num_classes-1)
plus an optional `ignore_index` (default 255) for unlabeled/void pixels. The
standard, tool-agnostic layout is **paired per-split image/mask directories**:

```
<base_dir>/
├── train/
│   ├── images/    img001.jpg, img002.jpg, …
│   └── masks/     img001.png, img002.png, …   (single-channel, value = class id)
├── val/
│   ├── images/
│   └── masks/
└── test/          (optional)
    ├── images/
    └── masks/
```

- Mask and image are paired by **filename stem** (`img001.jpg` ↔ `img001.png`),
  so masks are PNG (lossless — JPEG would corrupt integer class ids).
- `images_subdir` (default `"images"`) and `masks_subdir` (default `"masks"`) are
  configurable so non-standard layouts flex without a schema break.
- `ignore_index` (default 255) is excluded from both the loss and the metrics.

**Default chosen, flagged for review:** per-split `images/` + `masks/` dirs
keyed by filename stem. This mirrors the detection `base_dir` ergonomics and the
most common public-dataset layout (VOC-style, Cityscapes-style after flattening).
Alternative layouts (single combined dir + split file, palette/RGB masks needing
a color→id map) are deliberately deferred; the config field names are generic so
brick 2's `SegmentationDataModule` is where that flexes.

## 3. Model support — torchvision first, U-Net hand-rolled

Mirroring the detection torchvision backend (reuse torchvision rather than add a
heavy dep), the model factory wires the **torchvision segmentation family**:

- `deeplabv3_resnet50`, `deeplabv3_resnet101`, `deeplabv3_mobilenet_v3_large`
- `fcn_resnet50`, `fcn_resnet101`
- `lraspp_mobilenet_v3_large`

Plus a **hand-rolled `unet`** (the canonical encoder/decoder with skip
connections) so the roadmap's "U-Net, DeepLab" is satisfied without a new
dependency — U-Net is not in torchvision. Each torchvision head is resized to the
config `num_classes`; `weights_backbone=None` when not pretrained (no downloads
in CI, matching the detection factory convention).

## 4. New modules (mirrors the detection/regression layout)

```
src/visionforge/
├── utils/
│   └── segmentation_config.py    # SegmentationConfig + Model/Data/Training subtrees  ← brick 1
├── core/
│   ├── segmentation_data.py      # SegmentationDataModule — (image, mask) tensors     ← brick 2
│   └── segmentation_trainer.py   # CE/Dice loop, mIoU/Dice/pixel-acc                  ← brick 4
├── models/
│   └── segmentation_factory.py   # torchvision seg models + hand-rolled U-Net         ← brick 3
└── blocks/
    └── segmentation.py           # SegmentationBlock (setup/run/report)               ← brick 5
```

- `task` Literal is `"segmentation"`; the block dispatches via the dedicated path.
- `SegmentationTrainer` emits the **same SSE events** as classification, with
  segmentation metric fields (`miou`, `dice`, `pixel_acc`), so the GUI overlay
  and results view need no contract change.
- `run.json` stays the contract (ADR-013): metrics become `miou`, `dice`,
  `pixel_acc`; artifacts point at the loss/metric curves + `best.pt`.

## 5. Build sequence (each a small, tested, shippable brick)

1. **`SegmentationConfig` Pydantic models + tests** — no heavy deps; validates
   backbone choice, `num_classes`, dataset dir names, `ignore_index` coherence
   (must not collide with a real class id), loss choice, image size, epochs.
   ← **this brick.**
2. **`SegmentationDataModule`** — pair image+mask by stem, resize both (mask with
   nearest-neighbor to preserve class ids), return `(image_tensor, mask_long)`;
   reuse `TransformConfig`/preprocessing for the image branch. Picklable dataset
   (Windows spawn).
3. **`SegmentationModelFactory`** — torchvision DeepLab/FCN/LR-ASPP head resize +
   hand-rolled U-Net; reuse the shared backbone helpers where applicable.
4. **`SegmentationTrainer`** — CE / Dice / combined loss, per-epoch mean IoU /
   Dice / pixel accuracy (streaming confusion-matrix accumulator), best-by-val-
   mIoU checkpoint, `run.json`; mirror the regression/classification trainers.
5. **`SegmentationBlock`** — `setup/run/report`; dispatch via
   `/api/segmentation/run`. ADR-037 written.
6. **`/api/segmentation/{schema,run}`** — schema drives the form; run dispatches
   `SegmentationBlock` with `_progress_callback` → SSE; reuses the shared
   single-run state.
7. **GUI**: activate the Segmentação tab (currently placeholder); schema-driven
   form for `SegmentationConfig`; results view reads segmentation metrics + plots.

## 6. Open decisions (to ratify as ADRs)

- **ADR-037** — Semantic segmentation is a standalone config/block/run path
  (mirrors ADR-033 / ADR-036). To be written when brick 5 lands.
- **Loss default** — `cross_entropy` (the standard pixel-wise baseline);
  `dice`/`combined` selectable for class-imbalanced masks.
- **Mask format** — v1 assumes **integer-id single-channel masks**. RGB/palette
  masks (needing a color→class map) are deferred to brick 2 if a real dataset
  needs them; the config will not pre-commit to a palette.
- **Metric selection** — best checkpoint by **val mean IoU** (the segmentation
  standard), not val loss.
