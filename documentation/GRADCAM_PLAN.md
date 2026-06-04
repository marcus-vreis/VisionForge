# Grad-CAM Explainability — Design Plan (Backlog)

> Branch: `feat/gradcam` (off `development` — Grad-CAM is a classification
> explainability feature, independent of the Phase 6–9 task branches).
> Status: **In progress** — brick 1 (`core/gradcam.py`) landing first.
> Kickoff 2026-06-04 · author: tech-leader iteration

Grad-CAM (Gradient-weighted Class Activation Mapping) is the first item from the
**medium-term scientific expansion** backlog (CLAUDE.md §7.2 — "Explainability
Block: Grad-CAM, LIME, SHAP"). It answers *"where did the classifier look?"* — a
heatmap over the input highlighting the regions most responsible for a predicted
class. Essential for academic interpretation of a trained model.

## 1. Why Grad-CAM is a post-run action, not a new task

Unlike the Phase 6–9 tasks (each a standalone config/block/run path), Grad-CAM is
a **post-hoc analysis of an already-trained classification checkpoint** — it slots
into the existing per-run action surface alongside `/api/runs/{id}/test`,
`/runs/{id}/batch_predict`, and `/runs/{id}/export_onnx`. It needs no new config
tree: it loads a saved classification run, rebuilds the model, and produces
heatmap overlays for sample images. It is dependency-free — Grad-CAM is just two
hooks (forward activations + backward gradients) on the last conv layer plus a
weighted sum, all in torch.

## 2. How Grad-CAM works (the core, brick 1)

For a model, a target conv layer L, an input image x, and a target class c:

1. Forward pass; capture L's activations `A` (`[K, h, w]`) via a forward hook.
2. Backprop the logit for class c; capture the gradients `∂y_c/∂A` via a backward
   hook.
3. Channel weights `w_k = GAP(gradients)` (global-average-pool over h, w).
4. `CAM = ReLU(Σ_k w_k · A_k)` → `[h, w]`, normalized to [0, 1] and bilinearly
   upsampled to the input H×W.
5. Overlay the CAM (jet colormap) on the de-normalized input image.

The target layer is resolved as the **last `nn.Conv2d`** in the model (arch-
agnostic; correct for ResNet/EfficientNet/VGG/AlexNet — the conventional Grad-CAM
target). No per-architecture branching.

## 3. New modules

```
src/visionforge/
├── core/
│   └── gradcam.py          # GradCAM hooks + resolve_target_layer + overlay   ← brick 1
└── gui/api/
    └── routes.py           # POST /api/runs/{id}/gradcam (per-run action)     ← brick 2
```

## 4. Build sequence

1. **`core/gradcam.py` + tests** — `GradCAM` (forward/backward hooks, CAM
   compute), `resolve_target_layer` (last conv), `overlay_cam` (jet overlay on a
   de-normalized image). Pure backend, tested with a tiny CNN + a torchvision
   backbone. ← **this brick.**
2. **`POST /api/runs/{id}/gradcam`** — load the run's checkpoint + config, pick N
   sample images from a dataset path, compute overlays, write PNGs into the run
   dir, return paths; classification-gated (mirrors the per-run test guard).
   ADR for the per-run-action placement.
3. **GUI** — a "🔥 Grad-CAM" action in `RunDetailPanel` (mirrors the "+ testar" /
   ONNX export forms) that calls the endpoint and renders the overlay grid.

## 5. Open decisions

- **Target class** — default to the model's **predicted** class per image
  (argmax); a future param could pin a specific class to inspect.
- **Target layer** — last `nn.Conv2d`, arch-agnostic. A future param could expose
  layer choice for deeper inspection.
- **LIME / SHAP** — deferred; Grad-CAM is the highest-value, dependency-free
  first step of the explainability backlog item.
