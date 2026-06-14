# Custom models (drop-in)

Define your own architectures here and reference them from a config — no edits to
VisionForge's source. See ADR-048/049.

## How it works

1. Create a `.py` file in this folder (any name not starting with `_`).
2. Register a builder with `@register_model("your_name")`. The builder receives a
   single int — the task's **output dimension** — and returns an `nn.Module`
   emitting that many outputs:

   ```python
   import torch.nn as nn
   from visionforge.models.registry import register_model

   @register_model("your_name")
   def build(num_outputs: int) -> nn.Module:
       return MyNet(out=num_outputs)
   ```

3. Point a config at it via `model.custom_model` (classification, regression, or
   segmentation):

   ```yaml
   model:
     custom_model: your_name   # builtin `name`/`pretrained` are ignored
     num_classes: 10           # classification/segmentation; regression uses num_targets
     # weights_path: path/to/checkpoint.pth   # optional, loaded non-strictly
   ```

VisionForge imports every `.py` file in this folder on demand to discover the
registered builders. `example_custom_model.py` is a working template — copy it,
or delete it if you don't need it.

## Notes

- The builder's int is the task's output dimension: `num_classes` for
  classification, `num_classes` (per-pixel logits) for segmentation, `num_targets`
  for regression. One model can serve any CNN-headed task. Detection and anomaly
  use their own model paths and don't read `custom_model`.
- This runs your own local Python. VisionForge is local-first (ADR-005); nothing
  is fetched from the network. The trust boundary is your own machine.
- Files starting with `_` are ignored, so shared helpers can live here too.
