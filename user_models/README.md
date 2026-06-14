# Custom models (drop-in)

Define your own architectures here and reference them from a config — no edits to
VisionForge's source. See ADR-048.

## How it works

1. Create a `.py` file in this folder (any name not starting with `_`).
2. Register a builder with `@register_model("your_name")`. The builder receives
   `num_classes` and returns an `nn.Module` emitting that many logits.
3. Point a classification config at it via `model.custom_model`:

   ```yaml
   model:
     custom_model: your_name   # builtin `name`/`pretrained` are ignored
     num_classes: 10
     # weights_path: path/to/checkpoint.pth   # optional, loaded non-strictly
   ```

VisionForge imports every `.py` file in this folder on demand to discover the
registered builders. `example_custom_model.py` is a working template — copy it,
or delete it if you don't need it.

## Notes

- This runs your own local Python. VisionForge is local-first (ADR-005); nothing
  is fetched from the network. The trust boundary is your own machine.
- Custom models currently apply to the **classification** task (the `ModelFactory`
  path). Other tasks use their own factories.
- Files starting with `_` are ignored, so shared helpers can live here too.
