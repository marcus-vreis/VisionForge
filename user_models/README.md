# user_models/ — your own architectures · suas próprias arquiteturas

Drop a `.py` file here, register a builder, and any config can use your model:

```python
import torch.nn as nn

from visionforge.models.registry import register_model


@register_model("my_net")
def build(num_outputs: int) -> nn.Module:
    return MyNet(out=num_outputs)
```

Then set `model.custom_model: my_net` in a config, or pick it in the **Modelo**
section of the interface. The int is the task's output dimension —
`num_classes` for classification and segmentation, `num_targets` for
regression.

Coloque um `.py` aqui, registre um builder e qualquer config pode usar o seu
modelo. Depois é só apontar `model.custom_model: seu_nome`.

**Full guide · Guia completo:**
[`docs/custom/MODELS.md`](https://github.com/marcus-vreis/VisionForge/blob/main/docs/custom/MODELS.md)

`example_custom_model.py` is a working template — copy it, or delete it if you
don't need it. Files starting with `_` are ignored, so shared helpers can live
here too.
