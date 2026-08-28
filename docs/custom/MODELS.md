# Your own models · Seus próprios modelos

Drop a Python file into `user_models/` and any config can use your architecture
— for classification, regression or segmentation. No edits to VisionForge's
source, no fork, no reinstall. See ADR-048/049.

Coloque um arquivo Python em `user_models/` e qualquer config passa a poder
usar a sua arquitetura — em classificação, regressão ou segmentação. Sem editar
o código do VisionForge, sem fork, sem reinstalar.

## Three steps · Três passos

**1.** Create a `.py` file in `user_models/` (any name not starting with `_`).

**2.** Register a builder. It receives one int — the task's **output
dimension** — and returns an `nn.Module` producing that many outputs:

```python
import torch.nn as nn

from visionforge.models.registry import register_model


@register_model("my_net")
def build(num_outputs: int) -> nn.Module:
    return MyNet(out=num_outputs)
```

**3.** Point a config at it through `model.custom_model`:

```yaml
model:
  custom_model: my_net      # the builtin `name`/`pretrained` are ignored
  num_classes: 10           # classification/segmentation; regression uses num_targets
  # weights_path: path/to/checkpoint.pth   # optional, loaded non-strictly
```

In the interface the same field appears under **Modelo** — pick your model by
name and the built-in architecture selector steps aside.

VisionForge imports every `.py` file in the folder on demand to discover the
registered builders, so a new file is picked up without restarting the server.

## What the int means · O que o int significa

One builder can serve any CNN-headed task, because the argument is always
"how many numbers must the model emit":

| Task · Tarefa | `num_outputs` is · é |
|---|---|
| Classification · Classificação | `num_classes` |
| Segmentation · Segmentação | `num_classes` (per-pixel logits · logits por pixel) |
| Regression · Regressão | `num_targets` |

Detection and anomaly have their own model paths and do not read
`custom_model`.

Detecção e anomalia têm caminhos próprios de modelo e não leem `custom_model`.

## Notes · Observações

- Files starting with `_` are ignored, so shared helpers can live in the same
  folder. · Arquivos começando com `_` são ignorados, então utilitários
  compartilhados podem morar na mesma pasta.
- A file that fails to import logs a warning and is skipped; it never takes the
  server down. · Um arquivo que falha ao importar registra um aviso e é pulado;
  ele nunca derruba o servidor.
- `example_custom_model.py` in `user_models/` is a working template — copy it,
  or delete it if you don't need it. · O `example_custom_model.py` em
  `user_models/` é um template que funciona — copie, ou apague se não precisar.
- This runs **your own local Python** on your own machine. VisionForge is
  local-first (ADR-005) and fetches nothing from the network. · Isto executa o
  **seu Python local**, na sua máquina. O VisionForge é local-first (ADR-005) e
  não busca nada na rede.

## Guia em português

O passo a passo é o mesmo dos três blocos acima, e vale repetir o essencial:

1. Um arquivo `.py` em `user_models/`, com nome que não comece por `_`.
2. Uma função registrada com `@register_model("seu_nome")` que recebe um `int`
   — quantos números o modelo precisa emitir — e devolve um `nn.Module`.
3. No config (ou no campo **Modelo** da interface), `model.custom_model:
   seu_nome`. Os campos `name` e `pretrained` do modelo embutido passam a ser
   ignorados.

O mesmo modelo serve para classificação, segmentação e regressão, porque o
argumento é sempre a dimensão de saída da tarefa: `num_classes` nas duas
primeiras, `num_targets` na regressão.
