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

In the interface, the Modelo step of the Classification tab has a
**Modelo próprio** field (*Your own model* in English) that lists every name
`user_models/` registers. Picking one sets `custom_model` and hides Pesos
pré-treinados, which a custom model ignores; **Nenhum** (*None*) clears it,
and the Arquitetura field trains again. With nothing registered, the field
says how to add a model instead. The list comes from
`GET /api/model/custom`, which scans the folder each time the step opens.

In a YAML config file, set `custom_model` as above and run the file with
`visionforge run <file>.yaml`, or import it in the interface (**↑ Importar
YAML**, in the Revisão step of the Classification tab). Either way the Modelo
and Revisão steps name your model instead of the builtin architecture, and the
History lists the run under its name. On a shared server (`visionforge gui
--server`) the field is not shown and a run that sets `custom_model` is
refused, because a custom model is code a user brings.

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

For segmentation the module must return per-pixel logits, shaped
`[batch, num_classes, height, width]`: a classifier that ends in a linear layer
serves classification and regression, not segmentation.

Detection and anomaly have their own model paths and do not read
`custom_model`.

Na segmentação o módulo precisa devolver logits por pixel, no formato
`[batch, num_classes, altura, largura]`: um classificador que termina numa
camada linear serve para classificação e regressão, não para segmentação.

Detecção e anomalia têm caminhos próprios de modelo e não leem `custom_model`.

## Stability · Estabilidade

Before 1.0 the extension API may change, but not by accident. What is stable:
`@register_model("name")` on a function that takes one `int` (the task's output
dimension) and returns an `nn.Module`, and `build_custom_model(name, *,
num_outputs, user_models_dir=None)`. A change to either lands only in a minor
release, with an entry under "Changed" in the CHANGELOG, never in a patch.
`tests/tasks/test_task_api_contract.py` pins the signatures (ADR-190). There is
no API version number to check yet; the release notes are the signal.

Antes da 1.0 a API de extensão pode mudar, mas não por acidente. É estável:
`@register_model("nome")` numa função que recebe um `int` (a dimensão de saída
da tarefa) e devolve um `nn.Module`, e `build_custom_model(name, *,
num_outputs, user_models_dir=None)`. Uma mudança nesses pontos só entra numa
versão minor, com entrada em "Changed" no CHANGELOG, nunca num patch. O teste
`tests/tasks/test_task_api_contract.py` trava as assinaturas. Não há número de
versão da API; as notas de release são o aviso.

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
3. No config (YAML), `model.custom_model: seu_nome`. Na interface, a etapa
   Modelo da aba Classificação tem o campo **Modelo próprio**, que lista os
   nomes registrados em `user_models/`: escolher um preenche `custom_model` e
   esconde Pesos pré-treinados; **Nenhum** limpa o valor, e a Arquitetura volta
   a treinar. Sem nada registrado, o campo diz como adicionar um modelo. Os campos
   `name` e `pretrained` do modelo embutido passam a ser ignorados. Um YAML
   roda com `visionforge run arquivo.yaml` ou entra na interface por
   **↑ Importar YAML**, na etapa Revisão da aba Classificação. Nos dois casos as
   etapas Modelo e Revisão mostram o seu modelo, e o Histórico lista a execução
   com o nome dele. Num servidor compartilhado (`visionforge gui --server`) o
   campo não aparece e um treino com `custom_model` é recusado, porque um modelo
   próprio é código trazido pelo usuário.

O mesmo modelo serve para classificação, segmentação e regressão, porque o
argumento é sempre a dimensão de saída da tarefa: `num_classes` nas duas
primeiras, `num_targets` na regressão.
