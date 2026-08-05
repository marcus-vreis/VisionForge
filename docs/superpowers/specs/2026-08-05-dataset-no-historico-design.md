# Dataset no histórico de runs

**Data:** 2026-08-05
**Estado:** desenhado, aprovado, não implementado
**Escopo:** sub-projeto (a) de quatro. Ver "Decomposição" no fim.

## Problema

O histórico mostra nome do experimento, arquitetura, tarefa, épocas e métricas —
mas não diz **em qual dataset o run foi treinado**. Comparar duas linhas do
histórico sem essa informação obriga a abrir cada run e ler a config.

## O que já existe (verificado, não suposto)

Três peças estão construídas e não aparecem na interface:

1. **`config.data.base_dir`** está em **28 de 28** `run.json` de `outputs/models/`.
   O nome do dataset é recuperável para o histórico inteiro.
2. **`dataset_fingerprint`** (ADR-061) grava `digest`, `method`, `n_files`,
   `total_bytes`, `root` e `note` a cada treino — mas só a partir do commit
   `5fb8bbb`, de **2026-07-26**. Na prática: **2 de 28** runs têm. Todas as
   tarefas gravam, detecção inclusive (`cats-dogsv2.v1i.yolov8`, 559 arquivos).
3. **`same_dataset(a, b)`** (`src/visionforge/core/dataset_fingerprint.py:148`)
   já responde "mesmos dados?" e já devolve `None` quando a pergunta não pode ser
   respondida — digest ausente, ou métodos diferentes entre os dois runs.

A consequência de (1) e (2) juntos define o produto: **reconhecer e reencontrar
são retroativos; verificar não é.** O desenho não pode prometer o contrário.

## Decisões

### 1. Origem do nome

Helper `dataset_identity(run_json)` em
`src/visionforge/core/dataset_fingerprint.py` — ao lado de `same_dataset`, que é
a outra função que lê o mesmo bloco — com precedência:

```
dataset_fingerprint.root  →  se ausente  →  config.data.base_dir  →  se ausente  →  None
```

Nome exibido = último segmento do caminho. Vale para caminho relativo
(`datasets/USK-COFFEE`) e absoluto. `None` significa card sem selo, não card com
selo vazio.

**Ressalva registrada:** uma tarefa custom que *sintetiza* os próprios dados usa
`base_dir` como marcador, não como dataset (ver docstring de
`fingerprint_from_config`). O selo vai mostrar o nome dessa pasta. Não há nenhum
run custom em `outputs/models/` para confirmar como fica na prática, então a
regra fica uniforme e a ressalva fica escrita — inventar uma exceção sem
evidência seria pior.

### 2. Superfície da API

`RunSummary` ganha dois campos opcionais:

- `dataset_name: str | None`
- `dataset_root: str | None`

`RunDetail` ganha um bloco `dataset: DatasetInfo | None`, onde `DatasetInfo` é um
`BaseModel` novo em `schemas.py` com `name`, `root`, `n_files`, `total_bytes`,
`method`, `digest`, `note` — todos opcionais menos `name` e `root`, porque um run
sem fingerprint tem os dois primeiros e nenhum dos outros.

Nenhum endpoint novo. Ambos são derivados do `run.json` que a rota já lê.

### 3. Card do histórico

Selo `🗂 <nome>` na fileira de pills que já existe, ao lado de `⚗ N filtros`,
mesma linguagem visual (`HistoryOverlay.tsx`, segunda fileira do card). O
atributo `title` carrega o caminho completo — é a camada "reencontrar" sem gastar
largura. Nome longo é cortado com reticências no próprio selo (a fileira já tem
`flexWrap`, então o selo nunca empurra os outros para fora).

Alternativas descartadas: linha própria abaixo do nome (rouba a linha mais nobre
por informação de consulta ocasional) e agrupamento da lista por dataset (muda
navegação; ver "Fora de escopo").

### 4. Detalhe do run

Bloco "Dataset": nome, caminho completo, nº de arquivos, tamanho, método e os
**12 primeiros caracteres** do digest (o suficiente para conferir de olho a
diferença entre dois runs; o valor inteiro fica no `title`).

Sem fingerprint, mostra caminho e a frase `sem fingerprint — run anterior a
26/07/2026`. A data vem do commit que introduziu o campo, não de estimativa.

### 5. Comparador

`CompareRunsPanel` já busca `fetchRunDetail(id)` para cada run, então o veredito
sai no cliente sem endpoint novo. Três estados, espelhando `same_dataset`:

| Estado | Quando |
|---|---|
| `✓ mesmos dados` | digests presentes, mesmo método, iguais |
| `✗ dados diferentes` | digests presentes, mesmo método, diferentes |
| `⚠ mesmo caminho, não verificável` | falta digest em algum, ou métodos diferentes — com o motivo |

**Duplicação consciente:** a regra vive em Python (`same_dataset`) e será
reescrita em ~4 linhas de TypeScript, com testes vitest cobrindo os mesmos três
casos. Criar um endpoint para comparar dois campos custaria mais do que a
duplicação, e o terceiro estado — o que admite não saber — é o que não pode ser
perdido na tradução.

## Fora de escopo, deliberadamente

Registrado para não voltar por engano:

- **Agrupar/filtrar o histórico por dataset.** Boa ideia, mas é decisão de
  navegação e vale sozinha, depois que o selo existir.
- **Backfill de fingerprint em runs antigos.** Um botão "verificar agora"
  percorreria a pasta e geraria o digest. O problema é que o hash de *hoje* não
  descreve o que a pasta era *naquele dia*: se o dataset mudou desde então, a
  tela responderia "iguais" sobre dois runs que viram dados diferentes — o erro
  exato que o fingerprint existe para impedir. Só seria aceitável rotulado como
  "verificado agora, não no treino".

## Testes

**Backend**

- Derivação do nome: `root` do fingerprint presente; fallback para `base_dir`;
  ambos ausentes → `None`; caminho relativo e absoluto dando o mesmo nome.
- Contrato: `RunSummary` e `RunDetail` carregam os campos novos, e um `run.json`
  antigo (sem `dataset_fingerprint`) continua parseando.

**Frontend**

- Os três estados do veredito de comparação, incluindo métodos divergentes.
- Card com e sem selo.

`same_dataset` já tem cobertura em `tests/core/test_dataset_fingerprint.py:139`.

## Decomposição

Este é o primeiro de quatro sub-projetos acordados, nesta ordem:

- **(a) Dataset no histórico** — este documento.
- **(b) Toggle de data augmentation** nas cinco tarefas: desligado por padrão
  visualmente, parâmetros aparecem só quando ligado.
- **(c) Paridade do painel de detecção** com o de classificação: nº de classes
  junto do dataset, exemplos das classes, preview de augmentation,
  pré-processamento/filtros — hoje detecção é a única tarefa sem
  `TransformsSection`.
- **(d) Reapresentação dos parâmetros** — reorganizar como são exibidos, sem
  eliminar nenhum.

Auto-detecção de splits (`detectDatasetSplits`, `/dataset/detect`) já existe e é
usada só por `DatasetPicker`, que por sua vez só é usado pelo painel de
classificação. Reusá-la nas outras tarefas entra em (c).
