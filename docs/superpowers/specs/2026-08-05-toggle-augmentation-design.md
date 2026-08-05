# Toggle de data augmentation

**Data:** 2026-08-05
**Estado:** desenhado, aprovado, não implementado
**Escopo:** sub-projeto (b) de quatro.

## Problema

Os parâmetros de augmentation ficam sempre à vista, e não há forma de desligar a
augmentation sem zerar cada campo à mão. Em detecção são **15 campos**, então
"tirar um baseline sem augmentation" é um trabalho manual que ninguém faz.

## O que existe hoje

**Quatro tarefas compartilham o mesmo `TransformConfig`** (`utils/config.py:192`):
classificação, regressão, segmentação e anomalia.

| campo | é augmentation? |
|---|---|
| `image_size` | não |
| `normalize_mean` / `normalize_std` | não |
| `horizontal_flip` | sim |
| `rotation_degrees` | sim |
| `color_jitter` | sim |

A separação já existe no código: em `_build_transforms` (`core/data.py`) o flip,
a rotação e o jitter estão dentro do `if is_train`, enquanto resize e
normalização são aplicados aos três splits. **A UI é que não faz a distinção** —
a seção se chama "Augmentação & normalização" e mistura os dois.

**Detecção é o outlier:** `DetectionAugmentationConfig`
(`utils/detection_config.py:164`) tem 15 campos em nomenclatura Ultralytics
(`hsv_h/s/v`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`,
`fliplr`, `bgr`, `mosaic`, `mixup`, `copy_paste`, `auto_augment`, `erasing`).

## Decisões

### 1. Flag explícito, não valores neutros inferidos

`augment: bool = True` entra em `TransformConfig` e em
`DetectionAugmentationConfig`. Quando falso, `_build_transforms` pula os passos
de augmentation e o trainer de detecção passa os valores neutros à Ultralytics.

A alternativa considerada era inferir o estado ("está ligado se algum campo está
fora do neutro") e gravar valores neutros ao desligar. Recusada por três motivos:

- O `run.json` passa a **registrar** que a augmentation estava desligada, em vez
  de deixar isso para inferência posterior — que erra se alguém zerou os campos à
  mão por outro motivo.
- Os valores do pesquisador sobrevivem ao ciclo desliga/religa **e à
  exportação**: com valores neutros gravados, um YAML exportado com a
  augmentation desligada perde os ajustes, que só existiriam na memória do
  navegador.
- Em detecção são 15 valores a restaurar contra 1 booleano.

Campo novo com default é retrocompatível no Pydantic: config antigo carrega e
vira `augment=True`, sem migração.

### 2. Desligado esconde os parâmetros

Uma chave "Data augmentation", e enquanto desligada os campos de augmentation
**somem**, substituídos por um rótulo dizendo quantos estão guardados
(`3 parâmetros ocultos` / `15 parâmetros ocultos`).

Alternativa considerada: manter os campos visíveis e desabilitados. Recusada
porque em detecção deixaria 15 linhas cinzas intocáveis na tela — exatamente a
poluição visual que motivou o pedido.

Os valores continuam no formulário e no payload; esconder é só apresentação.

### 3. Tamanho e normalização nunca escondem

`image_size` e `normalize_mean`/`normalize_std` saem da seção de augmentation e
vão para uma seção "Imagem", sempre visível. Não são augmentation: valem para
treino, validação e teste. Escondê-los junto seria dizer que não se aplicam.

Isso divide a atual "Augmentação & normalização" em duas seções, o que é a
correção de um erro de rotulagem, não uma preferência.

## Fora de escopo

- Preview de augmentation em detecção — entra em (c), e lá a decisão é não
  fingir um preview aproximado do pipeline da Ultralytics.
- Presets de augmentation ("leve/médio/agressivo"). Ideia razoável, decisão
  separada.

## Testes

**Backend**

- `augment=False` remove flip, rotação e jitter do pipeline de treino, e mantém
  resize e normalização.
- `augment=True` (default) preserva o comportamento atual — um config sem o
  campo carrega e treina como antes.
- Detecção: `augment=False` passa os valores neutros à Ultralytics; os campos do
  config permanecem intactos.
- `run.json` registra o estado do flag.

**Frontend**

- Round-trip `formFromPayload(buildPayload(form)) == form` com o flag em ambos os
  estados (o contrato de import/export de YAML da ADR-059).
- Contagem de parâmetros ocultos por tarefa: 3 nas quatro que usam
  `TransformConfig`, 15 em detecção.
