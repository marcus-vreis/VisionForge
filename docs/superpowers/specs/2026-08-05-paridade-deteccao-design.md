# Paridade do painel de detecção

**Data:** 2026-08-05
**Estado:** desenhado, aprovado, não implementado
**Escopo:** sub-projeto (c) de quatro.

## Problema

O painel de detecção é o mais distante do de classificação, que a ADR-059 já
ratificou como contrato canônico. Quatro lacunas, relatadas de uso real.

## Decisões

### 1. `num_classes` vai para a seção Dataset

Classificação **já fez isso** (`ParamPanel.tsx:2319` — *"num_classes movido para
dataset → classes"*) e o deriva da contagem do dataset (`:2450`). Detecção ficou
para trás: o campo está na seção Modelo (`DetectionPanel.tsx:255`) e existe um
botão lá embaixo, no Dataset, que empurra o valor para cima — configura-se num
lugar e confere-se em outro.

Não há conflito com a ADR-059: ela ordena *seções*; classificação moveu o *campo*
entre seções depois, e detecção não acompanhou.

### 2. Exemplos das classes no dataset

`DetectionDatasetStats` mostra contagens; `DatasetStats` (classificação) mostra
imagens de exemplo por classe. Detecção passa a mostrar amostras também,
recortando a primeira caixa anotada de cada classe.

### 3. Auto-detecção de splits

Detecção **já tem a sua**, interna: `DetectionDataModule._detect_splits`
(`core/detection_data.py:67`). O que falta é a paridade de interface — o painel
não mostra o que foi detectado antes de treinar, como o `DatasetPicker` de
classificação faz. Expor o resultado, não reimplementar a detecção.

### 4. Pré-processamento: materializar uma cópia filtrada

`DetectionConfig` não tem campo `preprocessing`, e com o backend Ultralytics —
que é o default — `model.train(data=data_yaml)`
(`core/detection_trainer.py:274`) entrega o pipeline inteiro para a biblioteca.
Não há como injetar um filtro PIL no DataLoader dela sem subclassar sua dataset
interna, o que amarraria o projeto à versão da Ultralytics.

**Decisão:** aplicar os filtros uma vez, gravar o resultado numa pasta
temporária, treinar a partir dela e apagá-la ao fim.

Isso é mais barato do que parece, e não só um contorno: hoje o filtro roda **por
imagem, por época** (`_PreprocessingTransform` em `_build_transforms`).
Materializado, roda **uma vez** — em 30 épocas com um filtro caro (CLAHE,
bilateral) é ~30× menos CPU.

A integração com a Ultralytics é de uma linha: `resolve_data_yaml`
(`core/detection_data.py:50`) já sintetiza um `data.yaml` com `path: <base>`;
basta esse `path` apontar para a cópia.

Quatro guardas, sem as quais isso vira armadilha:

**Fingerprint do original, nunca da cópia.** Caso contrário o `run.json` grava o
digest e o caminho de algo que foi apagado, e o histórico do sub-projeto (a)
mostraria `🗂 tmp_a3f9`. O que se registra é: dataset original + a pipeline de
filtros aplicada.

**Limpeza em `finally` e varredura no startup.** Um run que morre no meio — o que
acontece de verdade, ver ADR-081 — deixaria uma cópia inteira do dataset em
disco. O `finally` cobre a exceção; a varredura ao subir a GUI cobre o processo
morto à força.

**PNG, não o formato original.** Regravar um dataset JPEG em JPEG aplica perda de
compressão *por cima* do filtro. O custo é real e deve ser dito: PNG de imagem
fotográfica costuma ocupar de 5 a 10 vezes o JPEG equivalente, então a
materialização mostra a estimativa de espaço antes de rodar. O formato fica
configurável para quem preferir trocar fidelidade por disco, com PNG no default.

**Chave por conteúdo, não por run.** A pasta é nomeada pelo hash de
(digest do dataset + pipeline canonicalizada), então um grid search de 20 trials
com os mesmos filtros materializa **uma vez**, não vinte. A pasta só é apagada
quando o último run que a usa termina.

Para detecção especificamente, a cópia leva junto os `.txt` de label e qualquer
arquivo não-imagem, verbatim. Filtrar imagem sem levar o rótulo dá treino
silenciosamente errado. A troca de extensão (`.jpg` → `.png`) é segura no
formato YOLO: o rótulo é casado pelo *stem* do arquivo, não pela extensão.

### 5. Preview de augmentation em detecção: não

Renderizar um preview exigiria reimplementar as transformações da Ultralytics
(mosaic, mixup, HSV, erasing) por fora dela — uma aproximação que pareceria
verdade sem ser. O painel mostra os valores; preview fica só onde o pipeline é
nosso.

## Consequência registrada

Detecção passa a usar um **mecanismo diferente de classificação para a mesma
feature**: on-the-fly lá, materializado aqui. Dois mecanismos para uma feature é
cheiro ruim e fica registrado como tal — a razão é concreta e não se aplica às
outras tarefas: a Ultralytics é dona do pipeline dela.

Migrar as outras quatro tarefas para materialização também (pelo ganho de CPU) é
uma decisão separada, deliberadamente fora deste escopo: o caminho on-the-fly
funciona e está coberto por testes.

## Testes

- A cópia materializada tem a mesma árvore de splits, os mesmos stems, e os
  `.txt` de label idênticos aos do original.
- Uma imagem da cópia difere da original quando há filtros, e o `data.yaml`
  aponta para a cópia.
- Pipeline vazia não materializa nada e o `data.yaml` aponta para o original.
- A mesma dupla (dataset, pipeline) reusa a pasta em vez de criar a segunda.
- Um run que levanta exceção não deixa a pasta para trás.
- A varredura de startup apaga pastas órfãs e preserva as em uso.
- O `run.json` registra o fingerprint do **original**.
- `num_classes` deriva do dataset, na seção Dataset.
