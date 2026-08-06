# Validação em datasets reais — 2026-07-29

Matriz completa de `(task, estratégia)` executada em **dados reais**, na GPU,
pelos mesmos endpoints que o navegador usa. Complementa o
`visionforge selftest`, que prova o mesmo caminho em dados sintéticos: aquele
roda em qualquer máquina em 90 s e entra no CI; este precisa dos datasets e da
GPU, e é o que se roda antes de confiar num release.

Registro do que foi medido, com o que passou, o que quebrou e o que os números
significam — e o que **não** significam.

## Corpus

Nada aqui é sintético. Todas as imagens e todos os rótulos vêm de datasets
públicos; o único trabalho local foi reorganizar arquivos no layout que cada
task lê.

| Task | Dataset | Origem | Tamanho usado |
|---|---|---|---|
| Classificação | **USK-COFFEE** | já no disco | 8.000 fotos, 4 classes |
| Detecção | **cats-dogs v2** (layout YOLO) | export Roboflow | já no disco |
| Segmentação | **Oxford-IIIT Pet** | `thor.robots.ox.ac.uk` | 600 pares imagem/máscara |
| Regressão | **IMDB-WIKI** `wiki_crop` | `data.vision.ee.ethz.ch` | 2.300 fotos, idade 10–89 |
| Anomalia | **USK-COFFEE** | já no disco | 1.200 treino / 800 teste |

Duas construções merecem explicação, porque em ambas o rótulo vem do próprio
dataset e não de uma invenção nossa:

- **Anomalia** — o USK-COFFEE já rotula `defect`. `premium` vira a classe
  normal (treino não-supervisionado vê só ela) e `defect` vira a anomalia do
  split de teste. É a rotulagem original reexpressa no layout MVTec.
- **Segmentação** — os trimaps do Oxford-IIIT Pet são anotados por humanos.
  A classe 1 é o animal, 2 o fundo e 3 a borda; mapeamos para
  `{0: fundo, 1: animal}` e mandamos a borda para `ignore_index`, de modo que
  um pixel ambíguo nunca pontue.
- **Regressão** — a idade é `photo_taken − date_of_birth`, ambos campos que
  acompanham o release. Foram descartadas as linhas que a própria distribuição
  marca como ruins (sem face detectada, ou com uma segunda face que torna o
  rótulo ambíguo).

## Matriz

21 casos: cinco tasks × as estratégias que cada uma realmente tem. Duas épocas
por treino — o bastante para a loss se mover e todo artefato ser escrito.

| Task | simple | k-fold | transfer | grid | random |
|---|---|---|---|---|---|
| Classificação | acc 0.79 | mean acc 0.769 | acc 0.873 | ✅ | ✅ |
| Detecção | best_epoch 2 | — | — | mAP50-95 0.147 | 0.147 |
| Regressão | mse 160.5 | r² −0.195 | mse 123.6 | r² 0.362 | r² −0.011 |
| Segmentação | mIoU 0.872 | mIoU 0.866 | mIoU 0.895 | 0.899 | 0.894 |
| Anomalia | auroc 0.740 | — | — | auroc 0.762 | auroc 0.741 |

**21/21 passaram** depois da correção descrita abaixo. Ausências são de
projeto, não pendências: detecção não tem K-fold (o Ultralytics é dono do loop)
nem seção de transfer learning; anomalia é não-supervisionada, e um fold de
validação sem anomalias não mede nada.

Critério de aprovação, herdado do `run_case` do selftest: o run completa, o
relatório carrega as chaves que aquela task de fato devolve, e o SSE entrega
progresso (`epoch_end` em run único, `trial_start`/`trial_end` em multi-trial).

## O defeito que a matriz encontrou

**`classification/transfer` não emitia nenhum evento SSE.** O treino rodava e o
relatório saía correto, mas a barra de progresso da GUI ficava morta o treino
inteiro. `TransferLearningBlock` nunca aceitou um `progress_callback`, e o
`routes.py` o deixava de fora do `isinstance` que liga a bomba de eventos —
com um comentário que declarava a lacuna como se fosse decisão.

O selftest sintético não pegava porque transfer learning é um *block* de
classificação, não uma das estratégias que ele enumera. Corrigido em ADR-065,
com dois testes de regressão: um garante que `epoch_end` chega ao callback, o
outro garante que `routes._execute_experiment` continua citando o block — um
block que aceita um callback que ninguém liga tem o mesmo defeito.

## Três falhas que eram do harness, não do produto

Registradas para não serem reinvestigadas:

- O relatório de anomalia é `train`/`test`, não `anomaly`.
- A métrica do sweep de anomalia é `auroc`, não `image_auroc`.

A segunda apareceu como
`RuntimeError: No sweep reported the metric 'image_auroc' — available: ['auroc', 'image_f1']`.
Isso é o guard de métrica não reportada (ADR-060) funcionando: em vez de
ranquear todos os trials como 0.0 e coroar um vencedor arbitrário, o sweep
falhou alto e nomeou as métricas disponíveis.

## O que estes números não são

Não são avaliação de modelo. Duas épocas sobre 1.500 faces não aprendem idade —
daí o r² negativo em algumas estratégias da regressão, que significa "pior que
prever a média". Isso é orçamento de treino, não defeito de sistema.

A matriz prova que **o sistema funciona**: cada modelo treina, cada estratégia
fecha, cada relatório sai com a forma certa e cada run transmite progresso. Um
número defensável exige o caminho de `docs/archive/TRAINING_PLAN.md` — camada
P2, com múltiplas seeds e teste de significância.
