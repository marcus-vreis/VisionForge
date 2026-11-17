# Quickstart — from zero to a defensible result

This walkthrough goes from nothing installed to a trained classifier with
multi-seed confidence intervals, using only what ships with VisionForge.
Time budget: ~10 minutes on any machine (CPU works; GPU is faster).

> 🇧🇷 **Este guia também está em português** — [role até a versão em
> português](#português-pt-br).

## 1. Install

The published package already contains the built interface, so this is all it
takes:

```bash
mkdir my-research && cd my-research       # runs and your code land here
python -m venv .venv
.venv\Scripts\activate                   # Linux/macOS: source .venv/bin/activate
pip install visionforge-studio
visionforge doctor --fix                  # reads your GPU, installs the matching torch
```

`doctor` reads your driver and any torch already present, names the build your
machine needs, and swaps it in once you answer `y`. On Windows the plain install
always brings a CPU torch first (PyPI has no other there), so on a GPU machine
that swap is expected. To skip it, install torch before the package:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`,
then `pip install visionforge-studio` (`cu130` · `cu126` · `cpu` also work in
place of `cu128`; `cu126` runs on any CUDA 12 driver, and a CUDA 11 driver needs
an update to train on the GPU, ADR-209).

Working from a source checkout instead? That path is in
[`CONTRIBUTING.md`](../CONTRIBUTING.md) — the rest of this walkthrough is
identical either way.

Before pointing it at real data, confirm the install actually trains — this
runs every task through the real API on synthetic data and needs no dataset,
GPU or network:

```bash
visionforge selftest --quick
```

## 2. Launch the GUI and get a dataset

```bash
visionforge gui                           # opens http://127.0.0.1:8000
```

The interface comes in English and Portuguese: the **PT** / **EN** switch in the
header changes it at any time, and the choice is remembered. This guide names
the English labels.

You don't need your own data to try it: click **⤓ datasets** in the bottom bar,
pick the **torchvision** provider, dataset `CIFAR10`,
choose an output folder and a per-class `limit` (e.g. 300 — keeps the first
run fast). VisionForge materializes it as an ImageFolder
(`<out>/train/<class>/*.png`, `<out>/test/...`) — the exact layout the
classification task consumes.

## 3. First training run

The task tabs configure a run in six steps — **Data**, **Strategy**,
**Model**, **Training**, **Image & augmentation** and **Review** — and the
**▶ Train** button on the bar below stays locked until a dataset is chosen in
the first one. On the **Classification** tab:

1. **Data**: point the dataset picker at the folder you just downloaded — split
   auto-detection and per-class stats render immediately; click
   **🎯 apply 10 classes** (the count it detected for CIFAR10) to inject the
   class count.
2. Leave **Strategy**, **Model** and **Image & augmentation** on their defaults:
   a single run of `resnet18` with pretrained weights. **Training** defaults
   to 20 epochs; lower it to 3 for a smoke run on CPU.
3. **Review**: check the summary, give the run a name, and press **▶ Train**.
   The overlay streams per-epoch loss/accuracy live.

When it finishes you get metric tiles, confusion matrices, ROC/PR curves, and
a markdown model card. Everything is also on disk under
`outputs/models/<run>/` — including `run.json` with the full config, seed,
per-epoch history and environment (torch/CUDA/cuDNN/GPU) that produced it.

## 4. Make the number defensible

A classification run already hands you part of the answer: each test metric
carries a 95% bootstrap interval (`0.8734 [0.8412, 0.9021]`) under the tile and
in the model card. Read it for what it is — it resamples the *test split* with
the model held fixed, so it says how much the number depends on which images
you happened to test on. It says nothing about whether retraining would land in
the same place.

For that, a single run is one sample from a noisy distribution. On every
task tab, **Classification** included, the strategy step has **Replicates**:
same config, N seeds, and the report gives you `metric = mean ± 95% CI` plus
the per-seed table — the citable version of your result. Classification ranks
the seeds on accuracy. Replicates run on this computer only: the Colab button
stays off, with the reason, while Replicates is picked. **K-fold (CV)** and
**Grid search** (add values to the *Architecture* field with “+ add to grid”)
sit on the same step, and the **History** overlay compares the runs.

To ask whether a change beats your configuration, pick **Replicated
comparison** on the same step. The form is the baseline: name one field
(`training.learning_rate`, `model.name`…) and 1 to 3 other values for it.
Every variant trains on the same seeds (2 to 20), so the cost is variants ×
seeds, which the card shows before you start. Each pair then gets a paired test
(t or Wilcoxon) with Holm correction, and History shows the matrix. Use 6 seeds
or more: with 5 or fewer, Wilcoxon cannot return a p below 0.05, and the card
warns you of it.

## 5. Reproduce it anywhere

The **Review** step has **↓ Export YAML** — the exported file is the exact
config the API received, so this reruns the experiment identically:

```bash
visionforge run experiment_my_test_2026-07-02.yaml
```

Import the same file back through **↑ Import YAML** to restore the form.

### Replicates, comparisons, sweeps and K-fold from the command line

The strategies of section 4 run without the GUI too. Wrap the task config under
`config:` and say what to do with it under `strategy:`; the options are the
ones the GUI sends, and `visionforge run --help` lists them. The job writes the
same report as the GUI (`*_summary.json`, `*_ranking.csv` and a `*_table.tex`
under `output.reports_dir`) and one History entry under `output.models_dir`.
A YAML without `strategy:` trains once, as before.

Replicates (mean ± 95% CI per metric, any task):

```yaml
strategy:
  kind: replicates
  seeds: [0, 1, 2, 3, 4]      # or n_replicates: 5, counted from training.seed
config:
  task: regression
  name: reg_resnet18
  # ... the rest of the exported task config
```

Paired comparison (every variant trains on the same seeds, then paired tests
with Holm correction; any task):

```yaml
strategy:
  kind: replicated_comparison
  seeds: [0, 1, 2, 3, 4]
  variants:
    baseline: {}
    low_lr: {training.learning_rate: 0.0001}
  alpha: 0.05
config:
  task: regression
  # ...
```

Sweep (ranked on the validation split; detection, regression, segmentation,
anomaly; classification uses `block: grid_search` or `random_search`):

```yaml
strategy:
  kind: sweep
  mode: grid                  # or random / optuna, with n_trials
  search_space:
    training.learning_rate: [0.001, 0.0003]
    training.batch_size: [16, 32]
config:
  task: segmentation
  # ...
```

K-fold (regression and segmentation; classification uses
`block: cross_validation`):

```yaml
strategy:
  kind: cv
  n_folds: 5
  fold_seed: 42
config:
  task: regression
  # ...
```

## 6. Define your own task (optional)

If your problem isn't classification/detection/regression/segmentation/anomaly:

```bash
visionforge new-task my_task      # generates user_tasks/my_task.py — it already trains
```

Fill the four hooks with your model/data/loss/metrics and the task gets the
run endpoint, live monitor, `run.json` provenance, sweeps and replicates for
free. Walkthrough: [`docs/custom/TASKS.md`](custom/TASKS.md).

## Where to go next

- [Export a checkpoint to a private Hugging Face repository (Portuguese)](gui/hugging-face-hub.md)
- [Track training in local MLflow (Portuguese)](gui/mlflow.md)
- [Send training to Weights & Biases (Portuguese)](gui/wandb.md)
- [Send training to Comet (Portuguese)](gui/comet.md)
- [Create a Zenodo draft (Portuguese)](gui/zenodo.md)
- [Send a run report to GitHub (Portuguese)](gui/github.md)
- [Version a run with DVC (Portuguese)](gui/dvc.md)

- [`README.md`](../README.md) — what VisionForge does, in English and Portuguese
- [`docs/DATASETS.md`](DATASETS.md) — dataset layouts and the download providers
- [`docs/custom/`](custom/) — your own architectures and whole task families
- [`docs/dev/ARCHITECTURE.md`](dev/ARCHITECTURE.md) — how the pieces fit

---

# Português (pt-BR)

Este roteiro vai de nada instalado a um classificador treinado, com intervalos
de confiança de várias seeds, usando só o que vem com o VisionForge. Tempo
estimado: ~10 minutos em qualquer máquina (funciona em CPU; GPU é mais rápido).

## 1. Instalar

O pacote publicado já traz a interface compilada, então basta:

```bash
mkdir my-research && cd my-research       # as execuções e o seu código ficam aqui
python -m venv .venv
.venv\Scripts\activate                   # Linux/macOS: source .venv/bin/activate
pip install visionforge-studio
visionforge doctor --fix                  # lê a sua GPU, instala o torch correspondente
```

O `doctor` lê o seu driver e qualquer torch já presente, diz qual build a sua
máquina precisa e faz a troca quando você responde `y`. No Windows, a
instalação simples sempre traz primeiro um torch de CPU (o PyPI não tem outro
lá), então em uma máquina com GPU essa troca é esperada. Para evitá-la,
instale o torch antes do pacote:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`,
depois `pip install visionforge-studio` (`cu130`, `cu126` e `cpu` também
servem no lugar de `cu128`; o `cu126` roda com qualquer driver CUDA 12, e um
driver CUDA 11 precisa ser atualizado para treinar na GPU, ADR-209).

Vai trabalhar a partir de um checkout do código-fonte? Esse caminho está em
[`CONTRIBUTING.md`](../CONTRIBUTING.md); o resto deste roteiro é igual nos dois
casos.

Antes de apontar para dados reais, confirme que a instalação realmente treina.
O comando abaixo roda todas as tarefas pela API real com dados sintéticos e não
precisa de dataset, GPU nem rede:

```bash
visionforge selftest --quick
```

## 2. Abrir a interface e obter um dataset

```bash
visionforge gui                           # abre http://127.0.0.1:8000
```

A interface existe em inglês e português: a chave **PT** / **EN** no cabeçalho
troca o idioma a qualquer momento, e a escolha fica guardada. Este roteiro
usa os nomes em português.

Você não precisa de dados próprios para experimentar: clique em **⤓ datasets**
na barra inferior, escolha o provedor **torchvision**, o dataset `CIFAR10`,
uma pasta de saída e um **Limite por classe** (`limit`, por exemplo 300, para
o primeiro treino ser rápido). O VisionForge o materializa como um ImageFolder
(`<out>/train/<class>/*.png`, `<out>/test/...`), exatamente o formato que a
tarefa de classificação consome.

## 3. Primeiro treino

As abas de tarefa configuram um treino em seis etapas: **Dados**,
**Estratégia**, **Modelo**, **Treinamento**, **Imagem e aumento** e
**Revisão**. O botão **▶ Treinar** da barra abaixo fica travado até um dataset
ser escolhido na primeira. Na aba **Classificação**:

1. **Dados**: aponte o seletor de dataset para a pasta que você acabou de
   baixar. A detecção automática dos splits e as estatísticas por classe
   aparecem na hora; clique em **🎯 aplicar 10 classes** (o número que ele
   detectou no CIFAR10) para preencher a contagem de classes.
2. Deixe **Estratégia**, **Modelo** e **Imagem e aumento** no padrão: um único
   treino de `resnet18` com pesos pré-treinados. **Treinamento** vem com 20
   épocas; baixe para 3 se quiser só um teste rápido em CPU.
3. **Revisão**: confira o resumo, dê um nome ao treino e pressione
   **▶ Treinar**. A tela de acompanhamento mostra loss e acurácia de cada
   época ao vivo.

Quando termina, você recebe os cartões de métricas, as matrizes de confusão, as
curvas ROC/PR e um model card em markdown. Tudo também fica em disco, em
`outputs/models/<run>/`, incluindo o `run.json` com a configuração completa, a
seed, o histórico por época e o ambiente (torch/CUDA/cuDNN/GPU) que o produziu.

## 4. Torne o número defensável

Um treino de classificação já entrega parte da resposta: cada métrica de teste
traz um intervalo bootstrap de 95% (`0.8734 [0.8412, 0.9021]`) sob o cartão e
no model card. Leia-o pelo que ele é: ele reamostra o *split de teste* com o
modelo fixo, então diz o quanto o número depende de quais imagens você usou no
teste. Ele não diz nada sobre se um novo treino cairia no mesmo lugar.

Para isso, um único treino é uma amostra de uma distribuição ruidosa. Em toda
aba de tarefa, **Classificação** incluída, a etapa de estratégia tem
**Réplicas**: mesma configuração, N seeds, e o relatório entrega
`métrica = média ± IC 95%` mais a tabela por seed, a versão citável do seu
resultado. A classificação ordena as seeds pela acurácia. As réplicas rodam só
neste computador: o botão do Colab fica desligado, com o motivo, enquanto
Réplicas estiver escolhida. **K-Fold (CV)** e **Grid search** (adicione
valores ao campo *Arquitetura* com “+ valor ao grid”) ficam na mesma etapa, e a
tela **Histórico** compara os treinos.

Para perguntar se uma mudança supera a sua configuração, escolha **Comparação
replicada** na mesma etapa. O formulário é a linha de base: nomeie um campo
(`training.learning_rate`, `model.name`…) e 1 a 3 outros valores para ele.
Cada variante treina nas mesmas seeds (2 a 20), então o custo é variantes ×
seeds, que o cartão mostra antes de você começar. Cada par recebe então um
teste pareado (t ou Wilcoxon) com correção de Holm, e o Histórico mostra a
matriz. Use 6 seeds ou mais: com 5 ou menos, o Wilcoxon não consegue devolver
um p abaixo de 0,05, e o cartão avisa disso.

## 5. Reproduza em qualquer lugar

A etapa **Revisão** tem **↓ Exportar YAML**: o arquivo exportado é exatamente a
configuração que a API recebeu, então este comando repete o experimento de
forma idêntica:

```bash
visionforge run experiment_my_test_2026-07-02.yaml
```

Importe o mesmo arquivo de volta com **↑ Importar YAML** para restaurar o
formulário.

### Réplicas, comparações, sweeps e K-fold pela linha de comando

As estratégias da seção 4 também rodam sem a interface. Envolva a configuração
da tarefa em `config:` e diga o que fazer com ela em `strategy:`; as opções são
as mesmas que a interface envia, e `visionforge run --help` as lista. O job
escreve o mesmo relatório da interface (`*_summary.json`, `*_ranking.csv` e um
`*_table.tex` em `output.reports_dir`) e uma entrada de Histórico em
`output.models_dir`. Um YAML sem `strategy:` treina uma vez, como antes.

Réplicas (média ± IC 95% por métrica, qualquer tarefa):

```yaml
strategy:
  kind: replicates
  seeds: [0, 1, 2, 3, 4]      # ou n_replicates: 5, contado a partir de training.seed
config:
  task: regression
  name: reg_resnet18
  # ... o resto da configuração da tarefa exportada
```

Comparação pareada (cada variante treina nas mesmas seeds, depois testes
pareados com correção de Holm; qualquer tarefa):

```yaml
strategy:
  kind: replicated_comparison
  seeds: [0, 1, 2, 3, 4]
  variants:
    baseline: {}
    low_lr: {training.learning_rate: 0.0001}
  alpha: 0.05
config:
  task: regression
  # ...
```

Sweep (ordenado pelo split de validação; detecção, regressão, segmentação e
anomalia; a classificação usa `block: grid_search` ou `random_search`):

```yaml
strategy:
  kind: sweep
  mode: grid                  # ou random / optuna, com n_trials
  search_space:
    training.learning_rate: [0.001, 0.0003]
    training.batch_size: [16, 32]
config:
  task: segmentation
  # ...
```

K-fold (regressão e segmentação; a classificação usa
`block: cross_validation`):

```yaml
strategy:
  kind: cv
  n_folds: 5
  fold_seed: 42
config:
  task: regression
  # ...
```

## 6. Defina a sua própria tarefa (opcional)

Se o seu problema não é classificação, detecção, regressão, segmentação nem
anomalia:

```bash
visionforge new-task my_task      # gera user_tasks/my_task.py, que já treina
```

Preencha os quatro hooks com o seu modelo, dados, loss e métricas, e a tarefa
ganha de graça o endpoint de execução, o monitor ao vivo, a proveniência em
`run.json`, sweeps e réplicas. Passo a passo:
[`docs/custom/TASKS.md`](custom/TASKS.md).

## Para onde ir depois

- [Exportar um checkpoint para um repositório privado do Hugging Face](gui/hugging-face-hub.md)
- [Acompanhar o treino no MLflow local](gui/mlflow.md)
- [Enviar o treino para o Weights & Biases](gui/wandb.md)
- [Enviar o treino para o Comet](gui/comet.md)
- [Criar um rascunho no Zenodo](gui/zenodo.md)
- [Enviar o relatório de um run para o GitHub](gui/github.md)
- [Versionar um run com o DVC](gui/dvc.md)

- [`README.md`](../README.md): o que o VisionForge faz, em inglês e português
- [`docs/DATASETS.md`](DATASETS.md): formatos de dataset e provedores de download
- [`docs/custom/`](custom/): suas próprias arquiteturas e famílias de tarefas
- [`docs/dev/ARCHITECTURE.md`](dev/ARCHITECTURE.md): como as peças se encaixam
