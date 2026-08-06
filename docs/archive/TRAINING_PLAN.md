# Plano de treinamento máximo — modelos e estratégias nativas

Plano operacional para exercitar **tudo o que o VisionForge oferece de fábrica**:
as cinco tasks nativas, todos os seus modelos e todas as suas formas de treino.
Tasks customizadas (ADR-058) estão fora de escopo — elas são definidas pelo
pesquisador e não têm matriz fixa.

O plano é **em camadas**: P0 → P3, custo crescente. Cada camada termina num
estado defensável, então parar no fim de qualquer uma delas é uma decisão
legítima, não um plano pela metade.

## O que este plano não é

Não é um benchmark publicável. Os números que ele produz medem *o sistema*
(cada modelo treina, cada estratégia fecha, cada relatório sai), não a
qualidade de um modelo num domínio real. Um resultado só vira reivindicação
científica depois de P2, onde entram réplicas com múltiplas seeds e teste de
significância.

## Pré-requisitos

```bash
visionforge doctor       # GPU/CUDA detectados + linha exata de instalação do torch
visionforge selftest     # o pipeline inteiro em dados sintéticos (~90s, CPU)
```

Se o `selftest` falhar, pare aqui: o plano abaixo só distingue "o modelo é
ruim" de "o sistema está quebrado" se o sistema estiver comprovadamente são.

Extras necessários para a matriz completa:

```bash
uv pip install -e ".[detection,timm,optuna,tensorboard]"
```

### Datasets — a restrição real

O download de um clique (`⤓ datasets` na barra inferior) cobre **apenas
classificação**: os built-ins do torchvision (CIFAR-10/100, MNIST,
Fashion-MNIST, KMNIST) materializam um `ImageFolder`. As outras quatro tasks
exigem layouts próprios, e cada uma precisa de um dataset providenciado antes
de começar:

| Task | Layout exigido | Origem prática |
|---|---|---|
| Classificação | `ImageFolder` (`train/val/test` × classe) | download embutido (CIFAR-10) |
| Detecção | layout YOLO (`images/` + `labels/`) ou `data.yaml` | Roboflow ou Kaggle |
| Regressão | manifesto CSV (coluna de imagem + colunas-alvo) | manifesto próprio |
| Segmentação | pares imagem/máscara (`images/` + `masks/`) | Kaggle / dataset próprio |
| Anomalia | layout MVTec (`train/good`, `test/<defeito>`) | MVTec-AD |

Escolha um dataset **pequeno** por task. O objetivo é cobertura da matriz, não
estado da arte; um dataset grande transforma P3 em semanas de GPU.

### Antes de rodar qualquer coisa

- **`training.deterministic`** (ADR-062): deixe **desligado** em P0/P1 (custa
  throughput) e **ligado** em P2/P3, onde a reprodutibilidade é o produto.
  Detecção já vem ligado por padrão, espelhando o `YOLO.train`.
- **Windows**: mantenha `training.workers` em 0–2. Cada worker é um processo
  que recarrega as DLLs CUDA do torch, e 8 deles esgotam o arquivo de paginação
  (`WinError 1455`).
- **Nomeie os experimentos.** O histórico agora tem uma aba por família; um
  nome como `cls_p1_resnet50` encontra o run meses depois, `experiment_001` não.

---

## Matriz oficial — o que existe para ser exercitado

### Modelos por task

| Task | Modelos nativos | Total |
|---|---|---|
| Classificação | resnet18/34/50/101, efficientnet_b1/b7, vgg16/19, alexnet | 9 |
| Regressão | mesmos 9 backbones CNN + cabeça linear | 9 |
| Segmentação | unet, deeplabv3_resnet50/101, deeplabv3_mobilenet_v3_large, fcn_resnet50/101, lraspp_mobilenet_v3_large | 7 |
| Anomalia | autoencoder; patchcore × backbone (resnet18/34/50, wide_resnet50_2) | 1 + 4 |
| Detecção · Ultralytics | YOLOv8/v9/v10/11/12/26 (5–6 variantes cada) + RT-DETR l/x | 32 |
| Detecção · torchvision | fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn, ssd300_vgg16, ssdlite320_mobilenet_v3_large | 5 |

`timm` e `user_models/` acrescentam centenas de backbones a classificação,
regressão e segmentação — fora da matriz nativa, mas o plano funciona igual
trocando o nome do modelo.

### Estratégias por task

Nem toda estratégia existe em toda task. Esta é a superfície real, conferida
contra as rotas da API:

| Estratégia | Class. | Detec. | Regr. | Segm. | Anom. |
|---|:--:|:--:|:--:|:--:|:--:|
| Treino simples | ✅ | ✅ | ✅ | ✅ | ✅ |
| Grid search | ✅ | ✅ | ✅ | ✅ | ✅ |
| Random search | ✅ | ✅ | ✅ | ✅ | ✅ |
| Optuna (TPE) | ✅ | ✅ | ✅ | ✅ | ✅ |
| K-fold (CV) | ✅ | — | ✅ | ✅ | — |
| Transfer learning | ✅ | — | ✅ | ✅ | — |
| Comparação de modelos | ✅ | ✅ | ✅ | ✅ | ✅ |
| Réplicas (N seeds) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Comparação replicada | ✅ | ✅ | ✅ | ✅ | ✅ |

Ausências são de projeto, não pendências: detecção delega o loop ao
Ultralytics (sem hook de fold), e anomalia treina só com imagens normais —
um fold de validação sem anomalias não mede nada.

Pós-treino, disponível conforme a task: teste em novo dataset, batch predict
para CSV, Grad-CAM, export ONNX com benchmark de latência, escalares no
TensorBoard.

---

## P0 — fumaça (≈ 15 runs, minutos)

**Objetivo:** provar que cada task treina de ponta a ponta *no seu dataset*,
antes de gastar GPU em qualquer matriz.

Para cada uma das cinco tasks: um treino simples, modelo mais barato, 2 épocas.

| Task | Modelo | Como |
|---|---|---|
| Classificação | resnet18 | painel Classificação → Treino simples |
| Detecção | yolo11n | painel Detecção → Treino simples |
| Regressão | resnet18 | painel Regressão → Treino simples |
| Segmentação | unet | painel Segmentação → Treino simples |
| Anomalia | autoencoder | painel Anomalia → Treino simples |

Depois, um de cada estratégia barata em **uma** task só (classificação):
grid 2×2, réplicas com 2 seeds, K-fold com 2 folds.

**Critério de saída:** cinco `run.json` gravados, cinco cards no histórico,
monitor ao vivo transmitindo épocas. Se o progresso não anda ou o relatório sai
vazio, o problema é de configuração — resolva antes de P1.

---

## P1 — linha de base por modelo (≈ 60 runs)

**Objetivo:** um número por modelo nativo, na mesma seed e no mesmo dataset.
Produz o ranking bruto que P2 vai testar estatisticamente.

- **Classificação** — 9 arquiteturas × treino simples, `seed=42`, mesmas
  épocas. Faça pela estratégia **Comparação** (grid de um eixo sobre
  Arquitetura), não à mão: sai um relatório único, ranqueado, com CSV.
- **Regressão** — os mesmos 9 backbones, mesma mecânica.
- **Segmentação** — 7 arquiteturas.
- **Anomalia** — autoencoder + patchcore nos 4 backbones = 5 configurações.
- **Detecção** — não rode as 37. Rode **uma variante `s` por família**
  (yolov8s, yolov9s, yolov10s, yolo11s, yolo12s, yolo26s) + `rtdetr-l` = 7, e
  os 5 do torchvision só se o backend importar para você.

**Ao terminar:** exporte o YAML de cada vencedor pelo `↓ EXPORTAR YAML` do
cabeçalho. Esses arquivos são a entrada de P2 e re-executam idênticos pela CLI.

---

## P2 — rigor (≈ 90 runs)

**Objetivo:** transformar "A ganhou de B" em afirmação defensável. É aqui que
o plano deixa de ser exercício e passa a produzir resultado citável.

Ligue `training.deterministic` a partir daqui.

1. **Réplicas** — top-3 de cada task × **5 seeds**. Sai
   `métrica = média ± IC 95%` (t de Student + bootstrap) em vez de um ponto.
2. **Comparação replicada** — os mesmos top-3 como variantes, sobre a
   **mesma lista de seeds**. Sai a matriz pareada com correção
   Holm-Bonferroni, Cohen's `d_z` e IC bootstrap da diferença.
3. **K-fold 5-fold** — no vencedor de classificação, regressão e segmentação.
   Mede sensibilidade à partição, que réplicas não medem.
4. **Transfer learning** — `feature_extraction` e `fine_tuning` no vencedor de
   classificação, regressão e segmentação: 3 tasks × 2 modos = 6 runs.

**Leia os avisos, não só os p-valores.** Com 5 seeds o Wilcoxon não alcança
p < 0.0625: o relatório marca `underpowered` e "não significativo" ali quer
dizer "poucas seeds", não "sem efeito". Se o resultado importa, suba para 8+
seeds nessa comparação específica.

**Saídas:** `replicates_summary.json`, matriz de significância, e a tabela
`.tex` (booktabs) gravada ao lado de cada relatório.

---

## P3 — varredura exaustiva (centenas de runs)

**Objetivo:** cobertura total. Só faz sentido com GPU sobrando e um dataset
pequeno.

- **Hiperparâmetros** — Optuna (TPE) por task, 50–100 trials sobre
  learning rate, batch size, otimizador, weight decay e scheduler. Prefira
  Optuna a grid: o grid gasta o mesmo orçamento em cantos do espaço que já se
  sabem ruins.
- **Detecção completa** — a escada de tamanhos (n/s/m/l/x) da família vencedora
  de P1, e depois as famílias restantes.
- **Todos os modelos × K-fold** — 9 + 9 + 7 arquiteturas × 5 folds.
- **Pós-treino** — no melhor checkpoint de cada task: teste em dataset novo,
  batch predict, Grad-CAM (classificação), export ONNX com benchmark de
  latência PyTorch × runtime.

---

## Orçamento e ordem

| Camada | Runs (ordem de grandeza) | Entrega |
|---|---|---|
| P0 | ~15 | o sistema treina nos seus dados |
| P1 | ~60 | um número por modelo nativo |
| P2 | ~90 | intervalos, p-valores corrigidos, tabelas LaTeX |
| P3 | centenas | cobertura exaustiva |

Rode nesta ordem. P1 sem P0 desperdiça horas quando o dataset está no layout
errado; P2 sem P1 replica modelos que nunca foram triados; P3 antes de P2 gera
volume sem nenhuma afirmação testada.

Hoje o VisionForge executa **um treino por vez** — um segundo envio recebe 409.
Uma fila FIFO está no backlog (`TASKS.md`); até lá, P3 é uma sessão supervisionada
ou uma sequência de chamadas de CLI num script:

```bash
visionforge run configs/cls_resnet50.yaml
visionforge run configs/cls_efficientnet_b1.yaml
```

## Registro dos resultados

Cada run grava um `run.json` versionado com config completa, seed, histórico
por época, `environment` (Python, torch, CUDA, cuDNN, GPU) e
`dataset_fingerprint`. É o que permite responder "quais dados produziram este
número" meses depois — desde que o dataset não seja re-exportado no meio do
plano. Se ele mudar, o digest muda, e comparar as duas metades vira comparação
entre datasets diferentes.

Trabalhe uma família por vez usando as abas do histórico, e exporte o YAML de
cada configuração que valeu a pena. O YAML é a unidade reproduzível; o
checkpoint, não.
