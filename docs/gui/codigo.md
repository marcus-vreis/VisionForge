# A aba Código — o código real de cada treino

Na tela de treino, a aba **Código** mostra as funções que o VisionForge
executa para treinar, validar e testar o modelo, com notas em português ao lado
das linhas. Tudo aqui sai do código atual (`core/code_tour.py` e
`gui/api/code_tour_routes.py`); o que o código não define está marcado como
*não verificado*.

## Como a aba funciona

- **Não é uma cópia.** O servidor lê cada função do pacote instalado com
  `inspect` a cada pedido. O texto que você vê é o que roda (ADR-173).
- **As notas apontam para o texto da linha, nunca para o número.** Um teste
  confere que toda nota ainda encontra a sua linha; se o código mudar e a nota
  ficar sem linha, a CI falha em vez de apontar para o lugar errado.
- **O caminho desta execução.** Uma nota que depende de uma configuração (binário
  ou multiclasse, precisão mista, aumento de dados, família do modelo, a perda, o
  método de anomalia) aparece como o caminho que o seu treino segue ou como um
  ramo que ele não toma. Uma configuração que o formulário não decide fica sem
  marca, em vez de adivinhada.
- **Sem acompanhamento linha a linha.** O treino informa uma vez por época; a aba
  diz que caminho o código toma, não qual linha roda agora.
- **Todas as tarefas.** Classificação, regressão, segmentação, anomalia e
  detecção têm um roteiro cada (ADR-181). A lista vem do servidor, então uma
  tarefa nova com roteiro aparece sem mudança na página.

## Rotas

| Rota | O que devolve |
|---|---|
| `GET /api/code-tour` | as tarefas com roteiro e as configurações que cada uma lê |
| `GET /api/code-tour/{task}` | as seções da tarefa: fonte, primeira linha, arquivo dentro do pacote e as notas |

A rota lê as configurações da consulta (`tour_settings`): uma chave que a tarefa
não conhece, ou um valor que ela não aceita, é descartado e as notas dessa
configuração ficam sem marca. `true` e `false` valem como `on` e `off`; `model`
é o nome do modelo, traduzido na configuração que ele decide (a família na
classificação, a arquitetura na segmentação, o detector na detecção). O arquivo
vem como caminho dentro do pacote, nunca o caminho absoluto da instalação.

Uma tarefa sem roteiro, ou um treino que não usa o código do projeto, devolve 404
com o motivo no idioma do pedido. A detecção só tem roteiro com
`backend=torchvision`: o laço do Ultralytics é da biblioteca, e uma execução que
não diz o backend recebe o motivo também, porque Ultralytics é o padrão.

O mapa de tarefas é este, em `core/code_tour.py`:

```python
TOURS: dict[str, tuple[Section, ...]] = {
    "classification": CLASSIFICATION,
    "regression": REGRESSION,
    "segmentation": SEGMENTATION,
    "anomaly": ANOMALY,
    "detection": DETECTION,
}
```

## Funções por tarefa

Cada item é uma seção da aba, na ordem em que aparece. Os arquivos ficam em
[`src/visionforge/`](https://github.com/marcus-vreis/VisionForge/tree/development/src/visionforge).

## Classificação (`classification`)

- `_build_transforms` ([`core/data.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/data.py)): monta a sequência de transformações da imagem (tamanho fixo, aumento de dados só no treino, tensor e normalização).
- `replace_final_layer` ([`models/factory.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/models/factory.py)): troca a última camada linear por uma com uma saída por classe do seu dataset; o ramo muda com a família do modelo.
- `Trainer._build_criterion` ([`core/trainer.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/trainer.py)): escolhe a perda, `BCEWithLogitsLoss` no binário e `CrossEntropyLoss` no multiclasse.
- `Trainer._train_epoch`: uma época de treino, com o passo de aprendizado (forward, perda, backward, step) e a contagem de acertos.
- `Trainer._eval_epoch`: a validação, em modo de avaliação e sem gradientes, que calcula a `val_loss`; a época de menor `val_loss` fica no `best_model.pth`.
- `Evaluator.evaluate` ([`core/evaluator.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/evaluator.py)): roda a inferência no conjunto de teste, uma vez, e devolve as métricas de classificação.

A escolha da perda, no código atual:

```python
if self._config.task == "binary":
    return nn.BCEWithLogitsLoss()
return nn.CrossEntropyLoss()
```

## Regressão (`regression`)

- `RegressionCsvDataset.__getitem__` ([`core/regression_data.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/regression_data.py)): abre a imagem da linha do CSV com as mesmas transformações da classificação e devolve o alvo numérico como `float32`.
- `RegressionModelFactory.create` ([`models/regression_factory.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/models/regression_factory.py)): monta o backbone com uma cabeça linear de uma saída por alvo e sem ativação.
- `RegressionTrainer._build_criterion` ([`core/regression_trainer.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/regression_trainer.py)): escolhe MSE, MAE ou Huber conforme a configuração.
- `train_epoch` ([`core/training_loop.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/training_loop.py)): o laço de treino compartilhado com outras tarefas, que recebe adaptadores de saída e de alvo.
- `RegressionTrainer._eval_epoch`: a validação, que devolve a `val_loss`, MSE, RMSE, MAE e R².
- `RegressionTrainer.evaluate_with_predictions`: o teste, que devolve as métricas junto com cada alvo e cada previsão.

## Segmentação (`segmentation`)

- `SegmentationDataset.__getitem__` ([`core/segmentation_data.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/segmentation_data.py)): lê imagem e máscara, leva as duas ao mesmo tamanho (a máscara por vizinho mais próximo) e aplica o aumento geométrico às duas juntas.
- `SegmentationModelFactory._build` ([`models/segmentation_factory.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/models/segmentation_factory.py)): monta a U-Net do projeto ou um modelo de segmentação do torchvision, com uma saída por classe.
- `UNet.forward`: o caminho da imagem pelo codificador, pelo gargalo e pelo decodificador com conexões de atalho, até os logits por pixel.
- `SegmentationTrainer._build_criterion` ([`core/segmentation_trainer.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/segmentation_trainer.py)): escolhe cross-entropy, Dice ou a combinação dos dois.
- `train_epoch`: o mesmo laço compartilhado, com `check_mask_ids` e `segmentation_logits` passados pela segmentação.
- `SegmentationTrainer._eval_epoch`: a validação, que devolve a `val_loss`, mIoU, Dice e acurácia de pixel; a época de maior mIoU fica no `best_model.pth`.
- `SegmentationTrainer.evaluate_with_confusion`: o teste, que guarda uma matriz de confusão por imagem para os intervalos de confiança.
- `SegmentationMetricAccumulator.compute`: transforma a matriz de confusão em mIoU, Dice e acurácia de pixel.

## Anomalia (`anomaly`)

- `AnomalyDataModule.__init__` ([`core/anomaly_data.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/anomaly_data.py)): separa as imagens normais do treino, as normais sem aumento usadas para calibrar o limiar e o teste com normais e defeitos.
- `AnomalyModelFactory.create` ([`models/anomaly_factory.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/models/anomaly_factory.py)): monta o autoencoder ou o PatchCore.
- `AnomalyTrainer._train_autoencoder_epoch` ([`core/anomaly_trainer.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/anomaly_trainer.py)): treina uma época de reconstrução e devolve a perda média dos lotes.
- `PatchCore._greedy_coreset`: escolhe `k` pedaços da imagem o mais espalhados possível para formar o banco de memória.
- `score_images`: dá a nota de anomalia de cada imagem, pelo erro de reconstrução ou pela distância dos pedaços do PatchCore.
- `PatchCore.score`: devolve a nota da imagem como a distância máxima entre os seus pedaços e o banco de normais.
- `compute_threshold`: calcula o limiar de decisão como um percentil das notas das imagens normais.
- `AnomalyTrainer.evaluate_with_scores`: o teste, que devolve as métricas, os rótulos e as notas de cada imagem.
- `AnomalyTrainer._metrics`: calcula a AUROC e o F1 a partir das notas, dos rótulos e do limiar.

O começo da seleção gulosa do banco de memória do PatchCore, no código atual:

```python
selected = [0]
min_dist = torch.cdist(features, features[0:1]).squeeze(1)
step = max(1, k // 100)
```

## Detecção (`detection`)

Só no backend `torchvision`; com Ultralytics a aba explica por que não há código
do VisionForge para mostrar.

- `DetectionDataset.__getitem__` ([`core/detection_dataset.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/detection_dataset.py)): devolve a imagem como tensor, no tamanho original, e o alvo com as caixas em pixels e os rótulos.
- `DetectionDataset._read_labels`: lê o rótulo no formato YOLO, converte para cantos em pixels, corta nos limites da imagem e soma 1 à classe, porque o torchvision reserva o 0 para o fundo.
- `detection_collate`: junta o lote em tuplas de imagens e de alvos, já que cada imagem tem o seu tamanho e o seu número de caixas.
- `build_torchvision_detector` ([`models/detection_factory.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/models/detection_factory.py)): devolve o detector do torchvision com a cabeça no tamanho de `num_classes`.
- `_build_faster_rcnn`: monta o Faster R-CNN, dois estágios, e troca a cabeça de classificação.
- `_build_ssd`: monta o SSD, um estágio, e troca a cabeça de classificação.
- `_build_retinanet`: monta a RetinaNet, um estágio com focal loss, e troca a cabeça de classificação.
- `DetectionTrainer._run_torchvision_epoch` ([`core/detection_trainer.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/detection_trainer.py)): uma época de treino em que o detector devolve um dicionário de perdas e o lote usa a soma delas.
- `DetectionTrainer._eval_torchvision_loss`: a `val_loss` da validação, calculada sem gradientes, sem passo do otimizador e com o BatchNorm em avaliação.
- `DetectionTrainer._eval_torchvision_map`: o mAP@0.5 da validação; a época de maior mAP@0.5 fica no `best_model.pth`.
- `_ap_for_class` ([`core/detection_metrics.py`](https://github.com/marcus-vreis/VisionForge/blob/development/src/visionforge/core/detection_metrics.py)): a precisão média de uma classe, a partir da curva precisão-revocação das detecções ordenadas por confiança.

## O que não foi verificado

- A aba não foi aberta no navegador ao escrever este guia. As frases acima vêm
  do código e das notas do catálogo de mensagens.
- Esta lista acompanha `TOURS`: o teste
  [`tests/test_docs_code_tab.py`](https://github.com/marcus-vreis/VisionForge/blob/development/tests/test_docs_code_tab.py)
  exige que cada função de cada roteiro esteja citada aqui e que os trechos de
  código deste guia existam literalmente na fonte.

Decisões: ADR-173 (a aba Código) e ADR-181 (a aba para todas as tarefas), em
[`docs/dev/DECISIONS.md`](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md).
