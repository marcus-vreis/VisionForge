# e2e_offline

- **ID da execução:** `run_1`
- **Data e hora:** 2026-10-23T08:05:12.214400
- **Situação:** concluída
- **Dispositivo usado:** `cpu`
- **Diretório da execução:** `fixture/20261023_080501_978649`
- **Arquivo do checkpoint:** `outputs\models\e2e_offline\20261023_080501_978649\best_model.pth`

## Configuração

- **Arquitetura:** resnet18
- **num_classes:** 1
- **Pré-treinado:** False
- **Tarefa:** binary
- **Otimizador:** adam
- **Taxa de aprendizado:** 0.001
- **Tamanho do batch:** 32
- **Épocas (máx.):** 8
- **Seed:** 42

## Data augmentation / Normalização

| Campo | Valor |
|---|---|
| image_size | `224` |
| augment | `True` |
| horizontal_flip | `True` |
| rotation_degrees | `10` |
| color_jitter | `False` |
| normalize_mean | `[0.485, 0.456, 0.406]` |
| normalize_std | `[0.229, 0.224, 0.225]` |

## Métricas finais

| Métrica | Valor |
|---|---|
| best_val_loss | 0.0000 |
| best_epoch | 3 |
| total_epochs | 8 |
| test_accuracy | 1.0000 |
| test_f1 | 1.0000 |
| test_precision | 1.0000 |
| test_recall | 1.0000 |
| test_auc_roc | 1.0000 |

## Histórico de treino (8 época(s))

| Época | train_loss | train_accuracy | val_loss | val_accuracy |
|---|---|---|---|---|
| 1 | 0.6376 | 0.9167 | 0.3084 | 1.0000 |
| 2 | 0.0211 | 1.0000 | 0.0000 | 1.0000 |
| 3 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 4 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 5 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 6 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 7 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 8 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |

## Gráficos

- `outputs\models\e2e_offline\20261023_080501_978649\loss.png`
- `outputs\models\e2e_offline\20261023_080501_978649\accuracy.png`
- `outputs\models\e2e_offline\20261023_080501_978649\confusion_matrix.png`
- `outputs\models\e2e_offline\20261023_080501_978649\confusion_matrix_normalized.png`
- `outputs\models\e2e_offline\20261023_080501_978649\roc_curve.png`
- `outputs\models\e2e_offline\20261023_080501_978649\precision_recall_curve.png`
