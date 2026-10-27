# e2e_stop

- **Run ID:** `run_1`
- **Timestamp:** 2026-10-23T08:04:54.398345
- **Status:** completed
- **Stopped:** yes (2 of 200 epochs)
- **Device used:** `cpu`
- **Run directory:** `fixture/20261023_080451_801587`
- **Checkpoint:** `outputs\models\e2e_stop\20261023_080451_801587\best_model.pth`

## Configuration

- **Architecture:** resnet18
- **num_classes:** 1
- **Pretrained:** False
- **Task:** binary
- **Optimizer:** adam
- **Learning rate:** 0.001
- **Batch size:** 32
- **Epochs (max):** 200
- **Seed:** 42

## Augmentation / Normalization

| Field | Value |
|---|---|
| image_size | `224` |
| augment | `True` |
| horizontal_flip | `True` |
| rotation_degrees | `10` |
| color_jitter | `False` |
| normalize_mean | `[0.485, 0.456, 0.406]` |
| normalize_std | `[0.229, 0.224, 0.225]` |

## Final metrics

| Metric | Value |
|---|---|
| best_val_loss | 0.0000 |
| best_epoch | 2 |
| total_epochs | 2 |
| test_accuracy | 1.0000 |
| test_f1 | 1.0000 |
| test_precision | 1.0000 |
| test_recall | 1.0000 |
| test_auc_roc | 1.0000 |

## Training history (2 epoch(s))

| Epoch | train_loss | train_accuracy | val_loss | val_accuracy |
|---|---|---|---|---|
| 1 | 0.6376 | 0.9167 | 0.3084 | 1.0000 |
| 2 | 0.0211 | 1.0000 | 0.0000 | 1.0000 |

## Plots

- `outputs\models\e2e_stop\20261023_080451_801587\loss.png`
- `outputs\models\e2e_stop\20261023_080451_801587\accuracy.png`
- `outputs\models\e2e_stop\20261023_080451_801587\confusion_matrix.png`
- `outputs\models\e2e_stop\20261023_080451_801587\confusion_matrix_normalized.png`
- `outputs\models\e2e_stop\20261023_080451_801587\roc_curve.png`
- `outputs\models\e2e_stop\20261023_080451_801587\precision_recall_curve.png`
