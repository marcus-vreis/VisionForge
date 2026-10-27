# exp

- **Run ID:** `run_1`
- **Timestamp:** 2026-05-22T10:00:00
- **Status:** completed
- **Stopped:** yes (2 of 5 epochs)
- **Device used:** `cpu`
- **Run directory:** `fixture\run_1`
- **Checkpoint:** `m.pth`

## Configuration

- **Architecture:** resnet18
- **num_classes:** 1
- **Pretrained:** False
- **Task:** binary
- **Optimizer:** adam
- **Learning rate:** 0.001
- **Batch size:** 4
- **Epochs (max):** 5
- **Seed:** 0
- **Scheduler:** step ({'kind': 'step'})
- **Mixed precision:** enabled

## Preprocessing pipeline

Applied to every loaded image before augmentation, in this order:

1. **resize**
2. **blur** (radius=2)

## Augmentation / Normalization

| Field | Value |
|---|---|
| flip | `True` |
| image_size | `64` |

## Final metrics

| Metric | Value | 95% CI (bootstrap) |
|---|---|---|
| accuracy | 0.8000 | [0.7000, 0.9000] |
| total_epochs | 2 | — |
| missing | — | — |
| test_loss | 0.2500 | — |

Intervals are percentile bootstrap over the 40 test images (1000 resamples): they quantify test-split sampling noise for this fixed model, not run-to-run variance. For the latter, train replicates under several seeds.

## Training history (2 epoch(s))

| Epoch | loss | phase | val_miou |
|---|---|---|---|
| 1 | 0.5000 | warmup | — |
| 2 | — | — | 0.8000 |

## Plots

- `cm.png`

## Test runs on additional datasets (1)

### ext
- **base_dir:** `d`
- **timestamp:** t
| Metric | Value |
|---|---|
| accuracy | 0.5000 |
