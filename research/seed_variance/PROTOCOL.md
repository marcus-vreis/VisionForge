# Seed variance in small-data image classification — protocol

Registered by the commit that adds this file. Any change after it is recorded
under **Deviations**, with the date and the reason, and reported in the paper.

## Question

On small image-classification datasets, fine-tuning ImageNet-pretrained
networks, how does the variation between training seeds compare with the
effect of common ablations, and how many seeds does a paired comparison need
before its conclusion stops depending on which seeds were drawn, and does that
number differ between convolutional and attention architectures?

## Hypotheses

- **H1.** In at least half of the cells (dataset × architecture × ablation), the
  baseline's standard deviation across seeds is at least half the absolute mean
  ablation effect (noise/effect ≥ 0.5).
- **H2.** With a single seed (k = 1), the reversal rate exceeds 10% in at least
  one cell per dataset.
- **H3.** The seed SD is larger on the 10% subset than on the full USK-Coffee
  training split, for every architecture.

## Falsification

H1 is rejected if noise/effect < 0.5 in more than half of the cells; H2 if every
cell has a k = 1 reversal rate ≤ 10%; H3 if any architecture has SD(10%) ≤
SD(full). A rejected hypothesis is reported as the finding.

## Design

| Factor | Levels |
|---|---|
| Dataset | USK-Coffee full (4 classes; 4,800 / 1,600 / 1,600); USK-Coffee 10% (stratified 10% of train drawn once with seed 2026 by `make_subset.py`; val and test unchanged); Beans (3 classes; Hugging Face, published splits) |
| Architecture | resnet18 (Adam, 1e-3), a classic CNN; convnext_tiny (AdamW, 1e-4), a modern CNN; swin_t (AdamW, 1e-4), a transformer — VisionForge's measured defaults (ADR-099/100) |
| Variant | `baseline` (horizontal flip + 10° rotation, 224 px); `no_aug` (no flip, no rotation); `res160` (160 px) |
| Seed | 0–9, identical across variants (paired) |

Fixed: ImageNet weights; at most 20 epochs with early stopping (patience 5) on
validation loss; batch 32; weight decay 0; deterministic cuDNN; one GPU.
Primary metric: test accuracy. Secondary, reported without hypothesis: macro-F1.
Total: 3 × 3 × 3 × 10 = 270 training runs, through VisionForge's replicated
comparison (`run_study.py`). Each run's `run.json` records seeds, library
versions and GPU; each report records the VisionForge version.

## Analysis, per cell

1. Baseline mean ± SD across seeds.
2. Paired effect Δ = mean(variant − baseline) over seeds both finished, with
   VisionForge's paired test, bootstrap CI and Holm correction within each
   (dataset, architecture) family.
3. noise/effect = SD(baseline) / |Δ|.
4. Reversal rate for k = 1, 3, 5: the share of k-seed subsets whose mean
   difference has the opposite sign of (or equals zero against) the 10-seed Δ.
5. Seeds needed for 80% power at α = 0.05 (normal approximation, paired).
5a. The smallest k (of 1, 3, 5, 10) whose reversal rate is at most 10%, per cell;
   reported per architecture family (CNN: resnet18, convnext_tiny; transformer:
   swin_t) without a hypothesis test, since one transformer cannot carry one.
6. Evaluation noise: half-width of the 95% bootstrap CI of test accuracy for the
   baseline at seed 0, against the seed SD — "which test images" versus "which
   seed".

## Sanity check before the grid

The baseline at seed 0 is trained twice per dataset with resnet18; the two test
accuracies must be identical (deterministic mode). If they are not, the study
stops and the cause is recorded before anything else runs.

## Exclusions

A failed run is repeated once with the same seed. If it fails again, that seed
is dropped for that variant and reported; paired statistics use the seeds every
variant finished.

## Prior work

The verified literature, and what this study adds to it, is in `LITERATURE.md`.
Seed variance swallowing small effects is established (Picard 2021; Bouthillier et
al. 2021; Summers & Dinneen 2021); what is not measured is the ratio of seed SD to
ablation effect and the reversal rate in small-data fine-tuning of ImageNet-pretrained
vision models under deterministic, paired seeds.

## Deviations

None yet.
