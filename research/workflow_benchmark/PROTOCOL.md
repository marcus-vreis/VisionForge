# Workflow benchmark — draft protocol

This is a measurement plan, not an executed study or a claim of superiority.
Freeze and commit the final design before collecting valid sessions. Do not
derive publication results from tests or synthetic examples.

## Question

How much setup effort and protocol error does VisionForge change when a
researcher executes the same local vision experiment with the corresponding
native PyTorch/Ultralytics workflow?

The native workflow includes reusable, documented scripts. Do not compare a
prepared GUI with writing an entire training pipeline from scratch and call the
difference tool overhead. MLflow may be evaluated separately as a tracker; it
does not define the complete execution baseline for this question.

## Matched conditions

Each pair fixes task, architecture, initial weights, dataset content and splits,
transforms, optimizer/scheduler, seed, batch size, epochs/early stopping, metric,
precision, workers, hardware and library versions. Audit splits before the pair
and save configs, source revisions, outputs and artifact hashes.

Measure cold installation separately from an installed-workspace task. Separate
download time, user setup time, training/evaluation runtime and report preparation.
Choose beforehand whether weights/data are cached; use the same condition in
both workflows. For accelerator runtime, synchronize before starting/stopping
the timer. Record memory/power settings and concurrent workloads.

## Workflow outcomes

- Completion without assistance and completion with assistance.
- Time from task instructions to a validated configuration.
- Number and category of protocol/configuration mistakes, using a checklist
  fixed before observing sessions.
- Runtime of the matched experiment; show each paired difference and disclose
  setup/evaluation work included in each measurement.
- Time and missing evidence when producing the final comparison/report.

Failures and timeouts remain in the dataset. Do not silently compute successful
time from only the easiest completions. Distinguish protocol correctness from
model quality: a higher test score does not by itself mean a better workflow.

## Procedure and reporting

Begin with one task and a technically verified baseline. Expand to the remaining
tasks only when the baseline matches the platform's computations. Counterbalance
workflow order across participants; record prior experience and assistance.
Task order and repetition policy must be decided before collection.

Use anonymous session IDs and the header-only `measurement_template.csv`.
Record metadata for every pair and link to result files; never invent timings.
With a small convenience sample, report individual paired observations and
descriptive summaries, without population-wide or causal claims.

## Pending before collection

Recruitment/consent process, sample and stopping rule, participant task script,
mistake checklist, human-session timing instrumentation and frozen software
versions. The restricted classification and segmentation CPU baselines and
machine-phase instrumentation are implemented (ADR-148/149). Full-data recipes
and native baselines for detection, regression and anomaly remain extensions.
No external participant has been recruited by this repository change.

## Engineering check before collecting workflow measurements

The public-data staging plan and reproducible CIFAR-10 subset preparation are in
[DATASETS.md](DATASETS.md). CPU pilots can run while the GPU is occupied, but
concurrent-workload timings cannot support performance comparisons.

`native_classification.py` provides an independent CPU PyTorch baseline;
`paired_classification.py` checks it against the public VisionForge Trainer and
DataModule. Supply real ImageFolder train/val/test splits and a torchvision
ResNet18 state dictionary with the corresponding output count:

```bash
python -m research.workflow_benchmark.paired_classification DATA INITIAL.pth OUTPUT --seed 37 --epochs 2 --batch-size 4
```

The CLI hides CUDA and sets one CPU thread before importing torch, then lowers
its process priority. `--concurrent-workload` records what else was running
(default `undeclared`). The restricted
recipe uses 32px resize/center crop, fixed normalization, no augmentation,
pretrained downloads, AMP or scheduler, SGD with zero momentum and workers=0.
At 32px the last training batch must contain at least two images for BatchNorm.
All splits must contain the same class vocabulary. Exact-byte image audit
findings must be resolved before pairing.

The harness requires identical histories, selected epochs, test predictions and
best-state tensors, while recording the shared initial-weights hash, dataset
content fingerprint and environment.

## Phase contract (ADR-148)

Both paths share one untimed preflight: argument checks, split and class layout,
the 32px final-batch rule and the initial-weights file. A preflight failure
creates nothing. Afterwards each path creates its own new output directory and
times four machine phases with the same boundaries:

| Phase | Starts | Ends |
|---|---|---|
| setup | after output creation | loaders, model with initial weights ready |
| training | seeding / training call | best checkpoint on disk |
| evaluation | reload of best checkpoint | test targets and predictions in memory |
| evidence | after evaluation | best-checkpoint hash and report assembled |

The single exclusive write of the report closes the evidence phase and is not
timed. The work inside a phase still differs between implementations, so every
report lists it under `phases.<name>.includes`. In particular, VisionForge's
training call also builds its optimizer and writes resume state, TensorBoard
logs and `run.json`; the native path writes only the best checkpoint.

Each report records `conditions`: cache (local pre-staged data and weights, no
downloads, OS file cache not controlled), library versions, CPU threads,
`CUDA_VISIBLE_DEVICES`, process priority and the declared concurrent workload.
The pair always runs native first, so the second path may read warm OS file
caches; the order is recorded as `execution_order`. Phase times are machine
wall-clock time (`timing_kind`); human effort is not measured by this harness.

A failure in any phase writes the partial report with `status: failed`,
`failed_phase` and the error, and keeps artifacts already written. The pair
report names the failed path and phase (e.g. `native.training`) and embeds the
partial sub-report. Reports are created exclusively; existing outputs are never
overwritten.

Reports without `phase_contract` predate ADR-148. Their `*_seconds` fields cover
implementation-specific work and keep that original scope: `report_scope()`
labels them `legacy_implementation_phases`, and they are not rewritten or
compared with contract reports.

These are numerical engineering checks. Synthetic test fixtures are not study
participants or a model-quality benchmark. No speed ratio or human-effort effect
is inferred from phase times, and timings taken under concurrent load do not
support performance comparisons. Counterbalance execution order before
collecting comparative measurements.

## Segmentation parity (ADR-149)

`native_segmentation.py` is an independent CPU PyTorch baseline for binary
segmentation; `paired_segmentation.py` checks it against the public
`SegmentationDataModule` and `SegmentationTrainer` under the same phase contract.
The dataset has `train`, `val` and `test` splits, each with `images/` (`.png`,
`.jpg`, `.jpeg`) and `masks/` (single-channel `.png`) paired by file stem. Mask
values are 0 (background), 1 (foreground) or 255 (ignored).

```bash
python -m research.workflow_benchmark.paired_segmentation DATA INITIAL.pth OUTPUT --seed 37 --epochs 2 --batch-size 2
```

The restricted recipe fixes, in both paths: the seed, the splits as supplied,
the initial weights of the harness network `tiny_fcn` (built by the harness, not
by VisionForge's model factory), batch order from the seeded loader shuffle,
bilinear image and nearest-neighbour mask resize to `--image-size` (at least
32), normalization with mean and std 0.5, no augmentation, cross-entropy with
`ignore_index=255`, SGD with zero momentum and weight decay, no scheduler, AMP
or early stopping, workers=0. The best checkpoint is the first epoch with the
highest validation mean IoU.

Metrics use one confusion matrix over non-ignored pixels of the split. Mean IoU
and Dice average the classes present in ground truth or prediction; pixel
accuracy divides correct pixels by labelled pixels. Ignored pixels never enter
the loss or the metrics. The shared preflight refuses, before creating output:
unpaired, duplicate-stem or unsupported files; masks that are not mode `L` or
`P`; mask values other than 0, 1 and 255; and any mask whose resized pixels are
all ignored. Such a mask would add nothing to the metrics, but a batch made only
of such masks makes cross-entropy undefined (NaN), so the recipe refuses it
explicitly instead of depending on batch composition.

The pair compares with these tolerances:

| Compared | Tolerance | Why |
|---|---|---|
| train/val losses, best epoch, initial-weights hash | exact | same torch operations, order and thread count |
| best-state tensors, test predictions, per-image confusion | exact | same reason; integers and stored tensors |
| val/test mean IoU, Dice, pixel accuracy | 1e-12 absolute | NumPy versus torch float64 reductions over identical integer matrices; one changed pixel moves a metric by at least 1/labelled pixels |

A divergence in any row fails the pair with `failed_phase: comparison` and the
list of mismatches. Each path writes `test_predictions.npy` (uint8, one mask per
test image in sorted stem order) and records its SHA-256. The pair report adds
the source revision and versions (`environment`), the dataset content
fingerprint, the tolerances and the trimap mapping. Outputs are created
exclusively and never overwritten.

Oxford-IIIT Pet trimaps are converted by `trimap_labels.py`: 1 (pet) becomes 1,
2 (background) becomes 0, 3 (border, not classified) becomes 255; any other
value is refused. The conversion reads local files only and is not wired to a
download. Synthetic fixtures prove the integration, not segmentation quality.
