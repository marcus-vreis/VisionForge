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

Recruitment/consent process, sample and stopping rule, task script, baseline
implementation, mistake checklist, timing instrumentation and exact software
versions. No external participant has been recruited by this repository change.

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

The CLI hides CUDA before importing torch and bounds CPU threads. The restricted
recipe uses 32px resize/center crop, fixed normalization, no augmentation,
pretrained downloads, AMP or scheduler, SGD with zero momentum and workers=0.
At 32px the last training batch must contain at least two images for BatchNorm.
All splits must contain the same class vocabulary. Exact-byte image audit
findings must be resolved before pairing.

The harness requires identical histories, selected epochs, test predictions and
best-state tensors, while recording the shared initial-weights hash, dataset
content fingerprint and environment. Failed pairs retain paired_result.json and
any completed artifacts; existing outputs are never overwritten.

These are numerical engineering checks. Synthetic test fixtures are not study
participants or a model-quality benchmark. Phase timings include different
artifact-writing responsibilities in each implementation; no speed ratio or
human-effort effect is inferred from them. Align the study's output/timing scopes
and counterbalance execution order before collecting comparative measurements.
