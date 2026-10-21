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
