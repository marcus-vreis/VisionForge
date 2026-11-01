---
title: 'VisionForge: local-first computer-vision experiments with the statistics built in'
tags:
  - Python
  - computer vision
  - deep learning
  - reproducibility
  - PyTorch
authors:
  - name: Marcus Vinícius Reis
    orcid: 0009-0002-1464-431X
    affiliation: 1
affiliations:
  - name: Universidade Federal de São Carlos (UFSCar)
    index: 1
date: <submission date, e.g. 9 October 2026>
bibliography: paper.bib
---

<!-- Draft. Every <...> must be filled before submission (Task D4 of the plan).
     Numbers cite research/seed_variance/results/summary.json; the training
     comparison cites results/real_summary.md, the browser series cite
     results/gui_summary.md, results/diag_summary.md and results/gui_v2_summary.md
     in the private research repository, whose recipes and per-run results must be
     public before submission. -->

# Summary

VisionForge is a desktop application for training, validating and comparing
computer-vision models on the researcher's own GPU. A single browser interface,
served locally, covers five task families: image classification, object
detection, image regression, semantic segmentation and anomaly detection, plus
task families a researcher defines in a Python file. Beyond training one model,
the interface runs the experiments a result needs before it can be reported:
the same configuration repeated over several seeds with 95% confidence
intervals, paired comparisons between variants with Holm-corrected tests,
K-fold cross-validation and hyperparameter sweeps. Every run writes a versioned
`run.json` that records its configuration, seeds, library versions, hardware
and dataset fingerprint, so a number in a paper can be traced to the run that
produced it.

# Statement of need

Applied researchers who train vision models on their own data (agronomy,
medicine, industrial inspection) usually work alone, on one GPU, without
experiment infrastructure. The tools within their reach train a model and
report one number. That number depends on the random seed as much as on the
choice being tested: seed variance in deep learning is documented
[@picard2021seed; @bouthillier2021accounting; @summers2021nondeterminism], yet a
single-seed ablation remains the default way to decide whether a change helps.

We measured how much this matters in small-data fine-tuning with VisionForge
itself, in a pre-registered study (`research/seed_variance/`): three datasets,
three architectures, two common ablations, ten paired seeds each, 270 training
runs. In all 18 cells the seed standard deviation was at least half the ablation
effect, and in 15 it exceeded the effect. Concluding from one seed would have
pointed the wrong way in 20% to 60% of the possible single-seed experiments,
depending on the cell (median 40%). No ablation effect survived Holm correction
with ten seeds, and in 11 of the 18 cells even five seeds left more than a 10%
chance of the wrong sign.

VisionForge exists so that a researcher without a statistics or MLOps
background runs the comparison that holds up, rather than the one that is easy:
replicates, paired tests and cross-validation are one button away, and the
interface warns when a conclusion rests on a single seed.

# State of the field

Training libraries make a single run easy. Ultralytics YOLO [@jocher2023yolo]
trains and validates detectors from one command, but repeating a run over seeds
and testing the difference is left to the user. Experiment trackers such as
MLflow [@zaharia2018mlflow] and Weights & Biases [@biewald2020wandb] record
what was run and compare runs side by side; they do not design the experiment,
pair the seeds or correct for multiple comparisons, and the hosted ones send
data off the machine. Hyperparameter optimisers such as Optuna
[@akiba2019optuna], which VisionForge uses for its sweeps, search for the best
configuration but report it from one evaluation per trial. No-code tools such as
Teachable Machine [@carney2020teachable] lower the entry barrier furthest, but
keep no provenance and offer no statistics.

We built VisionForge rather than contributing to one of these because the gap
is in how the pieces are joined, not in any one of them: a local graphical
interface, five task families behind one contract, paired multi-seed
statistics, and a provenance record per run. Each piece exists elsewhere; a
researcher who wants all of them today assembles them by hand. VisionForge
trains with PyTorch [@paszke2019pytorch] and Ultralytics underneath rather than
reimplementing them.

# Software design

The code is layered one way, `utils` ← `core` ← `blocks` ← `gui`: the statistics
and the trainers know nothing about the interface, so the same experiment runs
from the browser, the command line or a script. Every task family implements
one runner contract, and the experiment strategies (replicates, paired
comparison, K-fold, sweep) are written once against that contract, which is how
a task a researcher adds in a single Python file gets them all without further
code.

Results are the report's numbers, copied, never recomputed for display: the
mean on screen is the one in the report a paper table is built from. The
`run.json` contract is versioned and migrated on load, so a configuration
exported from an older release still runs. Determinism is on by default and
tested: the same configuration and seed give the same model and the same
metrics, a property the seed study depended on and whose failure it caught (the
classifier head was once initialised before the seed was set). Confidence
intervals are clipped to each metric's domain and flagged when clipped.

Every behaviour change is recorded as an architecture decision record in
`docs/dev/DECISIONS.md` (more than 150 so far). The project has about 5,200
backend and 1,050 frontend tests, a `visionforge selftest` command that trains every task
and strategy through the real API, and continuous integration on each push.
Everything runs locally; nothing leaves the machine unless the user downloads a
public dataset.

# Research impact statement

VisionForge produced the seed-variance study described above: the protocol was
committed before the 270 runs, and the reports, the analysis code and its
tests are public in `research/seed_variance/`, so each number above can be
re-derived. The study also changed the software: its determinism check exposed
the head-initialisation bug, and its findings are why the interface flags
single-seed conclusions.

A layer over the training libraries is only useful if it leaves their results
alone, so we trained three recipes both through VisionForge and on plain PyTorch
or Ultralytics, with the same configuration and seeds: ResNet-18 on CIFAR-10 (30
epochs, six seeds), DeepLabV3-ResNet50 on Oxford-IIIT Pet (40 epochs, six seeds)
and YOLO11s on VOC2007 (100 epochs, three seeds). The paired difference in test
accuracy was -0.0004 (95% CI -0.0012 to +0.0003) and in test mIoU +0.0004 (-0.0008
to +0.0015), with no margin for these differences set in advance; test mAP50
matched to full precision. The ratio of wall-clock time, VisionForge over native,
was 1.026 (90% CI 1.008 to 1.044), 1.005 (0.991 to 1.020) and 0.998 (0.995 to
1.000), inside a margin of 5%. Three runs that fell inside a slowdown of unknown
cause were repeated with the machine idle; with the original runs the segmentation
interval, 0.964 to 1.158, leaves the margin. These runs used the command line, so
they do not measure the browser interface.

We then repeated the classification and segmentation recipes through the browser,
with three seeds, importing the configuration and pressing Train in a visible
Chrome window, against margins fixed before the runs. Quality stayed inside its
margin of 0.005: the paired difference was -0.0005 (95% CI -0.0026 to +0.0016) in
test accuracy and +0.0007 (-0.0034 to +0.0049) in test mIoU. Time did not stay
inside its margin of 5%: the ratio, browser over native, was 1.168 (90% CI 1.017
to 1.341) and 1.120 (1.084 to 1.157). This setup kept the page rendering throughout,
which we chose as the worst case. A later exploratory series (three blocks, four
epochs, no margin set in advance) located the cost by comparing epoch time with
native training. With the browser closed after the job was submitted, the ratio was
1.009 (90% CI 1.004 to 1.015) for classification and 1.013 (1.007 to 1.018) for
segmentation; with the window minimized, 1.007 (1.002 to 1.012) and 1.015 (1.010 to
1.020); with the page visible, 1.168 (1.164 to 1.172) and 1.181 (1.181 to 1.182).
Looping interface animations accounted for most of that cost, so we ended them while
a job runs (ADR-157) and repeated the three-seed series, page visible, under the same
margins. Because the change was chosen after seeing these results, this series tests
a fix rather than an independent prediction. Quality stayed inside its margin: -0.0005
(95% CI -0.0026 to +0.0016) in test accuracy and +0.0000 (-0.0026 to +0.0027) in test
mIoU. The time ratio fell to 1.017 (90% CI 1.014 to 1.020) for segmentation, inside
the margin, and to 1.047 (1.006 to 1.089) for classification, whose interval still
leaves it. In the slowest classification run (ratio 1.074) other processes on the
machine used about 1.5 times the CPU of the other two browser runs; we did not identify
them or repeat the run, so we cannot attribute the excess to that load. Training
through the interface left the results unchanged, and its time cost, once large, is
now small but not shown to be within 5% for every recipe.

<Usability study: number of participants, mean SUS
and task completion rate, from `research/usability/`.> Each release is
archived on Zenodo (doi:10.5281/zenodo.23425765) and published on PyPI as
`visionforge-studio`.

# AI usage disclosure

Generative AI coding assistants, Anthropic's Claude through Claude Code, OpenAI's
Codex and Google's Gemini, were used throughout the development of VisionForge to draft code, tests,
refactorings and documentation, and to draft parts of this paper. The author
designed the architecture, the statistical methodology and the experiments,
reviewed and accepted every change, and verified behaviour through the project's
automated tests, the `selftest` command and real training runs. Every reference
was checked against its source. The author takes full responsibility for the
software and this paper. <Confirm whether other assistants (e.g. GitHub Copilot)
were used, and adjust.>

# Acknowledgements

<Testers (anonymous, as a group); institutional support, if any.>

# References
