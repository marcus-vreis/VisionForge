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
    orcid: <ORCID>
    affiliation: 1
affiliations:
  - name: <affiliation>
    index: 1
date: <submission date, e.g. 9 October 2026>
bibliography: paper.bib
---

<!-- Draft. Every <...> must be filled before submission (Task D4 of the plan).
     Numbers cite research/seed_variance/results/summary.json. -->

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
`docs/dev/DECISIONS.md` (122 so far). The project has about 4,450 backend and
1,010 frontend tests, a `visionforge selftest` command that trains every task
and strategy through the real API, and continuous integration on each push.
Everything runs locally; nothing leaves the machine unless the user downloads a
public dataset.

# Research impact statement

VisionForge produced the seed-variance study described above: the protocol was
committed before the 270 runs, and the reports, the analysis code and its
tests are public in `research/seed_variance/`, so each number above can be
re-derived. The study also changed the software: its determinism check exposed
the head-initialisation bug, and its findings are why the interface flags
single-seed conclusions. <Usability study: number of participants, mean SUS
and task completion rate, from `research/usability/`.> Each release is
archived on Zenodo (doi:10.5281/zenodo.23425765) and published on PyPI as
`visionforge-studio`.

# AI usage disclosure

Generative AI coding assistants, Anthropic's Claude through Claude Code, were
used throughout the development of VisionForge to draft code, tests,
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
