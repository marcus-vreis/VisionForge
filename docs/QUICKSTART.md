# Quickstart — from zero to a defensible result

This walkthrough goes from nothing installed to a trained classifier with
multi-seed confidence intervals, using only what ships with VisionForge.
Time budget: ~10 minutes on any machine (CPU works; GPU is faster).

## 1. Install

The published package already contains the built interface, so this is all it
takes:

```bash
mkdir my-research && cd my-research       # runs and your code land here
python -m venv .venv
.venv\Scripts\activate                   # Linux/macOS: source .venv/bin/activate
pip install visionforge-studio
visionforge doctor --fix                  # reads your GPU, installs the matching torch
```

`doctor` reads your driver and any torch already present, names the build your
machine needs, and swaps it in once you answer `y`. On Windows the plain install
always brings a CPU torch first (PyPI has no other there), so on a GPU machine
that swap is expected. To skip it, install torch before the package:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`,
then `pip install visionforge-studio` (`cu118` · `cu121` · `cu124` · `cu126` ·
`cpu` also work in place of `cu128`).

Working from a source checkout instead? That path is in
[`CONTRIBUTING.md`](../CONTRIBUTING.md) — the rest of this walkthrough is
identical either way.

Before pointing it at real data, confirm the install actually trains — this
runs every task through the real API on synthetic data and needs no dataset,
GPU or network:

```bash
visionforge selftest --quick
```

## 2. Launch the GUI and get a dataset

```bash
visionforge gui                           # opens http://127.0.0.1:8000
```

The interface comes in English and Portuguese: the **PT** / **EN** switch in the
header changes it at any time, and the choice is remembered. This guide names
the English labels.

You don't need your own data to try it: click **⤓ datasets** in the bottom bar,
pick the **torchvision** provider, dataset `CIFAR10`,
choose an output folder and a per-class `limit` (e.g. 300 — keeps the first
run fast). VisionForge materializes it as an ImageFolder
(`<out>/train/<class>/*.png`, `<out>/test/...`) — the exact layout the
classification task consumes.

## 3. First training run

The task tabs configure a run in six steps — **Data**, **Strategy**,
**Model**, **Training**, **Image & augmentation** and **Review** — and the
**▶ Train** button on the bar below stays locked until a dataset is chosen in
the first one. On the **Classification** tab:

1. **Data**: point the dataset picker at the folder you just downloaded — split
   auto-detection and per-class stats render immediately; click
   **🎯 apply 10 classes** (the count it detected for CIFAR10) to inject the
   class count.
2. Leave **Strategy**, **Model** and **Image & augmentation** on their defaults:
   a single run of `resnet18` with pretrained weights. **Training** defaults
   to 20 epochs; lower it to 3 for a smoke run on CPU.
3. **Review**: check the summary, give the run a name, and press **▶ Train**.
   The overlay streams per-epoch loss/accuracy live.

When it finishes you get metric tiles, confusion matrices, ROC/PR curves, and
a markdown model card. Everything is also on disk under
`outputs/models/<run>/` — including `run.json` with the full config, seed,
per-epoch history and environment (torch/CUDA/cuDNN/GPU) that produced it.

## 4. Make the number defensible

A classification run already hands you part of the answer: each test metric
carries a 95% bootstrap interval (`0.8734 [0.8412, 0.9021]`) under the tile and
in the model card. Read it for what it is — it resamples the *test split* with
the model held fixed, so it says how much the number depends on which images
you happened to test on. It says nothing about whether retraining would land in
the same place.

For that, a single run is one sample from a noisy distribution. On every
task tab, **Classification** included, the strategy step has **Replicates**:
same config, N seeds, and the report gives you `metric = mean ± 95% CI` plus
the per-seed table — the citable version of your result. Classification ranks
the seeds on accuracy. Replicates run on this computer only: the Colab button
stays off, with the reason, while Replicates is picked. **K-fold (CV)** and
**Grid search** (add values to the *Architecture* field with “+ add to grid”)
sit on the same step, and the **History** overlay compares the runs.

To ask whether a change beats your configuration, pick **Replicated
comparison** on the same step. The form is the baseline: name one field
(`training.learning_rate`, `model.name`…) and 1 to 3 other values for it.
Every variant trains on the same seeds (2 to 20), so the cost is variants ×
seeds, which the card shows before you start. Each pair then gets a paired test
(t or Wilcoxon) with Holm correction, and History shows the matrix. Use 6 seeds
or more: with 5 or fewer, Wilcoxon cannot return a p below 0.05, and the card
warns you of it.

## 5. Reproduce it anywhere

Every panel's header has **↓ Export YAML** — the exported file is the exact
config the API received, so this reruns the experiment identically:

```bash
visionforge run experiment_my_test_2026-07-02.yaml
```

Import the same file back through **↑ Import YAML** to restore the form.

### Replicates, comparisons, sweeps and K-fold from the command line

The strategies of section 4 run without the GUI too. Wrap the task config under
`config:` and say what to do with it under `strategy:`; the options are the
ones the GUI sends, and `visionforge run --help` lists them. The job writes the
same report as the GUI (`*_summary.json`, `*_ranking.csv` and a `*_table.tex`
under `output.reports_dir`) and one History entry under `output.models_dir`.
A YAML without `strategy:` trains once, as before.

Replicates (mean ± 95% CI per metric, any task):

```yaml
strategy:
  kind: replicates
  seeds: [0, 1, 2, 3, 4]      # or n_replicates: 5, counted from training.seed
config:
  task: regression
  name: reg_resnet18
  # ... the rest of the exported task config
```

Paired comparison (every variant trains on the same seeds, then paired tests
with Holm correction; any task):

```yaml
strategy:
  kind: replicated_comparison
  seeds: [0, 1, 2, 3, 4]
  variants:
    baseline: {}
    low_lr: {training.learning_rate: 0.0001}
  alpha: 0.05
config:
  task: regression
  # ...
```

Sweep (ranked on the validation split; detection, regression, segmentation,
anomaly; classification uses `block: grid_search` or `random_search`):

```yaml
strategy:
  kind: sweep
  mode: grid                  # or random / optuna, with n_trials
  search_space:
    training.learning_rate: [0.001, 0.0003]
    training.batch_size: [16, 32]
config:
  task: segmentation
  # ...
```

K-fold (regression and segmentation; classification uses
`block: cross_validation`):

```yaml
strategy:
  kind: cv
  n_folds: 5
  fold_seed: 42
config:
  task: regression
  # ...
```

## 6. Define your own task (optional)

If your problem isn't classification/detection/regression/segmentation/anomaly:

```bash
visionforge new-task my_task      # generates user_tasks/my_task.py — it already trains
```

Fill the four hooks with your model/data/loss/metrics and the task gets the
run endpoint, live monitor, `run.json` provenance, sweeps and replicates for
free. Walkthrough: [`docs/custom/TASKS.md`](custom/TASKS.md).

## Where to go next

- [Export a checkpoint to a private Hugging Face repository](gui/hugging-face-hub.md)
- [Track training in local MLflow (Portuguese)](gui/mlflow.md)
- [Send training to Weights & Biases (Portuguese)](gui/wandb.md)
- [Create a Zenodo draft (Portuguese)](gui/zenodo.md)
- [Send a run report to GitHub (Portuguese)](gui/github.md)
- [Version a run with DVC (Portuguese)](gui/dvc.md)

- [`README.md`](../README.md) — what VisionForge does, in English and Portuguese
- [`docs/DATASETS.md`](DATASETS.md) — dataset layouts and the download providers
- [`docs/custom/`](custom/) — your own architectures and whole task families
- [`docs/dev/ARCHITECTURE.md`](dev/ARCHITECTURE.md) — how the pieces fit
