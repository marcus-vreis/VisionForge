# Quickstart — from zero to a defensible result

This walkthrough goes from nothing installed to a trained classifier with
multi-seed confidence intervals, using only what ships with VisionForge.
Time budget: ~10 minutes on any machine (CPU works; GPU is faster).

## 1. Install

The published package already contains the built interface, so this is all it
takes:

```bash
mkdir my-research && cd my-research       # runs and your code land here
pip install "visionforge-studio[cu128]"   # or [cpu]; doctor names yours
visionforge doctor
```

`doctor` reads your driver and any torch already present, and prints the exact
install line for your machine. If it names a different hardware extra than the
one above, re-run the install with that one.

Prefer Docker (no PyTorch install at all) or a source checkout for development?
Both are in the [README](../README.md#installation) — the rest of this
walkthrough is identical either way.

<details>
<summary>From a clone instead</summary>

```bash
git clone https://github.com/marcus-vreis/VisionForge.git
cd VisionForge
uv venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
uv pip install -e ".[cu128,dev]"
cd frontend && npm install && npm run build && cd ..
```

The frontend build is only needed from source: the published wheel ships it.
</details>

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

You don't need your own data to try it: click **⤓ datasets** in the bottom bar,
pick the **torchvision** provider, dataset `CIFAR10`,
choose an output folder and a per-class `limit` (e.g. 300 — keeps the first
run fast). VisionForge materializes it as an ImageFolder
(`<out>/train/<class>/*.png`, `<out>/test/...`) — the exact layout the
classification task consumes.

## 3. First training run

On the **Classificação** tab:

1. Point the dataset picker at the folder you just downloaded — split
   auto-detection and per-class stats render immediately; click
   **🎯 aplicar** to inject the detected class count.
2. Keep the defaults (ResNet-50 pretrained, 10 epochs is plenty for a smoke
   run — lower it to 3 if you're on CPU).
3. Press **Treinar**. The overlay streams per-epoch loss/accuracy live.

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

For that, a single run is one sample from a noisy distribution. On a
standalone-task panel (e.g. **Regressão**), the strategy selector has
**Réplicas**: same config, N seeds, and the report gives you
`metric = mean ± 95% CI` plus the per-seed table — the citable version of your
result. For classification, use **K-Fold (CV)** in the strategy selector, or
sweep architectures with **Grid search** (add values to the *Arquitetura*
field with “+ valor ao grid”) and compare runs in the **History** overlay.

## 5. Reproduce it anywhere

Every panel's header has **↓ Exportar YAML** — the exported file is the exact
config the API received, so this reruns the experiment identically:

```bash
visionforge run experiment_meu_teste_2026-07-02.yaml
```

Import the same file back through **↑ Importar YAML** to restore the form.

## 6. Define your own task (optional)

If your problem isn't classification/detection/regression/segmentation/anomaly:

```bash
visionforge new-task my_task      # generates user_tasks/my_task.py — it already trains
```

Fill the four hooks with your model/data/loss/metrics and the task gets the
run endpoint, live monitor, `run.json` provenance, sweeps and replicates for
free. Walkthrough: `user_tasks/README.md`.

## Where to go next

- `README.md` — feature overview and optional extras (`detection`, `timm`,
  `optuna`, `tensorboard`, dataset providers)
- `user_models/README.md` — drop in your own architectures
- `user_tasks/README.md` — define whole new task families (ADR-058)
- `documentation/ARCHITECTURE.md` — how the pieces fit
