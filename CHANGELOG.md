# Changelog

All notable changes to VisionForge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0, the config schema and the HTTP API may change
between minor releases. Configs carry a `schema_version` and are migrated on
load (ADR-039), so a config exported from an older release keeps working.

Every entry links the ADR that records *why* the change was made; the full
reasoning lives in [`docs/dev/DECISIONS.md`](docs/dev/DECISIONS.md).

## [Unreleased]

## [0.6.0] — 2026-08-08

### Added

- **Detecção passa a reportar intervalo de confiança no mAP.** Era a única das
  cinco tarefas sem — e por um motivo real: mAP ordena todas as detecções do
  conjunto por confiança e percorre a curva precisão/recall, então é propriedade
  do *conjunto*, não média de números por imagem. Não há acumulador para somar,
  e a métrica é recomputada a cada reamostragem. Um sorteio que perde uma classe
  inteira é descartado em vez de entrar na média, e o `n_resamples` reportado
  conta só os sobreviventes ([ADR-087](docs/dev/DECISIONS.md)).

- **O painel de detecção mostra exemplos das classes e o layout detectado.**
  As caixas anotadas viram miniaturas recortadas, uma por classe — contagem não
  diz se o rótulo está no lugar certo, e a imagem inteira não diz a qual caixa
  ele se refere. Cada split também informa qual das duas convenções YOLO foi
  resolvida (`train/images` ou `images/train`), que é como um dataset convertido
  pela metade aparece antes de ocupar a GPU.

## [0.5.1] — 2026-08-06

## [0.5.0] — 2026-08-05

### Added

- **Cada hiperparâmetro tem uma linha explicando o que ele faz**, e os painéis
  passam a separar o básico (épocas, batch, learning rate, seed) do avançado,
  que começa recolhido. Nenhum parâmetro saiu da tela — o corte é por frequência
  de ajuste, não por importância, e um valor avançado fora do padrão abre a
  seção sozinho ([ADR-085](docs/dev/DECISIONS.md)).

- **Detecção aceita filtros de pré-processamento.** A Ultralytics é dona do
  próprio pipeline de dados, então os filtros são aplicados uma vez numa cópia
  temporária que o `data.yaml` passa a apontar — o que também é ~30x menos CPU
  que filtrar por imagem a cada época. A cópia é chaveada por conteúdo (um sweep
  de 20 trials materializa uma vez), removida ao fim mesmo quando o treino
  falha, e o `run.json` continua registrando o dataset **original**
  ([ADR-084](docs/dev/DECISIONS.md)).

## [0.4.0] — 2026-08-05

### Added

- **`/api/health`** informa a versão e o bundle da SPA com que o processo subiu.
  Serve para diagnosticar o caso em que um `visionforge gui` deixado aberto
  durante uma alteração passa a servir JavaScript novo com backend velho — os
  estáticos são lidos do disco a cada requisição, mas os módulos Python só na
  partida ([ADR-086](docs/dev/DECISIONS.md)).

- **Uma chave liga e desliga a data augmentation**, em todas as cinco tarefas.
  Desligada, os parâmetros somem da tela e o `run.json` registra o estado — em
  vez de você ter que zerar cada campo à mão, que em detecção são 15. Os valores
  ficam guardados, então religar devolve o ajuste. Normalização e tamanho da
  imagem saíram da seção: não são augmentation, valem para treino, validação e
  teste ([ADR-083](docs/dev/DECISIONS.md)).

- **O histórico mostra em qual dataset cada run foi treinado** — selo no card,
  caminho e fingerprint no detalhe do run, e um veredito "mesmos dados?" ao
  comparar runs. Nada novo é medido: os dados já estavam no `run.json`. O nome
  aparece em 69 dos 78 runs existentes porque cai para `config.data.base_dir`;
  a verificação por hash só vale de 26/07/2026 em diante, e o comparador diz
  isso em vez de adivinhar ([ADR-082](docs/dev/DECISIONS.md)).

## [0.3.1] — 2026-08-05

### Fixed

- **`WinError 1455: o arquivo de paginação é muito pequeno` while starting a
  run** now says what it means. Every DataLoader worker is a separate process
  that re-imports torch and its CUDA DLLs — on Windows the start method is
  spawn, not fork — so the cost is roughly a gigabyte per worker, per loader.
  Windows reported the shortfall against whichever DLL happened to be loading,
  which pointed at torch and hid the cause. The GUI now explains it and names
  `data.num_workers` ([ADR-081](docs/dev/DECISIONS.md)).

### Changed

- **Data modules own their loaders and shut the worker pools down.** Each call
  to `train_loader()` built a *new* DataLoader, so a second call meant a second
  set of worker processes for the same split; and nothing ever stopped them
  explicitly. Splits are now built once and every block closes its data module
  in a `finally`, so a run that raises does not leave its workers behind. The
  test split also no longer asks for `persistent_workers` — it is read once —
  which drops the peak worker count for a classification run from 12 to 8
  ([ADR-081](docs/dev/DECISIONS.md)).

## [0.3.0] — 2026-08-04

### Changed

- **"Testar modelo" takes one labelled folder** instead of a dataset root plus
  three split names, and dispatches per task — so regression, segmentation and
  anomaly can be tested at all, where they previously answered with a raw
  `Input should be 'binary' or 'multiclass'` and only classification and
  detection ever worked. The folder is given in the label shape the run was
  trained with; regression points at its `.csv` manifest, since its data model
  has manifests rather than split folders
  ([ADR-080](docs/dev/DECISIONS.md)).

### Fixed

- **`visionforge doctor` recommended the CPU wheel on a CUDA machine.** Driver
  6xx renamed the `nvidia-smi` header field to `CUDA UMD Version`, which the
  parser did not match, so a recent driver looked like no GPU at all — an
  RTX 5060 Ti on driver 610.74 was told to `pip install "visionforge-studio[cpu]"`.
  Both spellings are accepted now.
- **The documented frontend type-check was a no-op.** The root `tsconfig.json`
  is a solution file with `"files": []`, so `npx tsc --noEmit` type-checked
  nothing and always passed — which is how a `base_dir` reference that no longer
  existed survived a green check. The command is now `npm run typecheck`
  (`tsc -b`), and CI gained a `frontend` job running vitest and the SPA build;
  neither had ever run there.
- **Esc in the run history could desynchronise from the backdrop.** Its handler
  read the step-back function from a ref written during render; it is now a
  `useCallback` the effect depends on directly.
- **Detection runs produced no plots and no checkpoint.** Ultralytics resolves a
  relative `project` path under its own `runs_dir`, so every artifact landed in
  `runs/detect/outputs/...` and the real run directory held only `data.yaml` and
  `run.json` — which is why every post-training action on a detection run
  reported a missing `best.pt`. The path is now absolute, and the run collects
  both confusion matrices, all four Box curves and the validation prediction
  sample ([ADR-079](docs/dev/DECISIONS.md)).
- **Custom-task runs showed no plots** although the engine was drawing one:
  `run.json` hardcoded `graphics: []`, orphaning the primary-metric curve on
  disk. It is now declared, alongside a train-loss curve
  ([ADR-079](docs/dev/DECISIONS.md)).

## [0.2.0] — 2026-08-03

> Tagged locally and superseded before it was published; 0.3.0 is the first
> release after 0.1.0 to reach PyPI. The entries below shipped as part of it.

### Added

- **Bootstrap confidence intervals on a single run's test metrics.** Every
  classification run reports `0.8734 [0.8412, 0.9021]` instead of a bare number,
  written to `run.json` as `metric_cis` and shown in the result tiles, the
  run-detail panel and the model card. Always on — the metrics are recomputed
  with vectorized arithmetic (700x faster than one sklearn call per resample,
  pinned against sklearn to 1e-16), so it costs ~0.02 s and needs no knob
  ([ADR-074](docs/dev/DECISIONS.md)).
- **The same intervals for regression, segmentation and anomaly** — MSE/RMSE/MAE/R²,
  mIoU/Dice/pixel-accuracy, AUROC/F1. The image is always the resampling unit,
  never the smaller thing inside it: segmentation sums per-image confusion
  matrices rather than resampling pixels, which would report an interval far
  tighter than the evidence supports ([ADR-076](docs/dev/DECISIONS.md)).
- **A training queue.** A second submission no longer gets a 409 — it lines up and
  starts on its own when the GPU frees, so an evening's experiments can be
  submitted and left. `GET /api/queue` lists what is waiting, a not-yet-started
  job can be dropped, and the bottom bar shows `⧗ fila N` with a panel
  ([ADR-075](docs/dev/DECISIONS.md)).
- **Test-set diagnostics for every task, not just classification**: predicted-vs-actual
  scatter and residual histogram (regression), per-class IoU and confusion matrix
  (segmentation), score histogram with the decision threshold (anomaly)
  ([ADR-077](docs/dev/DECISIONS.md)).
- **Grad-CAM shows the true class next to the predicted one**, with wrong
  predictions outlined in red. Class names are recovered from the training
  folder and ground truth from each image's parent folder — never guessed: a
  count mismatch shows the index, and an unlabeled folder shows nothing
  ([ADR-077](docs/dev/DECISIONS.md)).

### Fixed

- **The markdown model card returned 500 for every task except classification.**
  It hardcoded the classification epoch columns and formatted each cell with
  `:.4f`, so the `"?"` fallback for a missing key always raised. Columns now come
  from the run's own history ([ADR-077](docs/dev/DECISIONS.md)).
- **Per-run actions crashed on a researcher-defined task's run.** `test`,
  `gradcam`, `batch_predict` and `export_onnx` read `config.task`, which a custom
  run does not have, fell through to the classification path and died rebuilding
  a ResNet. They now answer 400 naming the task
  ([ADR-077](docs/dev/DECISIONS.md)).
- **A custom task whose config used `Literal`, `Path` or any non-builtin
  annotation failed to load** with a confusing Pydantic "is not fully defined"
  error pointing at the user's file. Task modules are now registered in
  `sys.modules` before execution, which is where Pydantic resolves the
  stringified annotations the scaffold generates.
- The install docs recommended `cu121` and never listed `cu128`, walking anyone
  with an RTX 50-series card into a build that imports fine and fails at the
  first kernel launch. The `docker build` example also carried a literal `\n`
  instead of a line continuation.
- **A grid axis could not reach a scheduler's dependent parameters.** Putting
  `step` on the axis while the scalar control still said `none` left `step_size`
  and `gamma` unrendered, so the sweep ran them on defaults with no way to see or
  sweep them. The form now shows the union over the scalar kind and every kind on
  the axis ([ADR-078](docs/dev/DECISIONS.md)).
- **The image viewer covered the plot it was showing.** The caption and close
  button floated over the figure, hiding the strip matplotlib uses for the
  x-axis and legend. They now occupy their own rows, so nothing is covered and
  nothing is cut ([ADR-078](docs/dev/DECISIONS.md)).
- **An unknown preprocessing filter returned 500.** Typing `blur` — the natural
  guess for `gaussian_blur` or `median_blur` — produced a server fault with no
  hint. It is now a 422 that lists every registered filter
  ([ADR-078](docs/dev/DECISIONS.md)).
- **`bump-my-version bump` could never complete**, so the documented one-command
  release (ADR-073) had never once worked: the pytest pre-commit hook ran
  `uv run`, which re-locked because the version had just changed, and pre-commit
  aborted on the modified `uv.lock`. The hook now runs `--frozen`.

### Changed

- **Clicking outside a dialog steps back one level** instead of dismissing the
  whole stack: plot → run detail → history list → closed, with Esc mirroring it.
  The header's × still closes everything at once
  ([ADR-078](docs/dev/DECISIONS.md)).
- A busy server **queues** a submission instead of refusing it with 409. The
  per-endpoint validation errors (422 for a bad config, 404 for an unknown custom
  task) are unchanged and still happen before anything is enqueued
  ([ADR-075](docs/dev/DECISIONS.md)).

## [0.1.0] — 2026-07-29

First public release. Everything below already shipped on `main`; this is the
point at which it becomes a version other people can install and cite.

> **Installed from PyPI as `visionforge-studio`.** The bare `visionforge` name
> is an unrelated project by another author. The import name, the `visionforge`
> CLI command and the project itself are unchanged.

### Tasks

- **Classification** — ResNet 18/34/50/101, EfficientNet B1/B7, VGG 16/19,
  AlexNet, plus any `timm` backbone (ADR-051) or a drop-in custom model
  (ADR-048/049). Binary, multiclass and multilabel.
- **Object detection** — Ultralytics YOLOv8/9/10/11/12/26 and RT-DETR, plus a
  torchvision backend (Faster R-CNN, SSD, RetinaNet) with its own loop
  (ADR-035). Full Ultralytics hyperparameter surface (ADR-040).
- **Image regression** — CSV-manifest datasets, CNN backbone + linear head,
  MSE/RMSE/MAE/R².
- **Semantic segmentation** — DeepLabV3, FCN, LR-ASPP and a hand-rolled U-Net;
  mean IoU, Dice, pixel accuracy, with `ignore_index` respected everywhere.
- **Anomaly detection** — convolutional autoencoder and PatchCore, image-level
  AUROC on an MVTec-style layout.
- **Your own task** — define a whole new task family in one documented Python
  file (`visionforge new-task`), with sweeps, replicates and the live monitor
  for free (ADR-058).

### Strategies

- Single run, K-fold cross-validation (classification, regression,
  segmentation), grid / random / Optuna-TPE sweeps (ADR-052), transfer learning
  (ADR-046/047), model comparison, and multi-seed replicates reporting
  `mean ± 95% CI` (ADR-056).
- **Paired significance testing** (ADR-061) — compares N configurations over
  the *same* seeds, picks and justifies a paired t-test or Wilcoxon, reports
  Cohen's `d_z`, a bootstrap CI of the difference and Holm-Bonferroni
  correction. It refuses to compare runs whose seeds do not line up, and flags
  when the seed count makes significance unreachable, so "not significant" is
  never read as "no effect".
- **Paper-ready output** — every advanced report is also written as a
  `booktabs` LaTeX table with notes stating what each interval covers.

### Reproducibility

- Versioned `run.json` for every run (ADR-013) carrying the full config, seed,
  per-epoch history, `environment` (Python, torch, CUDA, cuDNN, GPU model —
  ADR-057) and a `dataset_fingerprint` (ADR-061), so "same data" is checkable
  rather than assumed.
- `training.deterministic` in **every** task and in the custom-task SDK
  (ADR-062). Detection defaults to `True` to mirror `YOLO.train`; the rest
  default to `False` because pinning cuDNN costs throughput.
- Config `schema_version` with migrations (ADR-039); YAML round-trips between
  the GUI and the CLI.

### Interface

- React SPA served by the same Python process — no separate frontend to run.
- Canonical panel layout across all tasks (ADR-059): experiment name, YAML
  export/import, strategy selector, model, training, dataset stats,
  preprocessing filters and augmentation with live preview.
- History grouped by task family with wrapping filters, multi-select delete and
  run comparison (ADR-063/064); every dropdown is drawn by the app, so no
  operating-system popup breaks the dark theme.
- Datasets is its own surface: one-shot download from torchvision, Roboflow,
  Kaggle or Hugging Face (ADR-055).
- Post-training: per-checkpoint testing on new data, batch prediction to CSV,
  Grad-CAM, ONNX export with a PyTorch-vs-runtime latency benchmark, and
  TensorBoard scalars per run (ADR-054).

### Verification

- `visionforge doctor` — detects GPU/CUDA and prints the exact torch install
  line for the machine (ADR-042).
- `visionforge selftest` — trains every task through the real API on synthetic
  data and asserts the run, the report shape and the live-progress contract
  (ADR-060). Offline, ~90 s, CI-ready.
- A full matrix on **real** datasets (ADR-065) is recorded in
  [`docs/dev/VALIDATION.md`](docs/dev/VALIDATION.md): 21 cases across
  five tasks and five strategies, all passing.
- 1274 backend tests, 102 frontend tests, ruff + mypy clean, gated in CI.

### Fixed in the run-up to this release

- Transfer learning trained correctly but streamed no live progress, leaving
  the GUI's progress bar dead for the whole run (ADR-065).
- Multi-trial strategies (K-fold, sweeps, replicates) emitted no progress
  events, so the bar advanced on wall-clock only.
- Opening the history after a K-fold returned a 500: cross-validation wrote
  timezone-aware timestamps while every other writer used naive local time.
- The preprocessing preview served a stale "final" image from the browser
  cache, one pipeline behind.
- The torchvision detection backend never seeded, so `seed: 42` was a claim
  nothing backed (ADR-062).
- Sweeps and replicates silently accepted a metric no trial reported, ranking
  every trial 0.0 and crowning an arbitrary winner (ADR-060).
- Replicated comparison ranked descending regardless of metric direction, so a
  MAE of 4.02 beat a MAE of 0.99 (ADR-061).

[Unreleased]: https://github.com/marcus-vreis/VisionForge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/marcus-vreis/VisionForge/releases/tag/v0.1.0
