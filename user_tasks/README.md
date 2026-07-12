# Custom tasks (drop-in) · Tarefas customizadas

Define a whole new task family — name, tab color, config fields, model, data,
loss, metrics — in **one documented Python file**, and VisionForge gives you
the GUI form, live training monitor, run history, `run.json` provenance,
TensorBoard, hyperparameter sweeps and multi-seed replicates for free. No
React, no FastAPI, no training loop. See ADR-058.

Defina uma família de tarefa inteira — nome, cor da aba, campos de config,
modelo, dados, loss, métricas — em **um arquivo Python documentado**, e o
VisionForge entrega de graça o formulário na GUI, monitor ao vivo, histórico,
proveniência em `run.json`, TensorBoard, sweeps de hiperparâmetros e réplicas
multi-seed. Sem React, sem FastAPI, sem loop de treino.

## Quickstart (EN)

```bash
visionforge new-task cell_counting          # flat file: user_tasks/cell_counting.py
visionforge new-task cell_counting --package  # or user_tasks/cell_counting/task.py
```

The generated file **already trains** (on synthetic data). Then:

1. Edit the `Config` class — add your task's fields. Each `Field(description=…)`
   becomes help text in the auto-generated GUI form; constraints (`ge`, `gt`,
   `le`) are validated before any GPU time is spent.
2. Fill the four hooks:
   - `build_model(cfg)` → any `nn.Module`
   - `build_loaders(cfg)` → `(train, val, test_or_None)` loaders
   - `compute_loss(model, batch, cfg)` → scalar loss for one batch
   - `compute_metrics(model, loader, cfg)` → `{name: value}`, names matching
     the `metrics=` declared in `@register_task`
3. Run `visionforge gui` — your task appears as a real tab with its own color.
   Files are discovered on every request, so no server restart is needed.

What the engine does around your hooks: seeding (+ optional determinism),
device resolution (CUDA/CPU), AMP, early stopping, best-checkpoint selection
by your `primary_metric` (direction-aware — `"lower"` for errors, `"higher"`
for scores), SSE live events, TensorBoard scalars, the metric curve PNG, and
the ADR-013 `run.json` contract with the full environment block.

### API surface

Every custom task `<key>` gets, with no extra code:

| Endpoint | What it does |
|---|---|
| `GET /api/tasks` | lists built-ins + your tasks (drives the GUI tabs) |
| `GET /api/custom/<key>/schema` | your Config's JSON Schema (drives the form) |
| `POST /api/custom/<key>/run` | one training run, SSE live monitor included |
| `POST /api/custom/<key>/sweep` | grid/random/Optuna sweep over **any Config field**, including the ones you declared (ADR-045) |
| `POST /api/custom/<key>/replicates` | N seeds → mean ± std ± 95% CI, never ranked (ADR-056) |

There is deliberately no `/compare` for custom tasks: comparing alternatives
is a one-axis sweep over whichever field your task declares.

### Level 2 — when training is not epoch-shaped

GANs, EM-style loops, two-stage pipelines: override `run(cfg, ctx)` instead of
the four hooks and own the whole loop. `ctx` gives you `run_dir`,
`emit(event)` (the SSE stream), `save_checkpoint(model, filename)` and
`write_run_json(...)` so your loop still honours every contract. Return the
final metrics dict.

### Rules that keep it working

- **`num_workers=0` for datasets defined in the task file** (ADR-030).
  `DataLoader` workers are spawned processes (always on Windows) that
  re-import the dataset's module by name — and task files are loaded from a
  path, not an importable package, so that re-import fails (the symptom is
  `EOFError: Ran out of input`). Datasets that need parallel workers must
  live in an installed/importable package; then pass `cfg.data.num_workers`.
- `key` is lowercase `[a-z0-9_]`; the five built-in keys are reserved.
- `accent` is a `#rrggbb` color; `primary_metric` must be one of `metrics`.
- Files (or package dirs) starting with `_` are ignored — shared helpers can
  live here.
- A broken file logs a warning and is skipped; it never crashes the server.
- This runs **your own local Python** on your own machine (ADR-005/048).
  Nothing is fetched from the network — the trust boundary is your filesystem.

## Guia rápido (PT)

```bash
visionforge new-task contagem_celulas
```

O arquivo gerado **já treina** (com dados sintéticos) — abra `visionforge gui`
e a aba nova aparece com a cor que você escolher, sem reiniciar o servidor.
Depois é substituir os TODOs:

1. **`Config`** — seus campos viram o formulário da GUI automaticamente
   (`description=` vira texto de ajuda; `ge`/`gt`/`le` validam antes de gastar
   GPU).
2. **Os quatro hooks** — `build_model`, `build_loaders`, `compute_loss`,
   `compute_metrics`. O engine cuida de todo o resto: seed, device, AMP, early
   stopping, melhor checkpoint pela sua métrica primária, monitor ao vivo,
   TensorBoard e `run.json` com proveniência completa.
3. **Sweeps e réplicas de graça** — `POST /api/custom/<key>/sweep` varre
   qualquer campo do seu Config (inclusive os que você declarou);
   `/replicates` treina N seeds e reporta média ± desvio ± IC 95%, nunca
   ranqueado (réplicas são amostras, não competidoras).

Treino que não é em épocas (GAN, EM)? Implemente `run(cfg, ctx)` — nível 2 —
e seja dono do loop inteiro, mantendo os contratos via `ctx`.

**Atenção (`num_workers`, ADR-030):** datasets definidos no próprio arquivo da
tarefa precisam de `num_workers=0` — os workers do DataLoader são processos
novos que re-importam o módulo do dataset pelo nome, e arquivos de tarefa são
carregados por caminho, não como pacote importável (o sintoma é
`EOFError: Ran out of input`). Dataset que precisa de workers paralelos deve
morar num pacote instalado/importável.

## Working example · Exemplo funcional

`example_counting/task.py` registers **Contagem (exemplo)**: a tiny CNN counts
bright dots in synthetic 32×32 images — no dataset on disk, trains in seconds
on CPU. Copy it as a starting point, or delete the folder.
