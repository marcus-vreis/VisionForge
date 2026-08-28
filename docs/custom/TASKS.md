# Your own task families · Suas próprias tarefas

Define a whole new task — name, tab colour, config fields, model, data, loss,
metrics — in **one documented Python file**, and VisionForge gives you the GUI
form, the live training monitor, the run history, `run.json` provenance,
TensorBoard, hyperparameter sweeps and multi-seed replicates for free. No
React, no FastAPI, no training loop. See ADR-058.

Defina uma tarefa inteira — nome, cor da aba, campos de config, modelo, dados,
loss, métricas — em **um arquivo Python documentado**, e o VisionForge entrega
de graça o formulário na interface, o monitor ao vivo, o histórico, a
procedência em `run.json`, TensorBoard, buscas de hiperparâmetros e réplicas
multi-seed. Sem React, sem FastAPI, sem laço de treino.

## Start · Começando

```bash
visionforge new-task cell_counting             # user_tasks/cell_counting.py
visionforge new-task cell_counting --package   # user_tasks/cell_counting/task.py
```

The generated file **already trains**, on synthetic data. Open
`visionforge gui` and the new tab is there, in the colour you chose — task
files are discovered on every request, so no server restart is needed.

O arquivo gerado **já treina**, com dados sintéticos. Abra o `visionforge gui`
e a aba nova já está lá, na cor que você escolheu — os arquivos de tarefa são
descobertos a cada requisição, então não é preciso reiniciar o servidor.

Then replace the TODOs · Depois é substituir os TODOs:

1. **The `Config` class** — your fields become the GUI form automatically. Each
   `Field(description=…)` becomes the help text behind the field's info dot,
   and constraints (`ge`, `gt`, `le`) are validated before any GPU time is
   spent. · Seus campos viram o formulário sozinhos; `description=` vira o
   texto do "i" ao lado do campo, e `ge`/`gt`/`le` validam antes de gastar GPU.
2. **The four hooks** · **Os quatro hooks**:
   - `build_model(cfg)` → any `nn.Module`
   - `build_loaders(cfg)` → `(train, val, test_or_None)`
   - `compute_loss(model, batch, cfg)` → scalar loss for one batch
   - `compute_metrics(model, loader, cfg)` → `{name: value}`, with names
     matching the `metrics=` you declared in `@register_task`

Everything around those hooks is the engine's job: seeding (and optional
determinism), device resolution, AMP, early stopping, best-checkpoint selection
by your `primary_metric` (direction-aware — `"lower"` for errors, `"higher"`
for scores), the SSE live events, the TensorBoard scalars, the metric curve
PNG, and the ADR-013 `run.json` contract with the full environment block.

Tudo em volta desses hooks é trabalho do engine: semente (e determinismo
opcional), escolha de device, AMP, early stopping, seleção do melhor checkpoint
pela sua métrica primária (sabendo a direção — `"lower"` para erros, `"higher"`
para escores), os eventos ao vivo, os escalares do TensorBoard, o PNG da curva
e o contrato do `run.json`.

## What you get for free · O que vem de graça

Every custom task `<key>` gets these endpoints with no extra code:

| Endpoint | What it does · O que faz |
|---|---|
| `GET /api/tasks` | lists built-ins + your tasks (drives the GUI tabs) |
| `GET /api/custom/<key>/schema` | your Config's JSON Schema (drives the form) |
| `POST /api/custom/<key>/run` | one training run, live monitor included |
| `POST /api/custom/<key>/sweep` | grid/random/Optuna over **any** Config field, including the ones you declared (ADR-045) |
| `POST /api/custom/<key>/replicates` | N seeds → mean ± std ± 95% CI, never ranked (ADR-056) |

There is deliberately no `/compare` for custom tasks: comparing alternatives is
a one-axis sweep over whichever field your task declares.

Não existe `/compare` para tarefas customizadas, de propósito: comparar
alternativas é uma busca de um eixo sobre o campo que a sua tarefa declarar.

## Level 2 — training that is not epoch-shaped

GANs, EM-style loops, two-stage pipelines: override `run(cfg, ctx)` instead of
the four hooks and own the whole loop. `ctx` hands you `run_dir`,
`emit(event)` (the live stream), `save_checkpoint(model, filename)` and
`write_run_json(...)`, so your loop still honours every contract. Return the
final metrics dict.

Treino que não é em épocas (GAN, EM, pipelines de dois estágios)? Implemente
`run(cfg, ctx)` em vez dos quatro hooks e seja dono do laço inteiro, mantendo
todos os contratos através do `ctx`.

## Rules that keep it working · Regras que mantêm isso funcionando

- **`num_workers=0` for datasets defined inside the task file** (ADR-030).
  DataLoader workers are spawned processes — always, on Windows — that
  re-import the dataset's module *by name*, and task files are loaded from a
  path rather than as an importable package, so that re-import fails. The
  symptom is `EOFError: Ran out of input`. A dataset that needs parallel
  workers has to live in an installed, importable package; then pass
  `cfg.data.num_workers`. · Datasets definidos no próprio arquivo da tarefa
  precisam de `num_workers=0`, pelo mesmo motivo.
- `key` is lowercase `[a-z0-9_]`, and the five built-in keys are reserved.
- `accent` is a `#rrggbb` colour; `primary_metric` must be one of `metrics`.
- Files (or package directories) starting with `_` are ignored, so shared
  helpers can live in the same folder.
- A broken file logs a warning and is skipped; it never crashes the server.
- This runs **your own local Python** on your own machine (ADR-005/048).
  Nothing is fetched from the network — the trust boundary is your filesystem.

## Working example · Exemplo funcional

`user_tasks/example_counting/task.py` registers **Contagem (exemplo)**: a tiny
CNN counting bright dots in synthetic 32×32 images. No dataset on disk, trains
in seconds on CPU. Copy it as a starting point, or delete the folder.

Ele registra a tarefa **Contagem (exemplo)**: uma CNN pequena contando pontos
claros em imagens sintéticas de 32×32. Sem dataset em disco, treina em segundos
na CPU. Copie como ponto de partida, ou apague a pasta.
