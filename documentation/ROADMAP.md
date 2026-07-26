# VisionForge — Roadmap & improvement plan

> Live planning doc owned by Cowork (judgment/decisions/docs). Each item below was
> grounded in the actual code, given a recommendation, and — where it's a real
> decision — an ADR. The agent team implements from here + the linked specs.
> Decisions here are **Proposed** until you approve them.

## Priority summary

| # | Item | Priority | Kind | Decision |
|---|------|:--:|------|----------|
| 1 | Task-aware history & results (detect missing info/plots) | **P0** | bug+feature | ADR-043 |
| 1b | Bulk-delete in history (multi-select + confirm) | **P0** | feature | ADR-043 |
| 2 | Live monitor: native metrics + wrong task label + grid-search logs blank | **P0** | bug+feature | ADR-044 |
| 2b | Per-model "test" should take one labeled folder, not a 3-split base_dir | **P1** | bug | ADR-047 |
| 3 | Detection: verify the rest beyond the basics | **P1** | testing | ROADMAP |
| 4 | New methods on detect & others (grid search, preprocessing) | **P1** | feature | ADR-041 + ROADMAP |
| 5 | `BaseTrainer` parent class / kill duplication | **P1** | refactor | ADR-045 |
| 6 | Docs cleanup (phase plans) + code comment policy | **P1** | hygiene | ADR-046 |
| 7 | Docker + `visionforge doctor` | **P2** | infra | ADR-042 |
| 8 | Online dataset download (Roboflow/Kaggle/HF) | **P2** | feature | ROADMAP |
| 9 | Design reformulation (simpler) | **P3** | design | ROADMAP |
| 10 | Animated SVG of the VisionForge icon | **P3** | polish | ROADMAP |

P0 = ship next · P1 = soon · P2 = planned · P3 = end of list.

---

## 1. Task-aware history & results — P0 (ADR-043)

**Your note:** reformular o histórico (pins por tipo de treino), mostrar os
principais resultados; no detect falta info; no detect (e provavelmente outros)
os gráficos e resultados não aparecem.

**What I found (three distinct root causes, all real):**

1. **Wrong/empty metrics for 3 tasks.** `routes.py::_SUMMARY_METRIC_KEYS` only has
   rows for `classification` and `detection`. Regression, segmentation, and
   anomaly fall through to the **classification** row (`accuracy`/`f1`/`val_loss`)
   — metrics they don't produce — so their history cards show nothing. Detection's
   row has only `map50`/`map50_95`, so precision/recall/box_loss never show.
2. **Detection plots don't render.** The frontend *does* render
   `artifacts.graphics` generically and even has labels for the detection plots
   (`results.png`, `BoxPR_curve.png`, `BoxF1_curve.png`). The failure is upstream:
   `detection_trainer._write_run_json` only lists those files `if (run_dir/p).exists()`
   — but **Ultralytics writes its plots to its own save dir, not VisionForge's
   `run_dir`**, so the check fails and `graphics` ends up empty. The torchvision
   backend produces **no plots at all** (no `MetricsPlotter` call).
3. **No pins / per-task grouping.** The history is one flat, time-sorted list with
   a single compact card shape. Your "pins per task" is a genuinely new feature.

**Decision (ADR-043):** introduce a **task descriptor registry** — one place that
declares, per task, (a) the history-card metric keys + labels and (b) the expected
plot filenames + human labels. Backend `_summary_metrics` and the frontend label
maps both read from it, so adding a task is one entry instead of edits in two
files. Separately, **fix detection plot collection**: after an Ultralytics run,
copy/relocate its generated plots into `run_dir` (or point `artifacts.graphics` at
the real Ultralytics dir), and give the torchvision backend a minimal loss/mAP
curve via `MetricsPlotter`. For **pins**: add an optional `pinned: bool` + `tags`
to the run summary (persisted in `run.json` or a sidecar), and let the history
overlay group by task and float pinned runs to the top.

**My refinement / question:** "pins per task" can mean two things — (a) *pin
individual runs* (favorites), or (b) *a per-task section/lane* in the history.
I'd do **both, cheaply**: group the list by task (lanes) and allow pinning runs
within a lane. Confirm that's what you pictured.

### 1b. Bulk-delete in history (your new request)

**Your note:** um botão de lixeira no histórico que funciona como o comparar —
seleciona vários treinamentos e exclui de verdade, com um "tem certeza?" antes.
Está lotado de runs de teste.

**What I found:** the backend already has `DELETE /api/runs/{run_id}` (single
run). So this is **frontend-only**: add a "🗑 selecionar" mode to `HistoryOverlay`
that reuses the exact multi-select machinery the "↔ Comparar" toggle already has,
then a confirm dialog ("Excluir N treinamentos? Esta ação é permanente.") that
calls the existing delete endpoint per selected run and refreshes the list. Low
risk, immediate declutter. Worth doing **with the history rework (ADR-043)** since
both touch `HistoryOverlay` selection state. One safety call I'd make: a
**bulk-delete endpoint** (`POST /api/runs/delete` taking a list) so N deletes are
one atomic request instead of N round-trips, and pinned runs are skipped unless
explicitly included.

---

## 2. Live monitor — native metrics, wrong label, grid-search logs blank — P0 (ADR-044)

**Your notes:** (a) detect logs só mostram val_loss/val_accuracy + um — devia
mostrar os vários resultados (você decidiu: logs estruturados, minha sugestão);
(b) a telinha do log diz "visionforge --classification" mesmo rodando detect —
está errado; (c) os logs do grid search de classificação não aparecem na telinha
(no terminal estão lá).

**What I found — all three are the same root cause: the overlay is hard-wired to
the classification event shape.**

- **Native metrics:** `detection_trainer` streams `map50`/`map50_95`/`box_loss`
  and then **fakes** `train_loss`/`val_loss`/`val_accuracy` = box_loss just so the
  classification overlay shows something. Precision, recall, cls_loss, dfl_loss,
  lr — all dropped.
- **Wrong label (confirmed bug):** `TrainingOverlay.tsx:93` literally hardcodes
  the line `` `$ visionforge train --task classification` `` — a fixed string,
  regardless of the real task. Trivial fix: interpolate the actual task.
- **Grid-search logs blank (confirmed cause):** the epoch-log effect
  (`TrainingOverlay.tsx:133-145`) destructures `train_loss`/`val_loss`/
  `val_accuracy` and calls `.toFixed(4)` on them **with no guard**. Any event that
  doesn't carry exactly those three numbers (the trial-wrapped grid-search stream,
  or any non-classification task) throws inside the effect → no log line is ever
  appended → the panel stays blank even though the SSE events arrive (and the
  terminal, which prints server-side, looks fine). It's brittle field access, not
  a missing stream.

**My answer to your question (and a refinement):** you're right that forcing
detection into `val_loss`/`val_accuracy` is wrong. But "show the raw Ultralytics
output" — meaning its full console spew — would be **noisy** and hard to read in
the overlay. Better: stream the **structured per-epoch metric dict** Ultralytics
already gives us (it's right there in the callback) and render it as labeled rows
the overlay understands per task. So: *don't filter down to classification's
shape, but do keep it structured* — not raw stdout. That gives you precision,
recall, the three loss components, lr, etc., cleanly.

**Decision (ADR-044):** the SSE `epoch_end` event carries a task-native
`metrics: {name: value}` map; the `TrainingOverlay` renders it via the same task
descriptor registry from ADR-043, **formatting defensively** (skip a metric that's
absent instead of crashing the render — this alone fixes the grid-search blank
logs). Drop the faked classification fields for detection, and **interpolate the
real task** into the command label instead of the hardcoded "classification". This
also unblocks regression/segmentation/anomaly showing *their* native metrics live
(today they piggyback on val_loss too).

---

## 2b. Per-model "test" should take one labeled folder — P1 (ADR-047)

**Your note:** no detalhe de um run, o botão "test" deixa escolher uma pasta pra
rodar o modelo, certo? Mas ele detecta train/val/test — o usuário deveria escolher
só a pasta específica de imagens. Analisar.

**What I found:** the "+ testar" form (`RunDetailPanel`) sends `base_dir` +
`train_dir`/`val_dir`/`test_dir` (defaults `train`/`val`/`test`) to
`POST /api/runs/{id}/test`, which reloads the checkpoint and runs the `Evaluator`.

**Questioning the rule (you asked me to):** there are actually *two different
things* hiding behind "test", and the current form conflates them:

1. **Evaluate with labels** — compute accuracy/F1/confusion/ROC on a *labeled*
   set. This genuinely needs labels, i.e. a class-labeled folder (ImageFolder
   shape). This is what "+ testar" does today and it's the right behavior — **but**
   asking for a 3-split `base_dir` is wrong: a *test* needs **one** labeled folder
   (the eval set), not train+val+test. You're right.
2. **Just predict** — run the checkpoint over a folder of *raw, unlabeled* images
   and dump a CSV. That already exists separately as **batch prediction**.

So the rule I'd set (ADR-047): "+ testar" asks for **one labeled folder** (the
class-subdir folder to evaluate on), not a base_dir with three split names. If the
user only has raw images and wants predictions, that's batch-predict, not test.
This keeps the two actions distinct and each one's input obvious. For tasks whose
"labels" aren't class subdirs (detection = YOLO labels, segmentation = masks,
regression = CSV), the single-folder picker adapts to that task's label shape via
the same task descriptor registry (ADR-043).

---

## 3. Detection — verify beyond the basics — P1

**Your note:** o básico do detect está funcional, não testei o resto.

**What I found:** Ultralytics path is well-covered by mocked unit tests; the
torchvision path has an opt-in real E2E (`VF_RUN_DETECTION_INTEGRATION=1`). What's
*not* verified is the **full GUI loop on a real dataset** end-to-end (the kind of
thing that surfaces the plot/log gaps above).

**Recommendation:** once ADR-043/044 land, run one real detection job per backend
(Ultralytics + torchvision) through the GUI on a small dataset and walk the whole
surface: live logs, history card, run detail, plots, per-model test, ONNX export.
This is a **manual verification checklist**, not new code — I can write it as a
`vf-verify`-style smoke script. Track as a P1 task after #1/#2.

---

## 4. New methods on detect & others — P1 (ADR-041 + ROADMAP)

**Your note:** adicionar métodos novos no detect e outros (grid search, talvez
seção de filtragem/preprocessing).

**What I found:** grid/random search + model comparison are exactly the
**cross-task parity** work already specced in `CROSS_TASK_PARITY_PLAN.md`
(ADR-041). The team already shipped model-comparison for regression/segmentation
(slice 2). Grid/random search is slice 4 via the generic sweep runner.

**Preprocessing/filtering for detection:** the classification preprocessing
pipeline (blur/edges/CLAHE/wavelet…) is **PIL-based and task-agnostic**, but
detection delegates augmentation to Ultralytics and uses a YOLO data path, so a
generic image-filter pipeline doesn't slot in cleanly. **Recommendation:** for
detection, expose Ultralytics' own augmentation knobs (already done in ADR-040)
rather than the classification filter pipeline; for regression/segmentation, the
existing `_build_transforms` pipeline *can* carry preprocessing — wire the panel
there. So "filtering section" = per-task: native knobs for detection, the shared
pipeline for the CNN tasks. This is a design call worth stating; folded into the
parity plan.

---

## 5. `BaseTrainer` parent class — P1 (ADR-045)

**Your note:** limpar código, classe pai, pouca repetição, mais eficiente.

**What I found:** five trainers totalling **2,347 lines**
(`trainer.py` 480, `detection` 587, `segmentation` 478, `regression` 409,
`anomaly` 393). They independently reimplement the same mechanics: device/seed
setup, the epoch loop skeleton, early-stopping + best-checkpoint bookkeeping,
scheduler stepping, SSE event emission, and `_write_run_json` (the ADR-013
contract). The *task-specific* parts — train/eval step, loss, metric — genuinely
differ.

**Decision (ADR-045):** extract a thin `BaseTrainer` in `core/` holding the
identical mechanics (device/seed, run.json writing, SSE emission, early-stop +
checkpoint bookkeeping, scheduler) with abstract hooks (`train_step`, `eval_step`,
`compute_metrics`, `primary_metric`) each task overrides. This is **DRY without
coupling task logic** — each task still owns its step/loss/metric. It pairs
naturally with the `TaskRunner` handle (ADR-041): `BaseTrainer` is the
implementation base, `TaskRunner` the orchestration interface; the run.json
writing should produce the typed `RunResult` ADR-041 wants.

**Tension I want to flag (and resolve):** ADR-033 deliberately chose *standalone*
trainers to avoid coupling task families. A shared parent reintroduces a common
ancestor — is that a regression? No: ADR-033's concern was coupling task
*config/logic*, not sharing *infrastructure*. A `BaseTrainer` that owns only the
mechanics every run needs (and zero task-specific branching) is the right kind of
sharing. The line to hold: **no `if task == ...` inside `BaseTrainer`.** If a hook
needs task knowledge, it's a method the subclass overrides, never a branch in the
base.

---

## 6. Docs cleanup + code comment policy — P1 (ADR-046)

**Your note:** as docs de tasks específicas estão separadas de um jeito que você
não concorda; e comentários no código "situando porque tomou a decisão baseada
numa regra, ou pq trocou a parte" não fazem sentido ali.

**What I found:** `documentation/` carries four per-task plans
(`PHASE6/7/8/9_*_PLAN.md`) plus `GRADCAM_PLAN.md`. They were implementation
scaffolds; their *decisions* now live in `DECISIONS.md` (ADRs) and their *status*
in `TASKS.md`. They're redundant with the canonical set. In code, there are ~35
comments referencing ADRs or narrating change history.

**Decision (ADR-046), two parts:**

- **Docs:** the live, canonical set is `CLAUDE.md` (root), `ARCHITECTURE.md`,
  `DECISIONS.md`, `TASKS.md`, `ROADMAP.md` (this file), plus the active specs
  (`CROSS_TASK_PARITY_PLAN.md`, `DOCKER_PLAN.md`). The `PHASE*_PLAN.md` and
  `GRADCAM_PLAN.md` move to `documentation/archive/` (history, not deleted — they
  show how a task was built). `PROJECT_CONTEXT.md` is also archive-tier (it lags
  the 5-task reality). *File moves happen on Windows* (the mount can't unlink).
- **Comment policy (refines CLAUDE.md §9):** a one-line **pointer** to an ADR is
  good (`# standalone per ADR-033`) — it links the code to its decision. What's
  noise and should go: comments that **re-litigate a decision inline** (a
  paragraph re-explaining *why* when the ADR already does) or **narrate the edit
  history** (`# was SGD before`, `# moved here`, `# previously used closure`).
  The *why* of a decision lives in its ADR; the *why* of a change lives in git.
  Inline comments are for non-obvious **local** logic only.

---

## 7–8. Docker · Online datasets — P2

**Docker (ADR-042):** already specced in `DOCKER_PLAN.md` — `visionforge doctor`
first, then the GPU image. No change; it sits at P2 behind the P0/P1 UX fixes.

**Online dataset download:** a `/api/dataset/download` with a provider interface
(Roboflow SDK, Kaggle, HuggingFace `datasets`, torchvision built-ins like CIFAR).
Paste a URL/key in the GUI → download → auto-detect splits → train. **Care:** API
keys stay local, never logged; downloads land under a user-chosen dir, not baked
into outputs. This is high usability value and fits "facilitate everything" —
promote from backlog to a spec when P1 clears. Roboflow is the natural first
provider (its export is already YOLO-format, which detection consumes directly).

---

## 9–10. Design reformulation · Animated icon — P3

**Design (simpler):** the frontend is **token-driven** — colors/fonts/radius are
CSS variables in `index.css` (`@theme inline`, shadcn/ui), and per-task identity
is one `accent` flowing from `App.tsx`. So "make it more basic" is low-effort and
low-risk: trim to a smaller palette, reduce the gradient/wave flourishes
(`Waves.tsx`, the radial-gradient background), flatten the per-task accents to one
neutral scheme. I can produce a `DESIGN.md` mapping each lever, or prototype a
restyle, when it reaches the top. Rebuild the SPA (`npm run build`) after.

**Animated SVG icon:** a nice end-of-list polish. A lightweight, looping SVG
(stroke-draw or a subtle forge/spark motion) for the header/loading state.
Pure-SVG/CSS, no dependency. I can design an original one when you want it.

---

## My additions (context-driven suggestions)

Things I'd put on the radar that you didn't list but the code is asking for:

- **Typed `RunResult` + formal run.json schema.** Today each trainer hand-builds
  the run.json dict; a typed object (validated) would kill a class of "missing
  field" bugs (exactly what broke the history metrics), and it's the natural
  return type for the `TaskRunner` handle (ADR-041) and `BaseTrainer` (ADR-045).
  This is the quiet keystone that makes items #1 and #5 cleaner.
- **Run notes/tags, not just pins.** Once you add pins (#1), a free-text note +
  tags per run is a tiny increment with big research value ("this is the one that
  overfit", "baseline for the paper").
- **A “reproduce this run” button.** Every run.json already has the full config;
  one click to re-load it into the form (or re-run) is trivial given the contract
  and is a killer reproducibility feature for a research tool / demo.
- **Custom model drop-in (`ModelRegistry`).** Already in backlog; it's the
  extensibility twin of the dataset-download item and fits the registry pattern
  you already use for blocks.

---

## How this maps to work

- ADRs **043–046** are recorded as **Proposed** in `DECISIONS.md`. Approve them to
  flip to Accepted, then the team implements from the specs.
- Suggested order: **#1 → #2** (the visible UX gaps), then **#5 + #6** (refactor +
  hygiene while the surface is fresh), then **#4** (parity sweep), then **#7/#8**,
  with **#3** verification interleaved after #1/#2.
- The `RunResult` keystone (my additions) is worth doing *with* #5 — it makes #1,
  #5, and ADR-041's handle line up instead of fighting.
