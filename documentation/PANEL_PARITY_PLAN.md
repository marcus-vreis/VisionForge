# Panel Parity — Canonical Task-Panel Contract (ADR-059) — Audit & Plan

> Status: **Proposed** — audit done 2026-07-01, implementation not started.
> Origin: user review — "classification serve como modelo para os outros; nos
> outros os hiperparâmetros estão organizados de forma errada, perdem
> pré-processamento, e os modos de treinamento estão embaralhados com os
> hiperparâmetros."
> Verdict after code audit: **the critique is factually correct** (details below).

## Audit (2026-07-01, branch claude/research-grade-review)

| Capacidade | Classif. | Detecção | Regressão | Segment. | Anomalia |
|---|---|---|---|---|---|
| Estratégia como seletor que muda o form | ✅ `BlockSelector` | ❌ cards empilhados | ❌ cards | ❌ cards | ❌ cards |
| Ordem canônica (Modelo → Treinamento → Dataset) | ✅ | ❌ Dataset antes | ❌ | ❌ | ❌ |
| Pré-processamento na GUI | ✅ | n/a (Ultralytics) | ❌ (backend ✅) | ❌ (backend ✅) | ❌ (backend ✅) |
| Augmentação exposta na GUI | ✅ | ✅ (card próprio) | ❌ **silenciosa** | ❌ **silenciosa** | ❌ **silenciosa** |
| Preview de augmentation | ✅ | ❌ | ❌ | ❌ | ❌ |
| Dataset stats | ✅ | ✅ | ❌ | ❌ | ❌ |
| YAML import/export | ✅ | ❌ | ❌ | ❌ | ❌ |

Evidence:
- `PreprocessingConfig` + `TransformConfig` are present in
  `utils/{regression,segmentation,anomaly}_config.py`, but `PreprocessingPanel`
  and the augment fields (`horizontal_flip`/`rotation_degrees`/`color_jitter`)
  render **only** in `ParamPanel.tsx` (grep: 0 occurrences in the three panels).
- `TransformConfig` defaults: `horizontal_flip=True`, `rotation_degrees=10` →
  every regression/segmentation/anomaly GUI run trains with augmentation the
  user never chose, sees, or can disable. For anomaly (PatchCore normal-patch
  memory bank; orientation-sensitive defects) this can change results — this is
  a correctness issue, not just UX.
- Section order: standalone panels render Experimento → Modelo → **Dataset →
  Treinamento**; classification renders Modelo → **Treinamento → Dataset**.
- Strategies: standalone tasks append `ComparisonCard` + `SweepCard` below the
  form (always visible, "modo avançado"), while classification switches the
  form via `BlockSelector`. Replicates (ADR-056) must NOT become a third
  stacked card — it lands inside the strategy selector.

## On "Comparar arquiteturas" (the user is right, with one nuance)

Functionally, a model comparison **is** a 1-axis grid sweep over `model.name`
(the SweepCard even suggests `model.name` as an axis), and every trial already
lands in the run history individually — so "grid + comparar no histórico" works
today. The card duplicates a concept and the surface. **Fold it into the Sweep
mode as a one-click preset** ("arquiteturas": multi-select → fills the
`model.name` axis), keeping the ranked report. Nuance: the backend comparison
runner/endpoints stay — they are the tested engine and the ranked-report shape
is good; only the duplicated GUI entry point goes away. Longer-term (Fase D),
ranking by a single run per arch is statistically weak anyway — comparison
should become "replicated comparison" (N seeds per arch, paired test), which
the strategy selector accommodates naturally.

## Canonical contract (what every panel renders, in order)

1. **Nome do experimento** (+ YAML import/export)
2. **Estratégia** — segmented: `Treino simples | K-fold* | Sweep | Réplicas`
   (*where the task has CV; sweep hosts the "arquiteturas" preset)
3. **Modelo** — arquitetura, pesos, campos específicos da tarefa
4. **Treinamento** — épocas/lr/batch/otimizador/scheduler/seed/deterministic/AMP
5. **Dataset** — fonte + stats da tarefa
6. **Pré-processamento** (filtros; exceto detecção)
7. **Augmentação** + preview (exceto detecção, que mantém o card Ultralytics)
8. Cards auxiliares (download de dataset)

Implementation rule: extract **shared section components** parametrized by
config path (`TrainingSection`, `TransformsSection` = PreprocessingPanel +
augment + preview, `StrategyBar`, `ExperimentHeader`) and compose the five
panels from them — so drift becomes structurally impossible and ADR-058's
generic custom-task panel is these same components pointed at a schema.

## Bricks (ordered by severity; each green on CPU CI + vitest + SPA rebuild)

- [ ] brick A — **stop the silent augmentation**: `TransformsSection`
  (pré-processamento + augment fields + preview) in regression, segmentation
  and anomaly panels; payload builders updated + vitest. Defaults unchanged —
  surfaced, not flipped.
- [ ] brick B — canonical section order in the four standalone panels
  (Modelo → Treinamento → Dataset) + consistent section labels.
- [ ] brick C — `StrategyBar` segmented selector (Simples | Sweep | Réplicas)
  replacing stacked cards; ComparisonCard becomes the Sweep "arquiteturas"
  preset; ReplicatesCard (ADR-056 GUI brick) is born inside it, not as a card.
- [ ] brick D — YAML import/export in the four standalone panels (reuse
  `lib/yaml-config` validation against each task's live schema).
- [ ] brick E — dataset stats parity: folder-based stats for segmentation
  (pares imagem/máscara por split) and anomaly (contagem normal/anômalos por
  split); CSV-manifest summary for regression (linhas, colunas-alvo, faixa).
- [ ] brick F — classification alignment pass: expose "Comparar modelos" as a
  grid preset there too (backend block untouched) so the mental model is one.
