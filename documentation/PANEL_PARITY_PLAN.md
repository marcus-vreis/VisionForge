# Panel Parity — Canonical Task-Panel Contract (ADR-059) — Audit & Plan

> Status: **In progress** — audit 2026-07-01; bricks A + B shipped 2026-07-02
> (browser-verified: canonical order + TransformsSection on the four panels).
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

- [x] brick A — **stop the silent augmentation**: shared `TransformsSection`
  (PreprocessingPanel + augment fields + AugmentPreview) in the regression,
  segmentation and anomaly panels; `lib/transforms-form.ts` mirrors
  `TransformConfig` (defaults unchanged — surfaced, not flipped); payload
  builders send `data.transforms` + `data.preprocessing` (validated against
  the Pydantic configs); `_pick_preview_image` gained flat-split and bounded
  recursive fallbacks so the previews work on CSV-manifest/MVTec/paired-mask
  layouts (tests in `tests/gui/test_routes_augment_preview.py`); vitest 81/81.
- [x] brick B — canonical section order in the four standalone panels
  (Modelo → Treinamento → Dataset; detection keeps its Ultralytics training
  cards between Treinamento and Dataset, augmentation card after Dataset).
  Browser-verified section sequences on all four tabs.
- [x] brick C — `StrategyBar` segmented selector (Simples | Sweep | Réplicas)
  right after Experimento in the four standalone panels (mirrors
  classification's BlockSelector position); stacked cards removed.
  `ComparisonCard` deleted — comparing architectures is now the Sweep
  "arquiteturas" preset (chip multi-select → upserts a `model.name` axis,
  valid in grid/random/optuna). `ReplicatesCard` (ADR-056 GUI) lives inside
  the selector: auto (n a partir do training.seed) ou seeds explícitas
  (validação espelha os 422 do backend), métrica destaque, launch →
  `POST /api/{task}/replicates`. `ReplicatesReport` renders the citable
  headline (mean ± IC 95%, n), per-metric aggregate table and per-seed table
  (detector shape-specific, checked before comparison/sweep). Overlay queue
  banner knows `replicates`. Browser-verified on regression + detection;
  vitest 85/85 (`lib/replicates-form.test.ts`). Shipped 2026-07-02.
- [x] brick D — **canonical experiment header + YAML parity** (shipped
  2026-07-02). The contract, made precise after user review: every task panel
  opens with the same `ExperimentHeader` card — row 1: experiment name +
  `↓ Exportar YAML` / `↑ Importar YAML` side by side; row 2 (same box, divided):
  the strategy selector. This is byte-for-byte the classification layout; the
  short-lived separate `StrategyBar` card was absorbed and deleted.
  - Export: `exportConfigToYaml(buildXPayload(form))` — the exact wire payload,
    so an exported file re-runs identically from the CLI too.
  - Import: parse (safe loader) → validate against the task's **live schema**
    (`GET /api/{task}/schema`, up to 5 issues shown, same UX as classification)
    → `xFormFromPayload(data)` rebuilds the form. The reverse converters are
    driven by a generic `mergeFormShape` (defaults define shape; mistyped
    leaves keep defaults — a malformed YAML can never corrupt the form) plus
    per-task adapters (target_columns list↔string, transforms arrays↔strings,
    schema-flat preprocessing steps, transfer_learning→mode,
    detection source derivation + auto_augment null↔"none" + backend/model
    coherence). **Round-trip is tested per task**: `formFromPayload(
    buildPayload(form)) == form` (vitest 90/90).
  - Browser-verified on Anomalia: header renders, real YAML import applies
    name/PatchCore/threshold/preprocessing to the form with a success note,
    malformed YAML shows the error banner, 0 console errors.
- [x] brick E — dataset stats parity. **Backend shipped 2026-07-02**:
  `POST /api/{segmentation,anomaly,regression}/dataset/stats`
  (mirrors the detection stats endpoint) —
  segmentation: pares imagem↔máscara por stem, unpaired counts, class ids
  amostrados de ≤20 máscaras de treino (sugere `num_classes` e expõe
  `ignore_index`); anomaly: contagem normal no treino + normal/por-defeito no
  teste, aviso de treino vazio; regression: linhas por CSV (cap 50k), colunas
  ausentes, distribuição dos alvos (n/média/min/max) e checagem amostrada de
  500 caminhos de imagem. Tests em `tests/gui/test_task_dataset_stats.py` (8).
  **GUI shipped 2026-07-02** (brick E complete): `TaskDatasetStats.tsx` (três
  componentes com primitivas compartilhadas + fetch com debounce de 400ms)
  dentro da seção Dataset de cada painel — segmentação: cards de pareamento
  por split + chips de ids (255 marcado como provável ignore_index) + botão
  "🎯 aplicar N classes" + **guarda de máscaras interpoladas** (>32 ids
  distintos → aviso de anti-aliasing em vez de sugerir classes; chips
  limitados a 12+N); anomalia: treino-normal / teste-normal / teste-anômalo +
  chips por defeito; regressão: linhas, colunas ausentes, imagens faltantes e
  μ/[min,max]/n por alvo. Browser-verified com datasets sintéticos (pareamento
  2 pares + aviso de img sem máscara, aplicar → num_classes=3, caminho
  interpolado → aviso, anomalia 3/2/4 + chip por defeito), 0 console errors.
- [ ] brick F — classification alignment pass: expose "Comparar modelos" as a
  grid preset there too (backend block untouched) so the mental model is one.
