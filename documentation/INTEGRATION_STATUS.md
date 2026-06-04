# Integration status — feature branches → `development`

> Generated 2026-06-04 by a tech-leader iteration. Non-destructive analysis
> (`git merge-tree`, no merges performed). Regenerate after any branch changes.

A long autonomous build session produced several independent branches off
`development` (HEAD `1c35fb1`). None are merged yet. **All share the same
merge-base = `development` HEAD**, so each one merges into `development`
**individually with zero conflicts** — the only conflicts are *between* branches
when merged sequentially, and they are limited to docs + the built SPA bundle.

## Branch landscape

| Branch | Carries | Δ commits |
|---|---|---|
| `feat/anomaly` | **Task sweep**: Phase 6 regression + Phase 8 segmentation + Phase 9 anomaly (supersedes `feat/segmentation`, which is a subset) | +21 |
| `feat/gradcam` | Grad-CAM explainability (core + `/api/runs/{id}/gradcam` + GUI) **and** dataset augmentation preview | +4 |
| `feat/config-schema-version` | `schema_version` + config forward-migration hook (ADR-039) | +1 |
| `feat/classification-e2e-test` | Opt-in real-training integration smoke test | +1 |
| `feat/onnx-benchmark-speedup` | PyTorch-vs-ONNX latency speedup in `ExportONNXBlock` | +1 |
| `feat/run-environment-capture` | `environment` (lib versions) block in `run.json` (ADR-013 update) | +1 |
| `docs/readme-accuracy` | README rewritten to cover detection + blocks + post-run actions | +1 |

`feat/segmentation` (+14) is a strict subset of `feat/anomaly` — **merge
`feat/anomaly` and skip `feat/segmentation`**.

## Conflict analysis

**No source-code (`.py`/`.ts`/`.tsx`) content conflicts exist between any pair.**
Files like `routes.py`, `schemas.py`, `client.ts`, `RunDetailPanel.tsx`,
`App.tsx`, `useExperiment.ts` **auto-merge cleanly** everywhere (each branch
touched different regions). The only real conflicts:

- **`documentation/TASKS.md`** — most off-`development` branches appended to the
  same backlog region. Trivial textual conflict: keep all sections.
- **`documentation/DECISIONS.md`** — `feat/anomaly` (ADR-036/037/038) vs
  `feat/config-schema-version` (ADR-039) added ADRs at the same tail. Trivial:
  keep both; ADR numbers don't collide.
- **`.codespellrc`** — `feat/anomaly` and `feat/gradcam` both extended the
  `ignore-words-list`. Trivial: union the words.
- **`src/visionforge/gui/static/**`** — every frontend branch rebuilt the SPA
  bundle (hashed filenames + `index.html`). **Do not hand-resolve** — after
  merging all frontend-touching branches, run `cd frontend && npm run build`
  once and commit the result.

## Recommended merge order

1. **`feat/anomaly`** — largest, brings the three new tasks; clean into `development`.
2. **`feat/config-schema-version`**, **`feat/run-environment-capture`** — core/infra.
3. **`feat/gradcam`**, **`feat/onnx-benchmark-speedup`** — GUI/post-run features.
4. **`feat/classification-e2e-test`** — test-only.
5. **`docs/readme-accuracy`** — last, so it can describe the merged whole.

At each step the only conflicts are `TASKS.md`/`DECISIONS.md`/`.codespellrc`
(accept both sides) and the `gui/static` bundle (rebuild once at the end). After
all merges: `npm run build`, then `pytest -q && ruff check && mypy src` and
`cd frontend && npx vitest run && npx tsc --noEmit`.

## Follow-ups to reconcile after integration (consistency, not conflicts)

- The standalone task trainers (regression/segmentation/anomaly/detection) should
  adopt the `schema_version` field (ADR-039) and the `run.json` `environment`
  block (ADR-013 update) — both currently only on the classification path.
- The ONNX benchmark frontend type fix (`feat/onnx-benchmark-speedup`) supersedes
  the stale `std_ms`/`n_runs` fields; ensure the merged `client.ts` keeps the new
  `torch_mean_ms`/`speedup`/`p50_ms`/`p95_ms` shape.
