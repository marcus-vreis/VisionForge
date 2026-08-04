# Ground-truth verification — 2026-06-06

> **Superseded (2026-07-27).** This snapshot describes a mid-integration state
> that no longer exists: the branches below were merged long ago and both
> follow-ups listed at the end are done (every standalone trainer carries
> `schema_version` and the `run.json` environment block, and the ONNX benchmark
> shape landed). Kept for the audit trail. **Current state:**
>
> - `main` and `development` are in sync; everything below is merged.
> - The full suite runs green on the maintainer's Windows/CUDA machine and in
>   CI (`1323 passed, 2 skipped` as of 2026-07-29, plus `slow` cases deselected
>   by default).
> - End-to-end health is now a command rather than a document:
>   `visionforge selftest` trains all six tasks through the real API across
>   five strategies — **27/27 cases** on CPU, offline (ADR-060). Verified again
>   inside the GPU container (`--quick`, 5/5).
> - Live phase/status tracking lives in `TASKS.md`; decisions in
>   `DECISIONS.md` (ADR-001..073).
> - v0.1.0 is on PyPI as `visionforge-studio` (ADR-067); releases go through
>   `bump-my-version` (ADR-073, `documentation/RELEASING.md`).

Independent state check on `feat/detection-hyperparams-yolo-family`.

**What was verifiable in an isolated env (no GPU, PyTorch index blocked, Python
3.13 unavailable → ran the config/utils layer on 3.10):**

- **Config/utils layer: 205 passed, 0 failed** (`tests/utils/` + `tests/test_example_configs.py`).
  Covers Pydantic validation for all five tasks, `schema_version` forward-migration,
  `DeviceConfig`, `cuda` detection guards, and every example YAML in `configs/`.
  This is the project's correctness backbone and it is fully green.
- The single "failure" seen was `test_environment::test_torch_version_present`,
  which asserts torch resolves — it only fails because torch is absent in this
  sandbox, **not** a project defect.

**Not reproducible here (documented, not a regression):** the ~600 torch/ultralytics
tests (the "814 passed" figure) need a CUDA-capable or CPU-torch env from the
PyTorch index — which this sandbox blocks — plus Python 3.13. These run in CI
(GitHub Actions, ADR-010) and must be the source of truth for that figure.

**Fix applied this pass — deprecated AMP API:**
`core/trainer.py` used `torch.cuda.amp.GradScaler(...)` / `torch.cuda.amp.autocast()`,
which emit a `FutureWarning` since torch 2.4 and are slated for removal. Switched to
the non-deprecated `torch.amp.GradScaler("cuda", ...)` / `torch.amp.autocast("cuda")`
(generic namespace exists since torch 2.3 = the project floor, so it's safe and
behaviour-preserving). Help text in `utils/config.py` updated to match. **Requires
CI confirmation** (the AMP path only executes under CUDA, untestable in this sandbox).

**Noted, not a bug:** regression/segmentation/anomaly trainers don't expose or
honour `mixed_precision` — config and trainer agree (no silent no-op). This is a
**feature gap** (AMP speedup is classification-only), a candidate for the
"use the GPU fully" performance workstream, not a defect.

---

# Integration status — feature branches → `development`

> Generated 2026-06-04 by a tech-leader iteration. Regenerate after any branch changes.
>
> **✅ Executed & verified.** The merge-readiness analysis below was confirmed by
> actually building the local branch **`integ/all-features`** (off `development`,
> not pushed): all 8 branches merged in the recommended order — the only conflicts
> were the predicted docs (`TASKS.md`/`DECISIONS.md`/`.codespellrc`) + the rebuilt
> `gui/static` bundle, **zero source-code conflicts**. The integrated branch is
> **fully green**: backend **814 passed / 2 skipped**, ruff + format + mypy clean;
> frontend tsc + eslint + **54 Vitest** clean, SPA builds; the GUI server boots and
> all five task schema endpoints respond 200. `integ/all-features` is a
> ready-to-review snapshot; the individual branches are untouched for clean PRs.

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
`cd frontend && npx vitest run && npm run typecheck`.

## Follow-ups to reconcile after integration (consistency, not conflicts)

- The standalone task trainers (regression/segmentation/anomaly/detection) should
  adopt the `schema_version` field (ADR-039) and the `run.json` `environment`
  block (ADR-013 update) — both currently only on the classification path.
- The ONNX benchmark frontend type fix (`feat/onnx-benchmark-speedup`) supersedes
  the stale `std_ms`/`n_runs` fields; ensure the merged `client.ts` keeps the new
  `torch_mean_ms`/`speedup`/`p50_ms`/`p95_ms` shape.
