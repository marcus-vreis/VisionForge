# VisionForge — project contract (single source of truth)

Auto-loaded each session. This is the **one** canonical `CLAUDE.md`. Deep
narrative lives in the docs linked below; this file is the operating contract the
agent team and Cowork both follow. When any instruction says "CLAUDE.md §N", it
means a section of **this** file.

## What it is
Local-first computer-vision training/experimentation platform (PyTorch + FastAPI +
React). **Local-only by design** — runs on the researcher's own GPU via CUDA, no
cloud. Five tasks today: **classification, detection, regression, segmentation,
anomaly**. Classification is the original `ExperimentConfig`/`ExperimentBlock`
path; the other four are standalone tasks (own config/trainer/block/route).

## Canonical docs (read for depth)
- `documentation/ARCHITECTURE.md` — layers, modules, boundaries
- `documentation/DECISIONS.md` — ADR-001..073 (every decision + reason)
- `documentation/INTEGRATION_STATUS.md` — branch/merge state + verified test state
- `documentation/PROJECT_CONTEXT.md` — long-form project narrative (historical depth)

## Architecture rules (don't violate)
- Layer imports are one-directional: `utils/` ← `core/` ← `blocks/` ← `gui/`.
  `utils/` imports nothing internal; `outputs/` is write-only, never imported.
- **Task-standalone pattern** (ADR-033/036/037/038): each non-classification task
  has its own `*_config.py` + `*_trainer.py` + `/api/<task>/*` route. Adding a task
  = new modules, never edit existing ones. Don't fold tasks into `ExperimentConfig`.
- Config is validated by Pydantic v2 at load time (fail before GPU time, ADR-002).
- Every run writes the `run.json` contract (ADR-013) incl. `environment` + `schema_version`.
- Decisions get an ADR in `DECISIONS.md` (ADR-012). Don't change behavior silently.

## §9 Code-writing rules (enforced by backend-dev + reviewer)
- Every public class/function gets a single-line description.
- `Args:`/`Returns:` only when the signature or return is non-obvious; `Raises:`
  only when the caller must handle a specific exception.
- Pydantic fields self-document via name + `Field()` — no `Args:` block on models.
- Comments explain *why*, not *what*; skip the comment if the code is clear.
- No filler, no AI-flavored boilerplate prose; clean, concise, professional.
- Document decisions, not options. Docs and code must always agree.
- Prefer modern, clean APIs (e.g. `torch.amp` over deprecated `torch.cuda.amp`).

## Commands
- Backend tests: `pytest` (CI gates at `--cov-fail-under=70`)
- Lint/format/types: `ruff check src/ tests/` · `ruff format --check` · `mypy src/`
- Frontend: `cd frontend && npx vitest run && npm run typecheck`
- Build SPA (served by FastAPI): `cd frontend && npm run build`
- Run GUI: `visionforge gui` · Run CLI: `visionforge run configs/<task>.yaml`
- **Restart `visionforge gui` after any change under `src/`.** Static files are
  re-read from disk per request, so a rebuilt SPA reaches the browser at once —
  but Python modules are imported once, at process start. A server left running
  across a change serves new JavaScript from old Python, and the feature
  silently does nothing. The GUI now detects this and says so, but the fix is
  always to restart the server, never to rebuild or hard-reload.

## Environment reality
- torch/torchvision are **user-managed** per hardware (ADR-005) — not pinned deps.
- The Cowork Linux sandbox **cannot run the full torch/ultralytics suite**: the
  PyTorch wheel index and Python 3.13 downloads are blocked by network policy.
  Only the config/utils layer runs here (~205 tests, green). **CI is the source of
  truth** for the full figure — currently `1323 passed, 2 skipped` (plus `slow`
  cases deselected by default).
- Published on PyPI as **`visionforge-studio`** (ADR-067); the import name stays
  `visionforge`. `pip install visionforge-studio[cpu]`, then `visionforge doctor`.
- A Docker image exists for both variants (ADR-071/072): GPU defaults to **cu128**,
  which is the floor for RTX 50-series (sm_120). See `documentation/DOCKER_PLAN.md`.
- Releases are one command: `bump-my-version bump <part>` (ADR-073, see
  `documentation/RELEASING.md`). Never tag by hand.

## Cowork operating rules
- **Never run git write ops through the mount** (`git add`/`commit`/`--renormalize`):
  the Windows→Linux mount denies the unlink git needs and **corrupts `.git/index`**.
  Cowork does file edits (Read/Write/Edit work fine); the **user runs git on Windows**.
  Recovery: `rm -f .git/index && git reset` (working tree is untouched).
- Division of labor: the `.claude/` **agent team** (lead → backend-dev/frontend-dev
  in worktrees, reviewer) runs the autonomous `/loop` implementation. **Cowork owns
  judgment/interaction work**: triage, architecture, ADRs/docs, review, writing,
  one-off fixes — and turning vague ideas into crisp specs/tasks for the team.
- Branches: agents PR into `development`, merge-when-green; **`main` is promoted
  manually** — agents/Cowork never commit to `main`.

## Style
- Portuguese for prose/docs with the user; English fine in code/comments/ADRs.
- **Chat replies use "caveman" prose to save tokens**: telegraphic, no articles or
  filler, no preamble/recap, no restating the question. Say the finding, the
  action, the number. This applies **only to conversation with the user** —
  code, comments, docstrings, commit messages, ADRs and docs stay full, correct
  prose per §9. Never drop a fact to be terse: shorten the wording, not the
  content, and keep exact numbers, file paths and caveats.
