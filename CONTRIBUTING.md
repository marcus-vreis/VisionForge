# Contributing

## Code style

All code is formatted and linted with **ruff**
Type annotations are required on all public functions and validated by **mypy**.
Spelling is checked with **codespell**.

Never run these tools manually — the pre-commit hooks handle everything automatically on every `git commit`.

## Docstrings and comments

- Every public class and function gets a single-line description on the first line.
- `Args:` and `Returns:` only on functions with a non-obvious signature or return value.
- `Raises:` only when the caller needs to handle a specific exception.
- No multi-line `Args:` on Pydantic models — fields document themselves via name and `Field()`.
- Comments explain *why*, not *what*. If the code is clear, skip the comment.
- No unnecessary capitalization. No filler phrases ("this function...", "note that...").
- Keep it short. Docstrings should be clean, concise, and professional, avoiding automated boilerplate.

## Naming conventions

### Python

| Element | Convention | Example |
|---|---|---|
| Modules | `snake_case` | `model_factory.py` |
| Classes | `PascalCase` | `ExperimentBlock` |
| Functions / methods | `snake_case` | `load_config()` |
| Constants | `UPPER_SNAKE_CASE` | `LOG_FORMAT_FILE` |
| Private attributes | `_leading_underscore` | `_DEFAULT_LOG_DIR` |
| Type aliases | `PascalCase` | `ConfigDict` |

### Files and directories

| Element | Convention | Example |
|---|---|---|
| Source modules | `snake_case.py` | `trainer.py` |
| Test files | `test_<module>.py` | `test_trainer.py` |
| Config files | `<experiment_name>.yaml` | `resnet50_baseline.yaml` |
| Output dirs | `<experiment_name>/<timestamp>/` | `resnet50_baseline/20260314_142300/` |

## Commit conventions

Commits follow the [Conventional Commits](https://www.conventionalcommits.org/) standard.

```
<type>: <short description in imperative, lowercase, English>
```

| Type | When to use |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `test` | Adding or updating tests |
| `chore` | Config, tooling, dependencies |
| `ci` | GitHub Actions changes |
| `docs` | Documentation only |
| `refactor` | Code change with no behavior change |
| `style` | Formatting, no logic change |

**Examples:**
```
feat: add CrossValidationBlock with stratified k-fold support
fix: apply review suggestions to config.py
test: add unit tests for config manager
chore: migrate from black+flake8 to ruff
ci: add SonarCloud scan on pull requests
docs: update ARCHITECTURE with task expansion model
```

## Branch conventions

```
phase/<number>-<description>    # phase work
feat/<description>              # new feature
fix/<description>               # bug fix
chore/<description>             # tooling / config
```

## Pull Requests

- One PR per issue.
- PR title follows the same Conventional Commits format.
- Merge only when CI is green (ruff, mypy, codespell, pytest all passing).
- Branches are deleted automatically after merging (configured in GitHub settings).
- Version bump is checked on every PR — run `bump-my-version bump patch` if needed.

## Documentation

When a decision is made (framework, library, pattern), document the decision — not a comparison of alternatives. If the decision changes, update the doc to reflect the new reality. Documentation and code must always agree.

Docs and comments describe the **current state and its rationale**, never the edit history. Write "X uses Y because Z", not "changed X from W to Y" or "swapped W for Y". Change history lives in git and the commit message; a doc that narrates its own diffs rots fast and adds no signal to the reader.

## Pre-commit hooks

Install once after cloning:

```bash
pre-commit install
```

Hooks that run on every `git commit`:

| Hook | Action |
|---|---|
| ruff | Lints and auto-fixes code |
| ruff-format | Auto-formats code |
| mypy | Fails on type errors |
| pytest | Fails if any test breaks |

`pytest` here runs the fast suite: cases marked `slow` (real trainings behind
a live server, ADR-060) are deselected by `addopts` so committing stays quick.
Before opening a PR that touches a trainer, a block, the API or the SSE
contract, also run the end-to-end matrix:

```bash
visionforge selftest          # every task x strategy, real API, synthetic data
pytest -m slow                # the harness's own live cases
```

Anything that changes what the browser receives — report shapes, event names,
endpoints — should be reflected in `utils/selftest.py`'s case table, so the
next run catches a regression the mocked route tests cannot.

## Adding a new ExperimentBlock

1. Create `src/visionforge/blocks/<your_block>.py`
2. Implement `ExperimentBlock` ABC (`setup`, `run`, `report`)
3. Add tests in `tests/blocks/test_<your_block>.py`
4. The `BlockRegistry` and GUI pick it up automatically — no other changes needed

## Adding a new task (Detection, Segmentation, etc.)

Every task after classification is **standalone** (ADR-033/036/037/038): it adds
new modules and never edits an existing task's code or folds into
`ExperimentConfig`. The brick sequence:

1. `<Task>Config` Pydantic tree in `src/visionforge/utils/<task>_config.py` (reuse
   `OutputConfig`/`DeviceConfig`).
2. `<Task>DataModule` in `src/visionforge/core/<task>_data.py`.
3. `<Task>ModelFactory` in `src/visionforge/models/<task>_factory.py`.
4. `<Task>Trainer` in `src/visionforge/core/<task>_trainer.py` (reuse
   `resolve_device`/`_seed_everything`; write the ADR-013 `run.json`; emit the
   `start`/`epoch_end`/`end` SSE events).
5. `<Task>Block` (`setup`/`run`/`report`) in `src/visionforge/blocks/<task>.py`,
   plus a new ADR for the task.
6. `/api/<task>/{schema,run}` in `src/visionforge/gui/api/routes.py` (reuse the
   shared single-run state + `/experiment/{status,events,result}`).
7. Frontend `<Task>Panel` + `lib/<task>-models.ts`, wired into `App.tsx` /
   `useExperiment` / `client.ts`.
8. Rebuild the SPA: `cd frontend && npm run build`.

## Frontend development

The frontend lives in `frontend/` and is built with React + TypeScript + Vite + shadcn/ui.

**Prerequisites:** Node.js >= 18 (dev-time only — users never need Node.js).

**Development workflow:**
1. Start the API server: `python -m visionforge gui`
2. In another terminal: `cd frontend && npm run dev` (Vite dev server with hot-reload)
3. Vite proxies `/api` requests to the FastAPI server automatically

**Building for production:**
```bash
cd frontend && npm run build
```
This outputs to `src/visionforge/gui/static/`, which FastAPI serves as static files.

**Adding shadcn/ui components:**
```bash
cd frontend && npx shadcn@latest add <component-name>
```
