# Contributing

Everyone taking part in the project follows the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security vulnerabilities privately
as described in [SECURITY.md](SECURITY.md), never in a public issue.

## Getting set up

```bash
git clone https://github.com/marcus-vreis/VisionForge.git
cd VisionForge
uv venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
```

PyTorch is deliberately **not** a plain dependency: its build has to match the
hardware, and a resolver cannot choose between the CPU and CUDA wheels. Pick
one through a hardware extra — the right index is already wired up for it:

```bash
uv pip install -e ".[cu128,dev]"    # cu118 / cu121 / cu124 / cu126 / cu128 / cpu
```

Not sure which? `visionforge doctor` reads the driver *and* any torch already
installed, and prints the exact line for the machine.

The web UI is served by the Python backend, so it has to be built once (end
users never need Node — the published wheel ships it built):

```bash
cd frontend && npm install && npm run build && cd ..
```

Then confirm the checkout actually trains, on synthetic data, offline:

```bash
visionforge selftest --quick
```

**While working:** rebuild after changing anything under `frontend/`, and
**restart `visionforge gui` after changing anything under `src/`**. Python
imports its modules once, at start, so a running server keeps serving the old
backend no matter how many times you rebuild or reload the page. The GUI shows
a warning when it detects this.

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

## Branch model

`main` holds what was released; `development` is the integration branch for the
next release. Both are protected: nothing is pushed to them directly, every change
arrives through a pull request.

- Branch from `development` and open the PR against `development`.
- Releases are promoted from `development` to `main` by a pull request, then
  tagged (see [`docs/dev/RELEASING.md`](docs/dev/RELEASING.md)).
- Settings of the protection (required checks, bypass) are documented in
  `docs/dev/RELEASING.md`.

## Branch conventions

```
phase/<number>-<description>    # phase work
feat/<description>              # new feature
fix/<description>               # bug fix
chore/<description>             # tooling / config
```

## Pull Requests

- One PR per issue, against `development`.
- PR title follows the same Conventional Commits format.
- Merge only when CI is green (ruff, mypy, codespell, pytest all passing).
- Delete your branch after merge; GitHub does not do it for you.
- CI warns when the version was not bumped, but the warning does not block the
  PR. Releases are cut with `bump-my-version` (see `docs/dev/RELEASING.md`).

## Documentation

When a decision is made (framework, library, pattern), document the decision — not a comparison of alternatives. If the decision changes, update the doc to reflect the new reality. Documentation and code must always agree.

Decisions are recorded as ADRs in `docs/dev/DECISIONS.md`. The rules at the top of that file say when an ADR is required and which status values are valid; copy `docs/dev/ADR-TEMPLATE.md` for a new one and append it at the end of the file.

Docs and comments describe the **current state and its rationale**, never the edit history. Write "X uses Y because Z", not "changed X from W to Y" or "swapped W for Y". Change history lives in git and the commit message; a doc that narrates its own diffs rots fast and adds no signal to the reader.

## Pre-commit hooks

Install once after cloning, and again in every new clone or `git worktree`
(hooks live in each checkout's `.git`, not in the repository):

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Plain `pre-commit install` sets up only the commit stage, so the `pytest` hook
would never run.

| Hook | Stage | Action |
|---|---|---|
| ruff | `git commit` | Lints and auto-fixes code |
| ruff-format | `git commit` | Auto-formats code |
| codespell | `git commit` | Fixes spelling mistakes in place |
| mypy | `git commit` | Fails on type errors |
| pytest | `git push` | Fails if any test breaks |

`pytest` runs the fast suite, 5 to 8 minutes: cases marked `slow` (real
trainings behind a live server, ADR-060) are deselected by `addopts`. It runs on
push, not on commit, so a commit takes seconds (ADR-186). To run it by hand:
`pre-commit run pytest --hook-stage pre-push --all-files`.
The coverage gate is `--cov-fail-under=93` in the `tests` job of
`.github/workflows/ci.yml`, the only place it is set (measured 95% on the full
fast suite). It lives there and not in `.coveragerc` on purpose: `pytest` always
runs with `--cov`, so a gate in the config would fail every partial run
(`pytest tests/x`).
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
4. `BlockRegistry` discovers the subclass by itself, but the GUI does not: it
   dispatches on `config.block`. Add the new name to the `block` `Literal` in
   `ExperimentConfig` (`src/visionforge/utils/config.py`) and to `_block_types()`
   in `src/visionforge/gui/api/execution.py` (the Colab runner has the same table in
   `colab_remote_runner.py`). Without that step the GUI runs a classification.
5. If the block needs a form in the GUI, add it under `frontend/` and rebuild the SPA.

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
6. `/api/<task>/{schema,run}` in `src/visionforge/gui/api/training_routes.py`
   (`routes.py` only assembles the domain routers now; reuse the
   shared single-run state + `/experiment/{status,events,result}`).
7. Frontend `<Task>Panel` + `lib/<task>-models.ts`, wired into `App.tsx` /
   `useExperiment` / `client.ts`.
8. Rebuild the SPA: `cd frontend && npm run build`.

## Frontend development

The frontend lives in `frontend/` and is built with React + TypeScript + Vite + shadcn/ui.

**Prerequisites:** Node.js >= 22.12 (or 20.19, the range Vite 8 requires; dev-time only — users never need Node.js).

**Development workflow:**
1. Start the API server without opening a browser tab: `python -m visionforge gui --no-browser`
2. In another terminal: `cd frontend && npm run dev` (Vite dev server with hot-reload)
3. Vite proxies `/api` requests to `http://localhost:8000`; if the server runs elsewhere, set `VF_API_URL` (for example `VF_API_URL=http://127.0.0.1:9000 npm run dev`). See `frontend/README.md`.

**Building for production:**
```bash
cd frontend && npm run build
```
This outputs to `src/visionforge/gui/static/`, which FastAPI serves as static files.

**Adding shadcn/ui components:**
```bash
cd frontend && npx shadcn@latest add <component-name>
```
