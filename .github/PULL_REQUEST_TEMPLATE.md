<!-- Open the PR against `development`, not `main` (see CONTRIBUTING.md, "Branch model"). -->

## What and why

<!-- One or two sentences. Link the issue: "Closes #N". -->

## Checklist

- [ ] The PR targets `development`
- [ ] One issue per PR; the title follows Conventional Commits
- [ ] Tests cover the change (a bug fix has a test that failed before it)
- [ ] `pytest`, `ruff check`, `ruff format --check` and `mypy src/` pass locally
- [ ] Frontend changes: `npm run lint`, `npx tsc -b`, `npx vitest run` pass and the bundle in `src/visionforge/gui/static` is rebuilt and committed
- [ ] Docs and `CHANGELOG.md` (`## [Unreleased]`) updated when behavior changes
- [ ] A new design decision has an ADR in `docs/dev/DECISIONS.md`
- [ ] No secrets, tokens or personal data in the diff
