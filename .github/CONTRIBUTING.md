# Contributing to VisionForge

The full guide — dev setup, the test/lint gauntlet, and the PR flow — lives at
[`documentation/CONTRIBUTING.md`](../documentation/CONTRIBUTING.md).

This file exists so GitHub finds it: the "Contribute" links on the issue and
pull-request pages only look at the repository root, `.github/`, or `docs/`.

The short version, before opening a PR:

```bash
pytest                                   # backend
ruff check src/ tests/ && mypy src/      # lint + types
cd frontend && npx vitest run && npx tsc --noEmit
visionforge selftest                     # every task through the real API
```

Behaviour changes get an ADR in
[`documentation/DECISIONS.md`](../documentation/DECISIONS.md) — the reasoning
matters as much as the diff.
