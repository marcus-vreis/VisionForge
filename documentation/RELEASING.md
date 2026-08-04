# Releasing

One command. The version is never edited by hand.

```bash
bump-my-version bump patch     # 0.1.0 -> 0.1.1   bug fixes
bump-my-version bump minor     # 0.1.0 -> 0.2.0   new features
bump-my-version bump major     # 0.1.0 -> 1.0.0   breaking changes
```

That single command rewrites every place a version literal lives, adds the
changelog heading, commits, and tags — all atomically. Then:

```bash
git push && git push --tags
```

The `v*.*.*` tag triggers `.github/workflows/cd.yml`, which builds the wheel
and sdist and publishes to PyPI via Trusted Publishing (OIDC — no token in the
repository).

## Why not `git tag` by hand

Because the tag and the packaged version are set in two different places, and
nothing makes them agree. Tagging `v0.2.0` while `pyproject.toml` still says
`0.1.0` builds `visionforge_studio-0.1.0` and either publishes it under a
`v0.2.0` release or dies inside the upload when PyPI rejects a version that
already exists.

CD now fails before publishing when those two disagree, and names both values —
but the fix is to not create the situation: `bump-my-version` is the one action.

## Where the version actually lives

Exactly two files hold a literal, and `bump-my-version` rewrites both:

| File | Why it cannot be derived |
|---|---|
| `pyproject.toml` | it *is* the package metadata |
| `CITATION.cff` | a data file — it cannot import anything |

Everything else reads it at runtime. `visionforge.__version__` comes from
`importlib.metadata`, and the CLI (`visionforge --version`), the FastAPI app,
the `/api/system/info` payload and the GUI header all read that one value. The
Docker image stamps its `org.opencontainers.image.version` label from a
`VF_VERSION` build arg passed from the package metadata, so the image cannot
disagree either.

## Before tagging

```bash
pytest && ruff check src/ tests/ && mypy src/
cd frontend && npx vitest run && npm run typecheck && npm run build && cd ..
visionforge selftest
```

The SPA build matters: the wheel ships `gui/static`, so a stale build would
publish an old interface with new code. Confirm the wheel carries it:

```bash
uv build
python -c "import zipfile,glob; z=zipfile.ZipFile(glob.glob('dist/*.whl')[0]); print(sum('gui/static' in n for n in z.namelist()), 'static files')"
```

## After publishing

```bash
pip install "visionforge-studio[cpu]" --force-reinstall
visionforge --version
visionforge selftest --quick
```

Install the published artifact in a throwaway environment and run it. The
package index is the only place a packaging mistake shows up, and it shows up
to users first otherwise.
