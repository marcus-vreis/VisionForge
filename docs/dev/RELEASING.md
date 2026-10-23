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

The `distribution` CI job builds a wheel and sdist, installs the wheel in a clean
CPU environment and runs `scripts/verify_distribution.py` outside the checkout
with PYTHONPATH unset. It verifies installed imports/static assets, a synthetic
training and the evidence CLI. This complements editable-install tests; it does
not publish packages or replace checking the final published artifact.

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

## CI action runtimes

ADR-139 replaces actions declaring Node 20 with releases declaring Node 24.
Versions were checked against upstream releases and `action.yml` on 2026-10-21:

| Action | Version |
|---|---|
| [checkout](https://github.com/actions/checkout/releases/tag/v7.0.1) | `v7.0.1` |
| [setup-python](https://github.com/actions/setup-python/releases/tag/v7.0.0) | `v7.0.0` |
| [setup-node](https://github.com/actions/setup-node/releases/tag/v7.1.0) | `v7.1.0` |
| [cache](https://github.com/actions/cache/releases/tag/v6.1.0) | `v6.1.0` |
| [upload-artifact](https://github.com/actions/upload-artifact/releases/tag/v7.0.2) | `v7.0.2` |
| [download-artifact](https://github.com/actions/download-artifact/releases/tag/v8.0.2) | `v8.0.2` |
| [setup-uv](https://github.com/astral-sh/setup-uv/releases/tag/v10.3.1) | `v10.3.1` |
| [action-gh-release](https://github.com/softprops/action-gh-release/releases/tag/v3.0.3) | `v3.0.3` |
| [sonarqube-scan-action](https://github.com/SonarSource/sonarqube-scan-action/releases/tag/v8.3.0) | `v8.3.0` |

The pre-commit composite action was replaced with explicit install/cache/run
steps because its internal cache action still declared Node 20. Hosted runners
meet the newer runner requirements. Artifact uploads remain ZIPs; download
digest mismatches now fail. Sonar's scanner signature verification stays enabled
with its new default. The Docker-based publishing and paper actions have no
Node runtime to migrate. Workflow permissions were not expanded.

## SonarCloud integration branch

The `marcus-vreis_VisionForge` project uses `development` as its SonarCloud main
branch (ADR-144). The Free plan limits PR analysis to that target. The GitHub
release branch remains `main`; changing the Sonar label does not publish code.

The scanner analyzes PRs and pushes to `development`. A push analysis checks the
base branch's own checkout and coverage after a merge. Renaming an old Sonar
branch preserves its old analysis; it does not create a current baseline.
Check the analyzed revision and the quality gate, not just a historical Passed badge.

CI authenticates with the repository Actions secret `SONAR_TOKEN`. Browser login
does not refresh that secret. Replace it directly in GitHub when rotating a token;
never paste credentials into issues, logs or PRs. Follow the
[token lifecycle documentation](https://docs.sonarsource.com/sonarqube-cloud/managing-your-account/managing-tokens).

The codespell allowlist includes `afterall` to preserve Playwright's `test.afterAll`
API. Review hook rewrites before pushing and rerun affected checks if code changed.

### Analyzed version and New Code reference

The Sonar job derives `sonar.projectVersion` from `project.version` in the
analyzed checkout's `pyproject.toml` (ADR-147). It does not import an installed
VisionForge package, which may have a different version, or maintain a second
version in `sonar-project.properties`. A package version bump needs no additional
Sonar edit. Local scanner invocations must pass the same property explicitly;
`python scripts/sonar_project_version.py` prints `version=<checkout version>`.

Before correcting the old `0.0.1` label on 2026-10-23, Administration > New code
showed **Previous version**. The overview showed 330 overall findings on the
older baseline. Correcting the label to the actual source version can move the
New Code reference. Compare the analyzed SCM revision, overall findings and
New Code reference before and after the scan; a changed New Code gate alone is
not evidence of debt removal. Do not reset dates, change the definition or
lower gate thresholds to make the transition pass. See the official
[version parameter](https://docs.sonarsource.com/sonarqube-cloud/analyzing-source-code/analysis-parameters/parameters-not-settable-in-ui)
and [New Code definition](https://docs.sonarsource.com/sonarqube-cloud/managing-your-projects/project-analysis/configuring-new-code-calculation).
