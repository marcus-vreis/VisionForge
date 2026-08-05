# Dataset no histórico — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mostrar em qual dataset cada run foi treinado — selo no card do histórico, bloco no detalhe do run, e veredito "mesmos dados?" no comparador.

**Architecture:** Nada de novo é medido. `config.data.base_dir` (28/28 runs) e `dataset_fingerprint` (ADR-061, a partir de 2026-07-26) já estão no `run.json`; o trabalho é derivar uma identidade a partir deles em `core/`, expô-la em dois schemas da API, e renderizá-la em três lugares do frontend. Nenhum endpoint novo: o comparador já busca `fetchRunDetail(id)` por run.

**Tech Stack:** Python 3.13 · Pydantic v2 · FastAPI · pytest · React 18 + TypeScript · vitest

**Spec:** `docs/superpowers/specs/2026-08-05-dataset-no-historico-design.md` — fonte da verdade. Leia antes de começar.

---

## Estrutura de arquivos

| Arquivo | Responsabilidade | Ação |
|---|---|---|
| `src/visionforge/core/dataset_fingerprint.py` | `dataset_identity()` — deriva nome+caminho de um `run.json` | Modificar |
| `src/visionforge/gui/api/schemas.py` | `DatasetInfo`; campos novos em `RunSummary` e `RunDetail` | Modificar |
| `src/visionforge/gui/api/routes.py` | Preencher os campos em `_parse_run_summary` e no handler `get_run_detail` | Modificar |
| `frontend/src/types/run.ts` | Campos novos em `RunSummary`; tipo `DatasetInfo` | Modificar |
| `frontend/src/api/client.ts` | Campo `dataset` em `RunDetail` | Modificar |
| `frontend/src/lib/dataset-identity.ts` | Veredito de comparação (espelha `same_dataset`) + formatação | Criar |
| `frontend/src/components/HistoryOverlay.tsx` | Selo `🗂 <nome>` na fileira de pills | Modificar |
| `frontend/src/components/RunDetailPanel.tsx` | Bloco "Dataset" | Modificar |
| `frontend/src/components/CompareRunsPanel.tsx` | Linha de veredito | Modificar |

A lógica de decisão fica em `core/` (Python) e `lib/` (TS) — os componentes só renderizam. É o padrão que `metric-ci.ts` e `queue-format.ts` já seguem.

---

## Task 1: `dataset_identity()` no core

**Files:**
- Modify: `src/visionforge/core/dataset_fingerprint.py` (adicionar ao fim, antes do `__all__`)
- Test: `tests/core/test_dataset_fingerprint.py`

- [ ] **Step 1: Write the failing tests**

Adicione ao fim de `tests/core/test_dataset_fingerprint.py`:

```python
class TestDatasetIdentity:
    """The name has to survive runs written before the fingerprint existed."""

    def test_prefers_the_fingerprint_root(self) -> None:
        run = {
            "dataset_fingerprint": {"root": "/data/USK-COFFEE", "digest": "abc"},
            "config": {"data": {"base_dir": "/outro/caminho"}},
        }

        assert dataset_identity(run) == ("USK-COFFEE", "/data/USK-COFFEE")

    def test_falls_back_to_config_base_dir(self) -> None:
        """26 of 28 existing runs have only this."""
        run = {"config": {"data": {"base_dir": "/data/USK-COFFEE"}}}

        assert dataset_identity(run) == ("USK-COFFEE", "/data/USK-COFFEE")

    def test_relative_path_gives_the_same_name(self) -> None:
        run = {"config": {"data": {"base_dir": "datasets/USK-COFFEE"}}}

        name, _ = dataset_identity(run)
        assert name == "USK-COFFEE"

    def test_windows_path_gives_the_same_name(self) -> None:
        run = {"config": {"data": {"base_dir": r"C:\Users\x\datasets\USK-COFFEE"}}}

        name, _ = dataset_identity(run)
        assert name == "USK-COFFEE"

    def test_trailing_separator_does_not_produce_an_empty_name(self) -> None:
        run = {"config": {"data": {"base_dir": "datasets/USK-COFFEE/"}}}

        name, _ = dataset_identity(run)
        assert name == "USK-COFFEE"

    def test_no_path_anywhere_is_none(self) -> None:
        """None means "no badge", never a badge with an empty label."""
        assert dataset_identity({"config": {}}) == (None, None)

    def test_unavailable_fingerprint_falls_back(self) -> None:
        """`method: unavailable` carries an empty root; base_dir still answers."""
        run = {
            "dataset_fingerprint": {"root": "", "method": "unavailable"},
            "config": {"data": {"base_dir": "datasets/USK-COFFEE"}},
        }

        assert dataset_identity(run) == ("USK-COFFEE", "datasets/USK-COFFEE")
```

E adicione `dataset_identity` ao import no topo do arquivo (junto de `same_dataset`).

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/core/test_dataset_fingerprint.py::TestDatasetIdentity -q --no-cov
```

Esperado: `ImportError: cannot import name 'dataset_identity'`.

- [ ] **Step 3: Implement**

Em `src/visionforge/core/dataset_fingerprint.py`, antes do `__all__`:

```python
def dataset_identity(run_json: dict[str, Any]) -> tuple[str | None, str | None]:
    """The dataset a run.json points at, as ``(display name, full path)``.

    Prefers the fingerprint's ``root`` and falls back to ``config.data.base_dir``,
    which every run.json has ever written — the fingerprint itself only exists
    from 2026-07-26 on, so the fallback is what makes the name work on the whole
    history rather than on the last few runs.

    ``(None, None)`` means there is nothing to show, which the UI renders as no
    badge at all rather than as a badge with an empty label.
    """
    fingerprint = run_json.get("dataset_fingerprint") or {}
    root = fingerprint.get("root") or ""
    if not root:
        root = ((run_json.get("config") or {}).get("data") or {}).get("base_dir") or ""
    if not root:
        return None, None
    # PurePath picks the separator of the *running* platform, so a Windows path
    # read on Linux (or the reverse, in CI) would keep its separators and yield
    # the whole string as the name. PureWindowsPath accepts both.
    name = PureWindowsPath(str(root)).name
    return (name or None), str(root)
```

Adicione o import no topo: `from pathlib import Path, PureWindowsPath`, e `"dataset_identity"` ao `__all__` (mantendo a ordem alfabética).

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/core/test_dataset_fingerprint.py -q --no-cov
```

Esperado: PASS, incluindo os testes que já existiam.

- [ ] **Step 5: Commit**

```bash
git add src/visionforge/core/dataset_fingerprint.py tests/core/test_dataset_fingerprint.py
git commit -m "feat(core): derive a run's dataset identity from run.json"
```

---

## Task 2: Campos novos nos schemas da API

**Files:**
- Modify: `src/visionforge/gui/api/schemas.py:160-175` (`RunSummary`) e `:236-257` (`RunDetail`)
- Test: `tests/gui/test_device_and_run_detail.py`

- [ ] **Step 1: Write the failing test**

Adicione ao fim de `tests/gui/test_device_and_run_detail.py`:

```python
class TestDatasetOnSchemas:
    def test_run_summary_accepts_the_dataset_fields(self) -> None:
        from visionforge.gui.api.schemas import RunSummary

        summary = RunSummary(
            run_id="r1",
            experiment_name="e",
            model_arch="resnet50",
            task="multiclass",
            status="completed",
            started_at=datetime(2026, 8, 5, 10, 0, 0),
            finished_at=None,
            epochs_completed=1,
            final_metrics={},
            dataset_name="USK-COFFEE",
            dataset_root="/data/USK-COFFEE",
        )

        assert summary.dataset_name == "USK-COFFEE"

    def test_dataset_fields_are_optional_for_older_runs(self) -> None:
        """A run.json with no path at all must still serialize."""
        from visionforge.gui.api.schemas import RunSummary

        summary = RunSummary(
            run_id="r1",
            experiment_name="e",
            model_arch="resnet50",
            task="multiclass",
            status="completed",
            started_at=datetime(2026, 8, 5, 10, 0, 0),
            finished_at=None,
            epochs_completed=1,
            final_metrics={},
        )

        assert summary.dataset_name is None

    def test_dataset_info_holds_only_a_path_when_there_is_no_fingerprint(self) -> None:
        from visionforge.gui.api.schemas import DatasetInfo

        info = DatasetInfo(name="USK-COFFEE", root="/data/USK-COFFEE")

        assert info.digest is None
        assert info.n_files is None
```

Garanta que `from datetime import datetime` está no topo do arquivo de teste.

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/gui/test_device_and_run_detail.py::TestDatasetOnSchemas -q --no-cov
```

Esperado: FAIL — `ImportError: cannot import name 'DatasetInfo'`.

- [ ] **Step 3: Implement**

Em `schemas.py`, adicione antes de `class RunSummary`:

```python
class DatasetInfo(BaseModel):
    """The dataset a run was trained on, as much as the run.json can prove.

    `name` and `root` come from the config and exist for every run. Everything
    below them comes from the fingerprint (ADR-061), which only runs written
    from 2026-07-26 on carry — so they are optional, and their absence is what
    the UI reports as "not verifiable" instead of guessing.
    """

    name: str
    root: str
    n_files: int | None = None
    total_bytes: int | None = None
    method: str | None = None
    digest: str | None = None
    note: str | None = None
```

Em `RunSummary`, depois de `block: str = "classification"`:

```python
    # Dataset the run was trained on. Derived, not stored: the name falls back
    # to config.data.base_dir so it works on runs written before the fingerprint.
    dataset_name: str | None = None
    dataset_root: str | None = None
```

Em `RunDetail`, depois de `tests: list[dict[str, Any]] = []`:

```python
    # None when the run.json records no dataset path at all.
    dataset: DatasetInfo | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/gui/test_device_and_run_detail.py -q --no-cov
```

Esperado: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visionforge/gui/api/schemas.py tests/gui/test_device_and_run_detail.py
git commit -m "feat(api): dataset fields on RunSummary and RunDetail"
```

---

## Task 3: Preencher os campos nas rotas

**Files:**
- Modify: `src/visionforge/gui/api/routes.py:335-350` (`RunDetail(...)`, dentro do handler `get_run_detail` que começa em `:320`) e `:3066-3080` (`RunSummary(...)`, em `_parse_run_summary`)
- Test: `tests/gui/test_device_and_run_detail.py`

- [ ] **Step 1: Write the failing test**

Adicione à classe `TestDatasetOnSchemas` criada na Task 2:

```python
    def test_summary_carries_the_dataset_of_a_legacy_run(self, tmp_path: Path) -> None:
        """A run.json with no fingerprint block still names its dataset."""
        from visionforge.gui.api.routes import _parse_run_summary

        data = {
            "experiment": "coffee",
            "status": "completed",
            "timestamp": "2026-08-05T10:00:00",
            "config": {
                "task": "multiclass",
                "model": {"name": "resnet50"},
                "data": {"base_dir": "datasets/USK-COFFEE"},
            },
            "metrics": {"total_epochs": 30},
        }

        summary = _parse_run_summary(tmp_path / "20260805_100000", data)

        assert summary.dataset_name == "USK-COFFEE"
        assert summary.dataset_root == "datasets/USK-COFFEE"

    def test_detail_carries_the_full_fingerprint_when_present(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        data["dataset_fingerprint"] = {
            "digest": "abc123def456789",
            "method": "manifest",
            "n_files": 8000,
            "total_bytes": 123456,
            "root": "C:/data/USK-COFFEE",
            "note": "paths+sizes only",
        }
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")

        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get(f"/api/runs/{run_dir.name}")

        assert resp.status_code == 200
        dataset = resp.json()["dataset"]
        assert dataset["name"] == "USK-COFFEE"
        assert dataset["n_files"] == 8000
        assert dataset["method"] == "manifest"

    def test_detail_dataset_is_none_without_any_path(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        data["config"].pop("data")
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")

        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get(f"/api/runs/{run_dir.name}")

        assert resp.status_code == 200
        assert resp.json()["dataset"] is None
```

`_write_run`, `app_and_routes`, `json`, `patch`, `Path` e `TestClient` já existem no
arquivo — esta classe reusa todos.

**Nota sobre a estrutura real:** não existe helper `_build_run_detail`. O
`RunDetail` é construído dentro do próprio handler assíncrono
`get_run_detail(run_id)` (`routes.py:320`), que resolve o `run_dir` via
`_find_run_dir` e lê o `run.json`. Por isso o teste vai pelo endpoint HTTP, que
é como todos os testes de detalhe deste arquivo já funcionam.

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/gui/test_device_and_run_detail.py::TestDatasetOnSchemas -q --no-cov
```

Esperado: FAIL — `assert None == 'USK-COFFEE'`.

- [ ] **Step 3: Implement**

Em `routes.py`, importe o helper junto dos outros imports de `visionforge.core`:

```python
from visionforge.core.dataset_fingerprint import dataset_identity
```

Em `_parse_run_summary`, antes do `return RunSummary(`:

```python
    dataset_name, dataset_root = dataset_identity(data)
```

e adicione ao constructor, depois de `block=block,`:

```python
        dataset_name=dataset_name,
        dataset_root=dataset_root,
```

Em `get_run_detail` (`routes.py:320`), antes do `return RunDetail(`:

```python
    dataset_name, dataset_root = dataset_identity(data)
    fingerprint = data.get("dataset_fingerprint") or {}
    dataset = (
        DatasetInfo(
            name=dataset_name,
            root=dataset_root or "",
            n_files=fingerprint.get("n_files"),
            total_bytes=fingerprint.get("total_bytes"),
            method=fingerprint.get("method"),
            digest=fingerprint.get("digest") or None,
            note=fingerprint.get("note"),
        )
        if dataset_name
        else None
    )
```

e ao constructor, depois de `tests=data.get("tests", []),`:

```python
        dataset=dataset,
```

Importe `DatasetInfo` junto dos outros schemas no topo de `routes.py`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/gui/ -q --no-cov
```

Esperado: PASS, incluindo os testes de histórico que já existiam.

- [ ] **Step 5: Commit**

```bash
git add src/visionforge/gui/api/routes.py tests/gui/test_device_and_run_detail.py
git commit -m "feat(api): serve each run's dataset from the history routes"
```

---

## Task 4: Tipos no frontend

**Files:**
- Modify: `frontend/src/types/run.ts:60-75` (`RunSummary`)
- Modify: `frontend/src/api/client.ts:740-766` (`RunDetail`)

Sem teste próprio: tipos são verificados por `tsc` na Task 5, que os consome.

- [ ] **Step 1: Adicione os campos**

Em `frontend/src/types/run.ts`, dentro de `RunSummary`, depois de `block?: string;`:

```typescript
  /** Dataset the run trained on. Derived server-side, falling back to
   *  config.data.base_dir so it works on runs older than the fingerprint. */
  dataset_name?: string | null;
  dataset_root?: string | null;
```

No mesmo arquivo, no fim (antes do próximo `export`):

```typescript
/** The dataset a run trained on, as much as its run.json can prove.
 *
 * `name` and `root` exist for every run; everything below comes from the
 * fingerprint (ADR-061) and is absent on runs written before 2026-07-26.
 */
export interface DatasetInfo {
  name: string;
  root: string;
  n_files?: number | null;
  total_bytes?: number | null;
  method?: string | null;
  digest?: string | null;
  note?: string | null;
}
```

Em `frontend/src/api/client.ts`, dentro de `RunDetail`, depois de `tests: TestRecord[];`:

```typescript
  dataset?: DatasetInfo | null;
```

e adicione `DatasetInfo` ao bloco `import type { ... } from "../types/run";` no topo.

- [ ] **Step 2: Verify types compile**

```bash
cd frontend && npm run typecheck
```

Esperado: sem erros.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/run.ts frontend/src/api/client.ts
git commit -m "feat(frontend): dataset types for run summary and detail"
```

---

## Task 5: Lógica de veredito e formatação (`lib/`)

**Files:**
- Create: `frontend/src/lib/dataset-identity.ts`
- Test: `frontend/src/lib/dataset-identity.test.ts`

- [ ] **Step 1: Write the failing tests**

Crie `frontend/src/lib/dataset-identity.test.ts`:

```typescript
import { describe, expect, it } from "vitest";

import { compareDatasets, formatBytes, shortDigest } from "./dataset-identity";
import type { DatasetInfo } from "../types/run";

function info(overrides: Partial<DatasetInfo> = {}): DatasetInfo {
  return {
    name: "USK-COFFEE",
    root: "C:/data/USK-COFFEE",
    n_files: 8000,
    total_bytes: 123456789,
    method: "manifest",
    digest: "abc123def456789",
    note: "paths+sizes only",
    ...overrides,
  };
}

describe("compareDatasets", () => {
  it("reports the same data when both digests match", () => {
    const verdict = compareDatasets(info(), info());

    expect(verdict.kind).toBe("same");
  });

  it("reports different data when the digests differ", () => {
    const verdict = compareDatasets(info(), info({ digest: "999" }));

    expect(verdict.kind).toBe("different");
  });

  it("refuses to answer when one run has no digest", () => {
    // 26 of 28 existing runs are in this state.
    const verdict = compareDatasets(info(), info({ digest: null }));

    expect(verdict.kind).toBe("unknown");
    expect(verdict.reason).toMatch(/fingerprint/i);
  });

  it("refuses to answer when the two used different methods", () => {
    // A manifest digest and a content digest of the same data do not match;
    // calling that "different data" would be a lie.
    const verdict = compareDatasets(info(), info({ method: "content" }));

    expect(verdict.kind).toBe("unknown");
    expect(verdict.reason).toMatch(/método/i);
  });

  it("refuses to answer when a run has no dataset at all", () => {
    const verdict = compareDatasets(info(), null);

    expect(verdict.kind).toBe("unknown");
  });
});

describe("formatBytes", () => {
  it("scales to a readable unit", () => {
    expect(formatBytes(123456789)).toBe("117,7 MB");
  });

  it("handles missing sizes", () => {
    expect(formatBytes(null)).toBe("—");
  });
});

describe("shortDigest", () => {
  it("keeps the first 12 characters", () => {
    expect(shortDigest("abc123def456789")).toBe("abc123def456");
  });

  it("handles missing digests", () => {
    expect(shortDigest(null)).toBe("—");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd frontend && npx vitest run src/lib/dataset-identity.test.ts
```

Esperado: FAIL — `Failed to resolve import "./dataset-identity"`.

- [ ] **Step 3: Implement**

Crie `frontend/src/lib/dataset-identity.ts`:

```typescript
import type { DatasetInfo } from "../types/run";

export type DatasetVerdict =
  | { kind: "same" }
  | { kind: "different" }
  | { kind: "unknown"; reason: string };

/** Whether two runs saw the same data — mirrors `same_dataset` in Python.
 *
 * Deliberately duplicated rather than served by an endpoint: the rule is four
 * lines, and comparing two fields over HTTP would cost more than the copy. The
 * part that must survive the translation is the third answer — most runs
 * predate the fingerprint, so "I cannot tell" is the common case, not an edge
 * one, and reporting it as "different" would be worse than saying nothing.
 */
export function compareDatasets(
  a: DatasetInfo | null | undefined,
  b: DatasetInfo | null | undefined,
): DatasetVerdict {
  if (!a?.digest || !b?.digest) {
    return {
      kind: "unknown",
      reason: "um dos runs não tem fingerprint (anterior a 26/07/2026)",
    };
  }
  if (a.method !== b.method) {
    return { kind: "unknown", reason: "os dois runs usaram método diferente" };
  }
  return a.digest === b.digest ? { kind: "same" } : { kind: "different" };
}

const UNITS = ["B", "KB", "MB", "GB", "TB"];

/** `123456789` → `117,7 MB`. Em pt-BR a vírgula é o separador decimal. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "—";
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const shown = unit === 0 ? String(value) : value.toFixed(1).replace(".", ",");
  return `${shown} ${UNITS[unit]}`;
}

/** The first 12 characters — enough to tell two digests apart by eye. */
export function shortDigest(digest: string | null | undefined): string {
  return digest ? digest.slice(0, 12) : "—";
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd frontend && npx vitest run src/lib/dataset-identity.test.ts
```

Esperado: PASS, 9 testes.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/dataset-identity.ts frontend/src/lib/dataset-identity.test.ts
git commit -m "feat(frontend): dataset comparison verdict and formatting"
```

---

## Task 6: Selo no card do histórico

**Files:**
- Modify: `frontend/src/components/HistoryOverlay.tsx` — fileira de pills, logo após o bloco `run.preprocessing_count` (por volta da linha 464)

- [ ] **Step 1: Adicione o selo**

Depois do bloco `{run.preprocessing_count !== undefined && ... }` e antes de `{run.block && ...}`:

```tsx
        {run.dataset_name && (
          <span
            title={run.dataset_root ?? run.dataset_name}
            style={{
              padding: "2px 8px",
              background: "oklch(0.70 0.12 250 / 0.14)",
              border: "1px solid oklch(0.70 0.12 250 / 0.45)",
              borderRadius: 999,
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "oklch(0.86 0.11 250)",
              letterSpacing: "0.10em",
              maxWidth: 180,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            🗂 {run.dataset_name}
          </span>
        )}
```

O `maxWidth` + `ellipsis` é o que impede um nome de dataset longo de empurrar os outros pills; a fileira já tem `flexWrap: "wrap"`.

- [ ] **Step 2: Verify it compiles and nothing regressed**

```bash
cd frontend && npm run typecheck && npx vitest run
```

Esperado: sem erros de tipo; todos os testes passam.

Não há teste automatizado do selo: o projeto não tem biblioteca de teste de
componente (`vitest` cobre `src/lib/` e `src/hooks/`; não há `@testing-library`
nem `jsdom`). A decisão toda está em `dataset_identity` e em
`lib/dataset-identity.ts`, ambos cobertos — o que sobra aqui é um condicional
sobre um campo. Confirme visualmente no Step 3.

- [ ] **Step 3: Confirme na interface**

```bash
cd frontend && npm run build
```

Suba a GUI (`.venv/Scripts/visionforge.exe gui`), abra o histórico e confirme que
os runs mostram `🗂 <nome>` e que o hover no selo revela o caminho completo.
Todos os 28 runs devem ter selo — nenhum deveria aparecer sem.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/HistoryOverlay.tsx
git commit -m "feat(gui): dataset badge on the history card"
```

---

## Task 7: Bloco "Dataset" no detalhe do run

**Files:**
- Modify: `frontend/src/components/RunDetailPanel.tsx`

- [ ] **Step 1: Localize o ponto de inserção**

```bash
grep -n "Ambiente\|environment" frontend/src/components/RunDetailPanel.tsx | head -5
```

O bloco "Dataset" vai **imediatamente antes** do bloco de ambiente — dataset e ambiente são as duas informações de proveniência e devem ficar juntas.

- [ ] **Step 2: Adicione o bloco**

Importe no topo do arquivo:

```typescript
import { formatBytes, shortDigest } from "../lib/dataset-identity";
```

E insira, no ponto identificado no Step 1:

```tsx
{detail?.dataset && (
  <div style={{ marginBottom: 16 }}>
    <div
      style={{
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        color: "var(--vf-text-muted)",
        marginBottom: 6,
      }}
    >
      Dataset
    </div>
    <div style={{ fontSize: 13, color: "var(--vf-text)", marginBottom: 2 }}>
      🗂 {detail.dataset.name}
    </div>
    <div
      style={{
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        color: "var(--vf-text-dim)",
        wordBreak: "break-all",
      }}
    >
      {detail.dataset.root}
    </div>
    {detail.dataset.digest ? (
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--vf-text-dim)",
          marginTop: 4,
        }}
        title={`${detail.dataset.digest} — ${detail.dataset.note ?? ""}`}
      >
        {detail.dataset.n_files} arquivos · {formatBytes(detail.dataset.total_bytes)} ·{" "}
        {detail.dataset.method} {shortDigest(detail.dataset.digest)}
      </div>
    ) : (
      <div style={{ fontSize: 11, color: "var(--vf-text-muted)", marginTop: 4 }}>
        sem fingerprint — run anterior a 26/07/2026
      </div>
    )}
  </div>
)}
```

- [ ] **Step 3: Verify**

```bash
cd frontend && npm run typecheck && npx vitest run
```

Esperado: sem erros; testes passam.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/RunDetailPanel.tsx
git commit -m "feat(gui): dataset block in the run detail panel"
```

---

## Task 8: Veredito no comparador

**Files:**
- Modify: `frontend/src/components/CompareRunsPanel.tsx`

- [ ] **Step 1: Adicione a linha de veredito**

Importe no topo:

```typescript
import { compareDatasets } from "../lib/dataset-identity";
```

O painel compara N runs, não dois. A regra: o veredito é sobre **todos** os pares, então compare cada run com o primeiro e reduza — qualquer `unknown` torna o conjunto `unknown`, qualquer `different` torna `different`. Adicione, depois do `details` estar carregado e antes da tabela de métricas:

```tsx
{details.length > 1 &&
  (() => {
    const verdicts = details
      .slice(1)
      .map((d) => compareDatasets(details[0].dataset, d.dataset));
    const verdict = verdicts.find((v) => v.kind === "unknown")
      ?? verdicts.find((v) => v.kind === "different")
      ?? verdicts[0];
    const style = {
      same: { icon: "✓", color: "oklch(0.80 0.15 150)", text: "mesmos dados" },
      different: { icon: "✗", color: "oklch(0.72 0.19 25)", text: "dados diferentes" },
      unknown: { icon: "⚠", color: "oklch(0.80 0.13 85)", text: "não verificável" },
    }[verdict.kind];
    return (
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: style.color,
          display: "flex",
          gap: 6,
          alignItems: "center",
        }}
      >
        <span>{style.icon}</span>
        <span>
          {style.text}
          {verdict.kind === "unknown" ? ` — ${verdict.reason}` : ""}
        </span>
      </div>
    );
  })()}
```

- [ ] **Step 2: Verify**

```bash
cd frontend && npm run typecheck && npx vitest run && npm run build
```

Esperado: sem erros; build conclui.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/CompareRunsPanel.tsx
git commit -m "feat(gui): dataset verdict when comparing runs"
```

---

## Task 9: ADR e verificação final

**Files:**
- Modify: `documentation/DECISIONS.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Escreva o ADR**

Adicione ao fim de `documentation/DECISIONS.md` um `ADR-082 — O dataset de cada run aparece no histórico`, cobrindo:

- O problema: o histórico não dizia em qual dataset o run treinou.
- Que nada novo é medido — `config.data.base_dir` (28/28) e `dataset_fingerprint` (ADR-061, de 2026-07-26, 2/28) já estavam no `run.json`.
- A precedência `fingerprint.root → config.data.base_dir → None`, e por quê: sem o fallback, o selo funcionaria em 2 runs em vez de 28.
- **A consequência que o desenho não pode esconder:** reconhecer e reencontrar são retroativos, verificar não é. Daí o terceiro estado do comparador.
- A duplicação consciente de `same_dataset` em TypeScript, e por que um endpoint seria pior.
- A ressalva das tarefas custom que sintetizam dados: `base_dir` é marcador, o selo mostra o nome daquela pasta, e não houve run custom em `outputs/models/` para confirmar na prática.
- O que ficou fora: agrupar o histórico por dataset, e backfill de hash — este porque o hash de hoje não descreve o que a pasta era naquele dia.

- [ ] **Step 2: Entrada no CHANGELOG**

Em `CHANGELOG.md`, sob `## [Unreleased]`, na seção `### Added`:

```markdown
- **O histórico mostra em qual dataset cada run foi treinado** — selo no card,
  caminho e fingerprint no detalhe do run, e um veredito "mesmos dados?" ao
  comparar runs. Nada novo é medido: os dados já estavam no `run.json`. O nome
  funciona no histórico inteiro porque cai para `config.data.base_dir`; a
  verificação por hash só vale de 26/07/2026 em diante, e o comparador diz isso
  em vez de adivinhar ([ADR-082](documentation/DECISIONS.md)).
```

- [ ] **Step 3: Verificação completa**

```bash
.venv/Scripts/python.exe -m pytest -q --no-cov
.venv/Scripts/ruff.exe check src/ tests/
.venv/Scripts/ruff.exe format --check src/ tests/
.venv/Scripts/mypy.exe src/
```

```bash
cd frontend && npx vitest run && npm run typecheck && npm run build
```

Esperado: suíte verde (baseline atual: `1431 passed, 2 skipped, 2 deselected`, mais os testes novos), ruff e mypy limpos, 121+ testes de frontend passando, build conclui.

- [ ] **Step 4: Verificação manual contra runs reais**

O ponto do trabalho é o histórico do usuário, então confirme nele:

```bash
.venv/Scripts/python.exe -c "
from pathlib import Path
import json
from visionforge.core.dataset_fingerprint import dataset_identity
for rj in sorted(Path('outputs/models').glob('*/*/run.json'))[:10]:
    d = json.loads(rj.read_text(encoding='utf-8'))
    print(dataset_identity(d), '<-', rj.parent.parent.name)
"
```

Esperado: **nenhum `(None, None)`** — todos os 28 runs têm `config.data.base_dir`. Um `(None, None)` significa que a precedência está errada.

- [ ] **Step 5: Commit**

```bash
git add documentation/DECISIONS.md CHANGELOG.md
git commit -m "docs: ADR-082, the dataset of each run in the history"
```

---

## Notas de execução

**Ordem importa.** As tasks 1→3 são backend e cada uma depende da anterior. A 4 destrava 5→8. A 9 fecha.

**Nada de `npx tsc --noEmit`.** O `tsconfig.json` da raiz é solution file com `"files": []` — esse comando type-checa zero arquivos e passa sempre. Use `npm run typecheck` (`tsc -b`).

**Codespell reescreve português.** Se um commit abortar com `word ==> outra`, a palavra vai para `.codespell-ignore-words.txt`, não se reescreve a frase.

**Pre-commit roda a suíte inteira** a cada commit (~4 min). Os commits frequentes deste plano são intencionais, mas conte com esse tempo.
