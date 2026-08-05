# Paridade do painel de detecção — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Trazer o painel de detecção ao contrato canônico: `num_classes` junto do dataset, exemplos das classes, splits detectados à vista, e filtros de pré-processamento que funcionam mesmo com a Ultralytics dona do pipeline.

**Architecture:** O grosso é um módulo novo em `core/` que materializa uma cópia filtrada do dataset, chaveada por conteúdo, e um `data.yaml` apontando para ela. O resto é reorganização de painel.

**Tech Stack:** Pillow · Pydantic v2 · pytest · React + TypeScript

**Spec:** `docs/superpowers/specs/2026-08-05-paridade-deteccao-design.md`

---

## Estrutura de arquivos

| Arquivo | Responsabilidade | Ação |
|---|---|---|
| `src/visionforge/core/materialized_dataset.py` | filtra o dataset para uma pasta temporária, com cache e limpeza | Criar |
| `src/visionforge/utils/detection_config.py` | `data.preprocessing` + `preprocessed_format` | Modificar |
| `src/visionforge/core/detection_data.py` | `resolve_data_yaml` aponta para a cópia | Modificar |
| `src/visionforge/core/detection_trainer.py` | materializa antes, libera no `finally` | Modificar |
| `src/visionforge/gui/server.py` | varredura de órfãos no startup | Modificar |
| `frontend/src/components/DetectionPanel.tsx` | `num_classes` no Dataset; `TransformsSection` | Modificar |
| `frontend/src/components/DetectionDatasetStats.tsx` | amostras por classe; splits detectados | Modificar |

O módulo novo é deliberadamente independente de detecção: recebe caminho + passos e devolve caminho. Isso o mantém testável sem Ultralytics e reusável se as outras tarefas migrarem depois.

---

## Task 1: O módulo de materialização

**Files:**
- Create: `src/visionforge/core/materialized_dataset.py`
- Test: `tests/core/test_materialized_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
"""A filtered copy that outlives its run, or loses a label, is worse than none."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visionforge.core.materialized_dataset import (
    MaterializedDataset,
    materialize_dataset,
    sweep_orphans,
)

STEPS = [{"kind": "grayscale"}]


def _dataset(root: Path) -> Path:
    """A YOLO-shaped dataset: images plus label files that must survive."""
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (16, 16), (i * 100, 40, 200)).save(
                root / split / "images" / f"img{i}.jpg"
            )
            (root / split / "labels" / f"img{i}.txt").write_text("0 .5 .5 .2 .2\n")
    return root


class TestMaterialize:
    def test_mirrors_the_tree_and_keeps_the_labels_byte_for_byte(
        self, tmp_path: Path
    ) -> None:
        src = _dataset(tmp_path / "src")

        with materialize_dataset(src, STEPS, cache_root=tmp_path / "cache") as out:
            for split in ("train", "val"):
                assert sorted(p.stem for p in (out.path / split / "images").iterdir()) == [
                    "img0",
                    "img1",
                ]
                original = (src / split / "labels" / "img0.txt").read_bytes()
                assert (out.path / split / "labels" / "img0.txt").read_bytes() == original

    def test_the_images_are_actually_filtered(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")

        with materialize_dataset(src, STEPS, cache_root=tmp_path / "cache") as out:
            img = Image.open(out.path / "train" / "images" / "img0.png")
            r, g, b = img.convert("RGB").getpixel((0, 0))
            assert r == g == b  # grayscale applied

    def test_empty_pipeline_returns_the_original_untouched(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")

        with materialize_dataset(src, [], cache_root=tmp_path / "cache") as out:
            assert out.path == src
            assert not (tmp_path / "cache").exists()

    def test_the_same_dataset_and_pipeline_reuse_one_folder(self, tmp_path: Path) -> None:
        """A 20-trial sweep must filter once, not twenty times."""
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, STEPS, cache_root=cache) as first:
            with materialize_dataset(src, STEPS, cache_root=cache) as second:
                assert first.path == second.path
                assert len(list(cache.iterdir())) == 1

    def test_a_different_pipeline_gets_its_own_folder(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, STEPS, cache_root=cache) as a:
            with materialize_dataset(
                src, [{"kind": "invert"}], cache_root=cache
            ) as b:
                assert a.path != b.path

    def test_the_folder_is_removed_when_the_last_user_leaves(
        self, tmp_path: Path
    ) -> None:
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, STEPS, cache_root=cache) as out:
            path = out.path
        assert not path.exists()

    def test_a_raising_run_does_not_leave_the_copy_behind(self, tmp_path: Path) -> None:
        """Runs die for real — see ADR-081."""
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"
        leaked: Path | None = None

        with pytest.raises(RuntimeError, match="boom"):
            with materialize_dataset(src, STEPS, cache_root=cache) as out:
                leaked = out.path
                raise RuntimeError("boom")

        assert leaked is not None and not leaked.exists()

    def test_an_inner_user_does_not_delete_the_folder_from_under_the_outer(
        self, tmp_path: Path
    ) -> None:
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, STEPS, cache_root=cache) as outer:
            with materialize_dataset(src, STEPS, cache_root=cache):
                pass
            assert outer.path.exists()


class TestSweepOrphans:
    def test_removes_folders_left_by_a_killed_process(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        (cache / "abc123").mkdir(parents=True)

        removed = sweep_orphans(cache)

        assert removed == 1
        assert not (cache / "abc123").exists()

    def test_leaves_a_folder_that_is_in_use(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")
        cache = tmp_path / "cache"

        with materialize_dataset(src, STEPS, cache_root=cache) as out:
            assert sweep_orphans(cache) == 0
            assert out.path.exists()

    def test_a_missing_cache_root_is_not_an_error(self, tmp_path: Path) -> None:
        assert sweep_orphans(tmp_path / "nope") == 0


class TestEstimate:
    def test_reports_the_source_size_so_the_caller_can_warn(self, tmp_path: Path) -> None:
        src = _dataset(tmp_path / "src")

        estimate = MaterializedDataset.estimate_bytes(src)

        assert estimate > 0
```

- [ ] **Step 2:** Rodar e ver falhar por `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`materialize_dataset` é um **context manager** com contagem de uso em processo:

- Chave: `sha256` de (digest do `fingerprint_dataset(src)` + JSON canônico dos passos), truncado.
- Ao entrar: se a pasta existe, incrementa o refcount e devolve; senão espelha a árvore, aplicando `apply_pipeline` a cada imagem e copiando todo o resto com `shutil.copy2`.
- Imagens saem em PNG por padrão (`preprocessed_format`), preservando o *stem* — o YOLO casa rótulo por stem, não por extensão.
- Ao sair: decrementa; no zero, `shutil.rmtree`.
- Um arquivo-sentinela `.in-use` com o PID permite que `sweep_orphans` distinga pasta viva de pasta órfã.
- `estimate_bytes(src)` soma o tamanho dos arquivos para o aviso de disco.

- [ ] **Step 4:** `pytest tests/core/test_materialized_dataset.py -q` verde.
- [ ] **Step 5: Commit.**

---

## Task 2: Config de detecção aceita pré-processamento

**Files:**
- Modify: `src/visionforge/utils/detection_config.py`
- Test: `tests/utils/test_detection_config.py`

- [ ] `DetectionDataConfig` ganha `preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)` e `preprocessed_format: Literal["png", "jpeg"] = "png"`.
- [ ] Teste: config sem os campos continua válido; com filtros, valida.
- [ ] Commit.

---

## Task 3: O `data.yaml` aponta para a cópia

**Files:**
- Modify: `src/visionforge/core/detection_data.py:50-80`
- Modify: `src/visionforge/core/detection_trainer.py:229`
- Test: `tests/core/test_detection_data.py`

- [ ] **Step 1: Teste** — com pipeline vazia, `spec["path"]` é o `base_dir` original; com filtros, é a pasta materializada, e os splits/names continuam iguais.
- [ ] **Step 2:** Falha.
- [ ] **Step 3:** O trainer abre o context manager em volta do treino inteiro e passa o caminho materializado ao `resolve_data_yaml`. O `finally` do `with` é o que garante a limpeza mesmo quando o treino levanta.
- [ ] **Step 4:** Verde. **Step 5:** Commit.

**Cuidado registrado:** o `fingerprint_from_config` continua apontando para o `base_dir` **original**. Se ele passar a ver a cópia, o histórico do ADR-082 mostra a pasta temporária. Um teste deve fixar isso:

```python
def test_run_json_fingerprints_the_original_not_the_copy(...) -> None:
    ...
    assert Path(data["dataset_fingerprint"]["root"]).name == "src"
```

---

## Task 4: Varredura de órfãos no startup

**Files:**
- Modify: `src/visionforge/gui/server.py`
- Test: `tests/gui/test_startup_sweep.py`

- [ ] Um evento de startup do FastAPI chama `sweep_orphans(cache_root)` e registra quantas pastas removeu. Cobre o caso do processo morto à força, que nenhum `finally` alcança.
- [ ] Commit.

---

## Task 5: `num_classes` na seção Dataset

**Files:**
- Modify: `frontend/src/components/DetectionPanel.tsx:255` (remover do Modelo) e `:497+` (adicionar ao Dataset)

- [ ] Espelhar o que classificação faz: o campo vive na seção Dataset e é preenchido pela contagem detectada, com o valor ainda editável. O `onApplyClasses` deixa de ser um botão que empurra valor para longe e passa a preencher o campo ao lado.
- [ ] `npm run typecheck && npx vitest run && npm run build`. Commit.

---

## Task 6: Amostras por classe e splits à vista

**Files:**
- Modify: `frontend/src/components/DetectionDatasetStats.tsx`
- Modify: rota de stats de detecção em `src/visionforge/gui/api/routes.py`

- [ ] A rota passa a devolver, por classe, o caminho de uma imagem de exemplo e a caixa a recortar; o componente renderiza a miniatura ao lado da contagem, como `DatasetStats` faz.
- [ ] Os splits detectados por `_detect_splits` aparecem no painel antes do treino.
- [ ] Testes de rota para o formato novo. Commit.

---

## Task 7: `TransformsSection` em detecção

**Files:**
- Modify: `frontend/src/components/DetectionPanel.tsx`

- [ ] Adicionar `TransformsSection` **apenas com a parte de pré-processamento** — a augmentation de detecção é a da Ultralytics e já tem a sua própria seção (sub-projeto b).
- [ ] Um aviso com a estimativa de espaço (`estimate_bytes`) aparece quando há filtros, dizendo que uma cópia filtrada será gravada e apagada ao fim.
- [ ] Commit.

---

## Task 8: ADR e verificação final

- [ ] `ADR-084 — Detection preprocesses by materializing a filtered copy`, cobrindo: por que a Ultralytics força isso; por que também é ~30× menos CPU que filtrar por época; as quatro guardas (fingerprint do original, limpeza dupla, PNG e o custo de disco, chave por conteúdo); e o registro consciente de que detecção passa a usar mecanismo diferente do de classificação para a mesma feature.
- [ ] CHANGELOG.
- [ ] Verificação completa (backend + frontend).
- [ ] **Verificação com dados reais:** rodar uma detecção curta em `datasets/cats-dogsv2.v1i.yolov8` com um filtro ativo e confirmar, ao fim: métricas produzidas, `run.json` com o fingerprint do dataset **original**, e nenhuma pasta sobrando no cache.
