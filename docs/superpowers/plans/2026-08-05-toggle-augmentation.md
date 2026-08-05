# Toggle de augmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Uma chave liga/desliga a data augmentation em cada tarefa; desligada, os parâmetros somem e o `run.json` registra que estavam desligados.

**Architecture:** Um campo `augment: bool = True` em `TransformConfig` (usado por classificação, regressão, segmentação e anomalia) e outro em `DetectionAugmentationConfig`. O backend passa a consultá-lo em `_build_transforms` e no trainer de detecção; o frontend esconde os campos quando falso.

**Tech Stack:** Pydantic v2 · pytest · React + TypeScript · vitest

**Spec:** `docs/superpowers/specs/2026-08-05-toggle-augmentation-design.md`

---

## Estrutura de arquivos

| Arquivo | Responsabilidade | Ação |
|---|---|---|
| `src/visionforge/utils/config.py` | `TransformConfig.augment` | Modificar |
| `src/visionforge/core/data.py` | `_build_transforms` respeita o flag | Modificar |
| `src/visionforge/utils/detection_config.py` | `DetectionAugmentationConfig.augment` | Modificar |
| `src/visionforge/core/detection_trainer.py` | valores neutros quando desligado | Modificar |
| `frontend/src/lib/transforms-form.ts` | `augment` no form e no payload | Modificar |
| `frontend/src/components/TransformsSection.tsx` | separa "Imagem" de "Augmentation"; esconde | Modificar |
| `frontend/src/components/ParamPanel.tsx` | mesma separação em classificação | Modificar |
| `frontend/src/components/DetectionPanel.tsx` | chave sobre os 15 campos | Modificar |

---

## Task 1: O flag em `TransformConfig` e em `_build_transforms`

**Files:**
- Modify: `src/visionforge/utils/config.py:192-200`
- Modify: `src/visionforge/core/data.py:76-82`
- Test: `tests/core/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
class TestAugmentToggle:
    """Turning augmentation off must not touch resize or normalization."""

    def _transform_names(self, *, augment: bool, is_train: bool) -> list[str]:
        from visionforge.core.data import _build_transforms
        from visionforge.utils.config import TransformConfig

        cfg = TransformConfig(
            augment=augment, horizontal_flip=True, rotation_degrees=25, color_jitter=True
        )
        return [type(t).__name__ for t in _build_transforms(cfg, is_train=is_train).transforms]

    def test_augment_off_drops_the_augmenting_steps(self) -> None:
        names = self._transform_names(augment=False, is_train=True)

        assert "RandomHorizontalFlip" not in names
        assert "RandomRotation" not in names
        assert "ColorJitter" not in names

    def test_augment_off_keeps_resize_and_normalize(self) -> None:
        names = self._transform_names(augment=False, is_train=True)

        assert "Resize" in names
        assert "Normalize" in names

    def test_augment_on_is_the_default_and_unchanged(self) -> None:
        from visionforge.utils.config import TransformConfig

        assert TransformConfig().augment is True
        names = self._transform_names(augment=True, is_train=True)
        assert "RandomHorizontalFlip" in names

    def test_the_flag_does_nothing_outside_training(self) -> None:
        """Val/test never augmented; the flag must not change that."""
        on = self._transform_names(augment=True, is_train=False)
        off = self._transform_names(augment=False, is_train=False)

        assert on == off
```

- [ ] **Step 2: Run it and see it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/core/test_data.py::TestAugmentToggle -q --no-cov
```

Esperado: `TypeError: unexpected keyword argument 'augment'`.

- [ ] **Step 3: Implement**

Em `utils/config.py`, dentro de `TransformConfig`, antes de `horizontal_flip`:

```python
    # Off skips the augmenting steps without discarding their values, so a
    # baseline run and the tuned run differ by one field in the run.json.
    augment: bool = True
```

Em `core/data.py`, trocar `if is_train:` por:

```python
    if is_train and config.augment:
```

- [ ] **Step 4: Passa**

```bash
.venv/Scripts/python.exe -m pytest tests/core/test_data.py -q --no-cov
```

- [ ] **Step 5: Commit**

```bash
git add src/visionforge/utils/config.py src/visionforge/core/data.py tests/core/test_data.py
git commit -m "feat(config): augment flag skips augmentation without discarding it"
```

---

## Task 2: O flag em detecção

**Files:**
- Modify: `src/visionforge/utils/detection_config.py:164-187`
- Modify: `src/visionforge/core/detection_trainer.py` (onde os kwargs de augmentation são montados para `model.train`)
- Test: `tests/core/test_detection_trainer.py`

**Descubra primeiro** como os 15 campos chegam ao `model.train`:

```bash
grep -n "augmentation\." src/visionforge/core/detection_trainer.py | head -20
```

- [ ] **Step 1: Write the failing test**

O teste afirma que, com `augment=False`, os kwargs passados à Ultralytics têm os valores neutros, e que o config permanece intacto:

```python
def test_augment_off_sends_neutral_values_to_ultralytics() -> None:
    """The 15 knobs keep their values; only what reaches Ultralytics changes."""
    from visionforge.core.detection_trainer import _augmentation_kwargs
    from visionforge.utils.detection_config import DetectionAugmentationConfig

    cfg = DetectionAugmentationConfig(augment=False, mosaic=1.0, fliplr=0.5, hsv_h=0.015)

    kwargs = _augmentation_kwargs(cfg)

    assert kwargs["mosaic"] == 0.0
    assert kwargs["fliplr"] == 0.0
    assert kwargs["hsv_h"] == 0.0
    assert kwargs["auto_augment"] is None
    assert cfg.mosaic == 1.0  # the config itself is untouched


def test_augment_on_passes_the_configured_values() -> None:
    from visionforge.core.detection_trainer import _augmentation_kwargs
    from visionforge.utils.detection_config import DetectionAugmentationConfig

    kwargs = _augmentation_kwargs(DetectionAugmentationConfig(mosaic=0.8))

    assert kwargs["mosaic"] == 0.8
```

- [ ] **Step 2: Falha**

- [ ] **Step 3: Implement**

Em `detection_config.py`, em `DetectionAugmentationConfig`, antes de `hsv_h`:

```python
    # Off sends the neutral value for every knob below to Ultralytics, which is
    # the only way to disable augmentation there — it owns its own pipeline.
    augment: bool = True
```

Em `detection_trainer.py`, extrair a montagem dos kwargs para uma função testável:

```python
_NEUTRAL_AUGMENTATION: dict[str, Any] = {
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0, "translate": 0.0,
    "scale": 0.0, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0,
    "bgr": 0.0, "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
    "auto_augment": None, "erasing": 0.0,
}


def _augmentation_kwargs(cfg: DetectionAugmentationConfig) -> dict[str, Any]:
    """Ultralytics train kwargs for augmentation, honouring the on/off flag.

    Disabling means sending neutral values, not omitting the keys: Ultralytics
    fills omitted arguments with its own (augmenting) defaults.
    """
    if not cfg.augment:
        return dict(_NEUTRAL_AUGMENTATION)
    return cfg.model_dump(exclude={"augment"})
```

e usar `**_augmentation_kwargs(self._config.training.augmentation)` na chamada de `model.train`.

- [ ] **Step 4: Passa e a suíte de detecção continua verde**

```bash
.venv/Scripts/python.exe -m pytest tests/core/test_detection_trainer.py tests/gui/test_routes_detection.py -q --no-cov
```

- [ ] **Step 5: Commit**

---

## Task 3: `augment` no form compartilhado

**Files:**
- Modify: `frontend/src/lib/transforms-form.ts`
- Test: `frontend/src/lib/transforms-form.test.ts` (criar se não existir)

- [ ] **Step 1: Teste**

```typescript
import { describe, expect, it } from "vitest";
import { buildTransformsPayload, makeDefaultTransformsForm } from "./transforms-form";

describe("augment flag", () => {
  it("defaults to on, matching the backend", () => {
    expect(makeDefaultTransformsForm().augment).toBe(true);
  });

  it("travels in the payload", () => {
    const form = { ...makeDefaultTransformsForm(), augment: false };
    expect(buildTransformsPayload(form).augment).toBe(false);
  });

  it("keeps the tuned values when off, so turning it back on restores them", () => {
    const form = { ...makeDefaultTransformsForm(), augment: false, rotation_degrees: 25 };
    expect(buildTransformsPayload(form).rotation_degrees).toBe(25);
  });
});
```

- [ ] **Step 2: Falha** · **Step 3:** adicionar `augment: boolean` a `TransformsForm`, `augment: true` ao default, e `augment: t.augment` ao payload · **Step 4: Passa** · **Step 5: Commit**

---

## Task 4: `TransformsSection` separa Imagem de Augmentation

**Files:**
- Modify: `frontend/src/components/TransformsSection.tsx`

- [ ] **Step 1:** Dividir a seção atual "Augmentação & normalização" em duas:
  - **"Imagem"**: `image_size`, `normalize_mean`, `normalize_std` — sempre visível.
  - **"Data augmentation"**: uma `Toggle` ligada a `transforms.augment`, e abaixo dela `horizontal_flip`, `rotation_degrees`, `color_jitter` **apenas quando ligado**. Desligado, renderiza `3 parâmetros ocultos — ligue para ajustar`.
  - O `AugmentPreview` só aparece com a augmentation ligada: prever o efeito de transformações desativadas mostraria algo que o treino não vai fazer.

- [ ] **Step 2:** `npm run typecheck && npx vitest run` · **Step 3: Commit**

---

## Task 5: Mesma separação em classificação

**Files:**
- Modify: `frontend/src/components/ParamPanel.tsx` (seção `// AUMENTOS & NORMALIZAÇÃO`)

Mesma divisão da Task 4, sobre os campos próprios do ParamPanel. Verificar com `npm run typecheck && npx vitest run && npm run build`, e commitar.

---

## Task 6: A chave em detecção

**Files:**
- Modify: `frontend/src/components/DetectionPanel.tsx:597` (`<div style={sectionLabel}>Data augmentation</div>`)

- [ ] **Step 1:** Chave ligada a `formData.training.augmentation.augment`; os 15 campos abaixo só renderizam quando ligada, e desligada aparece `15 parâmetros ocultos — ligue para ajustar`.

- [ ] **Step 2:** Verificar e commitar.

---

## Task 7: ADR e verificação final

- [ ] **Step 1:** `ADR-083 — Data augmentation has an explicit on/off flag` em `documentation/DECISIONS.md`, cobrindo: por que flag explícito e não inferência dos valores (registro no `run.json`; sobrevivência dos valores à exportação do YAML; 15 campos contra 1 booleano); por que esconder e não desabilitar; e por que `image_size`/`normalize_*` ficaram de fora da seção — não são augmentation, e `_build_transforms` já os tratava assim.
- [ ] **Step 2:** Entrada no CHANGELOG.
- [ ] **Step 3:** Verificação completa:

```bash
.venv/Scripts/python.exe -m pytest -q --no-cov
.venv/Scripts/ruff.exe check src/ tests/
.venv/Scripts/mypy.exe src/
```

```bash
cd frontend && npx vitest run && npm run typecheck && npm run build
```

- [ ] **Step 4:** Confirmar que um `run.json` novo carrega o flag, e que um config YAML antigo (sem o campo) ainda valida.
