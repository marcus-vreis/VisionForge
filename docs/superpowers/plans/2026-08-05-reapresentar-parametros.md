# Reapresentar os parâmetros — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Deixar os hiperparâmetros compreensíveis sem esconder nenhum: básico à vista, avançado colapsado, e uma linha explicando o que cada um faz.

**Architecture:** Um mapa único de explicações em `lib/`, consumido por todos os painéis, e um componente de seção colapsável reusando o padrão de (b). Nenhuma mudança de backend, nenhuma mudança de payload.

**Tech Stack:** React + TypeScript · vitest

**Spec:** `docs/superpowers/specs/2026-08-05-reapresentar-parametros-design.md`

**Depende de (b):** o componente colapsável nasce lá; aqui ele é reusado.

---

## Estrutura de arquivos

| Arquivo | Responsabilidade | Ação |
|---|---|---|
| `frontend/src/lib/param-help.ts` | uma linha de explicação por parâmetro, e o corte básico/avançado | Criar |
| `frontend/src/lib/param-help.test.ts` | completude: nenhum campo renderizado sem explicação | Criar |
| `frontend/src/components/CollapsibleSection.tsx` | seção colapsável com contador (extraída de (b)) | Criar |
| `frontend/src/components/ParamPanel.tsx` | aplica o corte e as explicações | Modificar |
| `frontend/src/components/{Detection,Regression,Segmentation,Anomaly}Panel.tsx` | idem | Modificar |

Colocar as explicações num módulo de dados, e não espalhadas pelo JSX, é o que torna testável a afirmação "todo campo tem explicação" — que é o requisito de verdade, já que o problema relatado foi não saber o que os campos fazem.

---

## Task 1: O mapa de explicações

**Files:**
- Create: `frontend/src/lib/param-help.ts`
- Test: `frontend/src/lib/param-help.test.ts`

- [ ] **Step 1: Teste**

```typescript
import { describe, expect, it } from "vitest";
import { PARAM_HELP, PARAM_TIER, isAdvanced } from "./param-help";

describe("param help", () => {
  it("explains every parameter it classifies", () => {
    for (const key of Object.keys(PARAM_TIER)) {
      expect(PARAM_HELP[key], `sem explicação: ${key}`).toBeTruthy();
    }
  });

  it("keeps the four everyday knobs visible", () => {
    for (const key of ["epochs", "batch_size", "learning_rate", "seed"]) {
      expect(isAdvanced(key)).toBe(false);
    }
  });

  it("collapses the set-once knobs", () => {
    for (const key of ["optimizer", "weight_decay", "num_workers", "pin_memory"]) {
      expect(isAdvanced(key)).toBe(true);
    }
  });

  it("warns about the parameter that stops a training instead of degrading it", () => {
    expect(PARAM_HELP.num_workers).toMatch(/mem[óo]ria|processo/i);
  });

  it("treats an unknown parameter as basic rather than hiding it", () => {
    // Hiding a field nobody classified would make it invisible by accident.
    expect(isAdvanced("um_campo_novo")).toBe(false);
  });
});
```

- [ ] **Step 2:** Falha. **Step 3:** Implementar `PARAM_HELP` (chave → uma frase, em pt-BR, dizendo o efeito prático de aumentar/diminuir) e `PARAM_TIER` (chave → `"basic" | "advanced"`), mais `isAdvanced(key)` com default `basic`.

`num_workers` recebe a advertência do ADR-081: cada worker é um processo que recarrega o torch e as DLLs da CUDA, ~1 GB cada, e valor alto demais derruba o treino em vez de deixá-lo lento.

- [ ] **Step 4:** Verde. **Step 5:** Commit.

---

## Task 2: `CollapsibleSection`

**Files:**
- Create: `frontend/src/components/CollapsibleSection.tsx`

- [ ] Extrair do que (b) construiu: título, chave de abrir/fechar, contador de campos ocultos, e uma prop `forceOpen` para o caso da Task 4.
- [ ] Commit.

---

## Task 3: Aplicar em classificação

**Files:**
- Modify: `frontend/src/components/ParamPanel.tsx`

- [ ] A seção `// TREINAMENTO` passa a mostrar épocas, learning rate, batch size e seed; o resto (otimizador, early stopping, weight decay, determinístico, AMP) vai para uma `CollapsibleSection` "Avançado". `// LEARNING-RATE SCHEDULER` inteiro entra em avançado. `workers` e `pin memory`, hoje na seção `// CLASSES`, mudam para avançado no bloco de treinamento — é onde pertencem.
- [ ] Cada campo renderiza `PARAM_HELP[key]` como linha de apoio.
- [ ] `npm run typecheck && npx vitest run && npm run build`. Commit.

---

## Task 4: Um valor não-default força a seção a abrir

**Files:**
- Modify: `frontend/src/components/ParamPanel.tsx`
- Test: `frontend/src/lib/param-help.test.ts`

- [ ] Se qualquer campo avançado difere do default, a seção nasce aberta. Esconder um ajuste deliberado seria pior que a poluição visual que motivou a mudança — e é o caso que acontece ao importar um YAML afinado.
- [ ] Uma função pura `hasNonDefaultAdvanced(form, defaults)` em `param-help.ts`, com teste. Commit.

---

## Task 5: Aplicar nos outros quatro painéis

**Files:**
- Modify: `DetectionPanel.tsx`, `RegressionPanel.tsx`, `SegmentationPanel.tsx`, `AnomalyPanel.tsx`

- [ ] Mesmo corte e mesmas explicações. Detecção tem os campos próprios da Ultralytics (`close_mosaic`, `lr0`, ganhos de loss) — todos avançados, todos com explicação, nenhum removido.
- [ ] Verificar e commitar.

---

## Task 6: ADR e verificação final

- [ ] `ADR-085 — Hyperparameters are tiered, not trimmed`, registrando: o pedido foi entender, não reduzir; o corte é por frequência de ajuste e não por importância; e o mapa de explicações vive em dados justamente para que "todo campo explicado" seja uma asserção de teste, não uma promessa.
- [ ] CHANGELOG.
- [ ] Verificação completa.
- [ ] **Conferência manual:** abrir cada uma das cinco tarefas e confirmar que a seção avançada nasce fechada num formulário novo, e aberta depois de importar um YAML com valores afinados.
