# Reorganização da seção Dataset + auto-detecção — design

> Data: 2026-05-24 · Escopo: **frontend apenas** + code-review/karpathy nos arquivos tocados

## Problema

1. **Overlap visual:** o painel `// pré-processamento` colide com os controles `WORKERS / PIN MEMORY` (sem divisória entre elas).
2. **Botões mal posicionados:** `task` (binário/multiclass) vive no header do topo e `num_classes` em `// modelo` — ambos longe de onde as classes são realmente detectadas (a seção dataset).
3. **Ações manuais que deviam ser automáticas:**
   - `🎯 aplicar` (DatasetStats) — aplicar nº de classes detectado é manual.
   - `↻ Detectar splits` (DatasetPicker) — detecção de train/val/test é manual.
4. **Código morto:** `ConfigForm.tsx` é o formulário original, totalmente substituído por `ParamPanel` e importado em lugar nenhum.

## Decisões

### Nova ordem da seção `// dataset`
```
// dataset
 1. DatasetPicker (base_dir + 📁 Escolher pasta)   → auto-detecta splits ao mudar base_dir
 2. Subpastas treino / validação / teste            (override manual; auto-preenchido)
 3. DatasetStats + amostras                          (auto, já era)
 4. // classes:  task + nº de classes                ← MOVIDO pra cá, auto-aplicado, editável
 5. DataLoader:  workers + pin_memory
 ───── divisória ─────
// aumentos & normalização
 6. Pré-processamento (filtros + ▶ Ver preview)      ← MOVIDO pra baixo da divisória
 7. Aumentos & normalização
```

- `task`/`num_classes` mudam só de **posição visual**; os paths em `formData` (`task`, `model.num_classes`) permanecem — YAML round-trip e backend intactos.
- 2 classes detectadas → binário (`num_classes=1`), mantendo o comportamento atual (decisão do usuário).

### Auto-detecção (DatasetPicker)
- Remove o botão `↻ Detectar splits`. `detectDatasetSplits` roda em `useEffect` ao mudar `base_dir`, com debounce de 400 ms (e após o seletor nativo, que muda `base_dir`).
- Mantém `📁 Escolher pasta` e os overrides manuais de subpasta.

### Auto-aplicar classes (DatasetStats)
- Remove o botão `🎯 aplicar`. `onApplyClasses` dispara automaticamente quando os stats carregam, **apenas quando `base_dir` muda** (guarda via `useRef`) — assim edições manuais posteriores não são sobrescritas. Nota discreta "✓ aplicado" substitui o botão.

### Limpeza
- Deletar `ConfigForm.tsx` (morto, substituído por `ParamPanel`).
- `▶ Ver preview` **continua manual** (render server-side caro; auto-rodar desperdiçaria ciclos).

## Verificação
- `tsc` + `eslint` + `vite build` limpos.
- Subir a GUI e confirmar **visualmente** que o overlap sumiu antes de declarar pronto.
- Testes vitest existentes seguem passando.
