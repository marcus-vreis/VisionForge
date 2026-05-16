# Spec C — Loop de longa duração

**Status:** Approved (2026-05-15)
**Author:** Marcus Reis (via brainstorming with Claude)
**Depends on:** Spec A (Multi-agent team), Spec B (GitHub integration)
**Blocks:** —

---

## 1. Goal

Lead roda em loop sustentado de até 10h cumulativas, processando `TASKS.md` até zerar ou bater condição de parada. Quando bate rate limit do plano Anthropic, dorme até o reset e retoma — re-spawnando o time, porque `/resume` e `/rewind` não restauram teammates in-process (limitação documentada de Agent Teams).

## 2. Scope

**In scope:**
- Loop autônomo via plugin Ralph Loop (`/loop` em modo autônomo)
- Persistência de estado em `outputs/LOOP_STATUS.md`
- Detecção de rate-limit Anthropic e wait com `ScheduleWakeup`
- Sinal de parada via arquivo `STOP_TEAM`
- Hard time limit (10h cumulativas)
- Re-spawn de time após sleep (necessário, limitação Agent Teams)
- Reconciliação de PRs em vôo após resume (`gh pr list --state=open`)
- Escalation triggers

**Out of scope:**
- Notificações externas (Slack, email, push) — só LOOP_STATUS.md
- Multi-day loops (>10h)
- Auto-restart após crash do processo Claude Code (você reinicia manualmente)
- Métricas de custo de token (sem dashboard, só log)
- Recuperação após corrupção de TASKS.md (manual)

## 3. Mecanismo

| Componente | Decisão |
|---|---|
| Loop runner | Plugin **Ralph Loop** (já instalado), via `/loop` em modo dinâmico |
| Pacing | Dinâmico — lead usa `ScheduleWakeup` decidindo o intervalo a cada ciclo |
| Intervalo normal | 30 min (1800s) entre ciclos quando time ativo |
| Intervalo rate-limited | 5h (18000s) — janela típica de reset Anthropic Max |
| Sinalização de parada | Arquivo `STOP_TEAM` na raiz do repo (existe = stop) |
| Hard time limit | 10h cumulativas desde primeiro start (timestamp em LOOP_STATUS) |
| Estado persistente | `outputs/LOOP_STATUS.md` (markdown) |
| Notificação final | Última escrita de LOOP_STATUS.md com `state=done` ou `state=escalated` |

## 4. Estado: `outputs/LOOP_STATUS.md`

Arquivo human-readable, lido pelo lead a cada ciclo. Atualizado ao final de cada ciclo.

```markdown
# VisionForge Loop Status

- **State:** running | waiting_reset | done | escalated | paused_by_user
- **First start:** 2026-05-15T14:32:00Z
- **Elapsed:** 2h 14m
- **Current phase:** Phase 5
- **Last completed:** GridSearchBlock (PR #42, merged at 16:21)
- **In flight:**
  - bk-1: RandomSearchBlock (planning)
  - bk-2: CrossValidationBlock (implementing)
  - fr-1: idle
- **Open PRs:** [#43, #44]
- **Failures this cycle:** 0
- **Failures lifetime:** 1 (RandomSearchBlock plan rejected 1x)
- **Next wake:** 2026-05-15T17:00:00Z (in 30m)
- **Wake reason:** normal cycle

## Cycle history (last 5)
- 16:30 (cycle 4): merged PR #42, started #43, #44
- 16:00 (cycle 3): planning approved for bk-1, bk-2
- 15:30 (cycle 2): all 4 teammates spawned, plans in progress
- 15:00 (cycle 1): startup, gh auth ok, development branch verified
- 14:32 (cycle 0): initial start
```

## 5. Fluxo de um ciclo

```
loop fire (Ralph Loop dispara conforme ScheduleWakeup)
  ↓
Lead executa loop-cycle.sh (read-only check)
  ↓
1. Check STOP_TEAM existe?
   → YES: state=paused_by_user, cleanup team, exit loop (no ScheduleWakeup)
  ↓
2. Check elapsed >= 10h?
   → YES: state=done (ou escalated se algo pendente), cleanup, exit
  ↓
3. Check último ciclo gravou rate_limit_detected?
   → YES: state=waiting_reset, ScheduleWakeup(5h), atualiza status, exit
  ↓
4. Check time spawned?
   → NO: spawna 4 teammates (Spec A), grava state=running
   → YES: pega status atual via /team
  ↓
5. Reconcilia com GitHub (Spec B):
   - gh pr list --state=open → lista PRs em vôo
   - Para cada PR: check CI status, merge se verde+approved
   - Se PR órfão (worktree não existe localmente), tenta re-checkout
  ↓
6. Despacha tasks da Spec A normalmente até teammate idle
  ↓
7. Se TASKS.md = 0 unchecked + 0 PRs abertos:
   state=done, cleanup, exit (sem reschedule)
  ↓
8. Atualiza LOOP_STATUS.md com snapshot atual
  ↓
9. ScheduleWakeup(1800s) ("normal cycle")
   OU se detectou rate-limit no meio do ciclo: ScheduleWakeup(18000s)
```

## 6. Detecção de rate-limit

Lead executa cada ação suspeita (mensagem entre teammates, prompt grande) wrapped por error handling. Quando captura erro com substring conhecida:

- `"rate limit"` ou `"rate_limit_error"` ou `"429"` → marca `rate_limit_detected=true` no status
- Lead grava timestamp e estimated_reset (timestamp + 5h)
- Próximo `ScheduleWakeup` é 5h, com `reason="awaiting Anthropic plan reset"`

Limitação aceita: se o rate-limit acontecer no MEIO de um turno do lead (não na borda), o turno corrente pode falhar. Ralph Loop reinicia o ciclo na próxima fire, então tolerável.

## 7. Re-spawn de time após sleep

Documentação Agent Teams: `/resume` não restaura teammates in-process. Então no resume:

1. Lead lê `outputs/LOOP_STATUS.md` (sabe a fase, último item completo, PRs em vôo)
2. Lead lê `gh pr list --state=open --label "agent-team"` (PRs abertos pelo time anterior)
3. Lead spawna 4 teammates **frescos** com o start-team prompt da Spec A
4. Lead anexa contexto no prompt inicial de cada teammate: "PRs em vôo do ciclo anterior: #43 (bk-1), #44 (bk-2). Resume from there."
5. Se um PR já está em estado mergeável, lead mescla antes de spawnar trabalho novo

## 8. Files to create

```
.claude/scripts/
├── loop-cycle.sh           # ciclo principal (chamado pelo lead a cada wake)
├── check-stop-signal.sh    # checa STOP_TEAM e elapsed time
├── detect-rate-limit.sh    # parser de erro Anthropic
└── update-status.sh        # atualiza outputs/LOOP_STATUS.md

outputs/
└── LOOP_STATUS.md          # criado no primeiro start
```

`.gitignore` ganha `outputs/LOOP_STATUS.md` e `STOP_TEAM`.

## 9. Start, stop, monitor

### Start
```bash
# Você abre o terminal com bypass
claude --dangerously-skip-permissions

# Dispara o loop autônomo
> /loop "Você é o lead do VisionForge agent team. Leia .claude/prompts/start-team.md e siga as instruções. Use loop-cycle.sh a cada ciclo. Pare quando TASKS.md zerar, 10h passarem, ou STOP_TEAM existir."
```

Ralph Loop assume controle, `ScheduleWakeup` agenda os ciclos.

### Stop manual
```bash
# Em outro terminal, na raiz do repo:
echo "stop" > STOP_TEAM
```

Lead pega no próximo ciclo (até 30 min depois), faz cleanup gracioso (não interrompe tarefa em vôo, mas não pega nova), e sai.

### Monitor
```bash
cat outputs/LOOP_STATUS.md
gh pr list --label agent-team
git log development --oneline -20
```

## 10. Stop conditions completas (consolidação)

| Trigger | Resultado |
|---|---|
| `TASKS.md` zerado + 0 PRs abertos | `state=done`, exit |
| `STOP_TEAM` existe | `state=paused_by_user`, exit |
| Elapsed >= 10h | `state=done` (ou `escalated` se pendente), exit |
| 3× falha consecutiva mesma task (Spec A) | `state=escalated`, exit, log na status |
| `gh auth` expira | `state=escalated`, exit |
| Conflito merge irresolvível (Spec B) | `state=escalated`, exit |
| Exceção não tratada no hook | `state=escalated`, exit |

## 11. Acceptance criteria

- [ ] `.claude/scripts/loop-cycle.sh` implementa todos os 9 passos da §5
- [ ] `outputs/LOOP_STATUS.md` é criado no primeiro start e atualizado a cada ciclo
- [ ] Smoke 1 — start curto: rodar `/loop`, esperar 2 ciclos (~1h), parar via `STOP_TEAM`, verificar cleanup
- [ ] Smoke 2 — re-spawn: simular rate-limit (forçar via env var de teste), verificar que loop dorme e re-spawna time
- [ ] Smoke 3 — fim natural: rodar com TASKS.md quase vazio (1 item), confirmar que loop termina com `state=done` e sai
- [ ] `STOP_TEAM` no .gitignore
- [ ] LOOP_STATUS.md no .gitignore

## 12. Risks

| Risco | Mitigação |
|---|---|
| Loop fica em rate-limit eterno (reset não chega) | Hard time limit 10h destrava — você reinicia manualmente depois |
| Re-spawn perde contexto crítico do teammate anterior | LOOP_STATUS.md + PRs em vôo no GitHub carregam o estado essencial; teammates fresh leem CLAUDE.md/TASKS.md normalmente |
| Detecção de rate-limit por substring é frágil | Aceito — se quebrar, próxima fire do loop tenta de novo. Hard cap de 10h impede prejuízo |
| Lead crasha entre ciclos | Você reinicia manualmente; LOOP_STATUS preserva onde parou |
| Custo de token explode em loop longo | Visível no LOOP_STATUS (não controlado). Você pode parar via `STOP_TEAM` a qualquer momento |
| 10h não cabe na janela de tokens do plano | Esperado — vai entrar em `waiting_reset` 1-2× durante as 10h. Por design |
| Múltiplos `/loop` rodando | Ralph Loop permite um por sessão; abrir 2 sessões com loop simultaneamente quebraria o estado. Apenas uma sessão `/loop` por vez |

## 13. Como Spec A + B + C compõem (resumo final)

```
Você: claude --dangerously-skip-permissions
Você: /loop "<lead prompt>"

Spec C: Ralph Loop assume, agenda ciclos via ScheduleWakeup
   ↓
Spec C: loop-cycle.sh roda — checa STOP, elapsed, rate-limit
   ↓ (se ok)
Spec A: lead spawna time (4 teammates com plan-mode)
   ↓
Spec A: lead lê TASKS.md, dispatch
   ↓
Spec A: teammate planeja → reviewer aprova → teammate implementa
   ↓
Spec A: TaskCompleted hook gate (pytest + ruff)
   ↓
Spec B: lead push branch, abre PR draft em development
   ↓
Spec B: CI roda (pre-commit, tests, sonarqube)
   ↓
Spec B: reviewer aprova via gh pr review, merge --squash --auto
   ↓
Spec C: lead atualiza LOOP_STATUS, ScheduleWakeup próximo ciclo
   ↓
Loop repete até stop condition
```
