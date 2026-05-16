# Agent Team Operator Runbook

Guia operacional para rodar o time de agentes do VisionForge em loop sustentado (Plans A + B + C).

## Start the loop

Em um terminal limpo na raiz do repo:

```bash
claude --dangerously-skip-permissions
```

No prompt do Claude:

```
/loop "You are the lead. Read .claude/prompts/start-team.md and follow it exactly. Run loop-cycle.sh at the start of every cycle. Stop when TASKS.md is empty, 10h elapsed, or STOP_TEAM exists."
```

O hook `SessionStart` valida `gh auth` e garante que a branch `development` existe antes do time spawnar.

## Monitor

Em qualquer outro terminal:

```bash
# Estado atual do loop (atualizado a cada ciclo)
cat outputs/LOOP_STATUS.md

# PRs abertos pelo time
gh pr list --base development

# Commits recentes em development
git log development --oneline -20

# Worktrees ativas
git worktree list
```

## Stop

```bash
echo stop > STOP_TEAM
```

O lead pega no próximo ciclo (até 30 min), faz cleanup do time e sai.

Pra parar imediatamente, mate o processo `claude`. Depois limpe worktrees órfãs:

```bash
git worktree list
git worktree remove <path>  # pra cada uma
```

## Promote `development` → `main`

Isso é **manual** por design — agentes nunca tocam `main`.

```bash
git checkout main
git pull origin main
git merge --ff-only origin/development
git push origin main
```

Se `--ff-only` falhar, revise `development` antes de mergear.

## Problemas comuns

| Sintoma | Diagnóstico | Fix |
|---|---|---|
| Lead não inicia: "gh CLI not authenticated" | `lead-startup.sh` bloqueia | `gh auth login` |
| Loop termina com state=escalated | Mesma task falhou 3× | Veja `outputs/LOOP_STATUS.md`; corrija manualmente ou remova item do TASKS.md |
| Worktrees acumulam em `.claude/worktrees/` | Cleanup não rodou (crash) | `for d in .claude/worktrees/*/; do git worktree remove "$d" --force; done` |
| PR fica em draft pra sempre | `wait-for-ci.sh` saiu cedo | `gh pr ready <n>` manualmente |
| Loop parece travado em rate-limit | Timestamp errado em `outputs/.rate_limit` | `rm outputs/.rate_limit` pra forçar retry |

## Tuning

- **Hard time limit:** export `LOOP_HARD_LIMIT_SECONDS=<segundos>` antes de iniciar. Default 36000 (10h).
- **Intervalo entre ciclos:** muda `delay=1800` no passo 7 da seção "Loop mode (Plan C)" do `start-team.md`.
- **Janela de reset de rate-limit:** muda o `+5 hours` na seção "Rate-limit detection".
