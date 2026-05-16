# Spec A — Time multi-agente local

**Status:** Approved (2026-05-15)
**Author:** Marcus Reis (via brainstorming with Claude)
**Depends on:** —
**Blocks:** Spec B (GitHub integration), Spec C (Long-running loop)

---

## 1. Goal

Coordinate a 4-teammate Agent Team that picks unchecked items from `TASKS.md`, plans them under reviewer scrutiny, implements them in isolated git worktrees, and gates completion behind `pytest` + `ruff`. The lead session is the tech lead — it does not implement code. This spec covers **local execution only**; pushing, PRs, and CI-time gates are Spec B; long-running loops are Spec C.

## 2. Scope

**In scope:**
- Subagent definitions for backend-dev, frontend-dev, reviewer
- Lead behavior: parse `TASKS.md`, dispatch, gate, mark done
- Plan-mode requirement before implementation
- Worktree isolation per implementer task
- `TaskCompleted` and `TeammateIdle` hooks
- Stop conditions

**Out of scope:**
- Push to remote, branch `development`, PRs, merge automation → Spec B
- Coverage / pre-commit / SonarCloud gates → Spec B (CI handles them)
- Long-running loops, token-reset handling, escalation notifications → Spec C
- Promoting teammates to lead, nested teams (Agent Teams doesn't support it)

## 3. Team composition

| Role | Count | Model | Isolation | Tools |
|---|---|---|---|---|
| lead (tech lead implícito) | 1 | opus | none (workspace principal) | all |
| backend-dev | 2 | sonnet 4.6 | `worktree` per task | Read, Edit, Write, Glob, Grep, Bash |
| frontend-dev | 1 | sonnet 4.6 | `worktree` per task | Read, Edit, Write, Glob, Grep, Bash |
| reviewer | 1 | opus | none (reads worktrees by path) | Read, Glob, Grep, Bash |

Reviewer has no `Edit`/`Write` — only reports findings via `SendMessage`. Team coordination tools and `SendMessage` are always available regardless of the `tools` allowlist.

## 4. Files

```
.claude/
├── settings.json
├── agents/
│   ├── backend-dev.md
│   ├── frontend-dev.md
│   └── reviewer.md
├── hooks/
│   ├── task-completed-gate.sh
│   └── teammate-idle-nudge.sh
└── prompts/
    └── start-team.md

.gitignore                # add .claude/worktrees/
.worktreeinclude          # copy .env, .env.local into worktrees
```

## 5. Task flow

```
lead reads TASKS.md
  ↓
lead picks current Phase (lowest Phase with [ ] items)
  ↓
lead lists independent items (no `(depends: X)` or deps already done)
  ↓
lead spawns up to 2 backend-devs + 1 frontend-dev with plan-mode required
  ↓
each teammate plans in read-only mode
  ↓
teammate sends plan to lead
  ↓
lead forwards plan to reviewer
  ↓
reviewer responds APPROVE or REJECT: <reasons>
  ↓
lead approves/rejects plan in Claude Code (autonomous decision, informed by reviewer)
  ↓ (if rejected, teammate revises and resubmits)
teammate exits plan-mode, implements in its worktree
  ↓
teammate marks task completed
  ↓
TaskCompleted hook runs: pytest -q && ruff check && ruff format --check
  ↓                                     ↓
fail → exit 2, error to teammate        pass → lead asks reviewer for final review
                                         ↓
                                         APPROVE → lead commits on worktree branch,
                                                   marks [x] in TASKS.md
                                         REJECT  → teammate fixes, reviewer re-reviews
```

## 6. Task selection strategy

Lead reads `TASKS.md`, picks the **lowest Phase containing at least one `- [ ]`**. Lists items in file order. An item is independent unless it has `(depends: X)` where X is another item that isn't yet `- [x]`.

**Dependency syntax** (you author it in `TASKS.md`):

```markdown
- [ ] RegressionConfig Pydantic models
- [ ] RegressionTrainer (depends: RegressionConfig)
- [ ] Regression tab in GUI (depends: RegressionTrainer)
```

Items without `(depends: ...)` are treated as independent and can run in parallel.

When the Phase is fully checked, lead moves to the next Phase (in file order).

## 7. Worktrees

Each implementer uses `isolation: worktree` in its subagent frontmatter. Claude Code creates a worktree under `.claude/worktrees/<auto-name>/` on a branch `worktree-<auto-name>` at spawn, based on `head` (not `origin/main`) so the worktree carries the current branch state.

Reviewer does **not** use a worktree — it reads any worktree by absolute path provided by the lead.

`.worktreeinclude`:

```
.env
.env.local
```

`.gitignore` gets `.claude/worktrees/` appended so worktree contents don't appear as untracked files in the main checkout.

## 8. Hooks (settings.json wired)

### TaskCompleted gate (`.claude/hooks/task-completed-gate.sh`)

```bash
#!/usr/bin/env bash
set -uo pipefail

WORKTREE_DIR="${CLAUDE_TEAMMATE_CWD:-$PWD}"
cd "$WORKTREE_DIR" || { echo "cannot enter $WORKTREE_DIR"; exit 2; }

if ! pytest -q 2>&1 | tee /tmp/pytest-output.txt; then
  echo "TaskCompleted blocked: pytest failed"
  exit 2
fi

if ! ruff check . 2>&1; then
  echo "TaskCompleted blocked: ruff check failed"
  exit 2
fi

if ! ruff format --check . 2>&1; then
  echo "TaskCompleted blocked: ruff format failed (run 'ruff format .' to fix)"
  exit 2
fi

exit 0
```

Coverage, pre-commit, and SonarCloud are intentionally **not** here — they run in CI on PRs (Spec B).

### TeammateIdle nudge (`.claude/hooks/teammate-idle-nudge.sh`)

```bash
#!/usr/bin/env bash
set -uo pipefail

# Count unchecked items in TASKS.md
REMAINING=$(grep -c '^- \[ \]' TASKS.md 2>/dev/null || echo 0)

if [ "$REMAINING" -gt 0 ]; then
  echo "TASKS.md still has $REMAINING unchecked items. Claim the next independent task from the team task list."
  exit 2
fi

exit 0
```

## 9. settings.json

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "teammateMode": "in-process",
  "worktree": {
    "baseRef": "head"
  },
  "hooks": {
    "TaskCompleted": [
      { "hooks": [{ "type": "command", "command": "bash .claude/hooks/task-completed-gate.sh" }] }
    ],
    "TeammateIdle": [
      { "hooks": [{ "type": "command", "command": "bash .claude/hooks/teammate-idle-nudge.sh" }] }
    ]
  }
}
```

Windows note: hooks call `bash` explicitly because Git Bash is the project shell. On WSL the same scripts work unchanged.

## 10. Subagent definitions (full text in implementation plan)

### `.claude/agents/backend-dev.md`

```markdown
---
name: backend-dev
description: Implements Python backend code for VisionForge — blocks, core, models, utils.
tools: Read, Edit, Write, Glob, Grep, Bash
model: sonnet
isolation: worktree
---

You are a backend-dev on the VisionForge team. Before implementing:

1. Read CLAUDE.md and follow §9 (code writing rules).
2. Examine existing patterns. For new Blocks, read src/visionforge/blocks/classification.py as reference. For new Pydantic configs, read src/visionforge/utils/config.py.
3. Write tests first (TDD). Tests live in tests/<module>/test_*.py.
4. Coverage: every public branch in your new module must have a test.

When done:
- Run `pytest -q` in your worktree.
- Run `ruff check . && ruff format --check .`.
- SendMessage to the lead with a summary and the list of modified files.
```

### `.claude/agents/frontend-dev.md`

```markdown
---
name: frontend-dev
description: Implements React + TypeScript + Tailwind + shadcn frontend for VisionForge GUI.
tools: Read, Edit, Write, Glob, Grep, Bash
model: sonnet
isolation: worktree
---

You are a frontend-dev on the VisionForge team. Before implementing:

1. Read frontend-design/VisionForge.html and the *.jsx mockups for visual reference. Match the per-task color theming (classification=red, detection=green, regression=blue, segmentation=violet).
2. Examine existing patterns in frontend/src/. Stack is React 18 + TypeScript + Vite + Tailwind v4 + shadcn/ui.
3. Add tests where relevant (frontend test setup is TBD — if absent, scope is unit-tested logic only).

When done:
- Run `npm run build` in frontend/ to verify the build passes.
- Run `ruff` is not applicable; instead run any frontend linter present.
- SendMessage to the lead with a summary and the list of modified files.
```

### `.claude/agents/reviewer.md`

```markdown
---
name: reviewer
description: Reviews plans and implementations. Does NOT write code.
tools: Read, Glob, Grep, Bash
model: opus
---

You review two things:

1. **Plans** (before implementation): focus on scope (not over-engineering), patterns (follows conventions), risks (what if X is None? what if no GPU?).
2. **Implementations** (after): focus on correctness (do tests cover branches?), regression risk (did this change something other modules depend on?), and CLAUDE.md §9 compliance.

ALWAYS respond in this format:
- `APPROVE` — no changes needed
- `REJECT: <numbered reasons>` — required changes

Never implement fixes. Only report.
```

## 11. Lead prompt template (`.claude/prompts/start-team.md`)

```text
You are the tech lead of the VisionForge agent team.

Spawn an agent team with:
- 2 teammates using the backend-dev subagent (names: bk-1, bk-2)
- 1 teammate using the frontend-dev subagent (name: fr-1)
- 1 teammate using the reviewer subagent (name: rv-1)

Require plan approval for ALL teammates before implementation.

Read TASKS.md. Find the lowest Phase with unchecked items.
List independent items in that Phase. An item is independent if it
has no `(depends: X)` marker, or all listed X are already checked.

Dispatch independent items in parallel up to team capacity:
- Backend items → bk-1 or bk-2 (whichever is free)
- Frontend items → fr-1
- After teammate sends a plan, forward to rv-1 for review.
- rv-1 responds APPROVE or REJECT. Decide based on its feedback.
- After implementation completes and TaskCompleted gate passes, ask rv-1
  for a final review of the code. If APPROVE, commit on the worktree
  branch with a conventional commit message and mark `- [x]` in TASKS.md.
  If REJECT, message the teammate with the findings.

Stop when:
- The current Phase has zero unchecked items, OR
- The same task fails 3 times in a row (notify the human and stop), OR
- The human messages you to stop.

Do not implement code yourself. Delegate everything.
```

## 12. Stop conditions

Lead ends the team when:
- Current Phase fully checked, **OR**
- 3 consecutive failures on the same task (escalation), **OR**
- User explicitly messages "stop the team".

Time limits and token-reset handling are Spec C.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Two backends edit same file → silent overwrite | Each in own worktree; conflicts resolved at merge (Spec B), not in-memory |
| Reviewer approves bad plan, code lands broken | TaskCompleted hook runs pytest+ruff; teammate must fix before marking done |
| Lead implements code instead of delegating | Prompt explicitly says "Do not implement code yourself" |
| Frontend-dev idle during Phase 5 | Accepted — `TeammateIdle` lets it shut down gracefully when no frontend items remain |
| Plan-mode rejected loop (teammate ↔ reviewer) | Stop after 3 plan rejections on same task → escalation |
| Worktree leftover after crash | `cleanupPeriodDays` setting sweeps orphaned worktrees with no uncommitted changes |

## 14. Acceptance criteria

This spec is implemented when:

- [ ] `.claude/settings.json` enables Agent Teams and wires both hooks
- [ ] `.claude/agents/{backend-dev,frontend-dev,reviewer}.md` exist with the frontmatter shown in §10
- [ ] `.claude/hooks/task-completed-gate.sh` and `teammate-idle-nudge.sh` exist and are executable
- [ ] `.claude/prompts/start-team.md` exists with the template from §11
- [ ] `.worktreeinclude` exists with `.env` and `.env.local`
- [ ] `.gitignore` contains `.claude/worktrees/`
- [ ] Manual smoke test: launch lead with `claude --dangerously-skip-permissions`, give it the start-team prompt, verify it spawns 4 teammates and picks an item from Phase 5 of `TASKS.md`
- [ ] One end-to-end task completes (item from Phase 5) — plan approved, implementation done, gate passes, item checked in `TASKS.md`, commit visible on the worktree branch
