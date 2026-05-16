# Plan A — Multi-agent Team Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a 4-teammate Agent Team (2 backend-dev, 1 frontend-dev, 1 reviewer) into `.claude/` so that running Claude Code in this repo can spawn the team and process `TASKS.md` items end-to-end locally (no GitHub interaction — that's Plan B).

**Architecture:** Subagent definitions in `.claude/agents/`, hooks in `.claude/hooks/`, team prompt in `.claude/prompts/`, settings wired in `.claude/settings.json`. Plan-mode enforced for all teammates. Worktree isolation per task via `isolation: worktree` frontmatter. TaskCompleted hook gates on `pytest -q && ruff check && ruff format --check`.

**Tech Stack:** Claude Code v2.1.143 (Agent Teams experimental flag), bash hooks, git worktrees, pytest, ruff, Python 3.13.

**Spec reference:** `docs/superpowers/specs/2026-05-15-multi-agent-team-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `.claude/settings.json` | Enables Agent Teams, sets teammate mode, wires hooks |
| `.claude/agents/backend-dev.md` | Subagent definition for Python backend implementation |
| `.claude/agents/frontend-dev.md` | Subagent definition for React/TS frontend implementation |
| `.claude/agents/reviewer.md` | Subagent definition for read-only plan and code reviews |
| `.claude/hooks/task-completed-gate.sh` | Gates `TaskCompleted` event on pytest+ruff |
| `.claude/hooks/teammate-idle-nudge.sh` | Keeps teammate working if TASKS.md still has items |
| `.claude/prompts/start-team.md` | Lead prompt to spawn the team |
| `.worktreeinclude` | Copies `.env` and `.env.local` into worktrees |
| `.gitignore` | Adds `.claude/worktrees/` (and forward-compat entries for Plans B/C) |
| `TASKS.md` | Annotate Phase 6+ items with `(depends: X)` markers |

`.claude/settings.local.json` is **not** touched — it holds your local permissions.

---

### Task 1: Update `.gitignore` (forward-compat for B and C)

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read current .gitignore tail**

Run: `tail -20 .gitignore`

- [ ] **Step 2: Append worktree + loop state entries**

Append to `.gitignore`:

```gitignore

# Claude Code agent team (Spec A/B/C)
.claude/worktrees/
STOP_TEAM
outputs/LOOP_STATUS.md
```

- [ ] **Step 3: Verify entries present**

Run: `grep -E '(.claude/worktrees|STOP_TEAM|LOOP_STATUS)' .gitignore`
Expected: 3 lines printed.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore agent team runtime artifacts"
```

---

### Task 2: Create `.worktreeinclude`

**Files:**
- Create: `.worktreeinclude`

- [ ] **Step 1: Check if .env files exist**

Run: `ls -la .env .env.local 2>/dev/null`
If neither exists, proceed anyway — `.worktreeinclude` is gracefully no-op for absent files.

- [ ] **Step 2: Write file**

Create `.worktreeinclude` with:

```
.env
.env.local
```

- [ ] **Step 3: Verify**

Run: `cat .worktreeinclude`
Expected: 2 lines as above.

- [ ] **Step 4: Commit**

```bash
git add .worktreeinclude
git commit -m "chore: copy local env files into agent worktrees"
```

---

### Task 3: Create backend-dev subagent definition

**Files:**
- Create: `.claude/agents/backend-dev.md`
- Test: ad-hoc Python parse check

- [ ] **Step 1: Write the parse-validation test**

Create a one-shot validator (don't commit this — discard after Task 6):

```bash
cat > /tmp/validate-agents.py <<'PYEOF'
import sys, pathlib, re
import yaml

agent_dir = pathlib.Path(".claude/agents")
required = {"backend-dev", "frontend-dev", "reviewer"}
found = set()

for path in agent_dir.glob("*.md"):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        print(f"FAIL: {path} missing frontmatter"); sys.exit(1)
    fm = yaml.safe_load(m.group(1))
    for key in ("name", "description", "tools", "model"):
        if key not in fm:
            print(f"FAIL: {path} missing key '{key}'"); sys.exit(1)
    found.add(fm["name"])

missing = required - found
if missing:
    print(f"FAIL: missing agents {missing}"); sys.exit(1)
print(f"OK: validated {sorted(found)}")
PYEOF
```

- [ ] **Step 2: Run test to verify it fails (no agents yet)**

Run: `python /tmp/validate-agents.py`
Expected: FAIL — `missing agents {'backend-dev', 'frontend-dev', 'reviewer'}`

- [ ] **Step 3: Create the backend-dev definition**

Create `.claude/agents/backend-dev.md`:

```markdown
---
name: backend-dev
description: Implements Python backend code for VisionForge — blocks, core modules, models, utils. Follows Pydantic v2 + loguru + strict typing patterns established in src/visionforge/.
tools: Read, Edit, Write, Glob, Grep, Bash
model: sonnet
isolation: worktree
---

You are a backend-dev on the VisionForge agent team.

Before implementing:

1. Read CLAUDE.md and follow §9 (code writing rules) strictly: single-line docstrings on public classes/functions, `Args:`/`Returns:` only when non-obvious, no AI-flavored prose, comments explain *why* not *what*.
2. Examine existing patterns:
   - New `ExperimentBlock`? Read `src/visionforge/blocks/classification.py` first.
   - New Pydantic config? Read `src/visionforge/utils/config.py`.
   - New trainer/evaluator? Read `src/visionforge/core/trainer.py` and `src/visionforge/core/evaluator.py`.
3. Write tests first (TDD). Tests live in `tests/<module>/test_*.py`. Every public branch needs a test.

When the lead approves your plan and you exit plan-mode:

- Implement on your worktree branch.
- Use `pin_memory=False` and `num_workers=0` in test configs (CI parity).
- For paths that must exist in tests, use `tmp_path` fixture.

When done:

- Run `pytest -q` from your worktree root.
- Run `ruff check . && ruff format --check .`.
- SendMessage to the lead with: (1) one-line summary, (2) list of modified files, (3) test output tail.

You are not responsible for git pushes, PRs, or merges. The lead handles those.
```

- [ ] **Step 4: Run test to verify partial pass**

Run: `python /tmp/validate-agents.py`
Expected: FAIL — `missing agents {'frontend-dev', 'reviewer'}` (backend-dev no longer in the missing set, but test only passes when all 3 exist).

- [ ] **Step 5: Commit**

```bash
git add .claude/agents/backend-dev.md
git commit -m "feat: add backend-dev subagent definition"
```

---

### Task 4: Create frontend-dev subagent definition

**Files:**
- Create: `.claude/agents/frontend-dev.md`

- [ ] **Step 1: Write the file**

Create `.claude/agents/frontend-dev.md`:

```markdown
---
name: frontend-dev
description: Implements React 18 + TypeScript + Tailwind v4 + shadcn/ui frontend for VisionForge GUI. Visual reference is frontend-design/ mockups with per-task color theming.
tools: Read, Edit, Write, Glob, Grep, Bash
model: sonnet
isolation: worktree
---

You are a frontend-dev on the VisionForge agent team.

Before implementing:

1. Read `frontend-design/VisionForge.html` and the `*.jsx` mockups in `frontend-design/` for the visual reference. Match the per-task color theming exactly:
   - classification → red (`oklch(0.74 0.18 22)`)
   - detection → green (`oklch(0.78 0.18 150)`)
   - regression → blue (`oklch(0.74 0.16 240)`)
   - segmentation → violet (`oklch(0.74 0.18 305)`)
2. Examine existing patterns in `frontend/src/`. Stack is React 18 + TypeScript + Vite + Tailwind v4 + shadcn/ui. Components live under `frontend/src/components/`.
3. The API client lives in `frontend/src/api/client.ts` — extend it instead of duplicating fetch logic.
4. Frontend test infrastructure is minimal today. If you add testable logic, scope your tests to pure functions (no DOM) until a proper test setup is added.

When done:

- Run `cd frontend && npm run build` to verify the build passes.
- If you added testable logic, run whatever test command the new test setup provides.
- SendMessage to the lead with: (1) one-line summary, (2) list of modified files, (3) build output tail.

You are not responsible for git pushes, PRs, or merges. The lead handles those.
```

- [ ] **Step 2: Run test to check progress**

Run: `python /tmp/validate-agents.py`
Expected: FAIL — `missing agents {'reviewer'}`

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/frontend-dev.md
git commit -m "feat: add frontend-dev subagent definition"
```

---

### Task 5: Create reviewer subagent definition

**Files:**
- Create: `.claude/agents/reviewer.md`

- [ ] **Step 1: Write the file**

Create `.claude/agents/reviewer.md`:

```markdown
---
name: reviewer
description: Reviews plans and implementations from teammates. Does NOT write code. Only reports findings.
tools: Read, Glob, Grep, Bash
model: opus
---

You are the reviewer on the VisionForge agent team.

You review two things, and only two things:

1. **Plans** (before implementation): focus on
   - Scope: is the teammate doing too much? Pulling in unrelated refactors?
   - Patterns: does the plan follow existing conventions? (read the referenced files in `src/visionforge/`)
   - Risks: what if X is None? what if no GPU? what if pin_memory=True with num_workers=0?
   - CLAUDE.md §9 alignment: is the plan respecting code writing rules?

2. **Implementations** (after `TaskCompleted` gate passes): focus on
   - Correctness: do tests actually cover the public branches?
   - Regression risk: did this change something other modules depend on? (use `grep` to check imports)
   - Test quality: are tests using real paths via `tmp_path`, not mocks where mocks lie?
   - CLAUDE.md §9 compliance: comments explain *why*? No AI-flavored prose?

ALWAYS respond in exactly this format (the lead parses your reply):

- `APPROVE` — no changes needed
- `REJECT: <numbered reasons>` — required changes, each reason actionable

Never implement fixes. Never write code. Never edit files. If you see a fix is obvious, name it in the REJECT reasons — let the teammate do it.

When asked to do final review of a PR diff (Plan B), use `gh pr diff <number>` to read the changes. Use the same APPROVE/REJECT format. After APPROVE for a PR, the lead will instruct you to run `gh pr review <number> --approve`.
```

- [ ] **Step 2: Run test to verify all 3 agents validate**

Run: `python /tmp/validate-agents.py`
Expected: `OK: validated ['backend-dev', 'frontend-dev', 'reviewer']`

- [ ] **Step 3: Cleanup validator**

Run: `rm /tmp/validate-agents.py`

- [ ] **Step 4: Commit**

```bash
git add .claude/agents/reviewer.md
git commit -m "feat: add reviewer subagent definition"
```

---

### Task 6: Create TaskCompleted hook (with shell test)

**Files:**
- Create: `.claude/hooks/task-completed-gate.sh`

- [ ] **Step 1: Write the failing shell test**

Create `/tmp/test-task-completed-gate.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail

GATE=".claude/hooks/task-completed-gate.sh"

# Test 1: script exists and is executable
if [ ! -x "$GATE" ]; then
  echo "FAIL: $GATE not executable"; exit 1
fi

# Test 2: passes in a clean repo (pytest+ruff should pass since repo is healthy)
cd "$(git rev-parse --show-toplevel)"
if ! CLAUDE_TEAMMATE_CWD="$PWD" bash "$GATE" >/tmp/gate-out.txt 2>&1; then
  echo "FAIL: gate failed in clean repo:"; cat /tmp/gate-out.txt; exit 1
fi

# Test 3: fails when a syntax error is introduced
echo "this is not valid python =" > src/visionforge/__broken.py
if CLAUDE_TEAMMATE_CWD="$PWD" bash "$GATE" >/tmp/gate-out.txt 2>&1; then
  rm src/visionforge/__broken.py
  echo "FAIL: gate passed despite broken Python"; exit 1
fi
rm src/visionforge/__broken.py

echo "OK: task-completed-gate.sh works"
```

- [ ] **Step 2: Run test to verify it fails (hook doesn't exist)**

Run: `bash /tmp/test-task-completed-gate.sh`
Expected: FAIL — `.claude/hooks/task-completed-gate.sh not executable`

- [ ] **Step 3: Write the hook script**

Create `.claude/hooks/task-completed-gate.sh`:

```bash
#!/usr/bin/env bash
# Gates Claude Code TaskCompleted event for VisionForge agent team.
# Exits 2 with stderr if pytest or ruff fail, blocking task completion.

set -uo pipefail

WORKTREE_DIR="${CLAUDE_TEAMMATE_CWD:-$PWD}"
cd "$WORKTREE_DIR" || { echo "ERROR: cannot enter $WORKTREE_DIR" >&2; exit 2; }

echo "[gate] running pytest in $WORKTREE_DIR"
if ! pytest -q 2>&1 | tail -30; then
  echo "[gate] BLOCKED: pytest failed" >&2
  exit 2
fi

echo "[gate] running ruff check"
if ! ruff check . 2>&1 | tail -20; then
  echo "[gate] BLOCKED: ruff check failed" >&2
  exit 2
fi

echo "[gate] running ruff format --check"
if ! ruff format --check . 2>&1 | tail -20; then
  echo "[gate] BLOCKED: ruff format failed (run 'ruff format .' to fix)" >&2
  exit 2
fi

echo "[gate] PASS"
exit 0
```

- [ ] **Step 4: Make executable**

Run: `chmod +x .claude/hooks/task-completed-gate.sh`

- [ ] **Step 5: Run the shell test**

Run: `bash /tmp/test-task-completed-gate.sh`
Expected: `OK: task-completed-gate.sh works`

- [ ] **Step 6: Cleanup**

Run: `rm /tmp/test-task-completed-gate.sh /tmp/gate-out.txt`

- [ ] **Step 7: Commit**

```bash
git add .claude/hooks/task-completed-gate.sh
git commit -m "feat: add TaskCompleted gate hook (pytest + ruff)"
```

---

### Task 7: Create TeammateIdle hook (with shell test)

**Files:**
- Create: `.claude/hooks/teammate-idle-nudge.sh`

- [ ] **Step 1: Write the failing shell test**

Create `/tmp/test-teammate-idle.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail

NUDGE=".claude/hooks/teammate-idle-nudge.sh"
TASKS="TASKS.md"

if [ ! -x "$NUDGE" ]; then
  echo "FAIL: $NUDGE not executable"; exit 1
fi

# Test 1: TASKS.md has unchecked items → exit 2 (keep working)
if bash "$NUDGE" >/dev/null 2>&1; then
  echo "FAIL: expected exit 2 when TASKS.md has unchecked items"; exit 1
fi

# Test 2: simulate fully-checked TASKS → exit 0
cp "$TASKS" "${TASKS}.bak"
sed -i 's/^- \[ \]/- [x]/' "$TASKS"
if ! bash "$NUDGE" >/dev/null 2>&1; then
  mv "${TASKS}.bak" "$TASKS"
  echo "FAIL: expected exit 0 with all tasks checked"; exit 1
fi
mv "${TASKS}.bak" "$TASKS"

echo "OK: teammate-idle-nudge.sh works"
```

- [ ] **Step 2: Run test to verify it fails (hook doesn't exist)**

Run: `bash /tmp/test-teammate-idle.sh`
Expected: FAIL — `.claude/hooks/teammate-idle-nudge.sh not executable`

- [ ] **Step 3: Write the hook script**

Create `.claude/hooks/teammate-idle-nudge.sh`:

```bash
#!/usr/bin/env bash
# Re-prompts an idle teammate if TASKS.md still has unchecked items.
# Exit 2 with stdout = prompt for the teammate; exit 0 lets shutdown proceed.

set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

if [ ! -f TASKS.md ]; then
  exit 0
fi

REMAINING=$(grep -c '^- \[ \]' TASKS.md 2>/dev/null || echo 0)

if [ "$REMAINING" -gt 0 ]; then
  echo "TASKS.md still has $REMAINING unchecked items. Claim the next independent task from the team task list (one without unresolved \`(depends: X)\` markers). If no task is available right now, message the lead and wait."
  exit 2
fi

exit 0
```

- [ ] **Step 4: Make executable**

Run: `chmod +x .claude/hooks/teammate-idle-nudge.sh`

- [ ] **Step 5: Run the shell test**

Run: `bash /tmp/test-teammate-idle.sh`
Expected: `OK: teammate-idle-nudge.sh works`

- [ ] **Step 6: Cleanup**

Run: `rm /tmp/test-teammate-idle.sh`

- [ ] **Step 7: Commit**

```bash
git add .claude/hooks/teammate-idle-nudge.sh
git commit -m "feat: add TeammateIdle nudge hook"
```

---

### Task 8: Create `.claude/settings.json` (Agent Teams config + hooks)

**Files:**
- Create: `.claude/settings.json`

- [ ] **Step 1: Write JSON validation test**

```bash
cat > /tmp/test-settings.py <<'PYEOF'
import json, sys, pathlib

p = pathlib.Path(".claude/settings.json")
if not p.exists():
    print(f"FAIL: {p} does not exist"); sys.exit(1)

data = json.loads(p.read_text(encoding="utf-8"))

# Required keys
checks = [
    ("env.CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS", data.get("env", {}).get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS") == "1"),
    ("teammateMode", data.get("teammateMode") == "in-process"),
    ("worktree.baseRef", data.get("worktree", {}).get("baseRef") == "head"),
    ("hooks.TaskCompleted", isinstance(data.get("hooks", {}).get("TaskCompleted"), list) and len(data["hooks"]["TaskCompleted"]) > 0),
    ("hooks.TeammateIdle", isinstance(data.get("hooks", {}).get("TeammateIdle"), list) and len(data["hooks"]["TeammateIdle"]) > 0),
]

for name, ok in checks:
    if not ok:
        print(f"FAIL: {name} missing or wrong"); sys.exit(1)

print("OK: settings.json valid")
PYEOF
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python /tmp/test-settings.py`
Expected: FAIL — file doesn't exist.

- [ ] **Step 3: Write the settings**

Create `.claude/settings.json`:

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
      {
        "hooks": [
          { "type": "command", "command": "bash .claude/hooks/task-completed-gate.sh" }
        ]
      }
    ],
    "TeammateIdle": [
      {
        "hooks": [
          { "type": "command", "command": "bash .claude/hooks/teammate-idle-nudge.sh" }
        ]
      }
    ]
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python /tmp/test-settings.py`
Expected: `OK: settings.json valid`

- [ ] **Step 5: Cleanup**

Run: `rm /tmp/test-settings.py`

- [ ] **Step 6: Commit**

```bash
git add .claude/settings.json
git commit -m "feat: enable Agent Teams and wire team hooks in project settings"
```

---

### Task 9: Create start-team prompt template

**Files:**
- Create: `.claude/prompts/start-team.md`

- [ ] **Step 1: Create the directory**

Run: `mkdir -p .claude/prompts`

- [ ] **Step 2: Write the prompt**

Create `.claude/prompts/start-team.md`:

```markdown
You are the tech lead of the VisionForge agent team.

## Spawn

Spawn an agent team with:
- 2 teammates using the `backend-dev` subagent definition (names: `bk-1`, `bk-2`)
- 1 teammate using the `frontend-dev` subagent definition (name: `fr-1`)
- 1 teammate using the `reviewer` subagent definition (name: `rv-1`)

**Require plan approval** for ALL teammates before implementation. (This means each implementer enters plan-mode by default and waits for your approval after they send a plan.)

## Loop

Repeat until a stop condition fires:

1. Read `TASKS.md`. Find the lowest Phase header with at least one `- [ ]` item.
2. List items in that Phase that are independent. An item is independent if it has no `(depends: X)` marker, or if all listed `X` are already `- [x]`.
3. For each independent unclaimed item, dispatch to a free teammate:
   - Backend items (`src/visionforge/**`, `tests/**`, `configs/**`) → `bk-1` or `bk-2`, whichever is idle.
   - Frontend items (`frontend/**`, `frontend-design/**`) → `fr-1`.
   - Spawn task in the team task list. Teammate enters plan-mode.
4. When a teammate sends a plan via SendMessage:
   - Forward the plan to `rv-1` with: "Review this plan for <task>. Respond APPROVE or REJECT: <reasons>."
   - When `rv-1` replies, decide approve/reject in Claude Code's plan approval UI.
   - If rejected, message the teammate with the reasons. Teammate revises in plan-mode and resubmits.
   - 3 rejected plans on the same task → escalate (stop the team, write summary, ask the human).
5. After the teammate exits plan-mode and marks the task as completed, the `TaskCompleted` hook automatically runs `pytest` + `ruff`. If it fails (exit 2), the teammate must fix and re-complete.
6. Once the gate passes, ask `rv-1` for final review: "Review the diff for <task> in the worktree at <path>. Respond APPROVE or REJECT."
   - If APPROVE: commit on the worktree branch with a Conventional Commits message (`feat:`, `fix:`, `test:`, etc.). Mark `- [x]` in `TASKS.md`.
   - If REJECT: message the teammate with reasons. They fix and complete again, triggering the gate and re-review.

## Stop conditions

End the team gracefully when:
- The current Phase has zero `- [ ]` items, **OR**
- The same task fails 3 times (plan rejected or gate failed in a row) — escalate, **OR**
- The human messages you to stop.

After stopping, run `clean up the team` so the team config and tmux/in-process state are removed cleanly.

## What you do NOT do

- Do not implement code yourself. Delegate every implementation to teammates.
- Do not push to remote, open PRs, or merge. That comes in Plan B.
- Do not modify `TASKS.md` except to mark `- [x]` on completed items.
- Do not touch `main` branch.

## Coordination notes

- Teammates work in their own worktrees under `.claude/worktrees/<auto-name>/`. The reviewer reads worktrees by path you provide.
- If two backend items touch the same file, dispatch them serially (one finishes, then the next). Don't rely on the worktree to magically merge — that's a Plan B concern.
- Phase 5 has 7 independent blocks → at most 2 backends + 1 frontend in flight at once (capacity).
```

- [ ] **Step 3: Verify file**

Run: `wc -l .claude/prompts/start-team.md`
Expected: ~50 lines.

- [ ] **Step 4: Commit**

```bash
git add .claude/prompts/start-team.md
git commit -m "feat: add start-team lead prompt template"
```

---

### Task 10: Annotate Phase 6+ in `TASKS.md` with `(depends: X)` markers

**Files:**
- Modify: `TASKS.md`

This is a one-time content edit so the lead can compute independence. Phase 5 items are already mutually independent — no markers needed there.

- [ ] **Step 1: Read current Phase 6-9**

Run: `sed -n '/^## Phase 6/,/^## Phase 9/p' TASKS.md`

- [ ] **Step 2: Edit Phase 6 (Regression)**

In `TASKS.md`, change Phase 6 block to:

```markdown
## Phase 6 — Regression task

- [ ] `RegressionConfig` Pydantic models
- [ ] `RegressionTrainer` with MSE/MAE/R² metrics (depends: RegressionConfig)
- [ ] Regression blocks (GridSearch, KFold, etc.) (depends: RegressionTrainer)
- [ ] Regression tab in GUI (depends: RegressionTrainer)
```

- [ ] **Step 3: Edit Phase 7 (Detection)**

Change Phase 7 block to:

```markdown
## Phase 7 — Object Detection task

- [ ] `DetectionConfig` Pydantic models
- [ ] Model support: YOLO, Faster R-CNN, SSD (depends: DetectionConfig)
- [ ] `DetectionTrainer` with mAP, IoU metrics (depends: DetectionConfig)
- [ ] Detection blocks (depends: DetectionTrainer)
- [ ] Detection tab in GUI (depends: DetectionTrainer)
```

- [ ] **Step 4: Edit Phase 8 (Segmentation)**

Change Phase 8 block to:

```markdown
## Phase 8 — Segmentation task

- [ ] `SegmentationConfig` Pydantic models
- [ ] Model support: U-Net, DeepLab (depends: SegmentationConfig)
- [ ] `SegmentationTrainer` with IoU, Dice metrics (depends: SegmentationConfig)
- [ ] Segmentation blocks (depends: SegmentationTrainer)
- [ ] Segmentation tab in GUI (depends: SegmentationTrainer)
```

- [ ] **Step 5: Edit Phase 9 (Anomaly Detection)**

Change Phase 9 block to:

```markdown
## Phase 9 — Anomaly Detection task

- [ ] `AnomalyConfig` Pydantic models
- [ ] Model support: Autoencoder, PatchCore (depends: AnomalyConfig)
- [ ] `AnomalyTrainer` with AUROC, threshold metrics (depends: AnomalyConfig)
- [ ] Anomaly Detection blocks (depends: AnomalyTrainer)
- [ ] Anomaly Detection tab in GUI (depends: AnomalyTrainer)
```

- [ ] **Step 6: Verify `(depends:` markers are present**

Run: `grep -c '(depends:' TASKS.md`
Expected: 16 (4 in Phase 6, 4 in Phase 7, 4 in Phase 8, 4 in Phase 9).

- [ ] **Step 7: Commit**

```bash
git add TASKS.md
git commit -m "docs: annotate Phase 6-9 tasks with explicit dependencies"
```

---

### Task 11: Smoke test (manual)

This task is **manual verification** — no checkbox to mark done programmatically. It's the proof Plan A works end-to-end.

**Pre-flight check:**

```bash
# Verify all 8 files exist
ls .claude/settings.json \
   .claude/agents/backend-dev.md \
   .claude/agents/frontend-dev.md \
   .claude/agents/reviewer.md \
   .claude/hooks/task-completed-gate.sh \
   .claude/hooks/teammate-idle-nudge.sh \
   .claude/prompts/start-team.md \
   .worktreeinclude
```

Expected: all 8 paths listed.

**Smoke run:**

In a fresh terminal at the repo root:

```bash
claude --dangerously-skip-permissions
```

Inside Claude Code:

```
Read .claude/prompts/start-team.md and follow it exactly. Use TASKS.md as the source of work.
```

**Expected observable behavior:**

1. Lead reads the prompt and TASKS.md.
2. Lead identifies Phase 5 as the current Phase (Phase 1-4 are checked).
3. Lead spawns 4 teammates: `bk-1`, `bk-2`, `fr-1`, `rv-1`. You can verify with Shift+Down to cycle through teammates.
4. Lead dispatches 2 independent items from Phase 5 (e.g., `GridSearchBlock` and `RandomSearchBlock`) to `bk-1` and `bk-2`.
5. Each backend teammate creates a worktree under `.claude/worktrees/<name>/` (verify with `git worktree list`).
6. Each teammate plans (read-only) and sends plan to lead via SendMessage.
7. Lead forwards plan to `rv-1`. `rv-1` responds APPROVE or REJECT.
8. Lead approves the plan in Claude Code's UI.
9. Teammate implements. On marking task done, the `TaskCompleted` hook fires `pytest` + `ruff`.
10. If gate passes, lead asks `rv-1` for final review, then commits on the worktree branch with conventional commits.
11. Lead marks `- [x]` in `TASKS.md` (commit on `main`).

**Stop the smoke after 1 task completes** — full Phase 5 will run in real loop with Plan C.

Stop with: `Stop the team. Run cleanup.`

**Smoke failure modes and what to check:**

| Symptom | Where to look |
|---|---|
| Lead doesn't spawn teammates | `claude --version` ≥ 2.1.32; env var `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is loaded |
| Worktree creation fails | `.gitignore` has `.claude/worktrees/`; no leftover worktrees from prior runs (`git worktree list`) |
| `TaskCompleted` doesn't fire | hook file is executable (`ls -l .claude/hooks/`); `settings.json` references correct path |
| Reviewer can't read worktree | path provided by lead is absolute (`pwd` from worktree, not relative) |

---

## Self-review

**Spec coverage check:**

- §3 team composition (4 teammates with roles/models/tools) → Tasks 3, 4, 5 ✓
- §4 file layout → all 9 files created in Tasks 1-10 ✓
- §5 task flow → Task 9 prompt template captures the loop ✓
- §6 task selection (Phase + dependencies) → Task 10 adds `(depends:)` markers; Task 9 prompt explains the algorithm ✓
- §7 worktrees → `isolation: worktree` in Tasks 3 & 4 frontmatter; `.worktreeinclude` in Task 2; `.gitignore` in Task 1 ✓
- §8 hooks → Tasks 6, 7 ✓
- §9 settings.json → Task 8 ✓
- §10 subagent definitions → Tasks 3, 4, 5 ✓
- §11 lead prompt template → Task 9 ✓
- §12 stop conditions → encoded in Task 9 prompt ✓
- §13 risks/mitigations → covered by Tasks 1-10; no new code needed ✓
- §14 acceptance criteria → Task 11 smoke ✓

**Placeholder scan:** No TBDs except the explicit "frontend test infrastructure is minimal today" callout in Task 4 — that's a documented limitation, not a placeholder.

**Type/name consistency:** Teammate names (`bk-1`, `bk-2`, `fr-1`, `rv-1`) match between Task 9 prompt and Task 11 smoke. Hook script paths in settings.json (Task 8) match the files created in Tasks 6 and 7.

**Gap noted:** The smoke (Task 11) is manual and not auto-verifiable. That's intentional — Agent Teams interaction is interactive by design. Plan C will add programmatic monitoring via `LOOP_STATUS.md`.
