# Plan C — Long-running Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task.

**Goal:** Wrap the Plan A+B team in a sustained loop (up to 10h) using the Ralph Loop plugin and `ScheduleWakeup`. Persist state in `outputs/LOOP_STATUS.md`. Handle Anthropic rate-limit by sleeping until reset. Stop on `STOP_TEAM` file, hard time limit, or completion.

**Architecture:** Lead is wrapped by `/loop` (Ralph Loop in dynamic mode). At each wake, lead runs `loop-cycle.sh` checks, reconciles open PRs, dispatches work via Plan A+B mechanics, updates `LOOP_STATUS.md`, schedules next wake via `ScheduleWakeup`. Team is re-spawned after sleep (Agent Teams limitation: no session resume for teammates).

**Tech Stack:** Ralph Loop plugin, `ScheduleWakeup` tool, bash, `gh` CLI.

**Spec reference:** `docs/superpowers/specs/2026-05-15-long-running-loop-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `.claude/scripts/check-stop-signal.sh` | Returns exit code: 0=continue, 1=STOP_TEAM exists, 2=elapsed≥10h |
| `.claude/scripts/update-status.sh` | Atomic write to `outputs/LOOP_STATUS.md` with current state |
| `.claude/scripts/loop-cycle.sh` | Top-level check: orchestrates stop-signal + reconcile open PRs |
| `outputs/LOOP_STATUS.md` | Created on first cycle; human-readable state snapshot |
| `.claude/prompts/start-team.md` | Extended with loop section: cycle protocol, rate-limit handling, ScheduleWakeup usage |
| `docs/runbook-agent-team.md` | Operator runbook: start, stop, monitor commands |

---

### Task 1: `check-stop-signal.sh`

**Files:**
- Create: `.claude/scripts/check-stop-signal.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Returns:
#   0 = continue (normal)
#   1 = STOP_TEAM file exists (user requested stop)
#   2 = elapsed time exceeds hard limit
# Prints reason to stdout when non-zero.

set -uo pipefail

HARD_LIMIT_SECONDS="${LOOP_HARD_LIMIT_SECONDS:-36000}"  # 10h default
STATUS_FILE="outputs/LOOP_STATUS.md"

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

if [ -f STOP_TEAM ]; then
  echo "STOP_TEAM file detected"
  exit 1
fi

if [ -f "$STATUS_FILE" ]; then
  FIRST_START=$(grep -E '^- \*\*First start:\*\*' "$STATUS_FILE" | head -1 | sed -E 's/.*\*\*First start:\*\* (.*)$/\1/')
  if [ -n "$FIRST_START" ]; then
    if FIRST_EPOCH=$(date -d "$FIRST_START" +%s 2>/dev/null); then
      NOW_EPOCH=$(date +%s)
      ELAPSED=$((NOW_EPOCH - FIRST_EPOCH))
      if [ "$ELAPSED" -ge "$HARD_LIMIT_SECONDS" ]; then
        echo "Elapsed $ELAPSED seconds exceeds hard limit $HARD_LIMIT_SECONDS"
        exit 2
      fi
    fi
  fi
fi

exit 0
```

- [ ] **Step 2: Make executable + syntax check**

```bash
chmod +x .claude/scripts/check-stop-signal.sh
bash -n .claude/scripts/check-stop-signal.sh && echo "syntax OK"
```

- [ ] **Step 3: Smoke test**

```bash
# Should exit 0 (no STOP_TEAM, no status)
bash .claude/scripts/check-stop-signal.sh; echo "exit=$?"
# Expected: exit=0

# STOP_TEAM check
touch STOP_TEAM
bash .claude/scripts/check-stop-signal.sh; echo "exit=$?"
# Expected: exit=1, "STOP_TEAM file detected"
rm STOP_TEAM
```

---

### Task 2: `update-status.sh`

**Files:**
- Create: `.claude/scripts/update-status.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Writes outputs/LOOP_STATUS.md atomically.
# Usage: update-status.sh <state> [extra-context-from-stdin]
#   state in: running | waiting_reset | done | escalated | paused_by_user
# Stdin (optional) is appended verbatim under "## Cycle context".

set -uo pipefail

STATE="${1:?usage: $0 <state>}"
STATUS_FILE="outputs/LOOP_STATUS.md"
TMP_FILE="$(mktemp)"

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 1
mkdir -p outputs

NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Preserve first start
if [ -f "$STATUS_FILE" ]; then
  FIRST_START=$(grep -E '^- \*\*First start:\*\*' "$STATUS_FILE" | head -1 | sed -E 's/.*\*\*First start:\*\* (.*)$/\1/')
else
  FIRST_START="$NOW"
fi

# Compute elapsed
if FIRST_EPOCH=$(date -d "$FIRST_START" +%s 2>/dev/null); then
  NOW_EPOCH=$(date +%s)
  ELAPSED_SECS=$((NOW_EPOCH - FIRST_EPOCH))
  H=$((ELAPSED_SECS / 3600))
  M=$(((ELAPSED_SECS % 3600) / 60))
  ELAPSED="${H}h ${M}m"
else
  ELAPSED="?"
fi

# Read context from stdin
CONTEXT=""
if [ ! -t 0 ]; then
  CONTEXT=$(cat)
fi

REMAINING=$(grep -c '^- \[ \]' TASKS.md 2>/dev/null || echo 0)

cat > "$TMP_FILE" <<EOF
# VisionForge Loop Status

- **State:** $STATE
- **First start:** $FIRST_START
- **Last update:** $NOW
- **Elapsed:** $ELAPSED
- **Unchecked items in TASKS.md:** $REMAINING

## Cycle context

$CONTEXT
EOF

mv "$TMP_FILE" "$STATUS_FILE"
echo "[status] wrote $STATUS_FILE (state=$STATE)"
```

- [ ] **Step 2: Make executable + syntax check**

```bash
chmod +x .claude/scripts/update-status.sh
bash -n .claude/scripts/update-status.sh && echo "syntax OK"
```

- [ ] **Step 3: Smoke test**

```bash
echo "cycle 1: started team" | bash .claude/scripts/update-status.sh running
cat outputs/LOOP_STATUS.md
# Expected: structured markdown with state=running, first start = now, context preserved
```

---

### Task 3: `loop-cycle.sh`

**Files:**
- Create: `.claude/scripts/loop-cycle.sh`

This is the convenience wrapper. The lead actually drives the cycle; this script gives it a deterministic check sequence.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Convenience wrapper run by the lead at the start of each cycle.
# Returns exit code reflecting overall cycle state:
#   0  = proceed normally
#   1  = stop (STOP_TEAM file)
#   2  = stop (elapsed >= 10h)
#   10 = waiting (rate-limit signal detected — caller schedules longer wake)

set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

# Check stop signal
bash .claude/scripts/check-stop-signal.sh
SS_EXIT=$?
case "$SS_EXIT" in
  0) ;;
  1)
    bash .claude/scripts/update-status.sh paused_by_user <<<"STOP_TEAM file detected. Lead should cleanup and exit."
    exit 1
    ;;
  2)
    bash .claude/scripts/update-status.sh done <<<"Hard time limit reached. Lead should cleanup and exit."
    exit 2
    ;;
esac

# Check rate-limit flag (set by lead in previous cycle on Anthropic 429)
if [ -f outputs/.rate_limit ]; then
  RESET_TS=$(cat outputs/.rate_limit 2>/dev/null || echo "")
  NOW_EPOCH=$(date +%s)
  RESET_EPOCH=$(date -d "$RESET_TS" +%s 2>/dev/null || echo 0)
  if [ "$NOW_EPOCH" -lt "$RESET_EPOCH" ]; then
    bash .claude/scripts/update-status.sh waiting_reset <<<"Rate-limit reset expected at $RESET_TS"
    exit 10
  else
    rm -f outputs/.rate_limit
  fi
fi

# Snapshot open PRs (Plan B integration)
OPEN_PRS=$(gh pr list --state=open --base development --json number,title,headRefName --jq 'length' 2>/dev/null || echo "?")

# Snapshot TASKS.md
REMAINING=$(grep -c '^- \[ \]' TASKS.md 2>/dev/null || echo 0)

if [ "$REMAINING" -eq 0 ] && [ "$OPEN_PRS" = "0" ]; then
  bash .claude/scripts/update-status.sh done <<<"All TASKS.md items completed. No open PRs. Loop terminating."
  exit 0
fi

bash .claude/scripts/update-status.sh running <<<"Cycle proceeding. Open PRs: $OPEN_PRS. Unchecked tasks: $REMAINING."
exit 0
```

- [ ] **Step 2: Make executable + syntax check**

```bash
chmod +x .claude/scripts/loop-cycle.sh
bash -n .claude/scripts/loop-cycle.sh && echo "syntax OK"
```

- [ ] **Step 3: Smoke test**

```bash
bash .claude/scripts/loop-cycle.sh; echo "exit=$?"
cat outputs/LOOP_STATUS.md
# Expected: exit=0, status updated to "running" with open PR / task counts
```

---

### Task 4: Extend `start-team.md` with loop instructions

**Files:**
- Modify: `.claude/prompts/start-team.md`

- [ ] **Step 1: Append loop section**

Append to `.claude/prompts/start-team.md`:

```markdown

## Loop mode (Plan C)

When invoked under `/loop`, you run in cycles. At each wake:

### Cycle start protocol

1. Run `bash .claude/scripts/loop-cycle.sh` and check the exit code:
   - `0` → proceed with the normal team loop.
   - `1` → user requested stop. Run `clean up the team` and exit without scheduling next wake.
   - `2` → elapsed ≥ 10h. Run `clean up the team` and exit without scheduling next wake.
   - `10` → rate-limit cooldown. Run `clean up the team` and call `ScheduleWakeup` with delay=18000 seconds (5h) and reason="awaiting Anthropic plan reset". Exit this cycle.

2. If proceed: check if the team exists. If not (first cycle, or post-sleep re-spawn), spawn 4 teammates per the Spawn section above. Teammates do not survive `/resume` per Agent Teams limitations.

### During the cycle

3. **Reconcile open PRs first**, before spawning new work. Run:
   ```bash
   gh pr list --state=open --base development --json number,title,headRefName
   ```
   For each open PR:
   - Run `bash .claude/scripts/wait-for-ci.sh <pr-number> 60` (short timeout — don't block the cycle).
   - If CI green and `gh pr view <n> --json reviewDecision` ≠ APPROVED, ask `rv-1` to review.
   - If approved + CI green, run `bash .claude/scripts/merge-when-green.sh <n>`.

4. Then proceed with normal Plan A + Plan B flow (dispatch new tasks).

### Rate-limit detection

If at any point you receive an Anthropic API error matching `rate_limit_error`, `429`, or `quota exceeded`:
1. Compute the reset timestamp: `date -u -d "+5 hours" +"%Y-%m-%dT%H:%M:%SZ"` and write to `outputs/.rate_limit`.
2. Stop dispatching new work.
3. Run `clean up the team`.
4. Call `ScheduleWakeup` with delay=18000s and reason="awaiting Anthropic plan reset".
5. Exit the cycle.

### Cycle end protocol

6. Run `bash .claude/scripts/update-status.sh running` with a stdin summary of this cycle:
   ```
   Cycle <N>: merged PR #X, started #Y, #Z. bk-1=planning, bk-2=idle.
   ```

7. Call `ScheduleWakeup`:
   - `delay=1800` (30 min), `reason="normal cycle continuation"`, `prompt="<<autonomous-loop-dynamic>>"`.

### How to stop the loop

The user stops the loop by:
- Creating `STOP_TEAM` file in repo root (lead picks up next cycle and exits).
- Letting the 10h hard limit expire.
- Letting `TASKS.md` empty out naturally (loop-cycle.sh detects and exits).
```

- [ ] **Step 2: Verify**

```bash
grep -c "## Loop mode (Plan C)" .claude/prompts/start-team.md
# Expected: 1
```

---

### Task 5: Operator runbook

**Files:**
- Create: `docs/runbook-agent-team.md`

- [ ] **Step 1: Write the runbook**

Create `docs/runbook-agent-team.md`:

```markdown
# Agent Team Operator Runbook

This is the operational guide for running the VisionForge agent team in long-running loop mode (Plans A + B + C).

## Start the loop

Open a fresh terminal at the repo root. Run:

\`\`\`bash
claude --dangerously-skip-permissions
\`\`\`

At the Claude prompt:

\`\`\`
/loop "You are the lead. Read .claude/prompts/start-team.md and follow it exactly. Run loop-cycle.sh at the start of every cycle. Stop when TASKS.md is empty, 10h elapsed, or STOP_TEAM exists."
\`\`\`

The `SessionStart` hook validates `gh auth` and ensures `development` branch exists before the team spawns.

## Monitor progress

In any other terminal:

\`\`\`bash
# Current loop state (updated every cycle)
cat outputs/LOOP_STATUS.md

# Open PRs from the agent team
gh pr list --base development --label agent-team

# Recent commits on development
git log development --oneline -20

# Live worktrees
git worktree list
\`\`\`

## Stop the loop

\`\`\`bash
echo "stop" > STOP_TEAM
\`\`\`

The lead picks this up on the next cycle (up to 30 min), cleans up the team, and exits.

To force stop immediately, kill the `claude` process. Then clean up any leftover worktrees:

\`\`\`bash
git worktree list
git worktree remove <path>  # for each
\`\`\`

## Promote development → main

This is **manual** by design. Agents never touch `main`.

\`\`\`bash
git checkout main
git pull origin main
git merge --ff-only origin/development
git push origin main
\`\`\`

If `--ff-only` fails, inspect `development` for any unwanted commits before merging.

## Common issues

| Symptom | Diagnosis | Fix |
|---|---|---|
| Lead won't start: "gh CLI not authenticated" | `lead-startup.sh` blocking | Run `gh auth login` |
| Loop stops with state=escalated | Same task failed 3 times | Read `outputs/LOOP_STATUS.md` for the failed task; fix manually or remove from TASKS.md |
| Worktrees pile up under `.claude/worktrees/` | Cleanup didn't run (crash) | `for d in .claude/worktrees/*/; do git worktree remove "$d" --force; done` |
| PR stuck in "draft" forever | `wait-for-ci.sh` exited early | Manually run `gh pr ready <n>` |
| Rate-limit loop seems stuck | `outputs/.rate_limit` may have wrong timestamp | `rm outputs/.rate_limit` to force retry next cycle |

## Tuning

- Hard time limit: edit `LOOP_HARD_LIMIT_SECONDS` env in your shell before starting the loop. Default 36000 (10h).
- Cycle interval: change the `delay=1800` in step 7 of the start-team prompt loop section.
- Rate-limit reset window: change the `+5 hours` computation in the rate-limit detection step.
```

---

### Task 6: Smoke test (manual)

**Pre-flight:**

```bash
ls .claude/scripts/check-stop-signal.sh \
   .claude/scripts/update-status.sh \
   .claude/scripts/loop-cycle.sh \
   docs/runbook-agent-team.md
```

**Smoke 1 — happy cycle:**

```bash
# In one terminal
claude --dangerously-skip-permissions
> /loop "You are the lead. Read .claude/prompts/start-team.md and follow it exactly. Run loop-cycle.sh at the start of every cycle."

# Wait ~2 minutes. In another terminal:
cat outputs/LOOP_STATUS.md
# Expected: state=running, elapsed > 0, cycle context populated

# Stop after observing 1 task being picked up
echo stop > STOP_TEAM
# Wait up to 30 min for next cycle, OR Ctrl+C the loop manually

# Verify cleanup
git worktree list  # should be empty or only main checkout
cat outputs/LOOP_STATUS.md  # state should be paused_by_user
rm STOP_TEAM
```

**Smoke 2 — hard limit (synthetic):**

```bash
LOOP_HARD_LIMIT_SECONDS=10 bash .claude/scripts/check-stop-signal.sh
echo "exit=$?"
# If outputs/LOOP_STATUS.md has an older first-start than 10s ago: exit=2
# If newly started: exit=0 — wait 15 seconds and re-run
```

**Smoke 3 — TASKS empty (synthetic):**

```bash
# Temporarily mark all as checked
cp TASKS.md TASKS.md.bak
sed -i 's/^- \[ \]/- [x]/' TASKS.md
bash .claude/scripts/loop-cycle.sh
cat outputs/LOOP_STATUS.md  # state should be 'done'
mv TASKS.md.bak TASKS.md
```

---

## Self-review

**Spec coverage:**
- §3 mechanism (Ralph Loop + ScheduleWakeup) → Task 4 prompt addition ✓
- §4 LOOP_STATUS format → Task 2 update-status.sh ✓
- §5 cycle flow (9 steps) → Tasks 1-3 + Task 4 prompt ✓
- §6 rate-limit detection → Task 3 + Task 4 prompt ✓
- §7 re-spawn after sleep → Task 4 prompt (cycle start protocol step 2) ✓
- §8 files → Tasks 1-3 ✓
- §9 start/stop/monitor → Task 5 runbook ✓
- §10 stop conditions table → covered by Tasks 1, 3, 4 ✓
- §11 acceptance criteria → Task 6 smoke ✓

**Placeholder scan:** None.

**Type consistency:** State enum values match between scripts and spec: `running | waiting_reset | done | escalated | paused_by_user`. Exit codes consistent: 0/1/2/10 in `loop-cycle.sh`.

**Gap noted:** `escalated` state is written by the lead in the start-team prompt (when 3 task failures happen), not by `update-status.sh` directly. The script accepts any state string, so this is OK.
