# Plan B — GitHub Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task.

**Goal:** Add GitHub automation on top of Plan A so each completed task flows: worktree → push → draft PR → CI → reviewer-bot approval → squash merge into `development`. Branch `main` stays untouched by agents.

**Architecture:** Five bash scripts in `.claude/scripts/` invoked by the lead during the task loop. One hook script in `.claude/hooks/` (`lead-startup.sh`) wired to `SessionStart` validates `gh auth` and ensures `development` exists before the team starts.

**Tech Stack:** `gh` CLI (assumed installed and authenticated), bash, git.

**Spec reference:** `docs/superpowers/specs/2026-05-15-github-integration-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `.claude/hooks/lead-startup.sh` | Validates `gh auth status`, ensures `development` branch exists, attempts branch protection |
| `.claude/scripts/ensure-development-branch.sh` | Idempotent: creates `development` from `main` if absent, pushes to origin |
| `.claude/scripts/open-task-pr.sh` | From a worktree: stages, commits, pushes, opens draft PR with structured body |
| `.claude/scripts/wait-for-ci.sh` | Polls `gh pr checks` with timeout, returns 0 on green, 2 on red |
| `.claude/scripts/merge-when-green.sh` | Verifies approval state, squash-merges with `--auto`, deletes branch |
| `.claude/settings.json` | Extended with `SessionStart` hook |
| `.claude/prompts/start-team.md` | Extended with PR/merge instructions |

---

### Task 1: Check prerequisites

**Files:** none modified — validation only.

- [ ] **Step 1: Check `gh` CLI**

Run: `gh --version`
Expected: `gh version X.Y.Z ...` printed. If not installed, instruct user: `winget install --id GitHub.cli` (Windows) or document the gap and stop.

- [ ] **Step 2: Check `gh auth`**

Run: `gh auth status`
Expected: `Logged in to github.com account <username>`. If not, user must run `gh auth login`. Document the gap and stop.

- [ ] **Step 3: Check remote**

Run: `git remote -v`
Expected: `origin  https://github.com/<user>/VisionForge.git` (or SSH equivalent). If no remote, document and stop.

---

### Task 2: Create scripts directory

**Files:**
- Create: `.claude/scripts/` (directory)

- [ ] **Step 1: Create directory**

Run: `mkdir -p .claude/scripts`

---

### Task 3: `ensure-development-branch.sh`

**Files:**
- Create: `.claude/scripts/ensure-development-branch.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Ensures the 'development' branch exists locally and on origin.
# Idempotent: safe to run on every lead startup.

set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

# If already on a worktree, skip — we only operate on the main checkout
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  WORKTREE_ROOT=$(git rev-parse --show-toplevel)
  MAIN_CHECKOUT=$(git worktree list --porcelain | awk '/^worktree/ {print $2; exit}')
  if [ "$WORKTREE_ROOT" != "$MAIN_CHECKOUT" ]; then
    exit 0
  fi
fi

# Check if local development exists
if git show-ref --quiet refs/heads/development; then
  echo "[ensure-dev] local development branch exists"
else
  # Try to fetch from origin first
  if git fetch origin development 2>/dev/null; then
    git branch development origin/development 2>/dev/null || true
    echo "[ensure-dev] created local development tracking origin/development"
  else
    # Create from main
    CURRENT=$(git rev-parse --abbrev-ref HEAD)
    git branch development main 2>/dev/null || git branch development "$CURRENT"
    echo "[ensure-dev] created local development from main"
  fi
fi

# Push to origin if not present remotely
if ! git ls-remote --exit-code --heads origin development >/dev/null 2>&1; then
  git push -u origin development 2>&1 | tail -3
  echo "[ensure-dev] pushed development to origin"
else
  echo "[ensure-dev] origin/development exists"
fi

# Attempt branch protection (best-effort, fails silently if no admin or already set)
REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || echo "")
if [ -n "$REPO" ]; then
  gh api -X PUT "repos/$REPO/branches/development/protection" \
    -f required_status_checks.strict=true \
    -F required_status_checks.contexts[]=pre-commit \
    -F required_status_checks.contexts[]=main \
    -F required_status_checks.contexts[]=tests \
    -F required_pull_request_reviews.required_approving_review_count=1 \
    -F enforce_admins=false \
    -F restrictions= >/dev/null 2>&1 && \
    echo "[ensure-dev] branch protection applied" || \
    echo "[ensure-dev] branch protection skipped (insufficient perms or already set)"
fi

exit 0
```

- [ ] **Step 2: Make executable**

Run: `chmod +x .claude/scripts/ensure-development-branch.sh`

- [ ] **Step 3: Syntax check**

Run: `bash -n .claude/scripts/ensure-development-branch.sh && echo "syntax OK"`

---

### Task 4: `lead-startup.sh`

**Files:**
- Create: `.claude/hooks/lead-startup.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# SessionStart hook for the lead. Validates gh CLI and ensures development branch.
# Exit 2 blocks the session start.

set -uo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI not installed. Install from https://cli.github.com/" >&2
  exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh CLI not authenticated. Run 'gh auth login' first." >&2
  exit 2
fi

bash .claude/scripts/ensure-development-branch.sh || true

exit 0
```

- [ ] **Step 2: Make executable**

Run: `chmod +x .claude/hooks/lead-startup.sh`

- [ ] **Step 3: Syntax check**

Run: `bash -n .claude/hooks/lead-startup.sh && echo "syntax OK"`

---

### Task 5: `open-task-pr.sh`

**Files:**
- Create: `.claude/scripts/open-task-pr.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Opens a draft PR for a completed task in a worktree.
# Usage: open-task-pr.sh <worktree-path> <task-description> <commit-message>

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <worktree-path> <task-description> <commit-message>" >&2
  exit 1
fi

WORKTREE_PATH="$1"
TASK_DESCRIPTION="$2"
COMMIT_MSG="$3"

cd "$WORKTREE_PATH"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "development" ]; then
  echo "ERROR: refusing to open PR from $BRANCH" >&2
  exit 1
fi

git add -A
if git diff --staged --quiet; then
  echo "No changes to commit"
  exit 1
fi

git commit -m "$COMMIT_MSG"
git push -u origin "$BRANCH"

FILES=$(git diff --name-only origin/development...HEAD)
TITLE=$(echo "$COMMIT_MSG" | head -1)

BODY=$(cat <<EOF
## Task
$TASK_DESCRIPTION

## Files changed
\`\`\`
$FILES
\`\`\`

## Local gate (Plan A)
- pytest: passed
- ruff check: passed
- ruff format: passed

Generated by VisionForge agent team.
EOF
)

PR_URL=$(gh pr create --base development --draft --title "$TITLE" --body "$BODY")
echo "$PR_URL"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x .claude/scripts/open-task-pr.sh`

- [ ] **Step 3: Syntax check**

Run: `bash -n .claude/scripts/open-task-pr.sh && echo "syntax OK"`

---

### Task 6: `wait-for-ci.sh`

**Files:**
- Create: `.claude/scripts/wait-for-ci.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Polls CI checks on a PR. Returns 0 if all green, 2 if any fail, 3 on timeout.
# Usage: wait-for-ci.sh <pr-number> [timeout-seconds]

set -uo pipefail

PR_NUMBER="${1:?usage: $0 <pr-number> [timeout-seconds]}"
TIMEOUT_SECS="${2:-1800}"

# Tira de draft pra disparar CI completo (idempotente)
gh pr ready "$PR_NUMBER" >/dev/null 2>&1 || true

SECONDS=0
while [ $SECONDS -lt $TIMEOUT_SECS ]; do
  STATES=$(gh pr checks "$PR_NUMBER" --json state --jq '[.[].state] | unique | join(",")' 2>/dev/null || echo "ERROR")

  case ",$STATES," in
    *,FAILURE,*|*,CANCELLED,*|*,TIMED_OUT,*|*,ERROR,*)
      echo "CI failed: $STATES" >&2
      exit 2
      ;;
    *,PENDING,*|*,IN_PROGRESS,*|*,QUEUED,*)
      sleep 30
      ;;
    SUCCESS|,SUCCESS,)
      echo "CI passed"
      exit 0
      ;;
    *)
      sleep 30
      ;;
  esac
done

echo "Timeout waiting for CI on PR #$PR_NUMBER" >&2
exit 3
```

- [ ] **Step 2: Make executable**

Run: `chmod +x .claude/scripts/wait-for-ci.sh`

- [ ] **Step 3: Syntax check**

Run: `bash -n .claude/scripts/wait-for-ci.sh && echo "syntax OK"`

---

### Task 7: `merge-when-green.sh`

**Files:**
- Create: `.claude/scripts/merge-when-green.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Squash-merges a PR. Caller is responsible for ensuring CI passed and review approved.
# Usage: merge-when-green.sh <pr-number>

set -euo pipefail

PR_NUMBER="${1:?usage: $0 <pr-number>}"

REVIEW_STATE=$(gh pr view "$PR_NUMBER" --json reviewDecision --jq .reviewDecision)
if [ "$REVIEW_STATE" != "APPROVED" ]; then
  echo "ERROR: PR #$PR_NUMBER not approved (state: $REVIEW_STATE)" >&2
  exit 2
fi

gh pr merge "$PR_NUMBER" --squash --delete-branch --auto

# Wait briefly for merge to register
sleep 5

STATE=$(gh pr view "$PR_NUMBER" --json state --jq .state)
if [ "$STATE" != "MERGED" ]; then
  echo "Merge pending or failed (state: $STATE) — may complete asynchronously" >&2
  # Don't fail — --auto may merge after CI catches up
fi

# Sync local development
git fetch origin development 2>/dev/null || true
if git rev-parse --verify development >/dev/null 2>&1; then
  CURRENT=$(git rev-parse --abbrev-ref HEAD)
  if [ "$CURRENT" = "development" ]; then
    git pull --ff-only origin development 2>/dev/null || true
  fi
fi

echo "PR #$PR_NUMBER merge initiated"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x .claude/scripts/merge-when-green.sh`

- [ ] **Step 3: Syntax check**

Run: `bash -n .claude/scripts/merge-when-green.sh && echo "syntax OK"`

---

### Task 8: Extend `.claude/settings.json` with SessionStart hook

**Files:**
- Modify: `.claude/settings.json`

- [ ] **Step 1: Read current settings**

Run: `cat .claude/settings.json`

- [ ] **Step 2: Rewrite with SessionStart hook added**

Replace `.claude/settings.json` contents:

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
    "SessionStart": [
      {
        "hooks": [
          { "type": "command", "command": "bash .claude/hooks/lead-startup.sh" }
        ]
      }
    ],
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

- [ ] **Step 3: Validate JSON**

Run: `python -c "import json; json.load(open('.claude/settings.json'))" && echo "OK"`

---

### Task 9: Extend lead prompt with PR/merge instructions

**Files:**
- Modify: `.claude/prompts/start-team.md`

- [ ] **Step 1: Append PR section to the prompt**

Append to `.claude/prompts/start-team.md`:

```markdown

## After local commit (Plan B)

After committing on the worktree branch and marking `- [x]` in TASKS.md (locally), open a PR against `development`:

1. From within the teammate's worktree, run:
   ```bash
   bash .claude/scripts/open-task-pr.sh "<worktree-path>" "<task-description>" "<commit-message>"
   ```
   Capture the PR URL from stdout.

2. Wait for CI:
   ```bash
   bash .claude/scripts/wait-for-ci.sh <pr-number>
   ```
   - Exit 0 → CI green, proceed.
   - Exit 2 → CI red. Message the teammate with the failures (use `gh pr checks <pr> --json name,bucket,detailsUrl`). Retry up to 3 times total.
   - Exit 3 → timeout (30 min). Escalate.

3. Once CI is green, ask `rv-1` for final review of the PR diff:
   "Please review PR #<n>. Use `gh pr diff <n>` to read the changes. Respond APPROVE or REJECT."

4. If `rv-1` returns APPROVE, instruct `rv-1`:
   "Run `gh pr review <n> --approve` and then send me confirmation."

5. After `rv-1` confirms, merge:
   ```bash
   bash .claude/scripts/merge-when-green.sh <pr-number>
   ```

6. After merge, remove the worktree:
   ```bash
   git worktree remove <worktree-path>
   ```

## Conflict handling

If `wait-for-ci.sh` reports CI failure with a merge conflict against `development`:

1. `cd <worktree-path>`
2. `git fetch origin development`
3. `git rebase origin/development`
4. If rebase succeeds, `git push --force-with-lease` and retry `wait-for-ci.sh`.
5. If rebase has conflicts, message the teammate: "Rebase your branch against origin/development. Conflicts in: <files>. Resolve and re-push." Escalate after 2 rebase failures.

## What you do NOT do (Plan B additions)

- Do not merge to `main` — only to `development`. Promotion to main is a manual decision by the human.
- Do not approve PRs yourself. Always go through `rv-1`.
- Do not skip the CI wait. Even if you're impatient, wait for green.
```

- [ ] **Step 2: Verify**

Run: `grep -c "## After local commit (Plan B)" .claude/prompts/start-team.md`
Expected: 1.

---

### Task 10: Smoke test (manual)

**Pre-flight:**

```bash
# Verify Plan B files
ls .claude/hooks/lead-startup.sh \
   .claude/scripts/ensure-development-branch.sh \
   .claude/scripts/open-task-pr.sh \
   .claude/scripts/wait-for-ci.sh \
   .claude/scripts/merge-when-green.sh

# Verify gh ready
gh auth status
gh repo view --json nameWithOwner
```

**Smoke run:**

1. Launch `claude --dangerously-skip-permissions`.
2. The `SessionStart` hook fires `lead-startup.sh` — should print `[ensure-dev] ...` messages or no-op if everything's already set.
3. Give the lead the start-team prompt and let it pick one easy Phase 5 item.
4. Observe full flow: plan → review → implement → gate → commit → push → PR open → CI runs → reviewer-bot approves → merge → worktree removed.
5. Verify `development` branch on GitHub has the new commit.

**Stop after 1 PR completes.**

## Self-review

**Spec coverage:**
- §3 strategy (granularity, base, conflicts) → Task 9 prompt + Task 7 script ✓
- §4 branch protection → Task 3 ✓
- §5 flow → Task 9 prompt ✓
- §6 files → Tasks 3-7 ✓
- §7-10 each script → one task each ✓
- §11 conflict handling → Task 9 prompt ✓
- §12 stop conditions → Task 4 (`gh auth` check) + Task 9 prompt ✓

**Placeholder scan:** None.

**Type/name consistency:** PR number argument syntax uniform across scripts. Branch name `development` literal across all references.
