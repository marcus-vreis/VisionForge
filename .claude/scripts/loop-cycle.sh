#!/usr/bin/env bash
# Convenience wrapper run by the lead at the start of each cycle.
# Returns exit code reflecting overall cycle state:
#   0  = proceed normally
#   1  = stop (STOP_TEAM file)
#   2  = stop (elapsed >= 10h)
#   10 = waiting (rate-limit signal detected — caller schedules longer wake)

set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

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

OPEN_PRS=$(gh pr list --state=open --base development --json number,title,headRefName --jq 'length' 2>/dev/null || echo "?")
REMAINING=$(grep -c '^- \[ \]' TASKS.md 2>/dev/null || echo 0)

if [ "$REMAINING" -eq 0 ] && [ "$OPEN_PRS" = "0" ]; then
  bash .claude/scripts/update-status.sh done <<<"All TASKS.md items completed. No open PRs. Loop terminating."
  exit 0
fi

bash .claude/scripts/update-status.sh running <<<"Cycle proceeding. Open PRs: $OPEN_PRS. Unchecked tasks: $REMAINING."
exit 0
