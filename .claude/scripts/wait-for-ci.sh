#!/usr/bin/env bash
# Polls CI checks on a PR. Returns 0 if all green, 2 if any fail, 3 on timeout.
# Usage: wait-for-ci.sh <pr-number> [timeout-seconds]

set -uo pipefail

PR_NUMBER="${1:?usage: $0 <pr-number> [timeout-seconds]}"
TIMEOUT_SECS="${2:-1800}"

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
