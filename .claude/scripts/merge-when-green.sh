#!/usr/bin/env bash
# Squash-merges a PR. Caller ensures CI passed and review approved.
# Usage: merge-when-green.sh <pr-number>

set -euo pipefail

PR_NUMBER="${1:?usage: $0 <pr-number>}"

REVIEW_STATE=$(gh pr view "$PR_NUMBER" --json reviewDecision --jq .reviewDecision)
if [ "$REVIEW_STATE" != "APPROVED" ]; then
  echo "ERROR: PR #$PR_NUMBER not approved (state: $REVIEW_STATE)" >&2
  exit 2
fi

gh pr merge "$PR_NUMBER" --squash --delete-branch --auto

sleep 5

STATE=$(gh pr view "$PR_NUMBER" --json state --jq .state)
if [ "$STATE" != "MERGED" ]; then
  echo "Merge pending or failed (state: $STATE) — may complete asynchronously" >&2
fi

git fetch origin development 2>/dev/null || true
if git rev-parse --verify development >/dev/null 2>&1; then
  CURRENT=$(git rev-parse --abbrev-ref HEAD)
  if [ "$CURRENT" = "development" ]; then
    git pull --ff-only origin development 2>/dev/null || true
  fi
fi

echo "PR #$PR_NUMBER merge initiated"
