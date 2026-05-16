#!/usr/bin/env bash
# Ensures the 'development' branch exists locally and on origin.
# Idempotent: safe to run on every lead startup.

set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

# If running from inside a worktree, skip — we only operate on the main checkout
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
  if git fetch origin development 2>/dev/null; then
    git branch development origin/development 2>/dev/null || true
    echo "[ensure-dev] created local development tracking origin/development"
  else
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

# Attempt branch protection (best-effort)
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
