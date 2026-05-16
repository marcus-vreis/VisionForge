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
