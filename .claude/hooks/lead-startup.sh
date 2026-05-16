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
