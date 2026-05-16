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
