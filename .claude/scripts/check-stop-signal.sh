#!/usr/bin/env bash
# Returns:
#   0 = continue (normal)
#   1 = STOP_TEAM file exists (user requested stop)
#   2 = elapsed time exceeds hard limit
# Prints reason to stdout when non-zero.

set -uo pipefail

HARD_LIMIT_SECONDS="${LOOP_HARD_LIMIT_SECONDS:-36000}"
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
