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

if [ -f "$STATUS_FILE" ]; then
  FIRST_START=$(grep -E '^- \*\*First start:\*\*' "$STATUS_FILE" | head -1 | sed -E 's/.*\*\*First start:\*\* (.*)$/\1/')
else
  FIRST_START="$NOW"
fi

if FIRST_EPOCH=$(date -d "$FIRST_START" +%s 2>/dev/null); then
  NOW_EPOCH=$(date +%s)
  ELAPSED_SECS=$((NOW_EPOCH - FIRST_EPOCH))
  H=$((ELAPSED_SECS / 3600))
  M=$(((ELAPSED_SECS % 3600) / 60))
  ELAPSED="${H}h ${M}m"
else
  ELAPSED="?"
fi

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
