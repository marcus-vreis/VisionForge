cd ~/vfsmoke/real-v1
PY=~/vfsmoke/.venv/Scripts/python.exe; B=~/Documents/VisionForge/.local-docs/worktrees/guibench
one() { O=diag-v1/probe/$1; mkdir -p $O; $PY make_config.py cls 37 3 $O; abs=$(cd $O && pwd -W)
  (cd $O && PYTHONPATH=$B/src exec $PY -m visionforge gui --port 8030 --no-browser > server.log 2>&1) &
  srv=$!; until curl -s -o /dev/null http://127.0.0.1:8030/; do sleep 1; done
  KILL=$2 node diag-v1/kill.mjs "$abs/config.yaml" "Classificação" 8030 x; rc=$?
  powershell -NoProfile -c "Get-NetTCPConnection -LocalPort 8030 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }"; wait $srv 2>/dev/null
  echo "$1 kill=$2 rc=$rc epochs:" $(grep "Epoch " $O/server.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c12-19); }
one base none
one noshimmer shimmer
one nospin spin
one nopulse pulse-dot
one noall shimmer,spin,pulse-dot
