cd ~/vfsmoke/real-v1
PY=~/vfsmoke/.venv/Scripts/python.exe; B=~/Documents/VisionForge/.local-docs/worktrees/fixanim
one() { O=diag-v1/probe/$1; rm -rf $O; mkdir -p $O; $PY make_config.py $2 37 $3 $O; abs=$(cd $O && pwd -W)
  (cd $O && PYTHONPATH=$B/src exec $PY -m visionforge gui --port 8030 --no-browser > server.log 2>&1) &
  srv=$!; until curl -s -o /dev/null http://127.0.0.1:8030/; do sleep 1; done
  KILL= node diag-v1/$4 "$abs/config.yaml" "$5" 8030 "~/vfsmoke/real-v1/$O/out.json"; rc=$?
  sleep 2; powershell -NoProfile -c "Get-NetTCPConnection -LocalPort 8030 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }"; wait $srv 2>/dev/null
  echo "$1 rc=$rc epochs:" $(grep "Epoch " $O/server.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c12-19); }
one fix_probe cls 2 probe.mjs "Classificação"
cut -c1-120 diag-v1/probe/fix_probe/out.json | head -14
one fix_cls cls 4 kill.mjs "Classificação"
one fix_seg seg 4 kill.mjs "Segmentação"
