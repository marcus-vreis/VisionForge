# usage: run_one.sh task seed epochs side outdir summary   (side: native | server | hidden | visible)
cd ~/vfsmoke/real-v1
B=~/Documents/VisionForge/.local-docs/worktrees/guibench
PY=~/vfsmoke/.venv/Scripts/python.exe
task=$1; seed=$2; ep=$3; side=$4; out=$5; sum=$6
mkdir -p $out
t0=$(date +%s)
if [ "$side" = native ]; then
  $PY native.py $task $seed $ep $out > $out/run.log 2>&1
  rc=$?
else
  $PY make_config.py $task $seed $ep $out
  case $task in cls) tab="Classificação";; seg) tab="Segmentação";; esac
  abs=$(cd $out && pwd -W)
  (cd $out && PYTHONPATH=$B/src exec $PY -m visionforge gui --port 8030 --no-browser > server.log 2>&1) &
  srv=$!
  until curl -s -o /dev/null http://127.0.0.1:8030/; do sleep 1; done
  node diag-v1/gui_run.mjs $side "$abs/config.yaml" "$tab" 8030 "$abs/timing.json" > $out/run.log 2>&1
  rc=$?
  powershell -NoProfile -c "Get-NetTCPConnection -LocalPort 8030 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }"
  wait $srv 2>/dev/null
fi
echo "$task seed=$seed side=$side exit=$rc wall=$(( $(date +%s)-t0 ))s $(date '+%F %T')" >> $sum
exit $rc
