# usage: run_gui.sh task seed epochs side outdir summary   (side: gui | native)
cd ~/vfsmoke/real-v1
B=~/Documents/VisionForge/.local-docs/worktrees/guibench2
PY=~/vfsmoke/.venv/Scripts/python.exe
task=$1; seed=$2; ep=$3; side=$4; out=$5; sum=$6
mkdir -p $out
t0=$(date +%s)
if [ "$side" = gui ]; then
  $PY make_config.py $task $seed $ep $out
  case $task in cls) tab="Classificação";; seg) tab="Segmentação";; esac
  abs=$(cd $out && pwd -W)
  (cd $out && PYTHONPATH=$B/src exec $PY -m visionforge gui --port 8030 --no-browser > server.log 2>&1) &
  srv=$!
  until curl -s -o /dev/null http://127.0.0.1:8030/; do sleep 1; done
  node gui-v2/gui_run.mjs "$abs/config.yaml" "$tab" 8030 "$abs/timing.json" "$abs/end.png" > $out/run.log 2>&1
  rc=$?
  powershell -NoProfile -c "Get-NetTCPConnection -LocalPort 8030 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }"
  wait $srv 2>/dev/null
else
  $PY native.py $task $seed $ep $out > $out/run.log 2>&1
  rc=$?
fi
wall=$(( $(date +%s)-t0 ))
echo "$task seed=$seed side=$side exit=$rc wall=${wall}s $(date '+%F %T')" >> $sum
exit $rc
