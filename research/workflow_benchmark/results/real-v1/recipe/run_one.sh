# usage: run_one.sh task seed epochs side outdir summary
cd ~/vfsmoke/real-v1
B=~/Documents/VisionForge/.local-docs/worktrees/bench2
PY=~/vfsmoke/.venv/Scripts/python.exe
task=$1; seed=$2; ep=$3; side=$4; out=$5; sum=$6
mkdir -p $out
t0=$(date +%s)
if [ "$side" = vf ]; then
  $PY make_config.py $task $seed $ep $out && PYTHONPATH=$B/src $PY -m visionforge run $out/config.yaml > $out/run.log 2>&1
else
  $PY native.py $task $seed $ep $out > $out/run.log 2>&1
fi
rc=$?
wall=$(( $(date +%s)-t0 ))
if [ "$task" = det ] && [ $rc -eq 0 ]; then $PY det_test.py $out >> $out/run.log 2>&1 || rc=9; fi
echo "$task seed=$seed side=$side exit=$rc wall=${wall}s $(date '+%F %T')" >> $sum
exit $rc
