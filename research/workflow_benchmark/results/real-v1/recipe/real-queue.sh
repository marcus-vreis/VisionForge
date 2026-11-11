# Real-training queue: VF vs native, same recipe, seeds 37-39, alternating order.
# Re-running skips every run that already ended with exit=0.
cd ~/vfsmoke/real-v1
B=~/Documents/VisionForge/.local-docs/worktrees/bench2
SUM=summary.log
echo "start $(date '+%F %T') bench2=$(git -C $B rev-parse --short HEAD)" >> $SUM
for spec in cls:30 seg:40 det:100; do
  task=${spec%:*}; ep=${spec#*:}
  for seed in 37 38 39; do
    if [ $seed = 38 ]; then order="native vf"; else order="vf native"; fi
    for side in $order; do
      if grep -q "^$task seed=$seed side=$side exit=0 " $SUM; then continue; fi
      bash run_one.sh $task $seed $ep $side $task/seed$seed/$side $SUM
    done
  done
done
echo "done $(date '+%F %T')" >> $SUM
