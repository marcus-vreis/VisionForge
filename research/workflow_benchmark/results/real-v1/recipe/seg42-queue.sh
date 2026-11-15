# Runs the seg seed 42 pair after ext-queue, with no deadline, unless ext-queue already ran it.
cd ~/vfsmoke/real-v1
SUM=ext-summary.log
until grep -q '^done' $SUM 2>/dev/null; do sleep 30; done
for side in native vf; do
  grep -q "^seg seed=42 side=$side exit=0 " $SUM || bash run_one.sh seg 42 40 $side seg/seed42/$side $SUM
done
echo "all-finished $(date '+%F %T')" >> $SUM
