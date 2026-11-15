# Extra cls/seg seeds after the rerun. A pair starts only if it should end before the deadline.
cd ~/vfsmoke/real-v1
SUM=ext-summary.log
END=$(date -d '2026-10-31 08:50:00' +%s)
until grep -q '^done' rerun-summary.log; do sleep 30; done
echo "start $(date '+%F %T')" >> $SUM
pair() { # task seed epochs first second estimated_pair_seconds
  if [ $(( $(date +%s) + $6 )) -gt $END ]; then echo "skip $1 seed=$2 $(date '+%F %T')" >> $SUM; return; fi
  bash run_one.sh $1 $2 $3 $4 $1/seed$2/$4 $SUM
  bash run_one.sh $1 $2 $3 $5 $1/seed$2/$5 $SUM
}
pair cls 40 30 native vf 3900
pair seg 40 40 native vf 5200
pair cls 41 30 vf native 3900
pair seg 41 40 vf native 5200
pair cls 42 30 native vf 3900
pair seg 42 40 native vf 5200
echo "done $(date '+%F %T')" >> $SUM
