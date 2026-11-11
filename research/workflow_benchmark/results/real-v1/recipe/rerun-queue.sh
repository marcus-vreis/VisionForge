# Repeats the runs inside the 2026-10-30 slowdown window; originals stay untouched.
cd ~/vfsmoke/real-v1
SUM=rerun-summary.log
echo "start $(date '+%F %T')" >> $SUM
bash run_one.sh cls 39 30 vf rerun/cls/seed39/vf $SUM
bash run_one.sh cls 39 30 native rerun/cls/seed39/native $SUM
bash run_one.sh seg 37 40 vf rerun/seg/seed37/vf $SUM
echo "done $(date '+%F %T')" >> $SUM
