cd ~/vfsmoke/real-v1
G=diag-v1; SUM=$G/summary.log
PY=~/vfsmoke/.venv/Scripts/python.exe
$PY $G/monitor.py $G/monitor.csv &
MON=$!
echo "start $(date '+%F %T')" >> $SUM
block() { for side in $2 $3 $4 $5; do for t in cls seg; do
  [ -f $G/STOP ] && { echo "stopped before $t seed=$1 $side" >> $SUM; continue; }
  bash $G/run_one.sh $t $1 4 $side $G/$t/seed$1/$side $SUM
done; done; }
block 37 native server hidden visible
block 38 server hidden visible native
block 39 hidden visible native server
kill $MON
echo "done $(date '+%F %T')" >> $SUM
