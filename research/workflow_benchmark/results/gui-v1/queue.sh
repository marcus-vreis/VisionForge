cd ~/vfsmoke/real-v1
G=gui-v1; SUM=$G/summary.log
PY=~/vfsmoke/.venv/Scripts/python.exe
$PY $G/monitor.py $G/monitor.csv &
MON=$!
echo "start $(date '+%F %T')" >> $SUM
one() { [ -f $G/STOP ] && { echo "stopped before $1 seed=$2 $4 $(date '+%F %T')" >> $SUM; return; }; bash $G/run_gui.sh $1 $2 $3 $4 $G/$1/seed$2/$4 $SUM; }
pair() { one $1 $2 $3 $4; one $1 $2 $3 $5; }
pair cls 37 30 gui native
pair seg 37 40 gui native
pair cls 38 30 native gui
pair seg 38 40 native gui
pair cls 39 30 gui native
pair seg 39 40 gui native
kill $MON
echo "done $(date '+%F %T')" >> $SUM
