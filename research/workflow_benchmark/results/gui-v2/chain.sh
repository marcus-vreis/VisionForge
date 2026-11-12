cd ~/vfsmoke/real-v1
G=gui-v2
bash $G/run_gui.sh cls 37 1 gui $G/smoke/cls $G/smoke.log || { echo SMOKE FAILED; tail -5 $G/smoke/cls/run.log; exit 1; }
cat $G/smoke/cls/timing.json; echo
bash $G/queue.sh
tail -14 $G/summary.log
