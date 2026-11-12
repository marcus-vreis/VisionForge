cd ~/vfsmoke/real-v1
G=diag-v1
for s in server hidden visible; do bash $G/run_one.sh cls 37 2 $s $G/smoke/$s $G/smoke.log; done
bash $G/run_one.sh cls 37 2 native $G/smoke/native $G/smoke.log
echo finished >> $G/smoke.log
