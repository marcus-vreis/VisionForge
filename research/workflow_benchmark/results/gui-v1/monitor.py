"""Sample GPU and CPU load every 30 s so contaminated runs can be identified afterwards."""
import csv, subprocess, sys, time
import psutil
with open(sys.argv[1], "a", newline="") as f:
    w = csv.writer(f)
    while True:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        top = sorted(((p.info["memory_info"].rss, p.info["name"]) for p in psutil.process_iter(["name", "memory_info"]) if p.info["memory_info"]), reverse=True)[:3]
        w.writerow([time.strftime("%F %T"), g, psutil.cpu_percent(interval=1), "|".join(n for _, n in top)])
        f.flush()
        time.sleep(29)
