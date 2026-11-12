"""Sample GPU and per-group CPU every 10 s (percent of one logical core)."""
import csv, subprocess, sys, time
import psutil
GROUPS = ("python", "chrome", "node")
seen = {}
with open(sys.argv[1], "a", newline="") as f:
    w = csv.writer(f)
    while True:
        total = dict.fromkeys(GROUPS + ("other",), 0.0)
        for p in psutil.process_iter(["name"]):
            try:
                proc = seen.setdefault(p.pid, p)
                pct = proc.cpu_percent(None)
            except psutil.Error:
                continue
            name = (p.info["name"] or "").lower()
            if p.pid == 0 or name == "system idle process":
                continue
            total[next((g for g in GROUPS if g in name), "other")] += pct
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        w.writerow([time.strftime("%F %T"), g, *(round(total[k], 1) for k in GROUPS + ("other",))])
        f.flush()
        time.sleep(10)
