# gui-v2: gui-v1 repeated after the animation fix (frozen before the runs)

Written 2026-11-01, before any run of this series. gui-v1 stays in the record as the measurement before the fix.

Question: with looping animations ended during a run (ADR-157, PR #107), does training through the browser stay within the margins against plain PyTorch?

- Everything as results/gui-v1/PROTOCOL.md: same recipes, seeds 37, 38, 39, same order inside each pair, same browser setup with the page visible and throttling off, same recipe check, same outcomes and margins (time ±5%, 90% CI; quality ±0.005, 95% CI; n = 3, so failing the quality margin means inconclusive).
- Code: VisionForge 75fc3027d10d7ec19e74139d0a22782c65e15aab (merge of PR #107 into development). Native: real-v1/native.py unchanged, rerun in this series, not reused from gui-v1.
- Difference from gui-v1 besides the fix: monitor.csv samples CPU per process group every 10 s instead of machine CPU every 30 s.
- The fix was chosen after seeing gui-v1 and diag-v1, so this series tests a change made because of those results; it is reported as such.
- All first runs stay in the main table. A run repeated because of outside load is a sensitivity row, and no run is dropped.

## Outcome (added after the runs)

12/12 runs exit 0; recipe check: only the path separator of data.base_dir differs. Results in results/gui_v2_summary.md. Classification time failed the margin (1.047, 90% CI 1.006 to 1.089). In cls seed 39 gui (ratio 1.074) monitor.csv shows the "other" process group at a mean of 110% of one core against 73% in seeds 37 and 38; the processes were not identified and the run was not repeated. Median epoch ratios from logs, exploratory: cls 1.028, 1.002, 1.065; seg 1.007, 1.004, 1.008.
