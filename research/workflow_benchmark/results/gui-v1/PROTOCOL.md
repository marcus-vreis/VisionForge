# gui-v1: VisionForge through the real GUI vs native (frozen before the runs)

Written 2026-10-31, before any full run of this series. Only 1-epoch smoke runs preceded it.

Question: does training through the browser interface change quality or time compared with the same recipe on plain PyTorch?
Why: in real-v1 the VisionForge side ran `visionforge run`, the command line, not the interface.

- VF side: a fresh `visionforge gui` server, installed Google Chrome with a clean profile, headed, 1366x768, occlusion and background throttling off so the animation keeps rendering (worst case for the interface). The script imports the YAML, opens Revisão, clicks Treinar and waits for "treino concluído".
- Code: VisionForge a2a957d (PR #106), venv ~/vfsmoke/.venv. Native: real-v1/native.py unchanged.
- Recipes: same as real-v1 (ResNet-18 CIFAR-10 30 ep; DeepLabV3-ResNet50 Pet 40 ep). Detection is not part of this series.
- Seeds 37, 38, 39. GUI runs first on 37 and 39, native first on 38. Same seeds as real-v1, so GUI vs CLI metrics can also be read per seed.
- Recipe check: every key of config.yaml must match run.json, apart from path separators, or the run is invalid.

Outcomes, set before the runs:
- Time: wall clock from process launch, including server start and page load on the GUI side. Paired VF/native ratio, 90% CI, margin ±5%. Secondary: click to end from timing.json.
- Quality: test accuracy and test mIoU, paired difference, 95% CI, margin ±0.005 absolute. With n = 3 this CI is wide: failing the margin means inconclusive, not worse.

Load: monitor.csv samples GPU and CPU every 30 s. The PC may be in use during the series. All first runs stay in the main table. A run repeated because of outside load is reported as a sensitivity row, as in real-v1, and no run is dropped.
