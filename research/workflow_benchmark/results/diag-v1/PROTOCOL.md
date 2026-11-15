# diag-v1: where the gui-v1 time cost comes from (frozen before the runs)

Written 2026-11-01, after gui-v1 and before any run of this series. gui-v1 stays as reported; this is a separate, exploratory question.

Question: of the 12-17% that gui-v1 measured, how much comes from the server and how much from the browser, and does hiding the tab remove it?

- Conditions, same submission path (import YAML, Revisão, Treinar) except native:
  - native: real-v1/native.py.
  - server: browser closed right after Treinar, job runs on the server with no client.
  - hidden: Chrome stays open with its window minimized and Chrome's default background throttling on. Playwright disconnects after the click, because it keeps a page visible while attached (found in the smoke run, before any counted run).
  - visible: exactly as gui-v1, page rendering, throttling off.
- Code: VisionForge a2a957d, as gui-v1. Recipes as gui-v1, 4 epochs instead of 30/40.
- Blocks: seeds 37, 38, 39, on classification and segmentation. Order inside a block rotates: 37 native, server, hidden, visible; 38 server, hidden, visible, native; 39 hidden, visible, native, server.
- Outcome: mean epoch time from the end of epoch 1 to the end of epoch 4, so start-up and test evaluation are out. Ratio to native per block, geometric mean, 90% CI. Native comes from history seconds, VisionForge from the second-resolution server log.
- With n = 3 and no margin set, this is descriptive: it points to a cause and proves no equivalence. Metrics after 4 epochs are not reported as quality results.
- Check: the hidden condition records document.visibilityState and requestAnimationFrame count over 2 s, or it does not count as hidden.
- Load: monitor.csv samples CPU per process group (python, chrome, node) every 10 s. No run is dropped; a repeat is a sensitivity row.
