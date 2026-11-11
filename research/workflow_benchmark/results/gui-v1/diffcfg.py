import json, sys, yaml
from pathlib import Path
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, p + k + "."))
        else: o[p + k] = v
    return o
for t in sys.argv[1:]:
    base = Path(t)
    want = flat(yaml.safe_load((base / "config.yaml").read_text(encoding="utf-8")))
    run = sorted(base.rglob("run.json"))[-1]
    d = json.loads(run.read_text(encoding="utf-8"))
    got = flat(d["config"])
    print("==", t, run.relative_to(base))
    bad = 0
    for k, v in want.items():
        if k.startswith("output.") or k == "name": continue
        if got.get(k, "<absent>") != v:
            bad += 1; print("  DIFF", k, "yaml=", v, "run=", got.get(k, "<absent>"))
    print("  checked", len(want), "diffs", bad, "| metrics", {k: round(v, 4) for k, v in d.get("metrics", {}).items() if isinstance(v, float) and k.startswith("test_")})
    print("  output:", {k: v for k, v in got.items() if k.startswith("output.")})
