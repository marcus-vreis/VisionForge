"""Score one best.pt on the VOC test split, identically for both workflows."""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    best = sorted(out.rglob("weights/best.pt"))[0]
    m = YOLO(str(best)).val(data="~/vfsmoke/real-v1/voc.yaml", split="test", imgsz=640, batch=16, workers=4,
                            device=0, project=str(out), name="test_eval", exist_ok=True, verbose=False)
    (out / "det_test.json").write_text(
        json.dumps({"best": str(best), "map50": float(m.box.map50), "map50_95": float(m.box.map)}, indent=1), encoding="utf-8"
    )
