"""Write ../counter/model.json: the settings the deployed counter needs, taken from the evaluation.

    uv run python -m aera_vision.export_counter

The counter (../counter) is a small server without PyTorch, so it fits a free hosting plan. It must run
the chosen model exactly as it was scored: same weights (checked by SHA-256), same input size, same
non-maximum suppression and the same confidence threshold, which was set on the tuning split.
"""

from __future__ import annotations

import hashlib
import json

from aera_vision import candidates
from aera_vision.candidates import yolov5_crowdhuman as y5
from aera_vision.chosen import winner

OUT = candidates.weights_path("yolov5m_ch_head").parents[2].parent / "counter" / "model.json"
URL = "https://github.com/yakhyo/yolov5-crowdhuman-onnx/releases/download/v0.0.1/crowdhuman.onnx"


def main() -> None:
    m = winner()
    if not m["key"].startswith("yolov5m_ch"):
        raise SystemExit(f"the lean counter only supports the YOLOv5 CrowdHuman ONNX model, not {m['key']}")
    weights = candidates.weights_path(m["key"])
    cfg = {
        "label": m["label"],
        "class": m["kind"],  # "head": class 1 in the ONNX metadata
        "threshold": m["threshold"],
        "imgsz": y5.IMGSZ,
        "nms_iou": y5.NMS_IOU,
        "max_side": 2048,
        "weights_url": URL,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "source": m["source"],
        "licence": m["licence"],
        "note": "Written by model/aera_vision/export_counter.py from model/results and model/chosen.json. Do not edit by hand.",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
