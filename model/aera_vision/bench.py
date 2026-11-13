"""Speed and size of every candidate on this machine. Writes results/speed.json.

    uv run python -m aera_vision.bench

One 1280 x 720 frame (a typical camera frame: a CrowdHuman tune image resized), batch 1, the
full predict call including pre- and post-processing. 10 warm-up runs, then the median of 50 timed
runs, on the CPU and on the Apple GPU (MPS). The ONNX model runs on the CPU only. Nothing else
should be running while this measures.

Edge devices (Raspberry Pi 5 + AI HAT, Jetson Orin Nano) were not available, so they are not
measured here; see references/edge_speed.json for published figures.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time

import cv2
import numpy as np

from aera_vision import candidates, data
from aera_vision.evaluate import RESULTS

WARMUP, RUNS = 10, 50


def params_of(model) -> float | None:
    """Parameters in millions, counted from the loaded model."""
    try:
        if hasattr(model, "session"):  # ONNX
            import onnx

            m = onnx.load(str(model.weights_path))
            return sum(int(np.prod(t.dims)) for t in m.graph.initializer) / 1e6
        if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "parameters"):
            return sum(p.numel() for p in model.model.model.parameters()) / 1e6  # Ultralytics / RF-DETR wrapper
        if hasattr(model, "model") and hasattr(model.model, "parameters"):
            return sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception as e:  # noqa: BLE001
        print("params:", type(e).__name__, e)
    return None


def time_it(model, img: np.ndarray, sync) -> dict:
    for _ in range(WARMUP):
        model.predict(img)
    sync()
    ts = []
    for _ in range(RUNS):
        t = time.perf_counter()
        model.predict(img)
        sync()
        ts.append((time.perf_counter() - t) * 1000)
    med = statistics.median(ts)
    return {"ms_median": round(med, 2), "ms_p90": round(float(np.percentile(ts, 90)), 2), "fps": round(1000 / med, 2),
            "runs": RUNS, "warmup": WARMUP}


def chip() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor()


def main() -> None:
    import torch

    frames = data.load_frames()
    tune = data.load_splits()["tune"]
    fid = sorted(tune, key=lambda i: frames[i].count)[len(tune) // 2]  # a middle-of-the-road image
    img = cv2.resize(cv2.imread(str(frames[fid].path)), (1280, 720), interpolation=cv2.INTER_AREA)
    out = {"machine": {"chip": chip(), "platform": platform.platform(), "torch": torch.__version__,
                       "threads": torch.get_num_threads()},
           "frame": {"source_image": fid, "size": [1280, 720]}, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "candidates": {}}
    syncs = {"cpu": lambda: None, "mps": torch.mps.synchronize}
    for spec in candidates.SPECS:
        row: dict = {}
        for dev in ("cpu", "mps"):
            if spec.key.startswith("yolov5m_ch") and dev == "mps":
                continue  # ONNX Runtime on the CPU only
            m = candidates.load(spec.key, dev)
            if "file_mb" not in row:
                row["file_mb"] = round(m.weights_path.stat().st_size / 1e6, 1)
                row["params_m"] = round(p, 2) if (p := params_of(m)) else None
            row[dev] = time_it(m, img, syncs[dev])
            del m
            print(spec.key, dev, row[dev]["ms_median"], "ms", flush=True)
        out["candidates"][spec.key] = row
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "speed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
