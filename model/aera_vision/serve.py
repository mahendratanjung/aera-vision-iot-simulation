"""Local live counter for the app's Live camera panel.

    uv run python -m aera_vision.serve          (listens on 127.0.0.1:8765 only)

The browser sends a JPEG frame to POST /count; the chosen model counts the people and the reply is
the count and the boxes. The frame is decoded in memory and dropped; nothing is written to disk.
Only the app's own local address may call it.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from aera_vision import candidates
from aera_vision.chosen import density_points, needed_keys, winner
from aera_vision.infer import predict

MAX_BYTES = 5_000_000
app = FastAPI(title="AERA Vision live counter")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3100", "http://127.0.0.1:3100"],
                   allow_methods=["POST"], allow_headers=["Content-Type"])
ROW = winner()
PARTS = {}


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@app.on_event("startup")
def load() -> None:
    import json

    from aera_vision.evaluate import RESULTS

    by_key = {c["key"]: c for c in json.loads((RESULTS / "summary.json").read_text())["candidates"]}
    for k in needed_keys(ROW):
        PARTS[k] = (by_key[k], candidates.load(k, _device()))


@app.post("/count")
async def count(request: Request) -> dict:
    body = await request.body()
    if not body or len(body) > MAX_BYTES:
        raise HTTPException(413, "Frame too large")
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a JPEG frame")
    t = time.perf_counter()
    h, w = img.shape[:2]
    outs = {k: predict(m, img) for k, (_, m) in PARTS.items()}
    spec = ROW
    if ROW["kind"] == "hybrid":
        det, den = (PARTS[k][0] for k in ROW["parts"])
        n = int((outs[det["key"]].scores >= det["threshold"]).sum())
        spec = det if n < ROW["switch"] else den
    o = outs[spec["key"]]
    if spec["kind"] == "density":
        pts = density_points(o, w, h)
        result = {"count": round(float(o.density.sum())), "boxes": [], "points": pts.round(1).tolist()}
    else:
        keep = o.scores >= spec["threshold"]
        result = {"count": int(keep.sum()), "boxes": np.round(o.boxes[keep], 1).tolist()}
    del img
    return {**result, "model": ROW["label"], "ms": (time.perf_counter() - t) * 1000}


def main() -> None:
    import uvicorn

    print("Counter ready on http://127.0.0.1:8765 (camera and photo counting). Press Ctrl+C to stop.", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
