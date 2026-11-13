"""AERA Vision counter: counts heads in a photo with the chosen model. Small enough for a free server.

    POST /count   body: a JPEG or PNG image   ->   {"count", "boxes", "model", "ms"}
    GET  /health

Runs the YOLOv5m CrowdHuman ONNX model exactly as it was evaluated (model.json, written from the
evaluation): letterboxed 640 px input, score = objectness x head probability, non-maximum suppression at
IoU 0.6, the confidence threshold set on the tuning split. Only ONNX Runtime, NumPy and OpenCV; no PyTorch.

Images are decoded in memory, counted and dropped. Nothing is written to disk or logged.
"""

from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "model.json").read_text())
WEIGHTS = HERE / "weights" / "crowdhuman.onnx"
MAX_BYTES = 8_000_000
SCORE_FLOOR = 0.01  # as in the evaluation; the real cut is CFG["threshold"]

# Browsers allowed to call the counter: local development and the Vercel site (set on the server).
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3100,http://127.0.0.1:3100").split(",") if o.strip()]
ORIGIN_REGEX = os.environ.get("ALLOWED_ORIGIN_REGEX", r"^https://aera-vision[a-z0-9-]*\.vercel\.app$")

app = FastAPI(title="AERA Vision counter", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_origin_regex=ORIGIN_REGEX or None,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

opts = ort.SessionOptions()
opts.intra_op_num_threads = int(os.environ.get("THREADS", "1"))
SESSION = ort.InferenceSession(str(WEIGHTS), sess_options=opts, providers=["CPUExecutionProvider"])
NAMES = ast.literal_eval(SESSION.get_modelmeta().custom_metadata_map["names"])
CLS = {v: k for k, v in NAMES.items()}[CFG["class"]]
INPUT = SESSION.get_inputs()[0].name


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    out = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) / 2, (size - nh) / 2
    top, left = round(dy - 0.1), round(dx - 0.1)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy non-maximum suppression, same rule as torchvision.ops.nms (drop IoU > iou), highest score first."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ious = inter / (area[i] + area[order[1:]] - inter + 1e-12)
        order = order[1:][ious <= iou]
    return np.array(keep, dtype=int)


def count_heads(img_bgr: np.ndarray) -> tuple[int, np.ndarray]:
    """Head boxes (original pixels) scoring at least the threshold, and how many there are."""
    h, w = img_bgr.shape[:2]
    s = min(1.0, CFG["max_side"] / max(h, w))
    if s < 1.0:
        img_bgr = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    img, r, (left, top) = letterbox(img_bgr, CFG["imgsz"])
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), np.float32)[None] / 255.0
    out = SESSION.run(None, {INPUT: x})[0][0]
    score = out[:, 4] * out[:, 5 + CLS]
    keep = score >= SCORE_FLOOR
    out, score = out[keep], score[keep]
    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    idx = nms(boxes, score, CFG["nms_iou"])[:1000]
    boxes, score = boxes[idx], score[idx]
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / r).clip(0, img_bgr.shape[1]) / s
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / r).clip(0, img_bgr.shape[0]) / s
    sel = score >= CFG["threshold"]
    return int(sel.sum()), boxes[sel]


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": CFG["label"]}


@app.post("/count")
async def count(request: Request) -> dict:
    body = await request.body()
    if not body or len(body) > MAX_BYTES:
        raise HTTPException(413, "Image missing or too large")
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not an image")
    t = time.perf_counter()
    n, boxes = count_heads(img)
    del img, body
    return {"count": n, "boxes": np.round(boxes, 1).tolist(), "model": CFG["label"], "ms": (time.perf_counter() - t) * 1000}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8765"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Counter ready on http://{host}:{port} (camera and photo counting). Press Ctrl+C to stop.", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")

