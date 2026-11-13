"""YOLOv5m trained on CrowdHuman (person + head), community ONNX export, run with ONNX Runtime.

Weights: https://github.com/yakhyo/yolov5-crowdhuman-onnx/releases/download/v0.0.1/crowdhuman.onnx
Classes from the model's own metadata: 0 person, 1 head. Decoding follows standard YOLOv5:
score = objectness x class probability, then non-maximum suppression per class at IoU 0.6
(YOLOv5's validation setting), letterboxed 640 px input.
"""

from __future__ import annotations

import ast

import cv2
import numpy as np

from aera_vision.candidates import SCORE_FLOOR, Output
from aera_vision.data import ROOT

WEIGHTS = ROOT / "weights" / "yolov5_crowdhuman" / "crowdhuman.onnx"
IMGSZ = 640
NMS_IOU = 0.6


def letterbox(img: np.ndarray, size: int = IMGSZ) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    out = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) / 2, (size - nh) / 2
    top, left = round(dy - 0.1), round(dx - 0.1)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    import torch
    import torchvision

    if len(boxes) == 0:
        return np.zeros(0, int)
    return torchvision.ops.nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), iou).numpy()


class YOLOv5CrowdHuman:
    _session = None

    def __init__(self, kind: str):
        import onnxruntime as ort

        if YOLOv5CrowdHuman._session is None:
            YOLOv5CrowdHuman._session = ort.InferenceSession(str(WEIGHTS), providers=["CPUExecutionProvider"])
        self.session = YOLOv5CrowdHuman._session
        names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map["names"])
        self.cls = {v: k for k, v in names.items()}[kind]  # "person" -> 0, "head" -> 1
        self.input = self.session.get_inputs()[0].name
        self.weights_path = WEIGHTS

    def predict(self, image_bgr: np.ndarray) -> Output:
        img, r, (left, top) = letterbox(image_bgr)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), np.float32)[None] / 255.0
        out = self.session.run(None, {self.input: x})[0][0]  # (anchors, 5 + classes): cx cy w h obj cls...
        score = out[:, 4] * out[:, 5 + self.cls]
        keep = score >= SCORE_FLOOR
        out, score = out[keep], score[keep]
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        idx = nms(boxes, score, NMS_IOU)[:1000]
        boxes, score = boxes[idx], score[idx]
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / r).clip(0, image_bgr.shape[1])
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / r).clip(0, image_bgr.shape[0])
        return Output(boxes.astype(np.float32), score.astype(np.float32))
