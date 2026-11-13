"""Ultralytics YOLO11 / YOLO26 COCO models, person class only."""

from __future__ import annotations

import numpy as np

from aera_vision.candidates import SCORE_FLOOR, Output
from aera_vision.data import ROOT

WEIGHTS = ROOT / "weights" / "ultralytics"
PERSON = 0  # COCO class id in Ultralytics models
IMGSZ = 640


class UltralyticsPerson:
    def __init__(self, key: str, device: str):
        from ultralytics import YOLO

        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(WEIGHTS / f"{key}.pt"))  # downloads the official release asset if missing
        self.device = device
        self.weights_path = WEIGHTS / f"{key}.pt"

    def predict(self, image_bgr: np.ndarray) -> Output:
        r = self.model.predict(image_bgr, imgsz=IMGSZ, conf=SCORE_FLOOR, classes=[PERSON], max_det=1000,
                               device=self.device, verbose=False)[0]
        return Output(r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32))
