"""Roboflow RF-DETR COCO checkpoints (Nano 384 px, Small 512 px, Medium 576 px), person class only.

Weights are the official Roboflow-hosted files, loaded with PyTorch's safe loader (the library default).
"""

from __future__ import annotations

import numpy as np

from aera_vision.candidates import SCORE_FLOOR, Output
from aera_vision.data import ROOT

WEIGHTS = ROOT / "weights" / "rfdetr"


class RFDETRPerson:
    def __init__(self, size: str, device: str):
        import rfdetr

        WEIGHTS.mkdir(parents=True, exist_ok=True)
        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[size]
        self.weights_path = WEIGHTS / f"rf-detr-{size}.pth"
        self.model = cls(pretrain_weights=str(self.weights_path), device=device)
        # detections carry COCO category ids (person = 1), not indexes into model.class_names
        from rfdetr.assets.coco_classes import COCO_CLASSES

        self.person = [k for k, v in COCO_CLASSES.items() if v == "person"]
        if not self.person:
            raise RuntimeError("no person class in RF-DETR COCO_CLASSES")

    def predict(self, image_bgr: np.ndarray) -> Output:
        det = self.model.predict(np.ascontiguousarray(image_bgr[:, :, ::-1]), threshold=SCORE_FLOOR)
        keep = np.isin(det.class_id, self.person)
        return Output(det.xyxy[keep].astype(np.float32), det.confidence[keep].astype(np.float32))
