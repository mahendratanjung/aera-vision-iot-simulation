"""Every candidate model behind one interface: ``predict(image_bgr) -> Output``.

All are used as published: no training or fine-tuning. Detectors are run with a very low score
floor (``SCORE_FLOOR``) so precision-recall curves can be drawn; the operating threshold is chosen
later on the tune split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SCORE_FLOOR = 0.01
Kind = Literal["person", "head", "density"]


@dataclass
class Output:
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # xyxy pixels
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    density: np.ndarray | None = None  # density counters: people per cell, (h, w)
    cell: float = 0.0  # density cell size in input pixels


@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    kind: Kind
    family: str
    licence: str
    trained_on: str
    source: str
    note: str = ""


SPECS: list[Spec] = [
    *[Spec(f"yolo11{s}", f"YOLO11{s}", "person", "Ultralytics YOLO11", "AGPL-3.0", "COCO 2017 (person class)",
           "https://docs.ultralytics.com/models/yolo11/",
           "The model AERA's edge camera counter uses today (yolo11n)." if s == "n" else "") for s in "nsm"],
    *[Spec(f"yolo26{s}", f"YOLO26{s}", "person", "Ultralytics YOLO26", "AGPL-3.0", "COCO 2017 (person class)",
           "https://docs.ultralytics.com/models/yolo26/") for s in "nsm"],
    *[Spec(f"rfdetr_{s.lower()}", f"RF-DETR {s}", "person", "Roboflow RF-DETR", "Apache-2.0",
           "COCO 2017 (person class)", "https://github.com/roboflow/rf-detr") for s in ("Nano", "Small", "Medium")],
    Spec("yolov5m_ch_person", "YOLOv5m CrowdHuman, people", "person", "YOLOv5m trained on CrowdHuman",
         "No licence file (YOLOv5 code is AGPL-3.0)", "CrowdHuman train (person and head)",
         "https://github.com/yakhyo/yolov5-crowdhuman-onnx",
         "Community weights. Training on CrowdHuman probably used this validation set to pick checkpoints."),
    Spec("yolov5m_ch_head", "YOLOv5m CrowdHuman, heads", "head", "YOLOv5m trained on CrowdHuman",
         "No licence file (YOLOv5 code is AGPL-3.0)", "CrowdHuman train (person and head)",
         "https://github.com/yakhyo/yolov5-crowdhuman-onnx",
         "Same model, counting heads. Heads stay visible when bodies hide each other."),
    Spec("zip_n_nwpu", "ZIP Nano (NWPU)", "density", "ZIP density counter", "MIT", "NWPU-Crowd",
         "https://github.com/Yiming-M/ZIP", "MobileNetV4-small backbone."),
    Spec("zip_t_nwpu", "ZIP Tiny (NWPU)", "density", "ZIP density counter", "MIT", "NWPU-Crowd",
         "https://github.com/Yiming-M/ZIP", "MobileNetV4-medium backbone."),
    Spec("zip_t_shb", "ZIP Tiny (ShanghaiTech B)", "density", "ZIP density counter", "MIT", "ShanghaiTech Part B",
         "https://github.com/Yiming-M/ZIP", "MobileNetV4-medium backbone, trained on sparse street scenes."),
]
BY_KEY = {s.key: s for s in SPECS}


def weights_path(key: str):
    """Where a candidate's weights file lives (no model is loaded)."""
    from aera_vision.data import ROOT

    w = ROOT / "weights"
    if key.startswith(("yolo11", "yolo26")):
        return w / "ultralytics" / f"{key}.pt"
    if key.startswith("rfdetr"):
        return w / "rfdetr" / f"rf-detr-{key.split('_')[1]}.pth"
    if key.startswith("yolov5m_ch"):
        return w / "yolov5_crowdhuman" / "crowdhuman.onnx"
    from aera_vision.candidates.zip_counter import PATHS

    return PATHS[key]


def file_mb(key: str) -> float | None:
    p = weights_path(key)
    return round(p.stat().st_size / 1e6, 1) if p.exists() else None


def load(key: str, device: str):
    """The predictor for a candidate, ready to call."""
    spec = BY_KEY[key]
    if key.startswith("yolo11") or key.startswith("yolo26"):
        from aera_vision.candidates.ultralytics_det import UltralyticsPerson
        return UltralyticsPerson(key, device)
    if key.startswith("rfdetr"):
        from aera_vision.candidates.rfdetr_det import RFDETRPerson
        return RFDETRPerson(key.split("_")[1], device)
    if key.startswith("yolov5m_ch"):
        from aera_vision.candidates.yolov5_crowdhuman import YOLOv5CrowdHuman
        return YOLOv5CrowdHuman(spec.kind)
    if key.startswith("zip"):
        from aera_vision.candidates.zip_counter import ZIPCounter
        return ZIPCounter(key, device)
    raise KeyError(key)
