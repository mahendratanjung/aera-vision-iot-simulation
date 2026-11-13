"""The chosen model, as the evaluation picked it, turned into a count plus what to draw.

Used by demo.py (on cached outputs, so the demo shows exactly what was scored) and serve.py (live).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from aera_vision import candidates, data, metrics
from aera_vision.evaluate import RESULTS, density_count, detector_count


@dataclass
class View:
    count: float
    boxes: np.ndarray  # (N, 4) xyxy, people the count is made of (detectors)
    points: np.ndarray  # (M, 3) x, y, weight (density counters)


CHOSEN = RESULTS.parent / "chosen.json"  # the model the app uses, if set by hand (with its reason)


def winner() -> dict:
    """The model the app uses: model/chosen.json if present, otherwise the evaluation's pick."""
    s = json.loads((RESULTS / "summary.json").read_text())
    key = json.loads(CHOSEN.read_text())["key"] if CHOSEN.exists() else s["winner"]["key"]
    return next(c for c in s["candidates"] if c["key"] == key)


def density_points(out: candidates.Output, width: int, height: int, min_weight: float = 0.05, top: int = 400) -> np.ndarray:
    h, w = out.density.shape
    ys, xs = np.nonzero(out.density >= min_weight)
    v = out.density[ys, xs]
    order = np.argsort(-v)[:top]
    return np.stack([(xs[order] + 0.5) * width / w, (ys[order] + 0.5) * height / h, v[order]], 1) if len(order) else np.zeros((0, 3))


def view_of(row: dict, outs: dict[str, candidates.Output], frame: data.Frame, by_key: dict[str, dict]) -> View:
    """Count and drawing for one image. ``outs`` holds each needed candidate's output for this image."""
    if row["kind"] == "hybrid":
        det_key, den_key = row["parts"]
        det = by_key[det_key]
        n = detector_count(outs[det_key], frame, det["threshold"])
        return view_of(det if n < row["switch"] else by_key[den_key], outs, frame, by_key)
    o = outs[row["key"]]
    if row["kind"] == "density":
        return View(density_count(o, frame), np.zeros((0, 4)), density_points(o, frame.width, frame.height))
    keep = o.scores >= row["threshold"]
    boxes = o.boxes[keep]
    boxes = boxes[~metrics.centres_inside(boxes, frame.zones)]
    return View(float(detector_count(o, frame, row["threshold"])), boxes, np.zeros((0, 3)))


def needed_keys(row: dict) -> list[str]:
    return list(row["parts"]) if row["kind"] == "hybrid" else [row["key"]]
