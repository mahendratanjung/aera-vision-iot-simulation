"""Metric functions used by the evaluation. Each one is checked on hand-worked examples in tests/.

Boxes are numpy arrays of shape (N, 4) in pixel xyxy order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from aera_vision.levels import NAMES

# ---- boxes -----------------------------------------------------------------------------------


def _area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union of every box in a with every box in b, shape (len(a), len(b))."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    inter = _inter(a, b)
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def ioa_matrix(a: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Share of each box in a that lies inside each region (intersection over a's area)."""
    a, regions = np.asarray(a, float).reshape(-1, 4), np.asarray(regions, float).reshape(-1, 4)
    inter = _inter(a, regions)
    area = _area(a)[:, None]
    return np.where(area > 0, inter / np.where(area > 0, area, 1), 0.0)


def centres_inside(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Whether each box's centre lies inside any region. Shape (len(boxes),)."""
    boxes, regions = np.asarray(boxes, float).reshape(-1, 4), np.asarray(regions, float).reshape(-1, 4)
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes), bool)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    inside = ((cx[:, None] >= regions[None, :, 0]) & (cx[:, None] <= regions[None, :, 2])
              & (cy[:, None] >= regions[None, :, 1]) & (cy[:, None] <= regions[None, :, 3]))
    return inside.any(axis=1)


# ---- detection ---------------------------------------------------------------------------------

TP, FP, IGNORED = 1, 0, -1


def match_image(dets: np.ndarray, scores: np.ndarray, gts: np.ndarray, ignore: np.ndarray,
                iou_thr: float = 0.5, ignore_ioa: float = 0.5) -> np.ndarray:
    """Label each detection TP, FP or IGNORED, COCO style.

    Detections are taken from the highest score down. Each one takes the unmatched ground-truth
    box it overlaps most, if that IoU is at least ``iou_thr`` (true positive). Otherwise, if at
    least ``ignore_ioa`` of it lies inside an ignore region (crowds or people the annotators did
    not label), it is ignored: neither right nor wrong. Anything else is a false positive.
    """
    dets = np.asarray(dets, float).reshape(-1, 4)
    gts = np.asarray(gts, float).reshape(-1, 4)
    labels = np.full(len(dets), FP, int)
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    ious = iou_matrix(dets, gts)
    ioa = ioa_matrix(dets, ignore).max(axis=1) if len(ignore) else np.zeros(len(dets))
    taken = np.zeros(len(gts), bool)
    for i in order:
        if len(gts):
            cand = np.where(taken, -1.0, ious[i])
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                taken[j] = True
                labels[i] = TP
                continue
        if ioa[i] >= ignore_ioa:
            labels[i] = IGNORED
    return labels


def prf_at(scores: np.ndarray, labels: np.ndarray, n_gt: int, thr: float) -> tuple[float, float, float]:
    """Precision, recall and F1 of all detections scoring at least ``thr``."""
    keep = (scores >= thr) & (labels != IGNORED)
    tp = int(((labels == TP) & keep).sum())
    fp = int(((labels == FP) & keep).sum())
    return _prf(tp, fp, n_gt)


def _prf(tp: int, fp: int, n_gt: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class PRCurve:
    precision: np.ndarray  # after each detection, best score first
    recall: np.ndarray
    scores: np.ndarray
    f1: np.ndarray


def pr_curve(scores: np.ndarray, labels: np.ndarray, n_gt: int) -> PRCurve:
    """Precision and recall as the confidence threshold falls through every detection."""
    keep = labels != IGNORED
    s, lab = np.asarray(scores, float)[keep], np.asarray(labels)[keep]
    order = np.argsort(-s, kind="stable")
    s, lab = s[order], lab[order]
    tp = np.cumsum(lab == TP)
    fp = np.cumsum(lab == FP)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_gt if n_gt else np.zeros_like(tp, float)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    return PRCurve(precision, recall, s, f1)


def average_precision(scores: np.ndarray, labels: np.ndarray, n_gt: int) -> float:
    """COCO-style AP: precision made monotone, then averaged at 101 recall points 0, 0.01 … 1."""
    c = pr_curve(scores, labels, n_gt)
    if n_gt == 0 or len(c.precision) == 0:
        return 0.0
    prec = np.maximum.accumulate(c.precision[::-1])[::-1]  # best precision at this recall or beyond
    total = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(c.recall, r, side="left")
        total += prec[idx] if idx < len(prec) else 0.0
    return total / 101


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray, n_gt: int) -> tuple[float, float]:
    """The confidence threshold with the highest F1, and that F1. Ties go to the higher threshold."""
    c = pr_curve(scores, labels, n_gt)
    if len(c.f1) == 0:
        return 1.0, 0.0
    # only thresholds at the last detection of a run of equal scores are reachable
    last_of_run = np.append(c.scores[1:] != c.scores[:-1], True)
    f1 = np.where(last_of_run, c.f1, -1)
    i = int(np.argmax(f1))
    return float(c.scores[i]), float(c.f1[i])


# ---- counting ----------------------------------------------------------------------------------


def counting_metrics(true: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    """MAE, RMSE, MAPE (images with people only), mean signed error (negative = undercount)."""
    t, p = np.asarray(true, float), np.asarray(pred, float)
    err = p - t
    nz = t > 0
    return {
        "n": int(len(t)),
        "mae": float(np.abs(err).mean()) if len(t) else float("nan"),
        "rmse": float(np.sqrt((err**2).mean())) if len(t) else float("nan"),
        "mape": float((np.abs(err[nz]) / t[nz]).mean() * 100) if nz.any() else float("nan"),
        "mean_signed_error": float(err.mean()) if len(t) else float("nan"),
        "n_zero_true": int((~nz).sum()),
    }


# ---- crowding levels ---------------------------------------------------------------------------


def confusion(true_levels: Sequence[int], pred_levels: Sequence[int], k: int = 5) -> np.ndarray:
    """k×k matrix: rows are true levels 1..k, columns predicted levels 1..k."""
    m = np.zeros((k, k), int)
    for t, p in zip(true_levels, pred_levels, strict=True):
        m[t - 1, p - 1] += 1
    return m


def level_metrics(true_levels: Sequence[int], pred_levels: Sequence[int]) -> dict:
    """Per-level precision/recall/F1, macro-F1 over levels that occur in the truth, accuracy."""
    m = confusion(true_levels, pred_levels)
    per = []
    for i, name in enumerate(NAMES):
        tp, fp, fn = m[i, i], m[:, i].sum() - m[i, i], m[i, :].sum() - m[i, i]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per.append({"level": i + 1, "name": name, "support": int(m[i, :].sum()), "predicted": int(m[:, i].sum()),
                    "precision": float(p), "recall": float(r), "f1": float(f)})
    present = [x["f1"] for x in per if x["support"] > 0]
    n = int(m.sum())
    t, p = np.asarray(true_levels), np.asarray(pred_levels)
    return {
        "per_level": per,
        "macro_f1": float(np.mean(present)) if present else 0.0,
        "accuracy": float(np.trace(m) / n) if n else 0.0,
        "within_one": float((np.abs(t - p) <= 1).mean()) if n else 0.0,
        "confusion": m.tolist(),
    }


# ---- uncertainty -------------------------------------------------------------------------------


def bootstrap_ci(stat: Callable[[np.ndarray], float], n_items: int, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap: resample item indices with replacement, recompute ``stat(indices)``."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n_items, n_items)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
