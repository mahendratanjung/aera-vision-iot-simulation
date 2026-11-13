"""Metric functions on small examples worked out by hand."""

import math

import numpy as np
import pytest

from aera_vision import metrics as M


def test_iou_by_hand():
    # two 2x2 squares overlapping in a 1x2 strip: inter 2, union 4 + 4 - 2 = 6
    a = np.array([[0, 0, 2, 2]])
    b = np.array([[1, 0, 3, 2]])
    assert M.iou_matrix(a, b)[0, 0] == pytest.approx(2 / 6)
    assert M.iou_matrix(a, a)[0, 0] == 1
    assert M.iou_matrix(a, np.array([[5, 5, 6, 6]]))[0, 0] == 0


def test_ioa_and_centres():
    det = np.array([[0, 0, 4, 2]])  # area 8
    region = np.array([[2, 0, 10, 10]])  # covers half of it
    assert M.ioa_matrix(det, region)[0, 0] == pytest.approx(0.5)
    assert M.centres_inside(det, region).tolist() == [True]  # centre (2, 1) on the edge counts
    assert M.centres_inside(det, np.array([[3, 0, 10, 10]])).tolist() == [False]
    assert M.centres_inside(det, np.zeros((0, 4))).tolist() == [False]


def test_match_image_tp_fp_ignored():
    gts = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
    ignore = np.array([[100, 100, 200, 200]])
    dets = np.array([
        [0, 0, 10, 10],      # exact match of gt 0 -> TP
        [1, 0, 11, 10],      # also overlaps gt 0 (IoU 0.82) but gt 0 is taken -> FP
        [110, 110, 120, 120],  # inside the ignore region -> ignored
        [50, 50, 60, 60],    # nothing there -> FP
    ])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert M.match_image(dets, scores, gts, ignore).tolist() == [M.TP, M.FP, M.IGNORED, M.FP]


def test_match_prefers_higher_score():
    gts = np.array([[0, 0, 10, 10]])
    dets = np.array([[1, 0, 11, 10], [0, 0, 10, 10]])
    # the lower-scoring box fits better, but the higher-scoring one claims the person first
    assert M.match_image(dets, np.array([0.9, 0.5]), gts, np.zeros((0, 4))).tolist() == [M.TP, M.FP]


def test_prf_and_best_threshold():
    # 3 people; detections best first: TP, FP, TP, FP
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([M.TP, M.FP, M.TP, M.FP])
    p, r, f = M.prf_at(scores, labels, 3, 0.7)
    assert (p, r) == pytest.approx((2 / 3, 2 / 3))
    assert f == pytest.approx(2 / 3)
    # F1 at each cut: 0.9 -> P1 R1/3 F0.5; 0.8 -> P.5 R1/3 F0.4; 0.7 -> F2/3; 0.6 -> P.5 R2/3 F4/7
    thr, best = M.best_f1_threshold(scores, labels, 3)
    assert thr == 0.7 and best == pytest.approx(2 / 3)


def test_ignored_detections_do_not_count():
    scores = np.array([0.9, 0.8])
    labels = np.array([M.TP, M.IGNORED])
    assert M.prf_at(scores, labels, 1, 0.0) == (1.0, 1.0, 1.0)


def test_average_precision_by_hand():
    # 2 people. Detections: TP, FP, TP -> precision 1, 1/2, 2/3 at recall 1/2, 1/2, 1.
    # Monotone precision: recall 0..0.5 -> 1.0, recall 0.51..1.0 -> 2/3.
    # 101 points: 0.00..0.50 is 51 points at 1.0, 0.51..1.00 is 50 points at 2/3.
    ap = M.average_precision(np.array([0.9, 0.8, 0.7]), np.array([M.TP, M.FP, M.TP]), 2)
    assert ap == pytest.approx((51 * 1.0 + 50 * (2 / 3)) / 101)
    # perfect detector
    assert M.average_precision(np.array([0.9, 0.8]), np.array([M.TP, M.TP]), 2) == pytest.approx(1.0)
    # missed half the people entirely: recall never passes 0.5
    assert M.average_precision(np.array([0.9]), np.array([M.TP]), 2) == pytest.approx(51 / 101)


def test_average_precision_matches_pycocotools():
    """Cross-check our AP against the reference COCO implementation on a random example."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(1)
    images, anns, dets = [], [], []
    all_scores, all_labels, n_gt = [], [], 0
    for img_id in range(1, 21):
        images.append({"id": img_id, "width": 500, "height": 500})
        g = rng.uniform(0, 400, (6, 2))
        gts = np.hstack([g, g + rng.uniform(20, 80, (6, 2))])
        for k, b in enumerate(gts):
            anns.append({"id": img_id * 100 + k, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": float((b[2] - b[0]) * (b[3] - b[1]))})
        d = np.vstack([gts[:4] + rng.normal(0, 6, (4, 4)), np.hstack([g2 := rng.uniform(0, 400, (3, 2)), g2 + 40])])
        s = rng.uniform(0, 1, len(d))
        for b, sc in zip(d, s, strict=True):
            dets.append({"image_id": img_id, "category_id": 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "score": float(sc)})
        all_scores.append(s)
        all_labels.append(M.match_image(d, s, gts, np.zeros((0, 4)), iou_thr=0.5))
        n_gt += len(gts)
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
    coco.createIndex()
    ev = COCOeval(coco, coco.loadRes(dets), "bbox")
    ev.params.iouThrs = np.array([0.5])
    ev.params.areaRng, ev.params.areaRngLbl = [[0, 1e10]], ["all"]
    ev.params.maxDets = [100]
    ev.evaluate()
    ev.accumulate()
    ref = float(ev.eval["precision"][0, :, 0, 0, 0].mean())
    ours = M.average_precision(np.concatenate(all_scores), np.concatenate(all_labels), n_gt)
    assert ours == pytest.approx(ref, abs=1e-6)


def test_counting_metrics_by_hand():
    m = M.counting_metrics([10, 20, 0], [12, 14, 1])
    # errors +2, -6, +1
    assert m["mae"] == pytest.approx(3.0)
    assert m["rmse"] == pytest.approx(math.sqrt((4 + 36 + 1) / 3))
    assert m["mape"] == pytest.approx((2 / 10 + 6 / 20) / 2 * 100)  # the empty image is left out
    assert m["mean_signed_error"] == pytest.approx(-1.0)
    assert m["n_zero_true"] == 1


def test_level_metrics_by_hand():
    true = [1, 1, 2, 3, 5]
    pred = [1, 2, 2, 3, 4]
    m = M.level_metrics(true, pred)
    per = {x["name"]: x for x in m["per_level"]}
    assert per["Quiet"]["precision"] == 1.0 and per["Quiet"]["recall"] == 0.5
    assert per["Comfortable"]["precision"] == 0.5 and per["Comfortable"]["recall"] == 1.0
    assert per["Packed"]["f1"] == 0.0 and per["Crowded"]["support"] == 0
    # macro over the four levels present in the truth: Quiet 2/3, Comfortable 2/3, Busy 1, Packed 0
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1 + 0) / 4)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["within_one"] == 1.0
    assert m["confusion"][4][3] == 1


def test_bootstrap_ci_contains_mean_and_is_repeatable():
    x = np.arange(100, dtype=float)
    stat = lambda idx: float(x[idx].mean())  # noqa: E731
    lo, hi = M.bootstrap_ci(stat, len(x), n_boot=500, seed=3)
    assert lo < 49.5 < hi
    assert (lo, hi) == M.bootstrap_ci(stat, len(x), n_boot=500, seed=3)
