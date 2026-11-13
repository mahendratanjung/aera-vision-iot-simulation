"""Score every candidate the same way and pick the model. Writes model/results/.

    uv run python -m aera_vision.evaluate

Order of work, so the test split never influences a choice:

1. On the **tune** split (1,000 images), each detector's score threshold is set to the value with
   the best F1 at IoU 0.5. The hybrid's switch point and the sensor's confidence are also set here.
2. The model is chosen on the tune split by the rule fixed in PLAN.md (see ``choose_winner``).
3. Every candidate is then scored once on the **test** split (3,370 images), with 95 % bootstrap
   intervals. Those are the numbers reported.

Inputs: cached outputs from ``aera_vision.infer`` and speeds from ``aera_vision.bench``.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass

import numpy as np

from aera_vision import candidates, data, metrics
from aera_vision.infer import load_preds
from aera_vision.levels import CAR_CAPACITY, NAMES, VIEW_CAPACITY, view_level

RESULTS = data.ROOT / "results"
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
N_BOOT = 2000
BOOT_SEED = 7
MIN_CPU_FPS = 2.0  # speed rule: at least 2 frames per second on this Mac's CPU alone (edge proxy)
TIE_MACRO_F1 = 0.01
AERA_CAMERA_SD = 8.0  # AERA's crowdsource/fusion.py SOURCE_SD["camera"], points of car capacity
AERA_MIN_CONFIDENCE = 0.3  # AERA's sensors.car_observation floor


# ---- per-image scoring -------------------------------------------------------------------------


def gt_for(frame: data.Frame, kind: str, box: str = "v") -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth boxes and ignore regions for a detector of this kind."""
    if kind == "head":
        return frame.hbox, np.vstack([frame.zones, frame.head_ignore])
    return (frame.vbox if box == "v" else frame.fbox), frame.zones


def detector_count(out: candidates.Output, frame: data.Frame, thr: float) -> int:
    keep = out.scores >= thr
    return int((~metrics.centres_inside(out.boxes[keep], frame.zones)).sum())


def density_count(out: candidates.Output, frame: data.Frame) -> float:
    """Density summed over cells whose centre is outside every do-not-count zone."""
    h, w = out.density.shape
    if len(frame.zones) == 0:
        return float(out.density.sum())
    cx = (np.arange(w) + 0.5) * frame.width / w
    cy = (np.arange(h) + 0.5) * frame.height / h
    inside = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in frame.zones:
        inside |= ((cy[:, None] >= y1) & (cy[:, None] <= y2)) & ((cx[None, :] >= x1) & (cx[None, :] <= x2))
    return float(out.density[~inside].sum())


@dataclass
class Matched:
    """Detections of one candidate on a set of images, labelled at every IoU threshold."""
    scores: list[np.ndarray]
    labels: dict[float, list[np.ndarray]]  # IoU threshold -> per image labels
    n_gt: np.ndarray  # per image

    def flat(self, iou: float) -> tuple[np.ndarray, np.ndarray, int]:
        return np.concatenate(self.scores), np.concatenate(self.labels[iou]), int(self.n_gt.sum())


def match_all(preds: dict, frames: dict, ids: list[str], kind: str, box: str = "v",
              ious=IOU_THRESHOLDS) -> Matched:
    scores, labels, n_gt = [], {float(t): [] for t in ious}, []
    for i in ids:
        gt, ign = gt_for(frames[i], kind, box)
        o = preds[i]
        scores.append(o.scores)
        n_gt.append(len(gt))
        for t in ious:
            labels[float(t)].append(metrics.match_image(o.boxes, o.scores, gt, ign, iou_thr=float(t)))
    return Matched(scores, labels, np.array(n_gt))


# ---- summaries ---------------------------------------------------------------------------------


def macro_f1_from_confusion(m: np.ndarray) -> float:
    tp = np.diag(m).astype(float)
    p = np.divide(tp, m.sum(0), out=np.zeros(5), where=m.sum(0) > 0)
    r = np.divide(tp, m.sum(1), out=np.zeros(5), where=m.sum(1) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(5), where=(p + r) > 0)
    present = m.sum(1) > 0
    return float(f[present].mean()) if present.any() else 0.0


def count_summary(true: np.ndarray, pred: np.ndarray, boot: bool) -> dict:
    tl = np.array([view_level(x) for x in true])
    pl = np.array([view_level(x) for x in pred])
    out = {"counting": metrics.counting_metrics(true, pred), "levels": metrics.level_metrics(tl, pl)}
    by = []
    for lv, name in enumerate(NAMES, start=1):
        sel = tl == lv
        row = {"level": lv, "name": name, **metrics.counting_metrics(true[sel], pred[sel])}
        row["level_recall"] = float((pl[sel] == lv).mean()) if sel.any() else float("nan")
        by.append(row)
    out["by_level"] = by
    if boot:
        err = pred - true
        idx_rng = np.random.default_rng(BOOT_SEED)
        draws = [idx_rng.integers(0, len(true), len(true)) for _ in range(N_BOOT)]

        def ci(stat):
            v = np.array([stat(ix) for ix in draws])
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            return [float(lo), float(hi)]

        def conf(ix):
            return np.bincount((tl[ix] - 1) * 5 + (pl[ix] - 1), minlength=25).reshape(5, 5)

        out["ci95"] = {
            "mae": ci(lambda ix: np.abs(err[ix]).mean()),
            "rmse": ci(lambda ix: math.sqrt((err[ix] ** 2).mean())),
            "mean_signed_error": ci(lambda ix: err[ix].mean()),
            "level_macro_f1": ci(lambda ix: macro_f1_from_confusion(conf(ix))),
            "level_accuracy": ci(lambda ix: (tl[ix] == pl[ix]).mean()),
        }
        cp = (tl >= 4)
        out["crowded_packed_mean_signed_error"] = float(err[cp].mean()) if cp.any() else float("nan")
    out["per_image"] = {"true": true.tolist(), "pred": [round(float(x), 2) for x in pred]}
    return out


def detection_summary(m: Matched, thr: float, boot: bool) -> dict:
    s, lab, n = m.flat(0.5)
    p, r, f = metrics.prf_at(s, lab, n, thr)
    out = {"threshold": thr, "precision": p, "recall": r, "f1": f,
           "ap50": metrics.average_precision(s, lab, n),
           "ap50_95": float(np.mean([metrics.average_precision(*m.flat(float(t))) for t in IOU_THRESHOLDS]))}
    curve = metrics.pr_curve(s, lab, n)
    pick = np.unique(np.linspace(0, len(curve.recall) - 1, min(200, len(curve.recall))).astype(int)) if len(curve.recall) else []
    out["pr_curve"] = {"recall": [round(float(x), 4) for x in curve.recall[pick]],
                       "precision": [round(float(x), 4) for x in curve.precision[pick]],
                       "score": [round(float(x), 4) for x in curve.scores[pick]]}
    if boot:
        per_tp = np.array([((lab == metrics.TP) & (sc >= thr)).sum() for sc, lab in zip(m.scores, m.labels[0.5], strict=True)])
        per_fp = np.array([((lab == metrics.FP) & (sc >= thr)).sum() for sc, lab in zip(m.scores, m.labels[0.5], strict=True)])
        rng = np.random.default_rng(BOOT_SEED)
        vals = []
        for _ in range(N_BOOT):
            ix = rng.integers(0, len(per_tp), len(per_tp))
            vals.append(metrics._prf(int(per_tp[ix].sum()), int(per_fp[ix].sum()), int(m.n_gt[ix].sum())))
        v = np.array(vals)
        out["ci95"] = {k: [float(np.percentile(v[:, j], 2.5)), float(np.percentile(v[:, j], 97.5))]
                       for j, k in enumerate(["precision", "recall", "f1"])}
    return out


def f1_by_threshold(m: Matched) -> dict:
    """Tune-split F1 against threshold, for the report's threshold chart."""
    s, lab, n = m.flat(0.5)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return {"threshold": grid.tolist(), "f1": [round(metrics.prf_at(s, lab, n, float(t))[2], 4) for t in grid]}


# ---- choices made on the tune split ------------------------------------------------------------


def choose_hybrid_switch(det_tune: np.ndarray, den_tune: np.ndarray, true_tune: np.ndarray) -> tuple[int, dict]:
    """Detector count below the switch, density count at or above it. Best macro-F1 on tune, then MAE."""
    tl = np.array([view_level(x) for x in true_tune])
    best = None
    for t in range(0, 201):
        pred = np.where(det_tune < t, det_tune, den_tune)
        pl = np.array([view_level(x) for x in pred])
        f1 = macro_f1_from_confusion(np.bincount((tl - 1) * 5 + (pl - 1), minlength=25).reshape(5, 5))
        mae = float(np.abs(pred - true_tune).mean())
        if best is None or (f1, -mae) > (best[1], -best[2]):
            best = (t, f1, mae)
    return best[0], {"macro_f1": best[1], "mae": best[2]}


def choose_winner(rows: list[dict]) -> dict:
    """The rule fixed in PLAN.md before any result was seen, applied to tune-split numbers only.

    1. Fast enough: at least MIN_CPU_FPS on this Mac's CPU (a stand-in for the edge device).
    2. Highest crowding-level macro-F1 on tune; candidates within TIE_MACRO_F1 of the best tie.
    3. Ties: smallest |mean signed error| on Crowded and Packed tune images, then the smaller model,
       then the more permissive licence.
    """
    licence_rank = {"MIT": 0, "Apache-2.0": 0}
    ok = [r for r in rows if r.get("cpu_fps") is None or r["cpu_fps"] >= MIN_CPU_FPS]
    best = max(r["tune"]["levels"]["macro_f1"] for r in ok)
    tied = [r for r in ok if r["tune"]["levels"]["macro_f1"] >= best - TIE_MACRO_F1]
    tied.sort(key=lambda r: (abs(r["tune"]["crowded_packed_mean_signed_error"]),
                             r.get("params_m") or 1e9, licence_rank.get(r["licence"], 1)))
    w = tied[0]
    return {"key": w["key"], "label": w["label"], "excluded_too_slow": [r["key"] for r in rows if r not in ok],
            "tied_on_macro_f1": [r["key"] for r in tied], "best_tune_macro_f1": best}


def calibrate_confidence(pred_tune: np.ndarray, true_tune: np.ndarray) -> dict:
    """Counting error by predicted level band on tune, and the confidence the sensor should send.

    AERA reads confidence as an error size: sd = 8 / max(0.3, confidence) points of car capacity.
    A car's count is the sum of its two camera views, so (independent errors) its error in points
    of the 200-person car is sqrt(sd_a^2 + sd_b^2) / 200 * 100. The sim inverts AERA's rule:
    confidence = clamp(8 / sd_car, 0.3, 1).
    """
    pl = np.array([view_level(x) for x in pred_tune])
    err = pred_tune - true_tune
    bands = []
    for lv, name in enumerate(NAMES, start=1):
        sel = pl == lv
        n = int(sel.sum())
        rmse = float(np.sqrt((err[sel] ** 2).mean())) if n else float("nan")
        bias = float(err[sel].mean()) if n else float("nan")
        bands.append({"level": lv, "name": name, "n": n, "rmse_people": rmse, "bias_people": bias})
    # a band with fewer than 10 tune images borrows the larger error of itself and its neighbour below
    for i, b in enumerate(bands):
        b["rmse_used"] = b["rmse_people"]
        if b["n"] < 10 and i > 0:
            known = [x for x in (b["rmse_people"], bands[i - 1]["rmse_used"]) if not math.isnan(x)]
            b["rmse_used"] = max(known) if known else float("nan")
            b["borrowed"] = True
        same = math.sqrt(2) * b["rmse_used"] / CAR_CAPACITY * 100  # both halves in this band
        b["car_sd_points_if_both_halves"] = same
        b["confidence_if_both_halves"] = min(1.0, max(AERA_MIN_CONFIDENCE, AERA_CAMERA_SD / same)) if same > 0 else 1.0
    return {"bands": bands, "aera_camera_sd": AERA_CAMERA_SD, "aera_min_confidence": AERA_MIN_CONFIDENCE,
            "car_capacity": CAR_CAPACITY, "view_capacity": VIEW_CAPACITY,
            "rule": "confidence = clamp(8 / (sqrt(rmse_a^2 + rmse_b^2) / 200 * 100), 0.3, 1), rmse by each half's predicted level"}


# ---- main --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Score every candidate and pick the model.")
    ap.add_argument("--available-only", action="store_true",
                    help="score only the test photos every model was run on (the tune split must be complete)")
    args = ap.parse_args(argv)
    t0 = time.time()
    frames = data.load_frames()
    splits = data.load_splits()
    tune_ids, test_ids = splits["tune"], splits["test"]
    test_planned = len(test_ids)
    if args.available_only:
        done = None
        for spec in candidates.SPECS:
            have = set(load_preds(spec.key))
            done = have if done is None else done & have
        if not set(tune_ids) <= done:
            raise SystemExit("the tune split is not complete; it is needed for every choice")
        test_ids = [i for i in test_ids if i in done]
        print(f"scoring {len(test_ids)} of {test_planned} test photos (the rest were not run)")
    true = {k: np.array([frames[i].count for i in ids], float) for k, ids in (("tune", tune_ids), ("test", test_ids))}
    speed = json.loads((RESULTS / "speed.json").read_text())["candidates"] if (RESULTS / "speed.json").exists() else {}

    rows, counts = [], {}
    for spec in candidates.SPECS:
        preds = load_preds(spec.key)
        missing = [i for i in tune_ids + test_ids if i not in preds]
        if missing:
            print(f"skip {spec.key}: {len(missing)} images not run yet")
            continue
        sp = speed.get(spec.key, {})
        row = {**spec.__dict__, "params_m": sp.get("params_m"), "file_mb": sp.get("file_mb") or candidates.file_mb(spec.key),
               "cpu_fps": sp.get("cpu", {}).get("fps"), "mps_fps": sp.get("mps", {}).get("fps")}
        if spec.kind == "density":
            c_tune = np.array([density_count(preds[i], frames[i]) for i in tune_ids])
            c_test = np.array([density_count(preds[i], frames[i]) for i in test_ids])
            row["threshold"] = None
        else:
            m_tune = match_all(preds, frames, tune_ids, spec.kind, ious=[0.5])
            thr, f1 = metrics.best_f1_threshold(*m_tune.flat(0.5))
            row["threshold"] = round(thr, 4)
            row["tune_f1_curve"] = f1_by_threshold(m_tune)
            m_test = match_all(preds, frames, test_ids, spec.kind)
            row["test_detection"] = detection_summary(m_test, thr, boot=True)
            if spec.kind == "person":  # the same detections against full-body boxes, for reference
                m_full = match_all(preds, frames, test_ids, "person", box="f", ious=[0.5])
                row["test_detection"]["ap50_full_body"] = metrics.average_precision(*m_full.flat(0.5))
            c_tune = np.array([detector_count(preds[i], frames[i], thr) for i in tune_ids], float)
            c_test = np.array([detector_count(preds[i], frames[i], thr) for i in test_ids], float)
        row["tune"] = count_summary(true["tune"], c_tune, boot=False)
        row["tune"]["crowded_packed_mean_signed_error"] = float((c_tune - true["tune"])[true["tune"] >= 75].mean())
        row["test"] = count_summary(true["test"], c_test, boot=True)
        counts[spec.key] = {"tune": c_tune, "test": c_test}
        rows.append(row)
        print(f"{spec.key:20s} tune macro-F1 {row['tune']['levels']['macro_f1']:.3f}  "
              f"test MAE {row['test']['counting']['mae']:.2f}  ({time.time() - t0:.0f} s)", flush=True)

    # hybrid: best detector and best counter by tune macro-F1
    dets = [r for r in rows if r["kind"] != "density"]
    dens = [r for r in rows if r["kind"] == "density"]
    hybrid = None
    if dets and dens:
        d = max(dets, key=lambda r: r["tune"]["levels"]["macro_f1"])
        c = max(dens, key=lambda r: r["tune"]["levels"]["macro_f1"])
        switch, tune_fit = choose_hybrid_switch(counts[d["key"]]["tune"], counts[c["key"]]["tune"], true["tune"])
        h_tune = np.where(counts[d["key"]]["tune"] < switch, counts[d["key"]]["tune"], counts[c["key"]]["tune"])
        h_test = np.where(counts[d["key"]]["test"] < switch, counts[d["key"]]["test"], counts[c["key"]]["test"])
        cpu = [d["cpu_fps"], c["cpu_fps"]]
        hybrid = {"key": "hybrid", "label": f"Hybrid: {d['label']} below {switch}, {c['label']} above",
                  "kind": "hybrid", "family": "Hybrid", "licence": f"{d['licence']} + {c['licence']}",
                  "trained_on": f"{d['trained_on']} + {c['trained_on']}", "source": "", "note": "",
                  "parts": [d["key"], c["key"]], "switch": switch, "switch_tune_fit": tune_fit, "threshold": d["threshold"],
                  "params_m": (d["params_m"] or 0) + (c["params_m"] or 0) if d["params_m"] and c["params_m"] else None,
                  "file_mb": (d["file_mb"] or 0) + (c["file_mb"] or 0) if d["file_mb"] and c["file_mb"] else None,
                  "cpu_fps": 1 / sum(1 / x for x in cpu) if all(cpu) else None, "mps_fps": None,
                  "tune": count_summary(true["tune"], h_tune, boot=False), "test": count_summary(true["test"], h_test, boot=True)}
        hybrid["tune"]["crowded_packed_mean_signed_error"] = float((h_tune - true["tune"])[true["tune"] >= 75].mean())
        counts["hybrid"] = {"tune": h_tune, "test": h_test}
        rows.append(hybrid)

    winner = choose_winner(rows)
    calib = calibrate_confidence(counts[winner["key"]]["tune"], true["tune"])
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": "uv run python -m aera_vision.evaluate" + (" --available-only" if args.available_only else ""),
        "dataset": {**data.SOURCE, "images": len(tune_ids) + len(test_ids), "tune": len(tune_ids), "test": len(test_ids),
                    "split_seed": splits["seed"], "splits_sha256": splits["sha256"],
                    "test_levels": {n: int((np.array([view_level(x) for x in true["test"]]) == i + 1).sum())
                                    for i, n in enumerate(NAMES)},
                    "tune_levels": {n: int((np.array([view_level(x) for x in true["tune"]]) == i + 1).sum())
                                    for i, n in enumerate(NAMES)},
                    "people_counted_test": int(true["test"].sum()),
                    "test_planned": test_planned, "test_not_run": test_planned - len(test_ids),
                    "test_note": None if test_planned == len(test_ids) else
                    f"{test_planned - len(test_ids)} of the {test_planned} test photos were not run (the laptop was "
                    f"overloaded and restarted); the scores use the other {len(test_ids)}. The split itself is unchanged."},
        "rules": {"view_capacity": VIEW_CAPACITY, "car_capacity": CAR_CAPACITY, "iou": 0.5,
                  "threshold": "F1-optimal at IoU 0.5 on the tune split", "score_floor": candidates.SCORE_FLOOR,
                  "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "unit": "image", "interval": "95% percentile"},
                  "speed_rule_cpu_fps": MIN_CPU_FPS, "speed_rule_applied": bool(speed),
                  "speed_note": None if speed else "Speed was not measured, to keep the laptop free, so the speed "
                  "rule could not be applied; published edge figures are quoted instead.",
                  "tie_macro_f1": TIE_MACRO_F1,
                  "macro_f1": "mean F1 over the levels that occur in the truth"},
        "winner": winner,
        "calibration": calib,
        "candidates": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(clean(summary), indent=1, allow_nan=False))
    write_csvs(rows, winner)
    print(f"winner: {winner['label']}  ({time.time() - t0:.0f} s)")


def clean(x):
    """NaN and infinity become null, so the JSON is valid for every reader (JavaScript included)."""
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    return x


def write_csvs(rows: list[dict], winner: dict) -> None:
    def ci(d, k):
        v = d.get("ci95", {}).get(k)
        return (round(v[0], 4), round(v[1], 4)) if v else ("", "")

    with open(RESULTS / "candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "label", "kind", "licence", "params_m", "file_mb", "cpu_fps", "mps_fps", "threshold",
                    "test_precision", "test_recall", "test_f1", "test_f1_lo", "test_f1_hi", "test_ap50", "test_ap50_95",
                    "test_mae", "test_mae_lo", "test_mae_hi", "test_rmse", "test_mape", "test_mean_signed_error",
                    "test_level_macro_f1", "test_level_macro_f1_lo", "test_level_macro_f1_hi", "test_level_accuracy",
                    "tune_level_macro_f1", "winner"])
        for r in rows:
            d = r.get("test_detection", {})
            t = r["test"]
            w.writerow([r["key"], r["label"], r["kind"], r["licence"], r.get("params_m"), r.get("file_mb"),
                        r.get("cpu_fps"), r.get("mps_fps"), r.get("threshold"),
                        d.get("precision", ""), d.get("recall", ""), d.get("f1", ""), *ci(d, "f1"),
                        d.get("ap50", ""), d.get("ap50_95", ""),
                        t["counting"]["mae"], *ci(t, "mae"), t["counting"]["rmse"], t["counting"]["mape"],
                        t["counting"]["mean_signed_error"], t["levels"]["macro_f1"], *ci(t, "level_macro_f1"),
                        t["levels"]["accuracy"], r["tune"]["levels"]["macro_f1"], r["key"] == winner["key"]])
    with open(RESULTS / "by_level.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "level", "name", "n", "mae", "rmse", "mape", "mean_signed_error", "level_recall"])
        for r in rows:
            for b in r["test"]["by_level"]:
                w.writerow([r["key"], b["level"], b["name"], b["n"], b["mae"], b["rmse"], b["mape"],
                            b["mean_signed_error"], b["level_recall"]])
    with open(RESULTS / "confusion.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "true_level", *[f"pred_{n}" for n in NAMES]])
        for r in rows:
            for n, line in zip(NAMES, r["test"]["levels"]["confusion"], strict=True):
                w.writerow([r["key"], n, *line])


if __name__ == "__main__":
    main()
