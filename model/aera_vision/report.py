"""Write MODEL_REPORT.md and results/charts/*.png from results/summary.json and results/speed.json.

    uv run python -m aera_vision.report

Every number in the report is read from those files. The prose around them is fixed text; where a
sentence depends on the results (which model won, whether it undercounts), the script decides it.
Chart colours come from the app's theme (app/styles/theme.css), the project's only colour source.
"""

from __future__ import annotations

import json
import re

import numpy as np

from aera_vision import candidates, data
from aera_vision.evaluate import RESULTS
from aera_vision.levels import NAMES

ROOT = data.ROOT
CHARTS = RESULTS / "charts"
THEME = ROOT.parent / "app" / "styles" / "theme.css"


def tokens() -> dict[str, str]:
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", THEME.read_text()))


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def num(x, d=1):
    return "n/a" if x is None else f"{x:.{d}f}"


def signed(x, d=1):
    return "n/a" if x is None else f"{x:+.{d}f}"


def ci(c, f=num, d=1):
    return "" if not c else f" ({f(c[0], d)} to {f(c[1], d)})"


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---- charts ------------------------------------------------------------------------------------


def charts(s: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tokens()
    series = [t["accent"], t["highlight"], t["accent-2"]]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": t["rule-strong"], "axes.labelcolor": t["ink-2"],
                         "xtick.color": t["ink-3"], "ytick.color": t["ink-3"], "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": t["rule"], "grid.linewidth": 0.8,
                         "font.family": "sans-serif"})
    CHARTS.mkdir(parents=True, exist_ok=True)
    by = {c["key"]: c for c in s["candidates"]}
    w = by[s["winner"]["key"]]
    base = by["yolo11n"]

    # level macro-F1 on test, every candidate, with 95% interval
    rows = sorted(s["candidates"], key=lambda c: c["test"]["levels"]["macro_f1"])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1))
    y = np.arange(len(rows))
    vals = [c["test"]["levels"]["macro_f1"] for c in rows]
    lo = [v - c["test"]["ci95"]["level_macro_f1"][0] for v, c in zip(vals, rows, strict=True)]
    hi = [c["test"]["ci95"]["level_macro_f1"][1] - v for v, c in zip(vals, rows, strict=True)]
    colors = [series[0] if c["key"] == w["key"] else (series[1] if c["key"] == "yolo11n" else t["rule-strong"]) for c in rows]
    ax.barh(y, vals, height=0.6, color=colors)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=t["ink-2"], elinewidth=1, capsize=2)
    ax.set_yticks(y, [c["label"] for c in rows])
    ax.set_xlabel("Crowding-level macro-F1 on the test split (95% interval)")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(CHARTS / "level_macro_f1.png", dpi=160)
    plt.close(fig)

    # error by level: winner vs today's model
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    x = np.arange(5)
    for ax, field, title in ((axes[0], "mae", "Average miscount (people)"), (axes[1], "mean_signed_error", "Signed error (people)")):
        for i, c in enumerate((w, base)):
            v = [b[field] if b[field] is not None else np.nan for b in c["test"]["by_level"]]
            ax.bar(x + (i - 0.5) * 0.36, v, width=0.34, color=series[i], label=c["label"])
        ax.set_xticks(x, NAMES, rotation=20)
        ax.set_title(title, fontsize=9, color=t["ink"])
        ax.axhline(0, color=t["ink-3"], linewidth=0.8)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(CHARTS / "error_by_level.png", dpi=160)
    plt.close(fig)

    # predicted vs true, winner
    pi = w["test"]["per_image"]
    fig, ax = plt.subplots(figsize=(4, 4))
    m = max(max(pi["true"]), max(pi["pred"])) * 1.05
    ax.scatter(pi["true"], pi["pred"], s=6, alpha=0.35, color=series[0], linewidths=0)
    ax.plot([0, m], [0, m], linestyle="--", color=t["ink-3"], linewidth=1)
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_xlabel("People in the labels")
    ax.set_ylabel("People counted")
    ax.set_title(w["label"], fontsize=8, color=t["ink"])
    fig.tight_layout()
    fig.savefig(CHARTS / "scatter_winner.png", dpi=160)
    plt.close(fig)

    # confusion matrix, winner
    cm = np.array(w["test"]["levels"]["confusion"])
    share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("acc", [t["surface-2"], t["accent"]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(share * 0.6, cmap=cmap, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", color=t["ink"], fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(5), NAMES, rotation=30)
    ax.set_yticks(range(5), NAMES)
    ax.set_xlabel("Level from the model's count")
    ax.set_ylabel("Level in the labels")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(CHARTS / "confusion_winner.png", dpi=160)
    plt.close(fig)

    # precision-recall, detectors (test)
    fig, ax = plt.subplots(figsize=(5, 4))
    dets = [c for c in s["candidates"] if c.get("test_detection")]
    for c in dets:
        pr = c["test_detection"]["pr_curve"]
        hl = {w.get("parts", [w["key"]])[0]: 0, "yolo11n": 1}
        color = series[hl[c["key"]]] if c["key"] in hl else t["rule-strong"]
        ax.plot(pr["recall"], pr["precision"], color=color, linewidth=2 if c["key"] in hl else 1,
                label=c["label"] if c["key"] in hl else None)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("All detectors (grey) on the test split, IoU 0.5", fontsize=8, color=t["ink"])
    fig.tight_layout()
    fig.savefig(CHARTS / "pr_curves.png", dpi=160)
    plt.close(fig)


# ---- report ------------------------------------------------------------------------------------


def report(s: dict, speed: dict | None) -> str:
    by = {c["key"]: c for c in s["candidates"]}
    w = by[s["winner"]["key"]]
    base = by["yolo11n"]
    ds = s["dataset"]
    t = w["test"]
    det_part = by[w["parts"][0]] if w["kind"] == "hybrid" else (w if w.get("test_detection") else None)
    frames = data.load_frames()
    dropped = sum(len(f.vbox) - f.count for f in frames.values())
    total = sum(len(f.vbox) for f in frames.values())
    by_level = {b["name"]: b for b in t["by_level"]}
    cp_bias = [by_level[n]["mean_signed_error"] for n in ("Crowded", "Packed") if by_level[n]["mean_signed_error"] is not None]
    under = cp_bias and all(b < 0 for b in cp_bias)
    worst = max((b for b in t["by_level"] if b["n"]), key=lambda b: b["mae"])
    specs = {sp.key: sp for sp in candidates.SPECS}
    L = []
    add = L.append

    add("# AERA Vision: model report\n")
    add(f"Generated by `aera_vision.report` from `results/summary.json` ({s['generated']}). Do not edit by hand: "
        "rerun `make eval` and every number below is rewritten.\n")

    add("## Summary\n")
    det_txt = ""
    if det_part:
        d = det_part["test_detection"]
        det_txt = (f" Its {'head' if det_part['kind'] == 'head' else 'person'} detector ({det_part['label']}) finds people with "
                   f"precision {pct(d['precision'])}{ci(d['ci95']['precision'], pct)}, recall {pct(d['recall'])}{ci(d['ci95']['recall'], pct)} "
                   f"and F1 {pct(d['f1'])}{ci(d['ci95']['f1'], pct)} at IoU 0.5.")
    n_ready = sum(1 for c in s["candidates"] if c["kind"] != "hybrid")
    add(f"We compared {n_ready} ready-trained people counters (no retraining), plus a hybrid of the best two, on {ds['test']:,} held-out photos of crowds "
        f"from {ds['name']}" + (f" (the other {ds['test_not_run']:,} planned test photos were not run)" if ds.get("test_not_run") else "") + f". The chosen model is **{w['label']}**. On the test photos it puts a camera view in the right crowding "
        f"level {pct(t['levels']['accuracy'])} of the time{ci(t['ci95']['level_accuracy'], pct)} (macro-F1 "
        f"{num(t['levels']['macro_f1'], 2)}{ci(t['ci95']['level_macro_f1'], num, 2)}) and miscounts by {num(t['counting']['mae'])} "
        f"people on average{ci(t['ci95']['mae'])}.{det_txt} The person detector AERA's edge code uses today ({base['label']}) "
        f"scores macro-F1 {num(base['test']['levels']['macro_f1'], 2)}{ci(base['test']['ci95']['level_macro_f1'], num, 2)} and miscounts "
        f"by {num(base['test']['counting']['mae'])} people{ci(base['test']['ci95']['mae'])}. The chosen model's hardest level is {worst['name']} "
        f"({num(worst['mae'])} people off on average"
        + (f"; it undercounts Crowded and Packed views by {', '.join(num(-b) for b in cp_bias)} people on average" if under else "")
        + "). Only "
        f"{ds['test_levels']['Crowded'] + ds['test_levels']['Packed']} test photos are Crowded or Packed, so those numbers are the least certain.\n")

    add("## 1. Question and set-up\n")
    add("Which ready-trained model counts people best in a crowded train car, seen by a camera on the ceiling, and could run on a "
        "Raspberry Pi 5 with an AI HAT or a Jetson Orin Nano? AERA shows riders five crowding levels by load (count / capacity): "
        "Quiet under 20 %, Comfortable 20-50, Busy 50-75, Crowded 75-95, Packed 95 % and over (`frontend/lib/crowd/levels.ts`). "
        f"A car holds {s['rules']['car_capacity']} people. We assume **two cameras per car**, one at each end, each counting its half, "
        f"so one camera view holds **{s['rules']['view_capacity']} people** at full load and one test photo stands in for one view: "
        f"level = level(count / {s['rules']['view_capacity']}). The same rule is applied to the labelled count and the model's count.\n")

    add("## 2. Data\n")
    add(f"- **Test data:** {ds['name']} ({ds['homepage']}; files from {ds['url']}), licence **{ds['licence']}**. "
        f"{ds['images']:,} photos with a box for every person (full body, visible part, head). SHA-256 of the download: "
        f"`CrowdHuman_val.zip` {ds['files']['CrowdHuman_val.zip']}, `annotation_val.odgt` {ds['files']['annotation_val.odgt']}.")
    add(f"- **Split:** fixed once, seed {ds['split_seed']}, stratified by level: **tune** {ds['tune']:,} photos (thresholds, hybrid "
        f"switch, sensor confidence, choice of model) and **test** {ds['test']:,} photos (reported numbers only). The split file is "
        f"hashed (`{ds['splits_sha256'][:16]}…`) and the code refuses to run if it changes."
        + (f" **{ds['test_note']}**" if ds.get("test_note") else ""))
    add(f"- **Do-not-count zones:** CrowdHuman marks areas it did not label (dense background crowds, reflections, posters) and some "
        f"people as ignored. A person is counted only if the centre of their box is outside every such zone, for the labels and for "
        f"every model alike (density counters lose the density inside zones). This removes {dropped:,} of {total:,} labelled people "
        f"({dropped / total:.1%}).")
    add("- **Training data of the models** (not used here): COCO 2017 for the YOLO and RF-DETR person detectors; CrowdHuman train for "
        "the YOLOv5m CrowdHuman model; NWPU-Crowd or ShanghaiTech Part B for ZIP. CrowdHuman, NWPU-Crowd and ShanghaiTech are for "
        "research use only.\n")
    add(table(["Level", "Tune photos", "Test photos"],
              [[n, ds["tune_levels"][n], ds["test_levels"][n]] for n in NAMES]) + "\n")

    add("## 3. Candidates\n")
    add("All used exactly as published. Detectors run at 640 px (Ultralytics, YOLOv5) or their native size (RF-DETR Nano 384, "
        "Small 512, Medium 576 px); ZIP runs on the whole photo. Photos longer than 2,048 px are first scaled down to 2,048 px "
        "for every model alike.\n")
    add(table(["Model", "What it does", "Licence", "Trained on", "Params (M)", "File (MB)", "Source"],
              [[c["label"], c["kind"], c["licence"], c["trained_on"], num(c.get("params_m")), num(c.get("file_mb"), 0),
                c["source"] or ", ".join(c.get("parts", []))] for c in s["candidates"]]) + "\n")
    notes = [f"- {sp.label}: {sp.note}" for sp in specs.values() if sp.note]
    add("Notes:\n\n" + "\n".join(notes) + "\n")
    add("Considered and left out: P2PNet (licence limits use to academic research), APGCC (only ShanghaiTech A weights), "
        "PET and CSRNet (no licence file), DEIMv2 (custom licence), community CrowdHuman head models with unclear training data.\n")

    add("## 4. Method\n")
    add(f"- **Detection.** Boxes down to a score of {s['rules']['score_floor']} are kept. Each is matched to an unmatched labelled box "
        f"(visible-body box for person detectors, head box for head detectors) at IoU >= {s['rules']['iou']}, highest score first; boxes "
        "mostly (>= 50 %) inside a do-not-count zone are ignored. Average precision is the COCO 101-point AP, checked against "
        "pycocotools in the tests. AP is given at IoU 0.5 and averaged over 0.5 to 0.95.")
    add(f"- **Operating threshold:** {s['rules']['threshold']}. The tune-split F1 curve for each detector is in `summary.json` "
        "(`tune_f1_curve`). The same threshold gives the count: boxes at or above it, outside the zones.")
    add("- **Counting:** MAE, RMSE, MAPE (photos with at least one person), mean signed error (negative = undercount), per photo.")
    add(f"- **Levels:** per-level precision, recall and F1, macro-F1 ({s['rules']['macro_f1']}), accuracy, and a 5 x 5 confusion matrix.")
    add(f"- **Uncertainty:** {s['rules']['bootstrap']['interval']} bootstrap intervals, {s['rules']['bootstrap']['resamples']:,} "
        f"resamples of photos, seed {s['rules']['bootstrap']['seed']}.")
    add("- **Hybrid:** the best detector and the best density counter on tune; below a switch count the detector's count is used, at or "
        "above it the counter's. The switch is the value from 0 to 200 with the best tune macro-F1. Because it is tuned on the same photos "
        "used to rank models, its tune score is slightly optimistic; its test score is not.")
    add(f"- **Choice of model (fixed before any result):** at least {s['rules']['speed_rule_cpu_fps']} frames a second on this "
        f"laptop's CPU alone (a stand-in for the edge device)"
        + (f" ({s['rules']['speed_note']})" if s['rules'].get('speed_note') else "")
        + "; then the best tune macro-F1; candidates within "
        f"{s['rules']['tie_macro_f1']} tie and are separated by the smaller undercount on Crowded and Packed tune photos, then the "
        "smaller model, then the more permissive licence.")
    if speed:
        m = speed["machine"]
        add(f"- **Speed:** {m['chip']}, PyTorch {m['torch']}; one {speed['frame']['size'][0]} x {speed['frame']['size'][1]} frame, "
            "batch 1, full predict call, median of 50 runs after 10 warm-ups, CPU and Apple GPU (MPS). The ONNX model runs on the "
            "CPU only.\n")

    add("## 5. Results (test split)\n")
    add("![Level macro-F1 of every candidate](results/charts/level_macro_f1.png)\n")
    add("### Counting and levels\n")
    add(table(["Model", "MAE", "RMSE", "MAPE", "Signed error", "Level macro-F1", "Level accuracy", "Within one level"],
              [[("**" + c["label"] + "**") if c["key"] == w["key"] else c["label"],
                num(c["test"]["counting"]["mae"]) + ci(c["test"]["ci95"]["mae"]), num(c["test"]["counting"]["rmse"]),
                num(c["test"]["counting"]["mape"]) + " %", signed(c["test"]["counting"]["mean_signed_error"]) + ci(c["test"]["ci95"]["mean_signed_error"], signed),
                num(c["test"]["levels"]["macro_f1"], 3) + ci(c["test"]["ci95"]["level_macro_f1"], num, 3), pct(c["test"]["levels"]["accuracy"]),
                pct(c["test"]["levels"]["within_one"])] for c in s["candidates"]]) + "\n")
    add("### Detection\n")
    add(table(["Model", "Threshold", "Precision", "Recall", "F1", "AP@0.5", "AP@0.5:0.95", "AP@0.5 vs full-body boxes"],
              [[c["label"], num(c["threshold"], 3), pct(c["test_detection"]["precision"]) + ci(c["test_detection"]["ci95"]["precision"], pct),
                pct(c["test_detection"]["recall"]) + ci(c["test_detection"]["ci95"]["recall"], pct),
                pct(c["test_detection"]["f1"]) + ci(c["test_detection"]["ci95"]["f1"], pct), pct(c["test_detection"]["ap50"]),
                pct(c["test_detection"]["ap50_95"]), pct(c["test_detection"].get("ap50_full_body"))]
               for c in s["candidates"] if c.get("test_detection")]) + "\n")
    add("Head detectors are scored against head boxes and person detectors against visible-body boxes, so their detection numbers "
        "are not directly comparable; the counting and level tables are.\n")
    add("![Precision-recall curves](results/charts/pr_curves.png)\n")
    add("### By crowding level\n")
    add(table(["Level", "Photos", f"{w['label']} MAE", "signed", f"{base['label']} MAE", "signed"],
              [[b["name"], b["n"], num(b["mae"]), signed(b["mean_signed_error"]), num(bb["mae"]), signed(bb["mean_signed_error"])]
               for b, bb in zip(t["by_level"], base["test"]["by_level"], strict=True)]) + "\n")
    add("![Error by level](results/charts/error_by_level.png)\n")
    add("Every candidate's breakdown is in `results/by_level.csv`.\n")
    add(f"### Chosen model: {w['label']}\n")
    add(table(["Level", "Photos", "Precision", "Recall", "F1"],
              [[p["name"], p["support"], pct(p["precision"]), pct(p["recall"]), pct(p["f1"])] for p in t["levels"]["per_level"]]) + "\n")
    add("![Confusion matrix](results/charts/confusion_winner.png) ![Counted against labelled](results/charts/scatter_winner.png)\n")
    if speed:
        add("### Speed on this laptop\n")
        add(table(["Model", "CPU ms", "CPU fps", "GPU (MPS) ms", "GPU fps", "Params (M)", "File (MB)"],
                  [[by[k]["label"] if k in by else k, num(v.get("cpu", {}).get("ms_median"), 0), num(v.get("cpu", {}).get("fps")),
                    num(v.get("mps", {}).get("ms_median"), 0) if v.get("mps") else "n/a", num(v.get("mps", {}).get("fps")) if v.get("mps") else "n/a",
                    num(v.get("params_m")), num(v.get("file_mb"), 0)] for k, v in speed["candidates"].items()]) + "\n")
        add("RF-DETR is run through its plain `predict` call, which the library says is not optimised for inference; its speeds are "
            "therefore conservative. On the train, models would be exported (TensorRT on Jetson, Hailo on the Pi), which this laptop "
            "cannot do. Published edge figures, not measured here, are in `references/edge_speed.json`.\n")

    add("## 6. Why this model, and its weaknesses\n")
    tied = [by[k]["label"] for k in s["winner"]["tied_on_macro_f1"] if k != w["key"]]
    add(f"- **Chosen by the rule:** best tune macro-F1 ({num(s['winner']['best_tune_macro_f1'], 3)}) among models fast enough"
        + (f"; tied within {s['rules']['tie_macro_f1']} with {', '.join(tied)}, separated by the tie rules" if tied else "")
        + (f"; excluded as too slow: {', '.join(by[k]['label'] for k in s['winner']['excluded_too_slow'])}" if s["winner"]["excluded_too_slow"] else "")
        + ".")
    add(f"- **Against today's model ({base['label']}):** MAE {num(t['counting']['mae'])} vs {num(base['test']['counting']['mae'])} people, "
        f"level macro-F1 {num(t['levels']['macro_f1'], 3)} vs {num(base['test']['levels']['macro_f1'], 3)}.")
    add(f"- **Weakest level:** {worst['name']}, {num(worst['mae'])} people off on average"
        + (f"; Crowded and Packed views are undercounted ({', '.join(signed(b) for b in cp_bias)} people)." if under
           else "; no consistent undercount in Crowded and Packed views on these photos."))
    add(f"- **Licence:** {w['licence']}. For anything beyond a non-commercial prototype the CrowdHuman-trained parts would need "
        "retraining on licensed data, and Ultralytics/YOLOv5 code needs an enterprise licence or replacing with the Apache/MIT "
        "models in the tables.")
    add("- **Data mismatch:** CrowdHuman photos are street and event crowds from the web, mostly at eye level, not ceiling views of "
        "a train car (see section 8).\n")

    add("## 7. What the sensor tells AERA\n")
    add("AERA turns a camera reading's `confidence` into an error of `8 / max(0.3, confidence)` points of car capacity "
        "(`backend/aera_twin/crowdsource/sensors.py`) and weighs readings by inverse variance. The simulated device therefore sends a "
        "confidence calibrated from the counting error measured on the tune split, by the level each camera's count falls in: "
        f"`{s['calibration']['rule']}`.\n")
    add(table(["Level counted", "Tune photos", "RMSE (people)", "Bias (people)", "Car error if both halves here (points)", "Confidence"],
              [[b["name"], b["n"], num(b["rmse_used"]) + (" *" if b.get("borrowed") else ""), signed(b["bias_people"]),
                num(b["car_sd_points_if_both_halves"]), num(b["confidence_if_both_halves"], 2)] for b in s["calibration"]["bands"]]) + "\n")
    add("\\* fewer than 10 tune photos: the larger error of this level and the one below is used.\n")

    add("## 8. Limitations\n")
    add("- **Not train interiors.** No public, freely licensed set of labelled ceiling views inside metro cars was found. CrowdHuman "
        "is the closest large, fully labelled crowd set. Accuracy in a real Innovia car (poles, windows, reflections, people seen "
        "through the doors, fish-eye lenses, night lighting) is not measured here; recording and labelling a few hundred frames in a "
        "car, with Prasarana's permission, is the next step.")
    add("- **One photo = one camera view of 100 people** is an assumption that sets which counts map to which level.")
    add("- **Few Crowded and Packed photos** in the test split; their intervals are wide.")
    add("- **Possible advantage for the YOLOv5m CrowdHuman model:** it was trained on CrowdHuman train and its training probably "
        "used this validation set to choose checkpoints.")
    add("- **Speed not measured**, neither on the laptop (to keep it free) nor on edge hardware (none available); the "
        "quoted figures in `references/edge_speed.json` are for similar YOLO models. Whether RF-DETR and ZIP compile for "
        "the Pi's Hailo chip is unconfirmed.")
    if ds.get("test_note"):
        add(f"- **Part of the test split was not run:** {ds['test_note']}")
    add("- **No retraining**, by decision: fine-tuning on CrowdHuman or on our own train footage would likely change the ranking.\n")

    add("## 9. Reproduce\n")
    add("```bash\nmake data   # CrowdHuman val + weights, checksums verified (model/scripts/get_data.sh)\n"
        "make eval   # inference (about 90 min on an M4 Pro, heavy: better on a free Colab GPU), speed, scoring, this report\n```\n")
    add(f"Command recorded in the results: `{s['command']}`. Inputs and outputs: `data/splits.json` (frozen split), "
        "`.cache/preds/` (raw model outputs, not committed), `results/summary.json`, `results/speed.json`, `results/*.csv`, "
        "`results/charts/`. Tests: `uv run pytest` (metric functions on hand-worked examples, AP against pycocotools, parity "
        "with AERA's levels and contract).\n")
    return "\n".join(L)


def main() -> None:
    s = json.loads((RESULTS / "summary.json").read_text())
    speed_path = RESULTS / "speed.json"
    speed = json.loads(speed_path.read_text()) if speed_path.exists() else None
    charts(s)
    text = report(s, speed)
    out = ROOT / "MODEL_REPORT.md"
    out.write_text(text)
    readme_block(text)
    print(f"wrote {out.relative_to(ROOT.parent)}, {len(list(CHARTS.glob('*.png')))} charts and the README results paragraph")


def readme_block(text: str) -> None:
    """Copy the report's summary paragraph into README.md between the results markers."""
    summary = text.split("## Summary\n", 1)[1].split("\n## ", 1)[0].strip()
    readme = ROOT.parent / "README.md"
    r = readme.read_text()
    a, b = "<!-- results:start -->", "<!-- results:end -->"
    readme.write_text(r[: r.index(a) + len(a)] + "\n" + summary + "\n" + r[r.index(b):])


if __name__ == "__main__":
    main()
