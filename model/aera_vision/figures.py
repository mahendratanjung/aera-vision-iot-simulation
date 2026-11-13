"""Report figures for the chosen model, drawn from results/summary.json in the plain style of Ultralytics'
training plots (white, framed, no grid, thick "all classes" line, legend outside). Writes ../figures/*.png and a
README with a caption for each figure. Every number comes from the results file; nothing is typed here.

    uv run python -m aera_vision.figures
"""

from __future__ import annotations

import json

import numpy as np

from aera_vision.chosen import winner
from aera_vision.evaluate import RESULTS
from aera_vision.levels import NAMES

OUT = RESULTS.parent.parent / "figures"
DPI = 250
CLASS = "head"
THIN, THICK = "#1f9bb6", "#0000ff"  # class curve, all-classes curve (as in Ultralytics' plots)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    plt.rcParams.update({"font.size": 10, "axes.grid": False, "axes.titlesize": 12})
    s = json.loads((RESULTS / "summary.json").read_text())
    m = winner()
    d, t = m["test_detection"], m["test"]
    pr = d["pr_curve"]
    P, Rc, S = (np.array(pr[k]) for k in ("precision", "recall", "score"))
    F1 = np.where(P + Rc > 0, 2 * P * Rc / np.maximum(P + Rc, 1e-12), 0)
    thr = d["threshold"]
    n_test = s["dataset"]["test"]
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    captions = []

    def curve_fig():
        fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig, ax

    def save(fig, ax, name, caption, legend=True):
        if legend:
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.savefig(OUT / name, dpi=DPI, facecolor="white")
        plt.close(fig)
        captions.append((name, caption))

    # 1 precision-recall
    fig, ax = curve_fig()
    ax.plot(Rc, P, lw=1, color=THIN, label=f"{CLASS} {d['ap50']:.3f}")
    ax.plot(Rc, P, lw=3, color=THICK, label=f"all classes {d['ap50']:.3f} mAP@0.5")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    save(fig, ax, "fig1_precision_recall.png",
         f"Precision-recall curve of {m['label']} on {n_test:,} held-out CrowdHuman photos, detections matched to labelled heads at IoU 0.5. "
         f"mAP@0.5 {d['ap50']:.3f}, mAP@0.5:0.95 {d['ap50_95']:.3f}. At the confidence threshold used ({thr:.3f}): precision {d['precision']:.3f}, "
         f"recall {d['recall']:.3f}.")

    # 2-4 by confidence (x = confidence, ascending)
    x = S[::-1]
    for i, (name, ys) in enumerate([("F1", F1), ("Precision", P), ("Recall", Rc)], start=2):
        y = ys[::-1]
        j = int(np.argmax(y))
        fig, ax = curve_fig()
        ax.plot(x, y, lw=1, color=THIN, label=CLASS)
        ax.plot(x, y, lw=3, color=THICK, label=f"all classes {y[j]:.2f} at {x[j]:.3f}")
        ax.set(xlabel="Confidence", ylabel=name, title=f"{name}-Confidence Curve")
        at_thr = {"F1": d["f1"], "Precision": d["precision"], "Recall": d["recall"]}[name]
        save(fig, ax, f"fig{i}_{name.lower()}_confidence.png",
             f"{name} of {m['label']} against the confidence threshold ({n_test:,} test photos). Highest {name.lower()} {y[j]:.3f} at confidence "
             f"{x[j]:.3f}. The threshold used, {thr:.3f}, was chosen as the best F1 on the separate tuning split; on the test photos it gives "
             f"{name.lower()} {at_thr:.3f}.")

    # 5 confusion matrix of crowding levels
    cm = np.array(t["levels"]["confusion"])
    share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 6.4), tight_layout=True)
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for a in range(5):
        for b in range(5):
            ax.text(b, a, cm[a, b], ha="center", va="center", color="white" if share[a, b] > 0.55 else "black")
    ax.set_xticks(range(5), NAMES)
    ax.set_yticks(range(5), NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix (crowding level)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, ax, "fig5_confusion_matrix.png",
         f"Crowding level from the model's count (predicted) against the labels (true), {n_test:,} test photos, one photo treated as half a car "
         f"(100 people when full). Cells show photo counts; colour is the share of each true level. Accuracy {t['levels']['accuracy']:.3f}, "
         f"macro-F1 {t['levels']['macro_f1']:.3f}.", legend=False)

    # 6 counted vs actual
    pi = t["per_image"]
    hi = max(max(pi["true"]), max(pi["pred"])) * 1.05
    fig, ax = plt.subplots(figsize=(7, 6), tight_layout=True)
    ax.scatter(pi["true"], pi["pred"], s=8, alpha=0.4, color=THICK, linewidths=0, label="test photo")
    ax.plot([0, hi], [0, hi], ls="--", color="black", lw=1, label="perfect count")
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="Actual people", ylabel="Detected people", title="Detected vs Actual Count")
    save(fig, ax, "fig6_counted_vs_actual.png",
         f"People detected by {m['label']} against the labelled count, one dot per test photo ({n_test:,}). Dots below the dashed line are "
         f"undercounts. Mean absolute error {t['counting']['mae']:.2f} people, mean signed error {t['counting']['mean_signed_error']:+.2f}.")

    # 7 comparison of all detectors
    dets = sorted([c for c in s["candidates"] if c.get("test_detection")], key=lambda c: c["test_detection"]["f1"])
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(dets) + 1.5), tight_layout=True)
    yy = np.arange(len(dets))
    for j, (k, col, lab) in enumerate([("precision", "#1f77b4", "Precision"), ("recall", "#ff7f0e", "Recall"), ("f1", "#2ca02c", "F1")]):
        ax.barh(yy + (1 - j) * 0.27, [c["test_detection"][k] for c in dets], height=0.25, color=col, label=lab)
    ax.set_yticks(yy, [c["label"] + ("  (chosen)" if c["key"] == m["key"] else "") for c in dets])
    ax.set(xlim=(0, 1), xlabel="Score on the test photos", title="Precision, Recall and F1 of Every Detector Tested")
    save(fig, ax, "fig7_model_comparison.png",
         f"Precision, recall and F1 of all {len(dets)} detectors tested, used as published without retraining, each at the threshold with the "
         f"best F1 on the tuning split. {m['label']} (chosen) scores highest on all three: precision {d['precision']:.3f}, recall "
         f"{d['recall']:.3f}, F1 {d['f1']:.3f}. Head detectors are scored against head boxes, person detectors against visible-body boxes.")

    # 8 error by crowding level
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    by = t["by_level"]
    ax.bar(NAMES, [b["mae"] or 0 for b in by], color=THICK, width=0.6)
    for k, b in enumerate(by):
        ax.text(k, (b["mae"] or 0) + 0.5, f"{b['mae']:.1f}", ha="center")
    ax.set(ylabel="Average miscount (people)", title="Counting Error by Crowding Level")
    save(fig, ax, "fig8_error_by_level.png",
         "Average counting error per test photo, by crowding level in the labels. Error grows as views fill: "
         + ", ".join(f"{b['name']} {b['mae']:.1f}" for b in by if b["mae"] is not None)
         + " people. Test photos per level: " + ", ".join(f"{b['name']} {b['n']}" for b in by) + ".", legend=False)

    lines = ["# Figures for the report", "",
             f"Generated by `cd model && uv run python -m aera_vision.figures` from `model/results/summary.json` ({s['generated']}).",
             f"Model: **{m['label']}** ({m['source']}, trained on {m['trained_on']}; licence: {m['licence']}). Not retrained.",
             f"Test data: {s['dataset']['name']} ({s['dataset']['licence']}), {n_test:,} held-out photos. {s['dataset'].get('test_note') or ''}", ""]
    for name, cap in captions:
        lines += [f"## {name}", "", f"![{name}]({name})", "", cap, ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print(f"wrote {len(captions)} figures to {OUT}")


if __name__ == "__main__":
    main()
