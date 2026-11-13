"""Build the app's camera photo bank from the held-out test split.

    uv run python -m aera_vision.demo

Writes ../app/public/demo/frames/*.jpg and ../app/public/demo/bank.json. Each photo:

- comes from the CrowdHuman **test** split (never used for any choice);
- has every face blurred before it is written: all labelled head boxes, heads found by the
  CrowdHuman head detector, and all do-not-count zones (unlabelled crowds, reflections);
- carries the labelled count and the chosen model's own output on it (from the evaluation's
  cached outputs), so the app shows exactly what was scored.

Photos are spread across counts 0 to 150 (two per 3-person step) so the simulation can show any
crowding level. Photos listed in data/demo_exclude.txt (stock-photo watermarks, found by looking at
every photo) are skipped, and only photos listed in data/demo_reviewed.txt are used, so a
skipped photo is never replaced by one nobody has looked at.
CrowdHuman is CC BY-NC 4.0: non-commercial use with attribution.
"""

from __future__ import annotations

import hashlib
import json
import shutil

import cv2
import numpy as np

from aera_vision import data
from aera_vision.chosen import needed_keys, view_of, winner
from aera_vision.evaluate import RESULTS
from aera_vision.infer import load_preds

APP = data.ROOT.parent / "app" / "public" / "demo"
EXCLUDE = data.ROOT / "data" / "demo_exclude.txt"  # photo ids left out of the demo, one per line, "# reason"
REVIEWED = data.ROOT / "data" / "demo_reviewed.txt"  # only photos looked at by eye may be used
MAX_SIDE = 960
STEP, TOP, PER_STEP = 3, 150, 2
HEAD_SCORE = 0.25


def blur_regions(img: np.ndarray, boxes: np.ndarray, grow: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        a, b = int(max(0, x1 - bw * grow)), int(max(0, y1 - bh * grow))
        c, d = int(min(w, x2 + bw * grow)), int(min(h, y2 + bh * grow))
        if c - a < 2 or d - b < 2:
            continue
        k = max(15, (max(c - a, d - b) // 2) | 1)
        img[b:d, a:c] = cv2.GaussianBlur(img[b:d, a:c], (k, k), 0)
    return img


def pick(frames: dict[str, data.Frame], ids: list[str]) -> list[str]:
    """Two photos per 3-person step, landscape-ish, few unlabelled areas; stable choice by hash."""
    def ok(f: data.Frame) -> bool:
        zone_area = sum((z[2] - z[0]) * (z[3] - z[1]) for z in f.zones) / (f.width * f.height)
        return 1.2 <= f.width / f.height <= 2.0 and f.width >= 600 and zone_area < 0.15

    def key(i: str) -> str:
        return hashlib.sha256(i.encode()).hexdigest()

    skip = set()
    if EXCLUDE.exists():
        skip = {line.split("#")[0].strip() for line in EXCLUDE.read_text().splitlines() if line.split("#")[0].strip()}
    if REVIEWED.exists():
        ids = [i for i in ids if i in {x.strip() for x in REVIEWED.read_text().splitlines() if x and not x.startswith("#")}]
    chosen = []
    for lo in range(0, TOP + 1, STEP):
        pool = sorted((i for i in ids if lo <= frames[i].count < lo + STEP and ok(frames[i]) and i not in skip), key=key)
        chosen += pool[:PER_STEP]
    return chosen


def main() -> None:
    row = winner()
    summary = json.loads((RESULTS / "summary.json").read_text())
    by_key = {c["key"]: c for c in summary["candidates"]}
    frames = data.load_frames()
    preds = {k: load_preds(k) for k in set(needed_keys(row)) | {"yolov5m_ch_head"}}
    test = [i for i in data.load_splits()["test"] if all(i in p for p in preds.values())]  # only photos that were run
    ids = pick(frames, test)
    if APP.exists():
        shutil.rmtree(APP)
    (APP / "frames").mkdir(parents=True)
    bank = []
    for n, fid in enumerate(ids):
        f = frames[fid]
        img = cv2.imread(str(f.path))
        v = view_of(row, {k: preds[k][fid] for k in needed_keys(row)}, f, by_key)
        heads_found = preds["yolov5m_ch_head"][fid]
        found = heads_found.boxes[heads_found.scores >= HEAD_SCORE]
        img = blur_regions(img, np.vstack([f.hbox, f.head_ignore, found]))
        img = blur_regions(img, f.zones, grow=0.0)
        s = min(1.0, MAX_SIDE / max(f.width, f.height))
        img = cv2.resize(img, (round(f.width * s), round(f.height * s)), interpolation=cv2.INTER_AREA)
        name = f"{n:03d}.jpg"
        cv2.imwrite(str(APP / "frames" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        entry = {"id": fid, "file": name, "w": img.shape[1], "h": img.shape[0], "truth": f.count, "model": int(round(v.count))}
        if len(v.boxes):
            entry["boxes"] = np.round(v.boxes * s, 1).tolist()
        if len(v.points):
            entry["points"] = [[round(x * s, 1), round(y * s, 1), round(float(w), 3)] for x, y, w in v.points]
        bank.append(entry)
    out = {"model": {"key": row["key"], "label": row["label"], "kind": row["kind"], "threshold": row.get("threshold")},
           "attribution": "CrowdHuman validation set (Shao et al., 2018), CC BY-NC 4.0, test split, faces blurred.",
           "generated_from": summary["generated"], "frames": bank}
    (APP / "bank.json").write_text(json.dumps(out))
    counts = [b["truth"] for b in bank]
    print(f"{len(bank)} photos, labelled counts {min(counts)} to {max(counts)}, model {row['label']}")


if __name__ == "__main__":
    main()
