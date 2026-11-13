"""The test data: CrowdHuman validation set, and its fixed split into tune and test.

CrowdHuman (Shao et al., 2018; https://www.crowdhuman.org, CC BY-NC 4.0, non-commercial research)
labels every person with three boxes: full body (fbox), visible part (vbox) and head (hbox).
Areas the annotators chose not to label (crowds too dense to box, reflections, posters) are
marked as ``mask`` boxes; some person boxes carry ``extra.ignore = 1``. Together these are the
**do-not-count zones**.

Counting rule, the same for truth and every model: a person is counted when the centre of their
box lies outside every do-not-count zone. Truth uses the visible box; detectors use their own
box; density counters drop the density that falls in the zones.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

from aera_vision.levels import view_level
from aera_vision.metrics import centres_inside

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "crowdhuman"
ANNOTATIONS = RAW / "annotation_val.odgt"
IMAGES = RAW / "Images"
SPLITS = ROOT / "data" / "splits.json"

SOURCE = {
    "name": "CrowdHuman validation set",
    "url": "https://huggingface.co/datasets/sshao0516/CrowdHuman",
    "homepage": "https://www.crowdhuman.org",
    "licence": "CC BY-NC 4.0 (non-commercial research and education)",
    "files": {
        "CrowdHuman_val.zip": "c0ab99bb80ac162cd3efdf94a1a6100c4f059a61d69596412c6b44ebc20d1363",
        "annotation_val.odgt": "be422c79a190ff7e30fe5cbd74cbf45a2dadda6c5af58c6ec11a038ba2993c04",
    },
}
SPLIT_SEED = 2026
TUNE_SIZE = 1000


def _xyxy(b: list[float]) -> list[float]:
    x, y, w, h = b
    return [x, y, x + w, y + h]


@dataclass
class Frame:
    id: str
    width: int
    height: int
    vbox: np.ndarray  # people to count, visible boxes (N, 4)
    fbox: np.ndarray  # the same people, full-body boxes
    hbox: np.ndarray  # heads of those people whose head is labelled (M, 4)
    zones: np.ndarray  # do-not-count zones (K, 4)
    head_ignore: np.ndarray  # head boxes to ignore when scoring head detectors
    counted: np.ndarray = field(default=None)  # mask over vbox: centre outside every zone

    @property
    def path(self) -> Path:
        return IMAGES / f"{self.id}.jpg"

    @property
    def count(self) -> int:
        return int(self.counted.sum())

    @property
    def level(self) -> int:
        return view_level(self.count)


def _parse(row: dict) -> Frame:
    v, f, h, zones, head_ignore = [], [], [], [], []
    for b in row["gtboxes"]:
        if b["tag"] != "person":
            zones.append(_xyxy(b["fbox"]))
            continue
        if b.get("extra", {}).get("ignore", 0) == 1:
            zones.append(_xyxy(b["fbox"]))
            if "hbox" in b:
                head_ignore.append(_xyxy(b["hbox"]))
            continue
        v.append(_xyxy(b["vbox"]))
        f.append(_xyxy(b["fbox"]))
        if "hbox" in b:
            if b.get("head_attr", {}).get("ignore", 0) == 1:
                head_ignore.append(_xyxy(b["hbox"]))
            else:
                h.append(_xyxy(b["hbox"]))
    arr = lambda x: np.asarray(x, float).reshape(-1, 4)  # noqa: E731
    with Image.open(IMAGES / f"{row['ID']}.jpg") as im:
        w, hh = im.size
    fr = Frame(row["ID"], w, hh, arr(v), arr(f), arr(h), arr(zones), arr(head_ignore))
    fr.counted = ~centres_inside(fr.vbox, fr.zones)
    return fr


@cache
def load_frames() -> dict[str, Frame]:
    """Every validation image, by id."""
    with open(ANNOTATIONS) as fh:
        rows = [json.loads(line) for line in fh]
    return {r["ID"]: _parse(r) for r in rows}


def make_splits(frames: dict[str, Frame]) -> dict:
    """Split once into tune and test, stratified by crowding level so both hold every level."""
    rng = np.random.default_rng(SPLIT_SEED)
    ids = sorted(frames)
    by_level: dict[int, list[str]] = {}
    for i in ids:
        by_level.setdefault(frames[i].level, []).append(i)
    tune: list[str] = []
    for level in sorted(by_level):
        group = by_level[level]
        k = round(len(group) * TUNE_SIZE / len(ids))
        tune += [group[j] for j in rng.permutation(len(group))[:k]]
    tune_set = set(tune)
    test = [i for i in ids if i not in tune_set]
    out = {"seed": SPLIT_SEED, "source": SOURCE, "tune": sorted(tune), "test": test}
    out["sha256"] = splits_hash(out)
    return out


def splits_hash(s: dict) -> str:
    return hashlib.sha256(json.dumps({"tune": s["tune"], "test": s["test"]}, sort_keys=True).encode()).hexdigest()


def load_splits() -> dict:
    """The frozen split. Refuses to go on if the file was edited by hand."""
    s = json.loads(SPLITS.read_text())
    if splits_hash(s) != s["sha256"]:
        raise SystemExit("data/splits.json changed since it was frozen; the test set must not move")
    return s
