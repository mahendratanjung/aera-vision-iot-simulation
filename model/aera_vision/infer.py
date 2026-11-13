"""Run every candidate on every CrowdHuman validation image and cache the raw outputs.

    uv run python -m aera_vision.infer            # all candidates, resumes where it stopped
    uv run python -m aera_vision.infer --only yolo11n zip_t_nwpu

Raw outputs (boxes and scores down to the 0.01 floor, or density maps) go to
``.cache/preds/<candidate>/<chunk>.npz``. The evaluation reads them; nothing is chosen here.
Each image is decoded once and given to every model in turn.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from aera_vision import candidates, data

CACHE = data.ROOT / ".cache" / "preds"
CHUNK = 250
MAX_SIDE = 2048  # a few photos are up to 6,000 px wide; no model looks at more than 2,048 px


def shrink(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale an oversized image down once, on the CPU, for every model alike. Returns the factor."""
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def predict(model, img: np.ndarray) -> candidates.Output:
    """Run one model on a full-resolution image; boxes come back in original pixels."""
    small, s = shrink(img)
    out = model.predict(small)
    if s < 1.0 and out.density is None:
        out.boxes = out.boxes / s
    return out


def chunk_path(key: str, i: int) -> Path:
    return CACHE / key / f"{i:03d}.npz"


def save_chunk(path: Path, ids: list[str], outs: list[candidates.Output]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if outs[0].density is not None:
        flat = np.concatenate([o.density.ravel() for o in outs])
        shapes = np.array([o.density.shape for o in outs])
        np.savez_compressed(path, ids=np.array(ids), density=flat, shapes=shapes, cells=np.array([o.cell for o in outs]))
    else:
        n = np.array([len(o.scores) for o in outs])
        np.savez_compressed(path, ids=np.array(ids), n=n,
                            boxes=np.concatenate([o.boxes for o in outs]).reshape(-1, 4),
                            scores=np.concatenate([o.scores for o in outs]))


def load_preds(key: str) -> dict[str, candidates.Output]:
    """Every cached output for one candidate, by image id."""
    out: dict[str, candidates.Output] = {}
    for f in sorted((CACHE / key).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        ids = [str(x) for x in z["ids"]]
        if "density" in z:
            pos = 0
            for i, (h, w), c in zip(ids, z["shapes"], z["cells"], strict=True):
                out[i] = candidates.Output(density=z["density"][pos:pos + h * w].reshape(h, w), cell=float(c))
                pos += h * w
        else:
            ends = np.cumsum(z["n"])
            starts = ends - z["n"]
            for i, a, b in zip(ids, starts, ends, strict=True):
                out[i] = candidates.Output(z["boxes"][a:b], z["scores"][a:b])
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="candidate keys (default: all)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--reverse", action="store_true", help="work from the last chunk back (to run a second worker)")
    args = ap.parse_args(argv)
    keys = args.only or [s.key for s in candidates.SPECS]
    frames = data.load_frames()
    splits = data.load_splits()
    ids = splits["tune"] + splits["test"]
    chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
    todo = [(ci, ch) for ci, ch in enumerate(chunks) if not all(chunk_path(k, ci).exists() for k in keys)]
    if args.reverse:
        todo.reverse()
    print(f"{len(ids)} images, {len(keys)} candidates, {len(todo)} of {len(chunks)} chunks to do", flush=True)
    models = {k: candidates.load(k, args.device) for k in keys}
    started = time.time()
    for n, (ci, ch) in enumerate(todo):
        need = [k for k in keys if not chunk_path(k, ci).exists()]  # another worker may have done it
        if not need:
            continue
        outs: dict[str, list[candidates.Output]] = {k: [] for k in need}
        for fid in ch:
            img = cv2.imread(str(frames[fid].path))
            for k in need:
                outs[k].append(predict(models[k], img))
        for k in need:
            save_chunk(chunk_path(k, ci), ch, outs[k])
        el = time.time() - started
        print(f"chunk {ci + 1}/{len(chunks)} done, {el / 60:.1f} min elapsed, "
              f"about {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min left", flush=True)
    meta = {"device": args.device, "machine": platform.platform(), "processor": platform.processor(),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "candidates": keys, "splits_sha256": splits["sha256"]}
    (CACHE / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
