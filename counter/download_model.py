"""Download the model file named in model.json and check its SHA-256. Run once at build time."""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / "model.json").read_text())
dest = HERE / "weights" / "crowdhuman.onnx"
dest.parent.mkdir(exist_ok=True)
if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != cfg["weights_sha256"]:
    print(f"downloading {cfg['weights_url']}", flush=True)
    urllib.request.urlretrieve(cfg["weights_url"], dest)
digest = hashlib.sha256(dest.read_bytes()).hexdigest()
if digest != cfg["weights_sha256"]:
    dest.unlink()
    sys.exit(f"checksum mismatch: got {digest}, expected {cfg['weights_sha256']}")
print(f"model ready: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB), checksum verified")
