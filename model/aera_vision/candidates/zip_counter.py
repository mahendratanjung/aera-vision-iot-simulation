"""ZIP density counters (Ma et al., 2025; MIT), published checkpoints, no training.

The checkpoint holds the model config and weights. It is read with ``torch.load(weights_only=True)``,
which accepts only tensors and plain data, so loading cannot run code. The model classes come from
the vendored ZIP code (aera_vision/vendor/zip_models). The backbone is built without ImageNet weights
(the checkpoint replaces them anyway). Preprocessing follows ZIP's evaluate.py: whole image,
normalised with ZIP's mean/std, upscaled if a side is under the training window, sliding window only
for very large images (none in CrowdHuman val).
"""

from __future__ import annotations

import numpy as np

from aera_vision.candidates import Output
from aera_vision.data import ROOT

WEIGHTS = ROOT / "weights" / "zip"
PATHS = {
    "zip_n_nwpu": WEIGHTS / "nwpu" / "ebc_n" / "best_mae.pth",
    "zip_t_nwpu": WEIGHTS / "nwpu" / "ebc_t" / "best_mae.pth",
    "zip_t_shb": WEIGHTS / "shb" / "ebc_t_best" / "best_mae.pth",
}
MEAN = (0.48145466, 0.4578275, 0.40821073)  # ZIP datasets/crowd.py
STD = (0.26862954, 0.26130258, 0.27577711)
MAX_SIDE = 2048  # longer images are scaled down to keep memory in check; counts are scale-free


class ZIPCounter:
    def __init__(self, key: str, device: str):
        import timm
        import torch

        import aera_vision.vendor.zip_models.ebc.timm_models as tm
        from aera_vision.vendor.zip_models.ebc import _ebc

        tm.create_model = lambda *a, **k: timm.create_model(*a, **{**k, "pretrained": False})
        self.weights_path = PATHS[key]
        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        c = ckpt["config"]
        self.window = int(c.get("input_size") or 224)
        model = _ebc(model_name=c["model_name"], block_size=c["block_size"], bins=c["bins"],
                     bin_centers=c["bin_centers"], zero_inflated=c["zero_inflated"],
                     norm=c.get("norm") or "none", act=c.get("act") or "none")
        model.load_state_dict(ckpt["weights"], strict=True)
        self.block = c["block_size"]
        self.device = device
        self.model = model.to(device).eval()
        self.torch = torch
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def predict(self, image_bgr: np.ndarray) -> Output:
        torch = self.torch
        import torch.nn.functional as F

        h, w = image_bgr.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(image_bgr[:, :, ::-1])).to(self.device)
        x = x.permute(2, 0, 1)[None].float() / 255.0
        scale = 1.0
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
        if min(h, w) * scale < self.window:
            scale = self.window / min(h, w)
        if scale != 1.0:
            x = F.interpolate(x, size=(round(h * scale), round(w * scale)), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            den = self.model(x)[0, 0].float().cpu().numpy()
        # one cell covers (input pixels / cells) of the original image
        cell = h / den.shape[0]
        return Output(density=den.astype(np.float32), cell=float(cell))
