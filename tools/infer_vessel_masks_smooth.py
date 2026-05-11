"""
Infer vessel masks and apply smoothing + small speck removal.

Goal: smoother masks and remove tiny white dots near image borders.

Default pipeline:
- predict probability map
- threshold (default 0.35)
- close (bridge gaps) -> open (remove tiny noise) -> median blur
- connected components filtering:
  - remove components with area < min_area
  - remove border-touching components with area < border_max_area

Outputs:
  <out-dir>/<basename>_thr<THR>_smooth.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def _touches_border(x: int, y: int, w: int, h: int, W: int, H: int, pad: int = 0) -> bool:
    return x <= pad or y <= pad or (x + w) >= (W - pad) or (y + h) >= (H - pad)


def _clean_binary_mask(
    mask01: np.ndarray,
    close_k: int,
    open_k: int,
    median_k: int,
    min_area: int,
    border_max_area: int,
    border_pad: int,
) -> np.ndarray:
    """
    mask01: uint8 {0,1}
    returns uint8 {0,255}
    """
    m = (mask01.astype(np.uint8) * 255)

    if close_k > 0:
        k = close_k if close_k % 2 == 1 else close_k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    if open_k > 0:
        k = open_k if open_k % 2 == 1 else open_k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    if median_k and median_k > 1:
        k = median_k if median_k % 2 == 1 else median_k + 1
        m = cv2.medianBlur(m, k)

    # connected components filtering
    bin01 = (m > 127).astype(np.uint8)
    H, W = bin01.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)

    keep = np.zeros(num, dtype=np.uint8)
    keep[0] = 0  # background label must remain 0
    for i in range(1, num):
        x, y, w, h, area = stats[i].tolist()
        if area < int(min_area):
            continue
        if _touches_border(x, y, w, h, W=W, H=H, pad=int(border_pad)) and area < int(border_max_area):
            continue
        keep[i] = 1

    out = keep[labels] * 255
    return out.astype(np.uint8)


def _load_state_dict(model: torch.nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="save/vessel_esd_sam3_fullsam_edl_hnm_0105/model_epoch_best.pth",
        help="checkpoint path (can be a dict with key 'model' or a raw state_dict)",
    )
    parser.add_argument("--model", type=str, default="sam", help="models.register() name, e.g. 'sam'")
    parser.add_argument("--inp-dir", type=str, default="visual")
    parser.add_argument("--out-dir", type=str, default="visual/output_masks_smooth_v2")
    parser.add_argument("--inp-size", type=int, default=1008, help="SAM3 ViTDet RoPE table expects 1008")
    parser.add_argument("--thr", type=float, default=0.35)
    parser.add_argument("--use-gated", action="store_true", help="if model has infer_prob_uncert(), use prob_gated")

    parser.add_argument("--close-k", type=int, default=5)
    parser.add_argument("--open-k", type=int, default=3)
    parser.add_argument("--median-k", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=30, help="remove components smaller than this (pixels)")
    parser.add_argument("--border-max-area", type=int, default=200, help="remove border-touching components smaller than this")
    parser.add_argument("--border-pad", type=int, default=0, help="border pad in pixels (0 means exact border)")

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    import models  # noqa

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    # The project registers at least: 'sam' (see models/sam.py)
    encoder_mode = {
        "name": "sam",
        "patch_size": 16,
        "prompt_embed_dim": 256,
        "embed_dim": 1024,
    }
    model_cfg = {"name": str(args.model), "args": {"inp_size": int(args.inp_size), "encoder_mode": encoder_mode, "loss": "bce"}}
    model = models.make(model_cfg).to(device)
    missing, unexpected = _load_state_dict(model, args.ckpt)
    print(f"loaded ckpt. missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((int(args.inp_size), int(args.inp_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_files = sorted([p for p in os.listdir(args.inp_dir) if p.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not img_files:
        raise RuntimeError(f"No images found in: {args.inp_dir}")
    print(f"found {len(img_files)} images in {args.inp_dir}")

    for fn in img_files:
        in_path = os.path.join(args.inp_dir, fn)
        img = Image.open(in_path).convert("RGB")
        W0, H0 = img.size
        inp = tfm(img).unsqueeze(0).to(device)

        # probability inference
        if hasattr(model, "infer_prob_uncert"):
            prob_v, _u, prob_g = model.infer_prob_uncert(inp)
            prob = prob_g if args.use_gated else prob_v
        elif hasattr(model, "infer"):
            logits = model.infer(inp)
            prob = torch.sigmoid(logits)
        else:
            logits = model(inp)
            prob = torch.sigmoid(logits)

        prob_up = F.interpolate(prob, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
        prob_np = prob_up.detach().float().cpu().numpy()
        bin01 = (prob_np > float(args.thr)).astype(np.uint8)

        cleaned = _clean_binary_mask(
            bin01,
            close_k=int(args.close_k),
            open_k=int(args.open_k),
            median_k=int(args.median_k),
            min_area=int(args.min_area),
            border_max_area=int(args.border_max_area),
            border_pad=int(args.border_pad),
        )

        base = os.path.splitext(os.path.basename(fn))[0]
        out_path = os.path.join(args.out_dir, f"{base}_thr{args.thr:.2f}_smooth.png")
        Image.fromarray(cleaned, mode="L").save(out_path)
        print("saved:", out_path)

    print("all masks saved to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()


