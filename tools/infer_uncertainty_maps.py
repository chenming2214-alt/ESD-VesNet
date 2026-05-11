"""
Batch inference script for exporting uncertainty maps only.

Features:
  - Recursively scans images under a data directory.
  - Runs the existing segmentation model and extracts uncertainty.
  - For EDL models, uses model.infer_prob_uncert() directly.
  - For non-EDL models, falls back to a proxy uncertainty: 4 * p * (1 - p).
  - Saves uncertainty maps at the original image resolution.
  - Optionally saves heatmaps, overlays, and raw float .npy files.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _iter_images(root: str) -> Iterable[Tuple[str, str]]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.lower().endswith(IMG_EXTS):
                yield dirpath, fn


def _to_uint8(arr01: np.ndarray) -> np.ndarray:
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0 + 0.5).astype(np.uint8)


def _normalize_for_vis(arr: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo + 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _save_gray_png(arr01: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(_to_uint8(arr01), mode="L").save(out_path)


def _make_heat_bgr(arr01: np.ndarray) -> np.ndarray:
    arr_u8 = _to_uint8(arr01)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    return cv2.applyColorMap(arr_u8, cmap)


def _save_heat_png(arr01: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    heat_bgr = _make_heat_bgr(arr01)
    cv2.imwrite(out_path, heat_bgr)


def _save_overlay(img_rgb: Image.Image, heat01: np.ndarray, out_path: str, alpha: float) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base_bgr = cv2.cvtColor(np.array(img_rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
    heat_bgr = _make_heat_bgr(heat01)
    if heat_bgr.shape[:2] != base_bgr.shape[:2]:
        heat_bgr = cv2.resize(heat_bgr, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(heat_bgr, float(alpha), base_bgr, 1.0 - float(alpha), 0.0)
    cv2.imwrite(out_path, overlay)


@torch.no_grad()
def _infer_uncertainty(model, inp: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Returns uncertainty map in [0,1], shape [B,1,H,W].
    Second return value indicates whether true EDL uncertainty was used.
    """
    if hasattr(model, "infer_prob_uncert"):
        _prob_v, u, _prob_g = model.infer_prob_uncert(inp)
        return u, True

    if hasattr(model, "infer"):
        logits = model.infer(inp)
    else:
        logits = model(inp)
    prob = torch.sigmoid(logits)
    u = (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)
    return u, False


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Batch export uncertainty maps from vessel model.")
    parser.add_argument("--data-dir", type=str, required=True, help="Root folder containing images in subfolders.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--out-dir", type=str, default="", help="Output root. Defaults to <data-dir>_uncertainty_maps.")
    parser.add_argument("--model", type=str, default="sam3-sam-edl", help="models.register() name.")
    parser.add_argument("--inp-size", type=int, default=1024, help="Square inference input size.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-heat", action="store_true", help="Also save contrast-stretched heatmap PNG.")
    parser.add_argument("--overlay", action="store_true", help="Also save heatmap overlay on original image.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Overlay alpha in [0,1].")
    parser.add_argument("--save-npy", action="store_true", help="Also save raw float32 uncertainty map as .npy.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if the main gray PNG already exists.")
    parser.add_argument("--max-images", type=int, default=0, help="Only process the first N images. 0 means all.")
    parser.add_argument("--p-lo", type=float, default=1.0, help="Low percentile for heatmap contrast stretch.")
    parser.add_argument("--p-hi", type=float, default=99.0, help="High percentile for heatmap contrast stretch.")
    args = parser.parse_args()

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    sam3_main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sam3-main"))
    if sam3_main_dir not in sys.path:
        sys.path.insert(0, sam3_main_dir)

    import models  # noqa

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    out_dir = str(args.out_dir).strip()
    if not out_dir:
        out_dir = data_dir.rstrip("/\\") + "_uncertainty_maps"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    if str(args.model) == "sam":
        encoder_mode = {
            "name": "sam",
            "patch_size": 16,
            "prompt_embed_dim": 256,
            "embed_dim": 1024,
        }
        model_cfg = {"name": "sam", "args": {"inp_size": int(args.inp_size), "encoder_mode": encoder_mode, "loss": "bce"}}
    else:
        model_cfg = {"name": str(args.model), "args": {"inp_size": int(args.inp_size)}}

    model = models.make(model_cfg).to(device)
    try:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((int(args.inp_size), int(args.inp_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_items = list(_iter_images(data_dir))
    if int(args.max_images) > 0:
        image_items = image_items[: int(args.max_images)]
    if not image_items:
        raise RuntimeError(f"no images found in: {data_dir}")

    used_true_edl = None
    total = 0
    saved = 0

    for dirpath, fn in image_items:
        total += 1
        img_path = os.path.join(dirpath, fn)
        rel_dir = os.path.relpath(dirpath, data_dir)
        out_subdir = out_dir if rel_dir == "." else os.path.join(out_dir, rel_dir)

        base = os.path.splitext(fn)[0]
        gray_path = os.path.join(out_subdir, f"{base}_uncertainty_gray.png")
        heat_path = os.path.join(out_subdir, f"{base}_uncertainty_heat.png")
        overlay_path = os.path.join(out_subdir, f"{base}_uncertainty_overlay.png")
        npy_path = os.path.join(out_subdir, f"{base}_uncertainty.npy")

        if args.skip_existing and os.path.exists(gray_path):
            continue

        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size
        inp = tfm(img).unsqueeze(0).to(device)

        uncert, is_true_edl = _infer_uncertainty(model, inp)
        if used_true_edl is None:
            used_true_edl = bool(is_true_edl)
            mode_text = "true EDL uncertainty" if used_true_edl else "proxy uncertainty from sigmoid probabilities"
            print(f"uncertainty mode: {mode_text}")

        uncert_up = F.interpolate(uncert, size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
        uncert_np = uncert_up.detach().float().cpu().numpy()
        uncert_np = np.clip(uncert_np, 0.0, 1.0)

        _save_gray_png(uncert_np, gray_path)

        if args.save_npy:
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, uncert_np.astype(np.float32))

        if args.save_heat or args.overlay:
            uncert_vis = _normalize_for_vis(uncert_np, p_lo=float(args.p_lo), p_hi=float(args.p_hi))
            if args.save_heat:
                _save_heat_png(uncert_vis, heat_path)
            if args.overlay:
                _save_overlay(img, uncert_vis, overlay_path, alpha=float(args.overlay_alpha))

        saved += 1
        if saved == 1 or saved % 50 == 0 or saved == len(image_items):
            print(f"[{saved}/{len(image_items)}] saved: {gray_path}")

    print(f"done. images_found={total}, maps_saved={saved}")
    print(f"output_dir: {out_dir}")


if __name__ == "__main__":
    main()
