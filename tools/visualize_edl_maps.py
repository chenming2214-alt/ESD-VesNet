import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Ensure project root is on sys.path when running as a script: `python tools/xxx.py`
_PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)

import models


def _list_images(path: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if os.path.isfile(path):
        return [path]
    out: List[str] = []
    for root, _dirs, files in os.walk(path):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def _make_preprocess(inp_size: int):
    return transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _save_gray_png(arr01: np.ndarray, out_path: str):
    img = Image.fromarray(_to_uint8(arr01), mode="L")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def _normalize_for_vis(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """
    Contrast-stretch by percentiles for visualization.
    This is crucial because many maps are heavily skewed toward 0, which looks 'black' if saved directly.
    """
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        # fallback: min-max
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo + 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _save_heat_png(arr01: np.ndarray, out_path: str, cmap: str = "turbo"):
    import matplotlib.cm as cm

    cmap_fn = cm.get_cmap(cmap)
    rgba = cmap_fn(np.clip(arr01, 0.0, 1.0))  # HxWx4 in [0,1]
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def _save_overlay_heat(img_rgb: Image.Image, heat01: np.ndarray, out_path: str, alpha: float = 0.45, cmap: str = "turbo"):
    # resize base to match heat
    base = img_rgb.convert("RGB").resize((int(heat01.shape[1]), int(heat01.shape[0])), resample=Image.BILINEAR)
    import matplotlib.cm as cm

    cmap_fn = cm.get_cmap(cmap)
    rgba = cmap_fn(np.clip(heat01, 0.0, 1.0))
    heat_rgb = Image.fromarray((rgba[..., :3] * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.blend(base, heat_rgb, alpha=float(alpha)).save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Export EDL maps: prob_vessel / uncertainty_u / prob_gated")
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint .pth")
    parser.add_argument("--input", type=str, required=True, help="image file or folder")
    parser.add_argument("--output", type=str, required=True, help="output folder")
    parser.add_argument("--model", type=str, default="sam3-sam-edl", help="model name in models.make()")
    parser.add_argument("--inp-size", type=int, default=1024, help="input size used in training/val (default 1024)")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for binary mask")
    parser.add_argument("--gamma", type=float, default=1.0, help="gating gamma for prob_gated = p*(1-u)^gamma")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save-mask", action="store_true", help="also save thresholded mask from prob_gated")
    parser.add_argument("--max-images", type=int, default=0, help="only export first N images (0 = no limit)")
    parser.add_argument("--stride", type=int, default=1, help="sample every k images from input folder (default 1)")
    parser.add_argument("--cmap", type=str, default="turbo", help="colormap for heatmaps (matplotlib), e.g. turbo/jet/viridis/magma")
    parser.add_argument("--overlay", action="store_true", help="also save overlay heatmaps on the resized input image")
    parser.add_argument("--alpha", type=float, default=0.45, help="overlay alpha (0..1)")
    parser.add_argument("--p-lo", type=float, default=1.0, help="low percentile for contrast stretching")
    parser.add_argument("--p-hi", type=float, default=99.0, help="high percentile for contrast stretching")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    preprocess = _make_preprocess(int(args.inp_size))

    # Build model
    # - sam / sam-edl: reuse train_vessel_esd.py's MODEL_CONFIG (contains encoder_mode), only override inp_size.
    # - others (sam3-edl / sam3-sam-edl): accept minimal args.
    model_name = str(args.model)
    if model_name in {"sam", "sam-edl"}:
        import train_vessel_esd as tv

        model_cfg = dict(tv.Config.MODEL_CONFIG)
        model_cfg["name"] = model_name
        model_cfg = {"name": model_cfg["name"], "args": dict(model_cfg["args"])}
        model_cfg["args"]["inp_size"] = int(args.inp_size)
        if "encoder_mode" in model_cfg["args"] and isinstance(model_cfg["args"]["encoder_mode"], dict):
            model_cfg["args"]["encoder_mode"] = dict(model_cfg["args"]["encoder_mode"])
            model_cfg["args"]["encoder_mode"]["img_size"] = int(args.inp_size)
    else:
        model_cfg = {"name": model_name, "args": {"inp_size": int(args.inp_size), "gate_gamma": float(args.gamma)}}
    model = models.make(model_cfg).to(device)

    # Load checkpoint
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _ = (missing, unexpected)
    model.eval()

    img_paths = _list_images(args.input)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found under: {args.input}")
    stride = max(1, int(args.stride))
    img_paths = img_paths[::stride]
    if int(args.max_images) > 0:
        img_paths = img_paths[: int(args.max_images)]

    for i, p in enumerate(img_paths):
        if (i % 10) == 0:
            print(f"[run] {i+1}/{len(img_paths)}: {p}")
        rel = os.path.relpath(p, args.input) if os.path.isdir(args.input) else os.path.basename(p)
        stem = os.path.splitext(rel)[0]
        out_base = os.path.join(args.output, stem)
        os.makedirs(os.path.dirname(out_base), exist_ok=True)

        img = Image.open(p).convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            target = model.module if hasattr(model, "module") else model
            if hasattr(target, "infer_prob_uncert"):
                prob_vessel, u, prob_gated = target.infer_prob_uncert(inp)
            else:
                # Fallback for non-EDL models (e.g. baseline `sam`):
                # - prob_vessel = sigmoid(logits)
                # - uncertainty proxy u = 4 p (1-p) in [0,1] (max at p=0.5)
                # - prob_gated = p * (1-u)^gamma
                logits = target.infer(inp)
                prob_vessel = torch.sigmoid(logits)
                u = (4.0 * prob_vessel * (1.0 - prob_vessel)).clamp(0.0, 1.0)
                gate = (1.0 - u).clamp(0.0, 1.0) ** float(args.gamma)
                prob_gated = (prob_vessel * gate).clamp(0.0, 1.0)

        pv = prob_vessel.detach().float().cpu().numpy()[0, 0]
        uu = u.detach().float().cpu().numpy()[0, 0]
        pg = prob_gated.detach().float().cpu().numpy()[0, 0]
        cert = np.clip(1.0 - uu, 0.0, 1.0)
        delta = np.clip(pv - pg, 0.0, 1.0)

        # Save raw grayscale (kept for exact-value debugging)
        _save_gray_png(pv, out_base + "_prob_vessel_gray.png")
        _save_gray_png(np.clip(uu, 0.0, 1.0), out_base + "_uncertainty_u_gray.png")
        _save_gray_png(pg, out_base + "_prob_gated_gray.png")
        _save_gray_png(cert, out_base + "_certainty_1mu_gray.png")
        _save_gray_png(delta, out_base + "_delta_p_minus_pg_gray.png")

        # Save heatmaps with contrast stretching (for paper/visualization)
        pv_vis = _normalize_for_vis(pv, p_lo=float(args.p_lo), p_hi=float(args.p_hi))
        uu_vis = _normalize_for_vis(np.clip(uu, 0.0, 1.0), p_lo=float(args.p_lo), p_hi=float(args.p_hi))
        pg_vis = _normalize_for_vis(pg, p_lo=float(args.p_lo), p_hi=float(args.p_hi))
        cert_vis = _normalize_for_vis(cert, p_lo=float(args.p_lo), p_hi=float(args.p_hi))
        delta_vis = _normalize_for_vis(delta, p_lo=float(args.p_lo), p_hi=float(args.p_hi))
        _save_heat_png(pv_vis, out_base + "_prob_vessel_heat.png", cmap=str(args.cmap))
        _save_heat_png(uu_vis, out_base + "_uncertainty_u_heat.png", cmap=str(args.cmap))
        _save_heat_png(pg_vis, out_base + "_prob_gated_heat.png", cmap=str(args.cmap))
        _save_heat_png(cert_vis, out_base + "_certainty_1mu_heat.png", cmap=str(args.cmap))
        _save_heat_png(delta_vis, out_base + "_delta_p_minus_pg_heat.png", cmap=str(args.cmap))

        if args.overlay:
            _save_overlay_heat(img, pv_vis, out_base + "_prob_vessel_overlay.png", alpha=float(args.alpha), cmap=str(args.cmap))
            _save_overlay_heat(img, uu_vis, out_base + "_uncertainty_u_overlay.png", alpha=float(args.alpha), cmap=str(args.cmap))
            _save_overlay_heat(img, pg_vis, out_base + "_prob_gated_overlay.png", alpha=float(args.alpha), cmap=str(args.cmap))
            _save_overlay_heat(img, cert_vis, out_base + "_certainty_1mu_overlay.png", alpha=float(args.alpha), cmap=str(args.cmap))
            _save_overlay_heat(img, delta_vis, out_base + "_delta_p_minus_pg_overlay.png", alpha=float(args.alpha), cmap=str(args.cmap))

        if args.save_mask:
            m = (pg >= float(args.threshold)).astype(np.uint8) * 255
            Image.fromarray(m, mode="L").save(out_base + f"_mask_th{args.threshold:.2f}.png")

    print(f"[ok] exported {len(img_paths)} image(s) to: {args.output}")


if __name__ == "__main__":
    main()


