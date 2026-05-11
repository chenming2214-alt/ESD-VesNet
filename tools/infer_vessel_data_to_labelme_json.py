"""
Batch inference on all patients under a data root and export Labelme JSON per frame.

Target data layout (observed in this repo):
  data/<PATIENT_ID>/*.png

Output (default):
  <out-dir>/<PATIENT_ID>/<frame_basename>.json

JSON format: Labelme (compatible with visual_masks/*.json in this repo)
  - version: "5.2.1"
  - shapes: list of polygons with label "vessel segmentation"
  - imagePath: absolute path to image (robust even if json is stored elsewhere)
  - imageData: null
  - imageHeight / imageWidth
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

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


def _mask_to_polygons(mask255: np.ndarray, eps_ratio: float, min_points: int) -> List[List[List[float]]]:
    """
    mask255: uint8 {0,255}
    returns list of polygons, each polygon is list of [x,y] float points
    """
    if mask255.ndim != 2:
        raise ValueError("mask255 must be HxW")
    bin01 = (mask255 > 127).astype(np.uint8)
    contours, _hier = cv2.findContours(bin01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[List[List[float]]] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        peri = float(cv2.arcLength(cnt, True))
        eps = max(0.0, float(eps_ratio)) * peri
        approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
        if approx is None or len(approx) < int(min_points):
            continue
        pts = approx.reshape(-1, 2)
        poly = [[float(x), float(y)] for (x, y) in pts]
        polys.append(poly)

    return polys


def _build_labelme_json(
    image_path: str,
    H: int,
    W: int,
    polygons: List[List[List[float]]],
    label: str,
    image_data_b64: str | None = None,
) -> Dict[str, Any]:
    shapes = []
    for poly in polygons:
        shapes.append(
            {
                "label": label,
                "line_color": None,
                "fill_color": None,
                "points": poly,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        )
    return {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": image_data_b64,
        "imageHeight": int(H),
        "imageWidth": int(W),
    }


def _load_state_dict(model: torch.nn.Module, ckpt_path: str) -> Tuple[int, int]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return len(missing), len(unexpected)


def _iter_patient_image_files(patient_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg")
    files = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.lower().endswith(exts)]
    files.sort()
    return files


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(default_project_dir, "data"),
        help="root dir that contains patient folders",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(default_project_dir, "pred_labelme_json"),
        help="output root (will create one subfolder per patient)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="save/vessel_esd_sam3_fullsam_edl_hnm_0105/model_epoch_best.pth",
        help="checkpoint path (can be a dict with key 'model' or a raw state_dict)",
    )
    parser.add_argument("--model", type=str, default="sam", help="models.register() name, e.g. 'sam'")
    parser.add_argument("--label", type=str, default="vessel segmentation", help="Labelme shape label text")
    parser.add_argument(
        "--image-path-mode",
        type=str,
        default="basename",
        choices=["abs", "rel", "basename"],
        help="how to write imagePath in json: abs (absolute), rel (relative to patient folder), basename (filename only)",
    )
    parser.add_argument(
        "--embed-image-data",
        action="store_true",
        help="embed imageData (base64) into json; makes json huge but portable even without imagePath",
    )

    parser.add_argument("--inp-size", type=int, default=1008, help="SAM3 ViTDet RoPE table expects 1008")
    parser.add_argument("--thr", type=float, default=0.35)
    parser.add_argument("--use-gated", action="store_true", help="if model has infer_prob_uncert(), use prob_gated")

    # smoothing / speck removal
    parser.add_argument("--close-k", type=int, default=5)
    parser.add_argument("--open-k", type=int, default=3)
    parser.add_argument("--median-k", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--border-max-area", type=int, default=200)
    parser.add_argument("--border-pad", type=int, default=0)

    # polygon extraction
    parser.add_argument("--poly-eps", type=float, default=0.01, help="approxPolyDP epsilon as a ratio of contour perimeter")
    parser.add_argument("--min-poly-points", type=int, default=3)

    parser.add_argument("--save-mask-png", action="store_true", help="also save predicted mask png next to json")
    parser.add_argument("--log-every", type=int, default=50, help="print progress every N frames per patient (0 disables)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    import models  # noqa

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    # build model (matches training script)
    encoder_mode = {"name": "sam", "patch_size": 16, "prompt_embed_dim": 256, "embed_dim": 1024}
    model_cfg = {"name": str(args.model), "args": {"inp_size": int(args.inp_size), "encoder_mode": encoder_mode, "loss": "bce"}}
    model = models.make(model_cfg).to(device)
    ckpt_abs = os.path.abspath(args.ckpt)
    miss_n, unexp_n = _load_state_dict(model, ckpt_abs)
    print(f"loaded ckpt: {ckpt_abs} (missing={miss_n}, unexpected={unexp_n})")
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((int(args.inp_size), int(args.inp_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    patient_ids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not patient_ids:
        raise RuntimeError(f"No patient folders found in: {data_dir}")

    print(f"found {len(patient_ids)} patients in {data_dir}")

    total = 0
    for pid in patient_ids:
        pdir = os.path.join(data_dir, pid)
        imgs = _iter_patient_image_files(pdir)
        if not imgs:
            continue

        out_pdir = os.path.join(out_dir, pid)
        os.makedirs(out_pdir, exist_ok=True)
        print(f"[{pid}] {len(imgs)} frames")

        t0 = time.time()
        for idx, img_path in enumerate(imgs, start=1):
            img = Image.open(img_path).convert("RGB")
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
            pos_frac = float(bin01.mean())

            cleaned = _clean_binary_mask(
                bin01,
                close_k=int(args.close_k),
                open_k=int(args.open_k),
                median_k=int(args.median_k),
                min_area=int(args.min_area),
                border_max_area=int(args.border_max_area),
                border_pad=int(args.border_pad),
            )

            polys = _mask_to_polygons(cleaned, eps_ratio=float(args.poly_eps), min_points=int(args.min_poly_points))
            if args.image_path_mode == "abs":
                image_path_field = os.path.abspath(img_path)
            elif args.image_path_mode == "rel":
                image_path_field = os.path.relpath(img_path, start=pdir)
            else:
                image_path_field = os.path.basename(img_path)

            image_data_b64 = None
            if args.embed_image_data:
                with open(img_path, "rb") as f:
                    image_data_b64 = base64.b64encode(f.read()).decode("utf-8")

            js = _build_labelme_json(
                image_path=image_path_field,
                H=H0,
                W=W0,
                polygons=polys,
                label=str(args.label),
                image_data_b64=image_data_b64,
            )

            base = os.path.splitext(os.path.basename(img_path))[0]
            out_json = os.path.join(out_pdir, f"{base}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(js, f, ensure_ascii=False, indent=2)

            if args.save_mask_png:
                out_mask = os.path.join(out_pdir, f"{base}_mask.png")
                Image.fromarray(cleaned, mode="L").save(out_mask)

            total += 1
            log_every = int(args.log_every)
            if log_every > 0 and (idx == 1 or idx % log_every == 0 or idx == len(imgs)):
                dt = max(1e-6, time.time() - t0)
                fps = float(idx) / dt
                warn = " [WARN: mask almost full, increase --thr?]" if pos_frac > 0.98 else ""
                print(f"[{pid}] {idx}/{len(imgs)} done (polys={len(polys)}; pos_frac={pos_frac:.3f}; {fps:.2f} fps){warn}")

    print(f"done. total frames processed: {total}")
    print(f"json saved under: {out_dir}")


if __name__ == "__main__":
    main()


