"""
ESD-VesNet vessel segmentation inference and visualization.

Paper:
ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for Endoscopic
Submucosal Dissection with Hard Negative Mining

默认：仅绘制绿色轮廓线（无区域填充），对概率图先平滑再阈值，轮廓略粗、LINE_AA 抗锯齿。
可选 --draw-mode fill：半透明颜色填充叠加（旧行为）。

默认输出到 <项目>/save/overlay_smooth/，目录结构与 infer_labelme_from_data 的镜像规则一致。
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


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@torch.no_grad()
def _infer_prob(model, inp: torch.Tensor, use_gated: bool) -> torch.Tensor:
    if hasattr(model, "infer_prob_uncert"):
        prob_v, _u, prob_g = model.infer_prob_uncert(inp)
        return prob_g if use_gated else prob_v
    if hasattr(model, "infer"):
        logits = model.infer(inp)
    else:
        logits = model(inp)
    return torch.sigmoid(logits)


@torch.no_grad()
def _infer_prob_tta(model, inp: torch.Tensor, use_gated: bool) -> torch.Tensor:
    prob_sum = _infer_prob(model, inp, use_gated=use_gated)
    inp_h = torch.flip(inp, dims=[3])
    prob_h = _infer_prob(model, inp_h, use_gated=use_gated)
    prob_h = torch.flip(prob_h, dims=[3])
    prob_sum = prob_sum + prob_h
    inp_v = torch.flip(inp, dims=[2])
    prob_v = _infer_prob(model, inp_v, use_gated=use_gated)
    prob_v = torch.flip(prob_v, dims=[2])
    prob_sum = prob_sum + prob_v
    inp_hv = torch.flip(inp, dims=[2, 3])
    prob_hv = _infer_prob(model, inp_hv, use_gated=use_gated)
    prob_hv = torch.flip(prob_hv, dims=[2, 3])
    prob_sum = prob_sum + prob_hv
    return prob_sum / 4.0


@torch.no_grad()
def _infer_prob_tta_hflip(model, inp: torch.Tensor, use_gated: bool) -> torch.Tensor:
    prob_sum = _infer_prob(model, inp, use_gated=use_gated)
    inp_h = torch.flip(inp, dims=[3])
    prob_h = _infer_prob(model, inp_h, use_gated=use_gated)
    prob_h = torch.flip(prob_h, dims=[3])
    return (prob_sum + prob_h) / 2.0


def _iter_images(root: str) -> Iterable[Tuple[str, str]]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith("._"):
                continue
            if fn.lower().endswith(IMG_EXTS):
                yield dirpath, fn


def _smooth_alpha_map(
    prob: np.ndarray,
    blur_sigma: float,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), 0.0, 1.0)
    if blur_sigma > 1e-6:
        p = cv2.GaussianBlur(p, (0, 0), float(blur_sigma))
        p = np.clip(p, 0.0, 1.0)
    if bilateral_d > 0:
        p8 = (p * 255.0 + 0.5).astype(np.uint8)
        p8 = cv2.bilateralFilter(
            p8,
            int(bilateral_d),
            float(bilateral_sigma_color),
            float(bilateral_sigma_space),
        )
        p = (p8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    return p


def _compose_overlay_bgr(
    base_bgr: np.ndarray,
    alpha01: np.ndarray,
    tint_bgr: Tuple[int, int, int],
    alpha_max: float,
) -> np.ndarray:
    """alpha01: H,W in [0,1], effective alpha = alpha_max * alpha01 per pixel."""
    a = (np.clip(alpha01, 0.0, 1.0) * float(alpha_max))[..., np.newaxis]
    base = base_bgr.astype(np.float32)
    tint = np.zeros_like(base, dtype=np.float32)
    tint[:, :, 0] = float(tint_bgr[0])
    tint[:, :, 1] = float(tint_bgr[1])
    tint[:, :, 2] = float(tint_bgr[2])
    out = base * (1.0 - a) + tint * a
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _draw_contours_only_bgr(
    base_bgr: np.ndarray,
    prob_smoothed: np.ndarray,
    thr: float,
    color_bgr: Tuple[int, int, int],
    line_thickness: int,
) -> np.ndarray:
    """prob_smoothed: float [0,1]；二值化后只画轮廓，不填充。"""
    m = (np.clip(prob_smoothed, 0.0, 1.0) >= float(thr)).astype(np.uint8) * 255
    contours, _h = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = base_bgr.copy()
    t = max(1, int(line_thickness))
    cv2.drawContours(
        out,
        contours,
        -1,
        (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])),
        thickness=t,
        lineType=cv2.LINE_AA,
    )
    return out


def _extract_mp4_frame_png(mp4_path: str, local_idx: int, out_png: str) -> None:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {mp4_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(local_idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"failed to read frame {local_idx} from {mp4_path}")
    d = os.path.dirname(out_png)
    if d:
        os.makedirs(d, exist_ok=True)
    if not cv2.imwrite(out_png, bgr):
        raise RuntimeError(f"failed to write {out_png}")


@torch.no_grad()
def _run_one_image(
    model,
    tfm: transforms.Compose,
    device: torch.device,
    img_path: str,
    out_path: str,
    args: argparse.Namespace,
) -> bool:
    if args.skip_existing and os.path.isfile(out_path):
        return False

    img = Image.open(img_path).convert("RGB")
    w0, h0 = img.size
    inp = tfm(img).unsqueeze(0).to(device)

    if args.tta_mode == "hv":
        prob = _infer_prob_tta(model, inp, use_gated=args.use_gated)
    elif args.tta_mode == "h":
        prob = _infer_prob_tta_hflip(model, inp, use_gated=args.use_gated)
    else:
        prob = _infer_prob(model, inp, use_gated=args.use_gated)

    prob_up = F.interpolate(prob, size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
    prob_np = prob_up.detach().float().cpu().numpy()

    g = float(args.prob_gamma)
    if abs(g - 1.0) > 1e-6:
        prob_np = np.power(np.clip(prob_np, 0.0, 1.0), g)

    lo = float(args.vis_floor)
    if lo > 1e-6:
        prob_np = np.clip((prob_np - lo) / max(1.0 - lo, 1e-6), 0.0, 1.0)

    alpha_map = _smooth_alpha_map(
        prob_np,
        blur_sigma=float(args.blur_sigma),
        bilateral_d=int(args.bilateral_d),
        bilateral_sigma_color=float(args.bilateral_sigma_color),
        bilateral_sigma_space=float(args.bilateral_sigma_space),
    )

    base_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if str(args.draw_mode) == "fill":
        tint = (int(args.tint_bgr[0]), int(args.tint_bgr[1]), int(args.tint_bgr[2]))
        overlay = _compose_overlay_bgr(base_bgr, alpha_map, tint, alpha_max=float(args.alpha_max))
    else:
        col = (int(args.contour_bgr[0]), int(args.contour_bgr[1]), int(args.contour_bgr[2]))
        overlay = _draw_contours_only_bgr(
            base_bgr,
            alpha_map,
            thr=float(args.mask_thr),
            color_bgr=col,
            line_thickness=int(args.line_thickness),
        )

    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    if not cv2.imwrite(out_path, overlay):
        raise RuntimeError(f"failed to write {out_path}")
    return True


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Vessel overlay / contour visualization from segmentation model.")
    parser.add_argument("--data-dir", type=str, default="", help="recursive image root")
    parser.add_argument("--max-images", type=int, default=0, help="0 = no limit (data-dir only)")
    parser.add_argument("--image", action="append", default=[], help="single image path (repeatable)")
    parser.add_argument("--mp4", action="append", default=[], help="mp4 path (repeatable)")
    parser.add_argument("--mp4-out-dir", type=str, default="", help="temp PNG dir for --mp4")
    parser.add_argument("--mp4-frame-idx", type=int, default=30)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="sam3-sam-edl")
    parser.add_argument("--inp-size", type=int, default=1024)
    parser.add_argument("--use-gated", action="store_true")
    parser.add_argument("--tta-mode", type=str, default="none", choices=["none", "h", "hv"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--blur-sigma", type=float, default=5.0, help="Gaussian sigma on prob map; 0 disables")
    parser.add_argument(
        "--bilateral-d",
        type=int,
        default=0,
        help="if >0, bilateral filter on prob (edge-preserving smooth); use odd, e.g. 9",
    )
    parser.add_argument("--bilateral-sigma-color", type=float, default=75.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=75.0)
    parser.add_argument(
        "--draw-mode",
        type=str,
        default="outline",
        choices=["outline", "fill"],
        help="outline=绿色(可调)轮廓无填充；fill=半透明区域叠加",
    )
    parser.add_argument(
        "--mask-thr",
        type=float,
        default=0.5,
        help="outline 模式：平滑后概率 >= 该值视为前景再提轮廓",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="outline 模式轮廓线宽（像素，LINE_AA）；1 最细",
    )
    parser.add_argument(
        "--contour-bgr",
        type=int,
        nargs=3,
        default=(0, 255, 0),
        metavar=("B", "G", "R"),
        help="outline 模式线条颜色 BGR，默认绿色",
    )
    parser.add_argument("--alpha-max", type=float, default=0.48, help="fill 模式：最大叠加强度")
    parser.add_argument("--prob-gamma", type=float, default=1.0, help=">1 slightly softens bright regions")
    parser.add_argument(
        "--vis-floor",
        type=float,
        default=0.0,
        help="subtract before normalize [0,1] to suppress very low prob haze",
    )
    parser.add_argument(
        "--tint-bgr",
        type=int,
        nargs=3,
        default=(40, 40, 255),
        metavar=("B", "G", "R"),
        help="fill 模式叠色 BGR",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="default <project>/save/overlay_smooth; ignored if --next-to-image",
    )
    parser.add_argument("--next-to-image", action="store_true", help="write <stem>_overlay.png beside source image")
    parser.add_argument("--suffix", type=str, default="_overlay", help="output filename: <stem><suffix>.png")
    args = parser.parse_args()

    mp4_list = [os.path.abspath(p) for p in args.mp4 if str(p).strip()]
    image_list = [os.path.abspath(p) for p in args.image if str(p).strip()]
    if not str(args.data_dir).strip() and not mp4_list and not image_list:
        parser.error("Provide --data-dir, --image, and/or --mp4")
    if mp4_list and not str(args.mp4_out_dir).strip():
        parser.error("--mp4 requires --mp4-out-dir")

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    sam3_main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sam3-main"))
    if sam3_main_dir not in sys.path:
        sys.path.insert(0, sam3_main_dir)

    import models  # noqa

    if args.next_to_image:
        out_root: str | None = None
    else:
        o = str(args.out_dir).strip()
        out_root = os.path.abspath(o) if o else os.path.join(proj_root, "save", "overlay_smooth")
        os.makedirs(out_root, exist_ok=True)
        print(f"out_root: {out_root}", flush=True)

    device = torch.device(args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("device:", device, flush=True)

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

    total = 0
    saved = 0
    suf = str(args.suffix).strip() or "_overlay"

    def out_name(base: str) -> str:
        return f"{base}{suf}.png"

    if str(args.data_dir).strip():
        data_dir = os.path.abspath(args.data_dir)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(data_dir)
        n_seen = 0
        max_img = int(args.max_images)
        for dirpath, fn in _iter_images(data_dir):
            n_seen += 1
            if max_img > 0 and n_seen > max_img:
                break
            base = os.path.splitext(fn)[0]
            img_path = os.path.join(dirpath, fn)
            if out_root is None:
                out_path = os.path.join(dirpath, out_name(base))
            else:
                rel_dir = os.path.relpath(dirpath, data_dir)
                if rel_dir in (".", ""):
                    out_path = os.path.join(out_root, out_name(base))
                else:
                    out_path = os.path.join(out_root, rel_dir, out_name(base))
            total += 1
            if _run_one_image(model, tfm, device, img_path, out_path, args):
                saved += 1

    for img_path in image_list:
        if not os.path.isfile(img_path) or not img_path.lower().endswith(IMG_EXTS):
            print(f"[skip] {img_path}", flush=True)
            continue
        fn = os.path.basename(img_path)
        base = os.path.splitext(fn)[0]
        dirpath = os.path.dirname(img_path)
        if out_root is None:
            out_path = os.path.join(dirpath, out_name(base))
        else:
            patient = os.path.basename(dirpath)
            modality = os.path.basename(os.path.dirname(dirpath))
            if patient and modality:
                out_path = os.path.join(out_root, modality, patient, out_name(base))
            else:
                out_path = os.path.join(out_root, out_name(base))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        total += 1
        if _run_one_image(model, tfm, device, img_path, out_path, args):
            saved += 1
            print(f"[ok] {img_path} -> {out_path}", flush=True)

    mp4_out = os.path.abspath(str(args.mp4_out_dir).strip()) if mp4_list else ""
    fidx = int(args.mp4_frame_idx)
    for mp4_path in mp4_list:
        if not os.path.isfile(mp4_path):
            print(f"[skip] {mp4_path}", flush=True)
            continue
        stem = os.path.splitext(os.path.basename(mp4_path))[0]
        png_fn = f"{stem}_local{fidx:03d}.png"
        png_path = os.path.join(mp4_out, png_fn)
        try:
            _extract_mp4_frame_png(mp4_path, fidx, png_path)
        except RuntimeError as e:
            print(f"[skip] {mp4_path}: {e}", flush=True)
            continue
        clip_base = f"{stem}_local{fidx:03d}"
        if out_root is None:
            out_path = os.path.join(mp4_out, out_name(clip_base))
        else:
            out_path = os.path.join(out_root, out_name(clip_base))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        total += 1
        if _run_one_image(model, tfm, device, png_path, out_path, args):
            saved += 1
            print(f"[ok] {stem} -> {out_path}", flush=True)

    print(f"done. items={total}, overlay_saved={saved}", flush=True)


if __name__ == "__main__":
    main()
