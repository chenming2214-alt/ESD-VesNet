"""
ESD-VesNet evaluation script: run inference on the ESD validation split and print metrics.

Paper:
ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for Endoscopic
Submucosal Dissection with Hard Negative Mining

Metrics (binary segmentation):
- dice_fg, dice_bg, mDice (= mean of fg/bg dice)
- iou_fg, iou_bg, mIoU (= mean of fg/bg IoU)
- COD-style: S-measure, E-measure, MAE (computed on probability map)

It follows the same val split logic as training code:
- collect all patient dirs from Config.DATA_ROOTS
- prefix match with Config.VAL_PATIENTS (e.g., 'P25' matches 'P25_XXX')

Additionally prints metrics by patient groups (G1: P38,P16; G2: P10,P35; G3: P17,P25)
plus a TOTAL line over the full val set.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple

# Validation subgroups (prefix match on patient folder id, same as training val split)
_VAL_GROUP_DEFS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("G1: P38,P16", ("P38", "P16")),
    ("G2: P10,P35", ("P10", "P35")),
    ("G3: P17,P25", ("P17", "P25")),
)
_N_VAL_GROUPS = len(_VAL_GROUP_DEFS)


def _canon_patient_prefix(pid: str) -> str:
    return str(pid).split("_", 1)[0]


def _val_group_index(canon_pid: str) -> Optional[int]:
    for i, (_name, prefs) in enumerate(_VAL_GROUP_DEFS):
        if canon_pid in prefs:
            return i
    return None


import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _mae(prob01: np.ndarray, gt01: np.ndarray) -> float:
    prob01 = np.asarray(prob01, dtype=np.float32)
    gt01 = np.asarray(gt01, dtype=np.float32)
    return float(np.mean(np.abs(prob01 - gt01)))


def _emasure(prob01: np.ndarray, gt01: np.ndarray, eps: float = 1e-12) -> float:
    """Enhanced-alignment measure (E-measure), continuous version on [0,1] map and binary GT."""
    P = np.asarray(prob01, dtype=np.float32)
    G = (np.asarray(gt01, dtype=np.float32) > 0.5).astype(np.float32)

    # Special cases (GT all 0 / all 1)
    if float(G.sum()) == 0.0:
        return float(1.0 - P.mean())
    if float(G.sum()) == float(G.size):
        return float(P.mean())

    mu_P = float(P.mean())
    mu_G = float(G.mean())
    P_c = P - mu_P
    G_c = G - mu_G
    align = (2.0 * P_c * G_c) / (P_c * P_c + G_c * G_c + eps)
    enhanced = ((align + 1.0) ** 2) / 4.0
    return float(enhanced.mean())


def _ssim(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """SSIM-like similarity used in S-measure region term."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mu_x = float(x.mean())
    mu_y = float(y.mean())
    var_x = float(((x - mu_x) ** 2).mean())
    var_y = float(((y - mu_y) ** 2).mean())
    cov_xy = float(((x - mu_x) * (y - mu_y)).mean())
    c1 = 0.01**2
    c2 = 0.03**2
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float(num / (den + eps))


def _smeasure(prob01: np.ndarray, gt01: np.ndarray, alpha: float = 0.5) -> float:
    """Structure-measure (S-measure), computed on probability map P in [0,1] and binary GT."""
    P = np.asarray(prob01, dtype=np.float32)
    G = (np.asarray(gt01, dtype=np.float32) > 0.5).astype(np.float32)

    # Special cases
    if float(G.sum()) == 0.0:
        return float(1.0 - P.mean())
    if float(G.sum()) == float(G.size):
        return float(P.mean())

    # Object-aware term
    fg = P[G == 1]
    bg = P[G == 0]
    mu_fg = float(fg.mean()) if fg.size > 0 else 0.0
    mu_bg = float((1.0 - bg).mean()) if bg.size > 0 else 0.0
    o_fg = (2.0 * mu_fg) / (mu_fg * mu_fg + 1.0 + 1e-12)
    o_bg = (2.0 * mu_bg) / (mu_bg * mu_bg + 1.0 + 1e-12)
    w_fg = float(G.mean())
    s_object = w_fg * o_fg + (1.0 - w_fg) * o_bg

    # Region-aware term (centroid split into 4 regions)
    ys, xs = np.where(G == 1)
    if ys.size == 0:
        cx, cy = (P.shape[1] // 2), (P.shape[0] // 2)
    else:
        cy = int(np.round(float(ys.mean())))
        cx = int(np.round(float(xs.mean())))
    H, W = G.shape
    regions = [
        (slice(0, cy), slice(0, cx)),
        (slice(0, cy), slice(cx, W)),
        (slice(cy, H), slice(0, cx)),
        (slice(cy, H), slice(cx, W)),
    ]
    ssim_sum = 0.0
    w_sum = 0.0
    for sl in regions:
        g_r = G[sl]
        p_r = P[sl]
        area = float(g_r.size)
        if area <= 0:
            continue
        w_sum += area
        ssim_sum += area * _ssim(p_r, g_r)
    s_region = (ssim_sum / (w_sum + 1e-12)) if w_sum > 0 else 0.0

    s = alpha * s_object + (1.0 - alpha) * s_region
    return float(np.clip(s, 0.0, 1.0))


def _cod_metrics(prob01: np.ndarray, gt01: np.ndarray) -> Tuple[float, float, float]:
    """Returns (S-measure, E-measure, MAE)."""
    return _smeasure(prob01, gt01), _emasure(prob01, gt01), _mae(prob01, gt01)


def _confusion_from_prob(prob: torch.Tensor, gt: torch.Tensor, thr: float) -> Tuple[int, int, int, int]:
    pred = prob > thr
    gt1 = gt > 0.5
    tp = int((pred & gt1).sum().item())
    fp = int((pred & (~gt1)).sum().item())
    fn = int(((~pred) & gt1).sum().item())
    tn = int(((~pred) & (~gt1)).sum().item())
    return tp, fp, fn, tn


def _binary_closing(pred01: torch.Tensor, k: int) -> torch.Tensor:
    """
    Simple binary closing: dilate then erode, implemented via max-pool.
    pred01: bool/float tensor [1,1,H,W] on any device.
    k: odd kernel size >= 3.
    Returns: bool tensor [1,1,H,W]
    """
    if k is None or k <= 0:
        return pred01 > 0.5
    k = int(k)
    if k < 3:
        return pred01 > 0.5
    if (k % 2) == 0:
        k += 1
    x = (pred01 > 0.5).float()
    pad = k // 2
    # dilation
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
    # erosion via min-pool = 1 - maxpool(1-x)
    x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=pad)
    return x > 0.5


@torch.no_grad()
def _infer_prob(model, inp: torch.Tensor, use_gated: bool) -> torch.Tensor:
    """
    Returns probability map in [0,1], shape [B,1,H,W].
    Supports both EDL models (infer_prob_uncert) and plain models (infer / forward logits).
    """
    if hasattr(model, "infer_prob_uncert"):
        prob_v, _u, prob_g = model.infer_prob_uncert(inp)
        prob = prob_g if use_gated else prob_v
        return prob
    if hasattr(model, "infer"):
        logits = model.infer(inp)
    else:
        logits = model(inp)
    return torch.sigmoid(logits)


@torch.no_grad()
def _infer_prob_tta(model, inp: torch.Tensor, use_gated: bool) -> torch.Tensor:
    """
    Simple test-time augmentation: average over {orig, hflip, vflip, hvflip}.
    inp: [B,3,H,W]
    Returns: prob [B,1,H,W]
    """
    prob_sum = _infer_prob(model, inp, use_gated=use_gated)

    # hflip
    inp_h = torch.flip(inp, dims=[3])
    prob_h = _infer_prob(model, inp_h, use_gated=use_gated)
    prob_h = torch.flip(prob_h, dims=[3])
    prob_sum = prob_sum + prob_h

    # vflip
    inp_v = torch.flip(inp, dims=[2])
    prob_v = _infer_prob(model, inp_v, use_gated=use_gated)
    prob_v = torch.flip(prob_v, dims=[2])
    prob_sum = prob_sum + prob_v

    # hvflip
    inp_hv = torch.flip(inp, dims=[2, 3])
    prob_hv = _infer_prob(model, inp_hv, use_gated=use_gated)
    prob_hv = torch.flip(prob_hv, dims=[2, 3])
    prob_sum = prob_sum + prob_hv

    return prob_sum / 4.0


@torch.no_grad()
def _infer_prob_tta_hflip(model, inp: torch.Tensor, use_gated: bool) -> torch.Tensor:
    """Faster TTA: average over {orig, hflip}."""
    prob_sum = _infer_prob(model, inp, use_gated=use_gated)
    inp_h = torch.flip(inp, dims=[3])
    prob_h = _infer_prob(model, inp_h, use_gated=use_gated)
    prob_h = torch.flip(prob_h, dims=[3])
    return (prob_sum + prob_h) / 2.0


def _dice_iou_from_conf(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-7):
    dice_fg = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    dice_bg = (2.0 * tn) / (2.0 * tn + fp + fn + eps)
    mdice = 0.5 * (dice_fg + dice_bg)
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    miou = 0.5 * (iou_fg + iou_bg)
    return dice_fg, dice_bg, mdice, iou_fg, iou_bg, miou


def _metrics_lines(
    label: str,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    vdr_hit: int,
    vdr_tot: int,
    cod_s: float,
    cod_e: float,
    cod_m: float,
    n: int,
    vdr_min_area: int,
    cod_s_raw: Optional[float] = None,
    cod_e_raw: Optional[float] = None,
    cod_m_raw: Optional[float] = None,
) -> List[str]:
    dice_fg, dice_bg, mdice, iou_fg, iou_bg, miou = _dice_iou_from_conf(tp, fp, fn, tn)
    prec_fg = tp / (tp + fp + 1e-7)
    bg_fpr_all = fp / (fp + tn + 1e-7)
    vdr = vdr_hit / (vdr_tot + 1e-7)
    nn = max(1, n)
    lines = [
        f"[VAL {label}] n_imgs={n} mIoU={miou:.4f}, mDice={mdice:.4f}, dice_fg={dice_fg:.4f}, iou_fg={iou_fg:.4f}, "
        f"prec_fg={prec_fg:.4f}, bg_fpr_all={bg_fpr_all:.4f}, VDR(TP>={vdr_min_area})={vdr:.4f}",
        f"[VAL {label}] S-measure={cod_s / nn:.4f}, E-measure={cod_e / nn:.4f}, MAE={cod_m / nn:.4f}",
    ]
    if cod_s_raw is not None and cod_e_raw is not None and cod_m_raw is not None:
        lines.append(
            f"[VAL {label}] S-measure(raw)={cod_s_raw / nn:.4f}, E-measure(raw)={cod_e_raw / nn:.4f}, "
            f"MAE(raw)={cod_m_raw / nn:.4f}"
        )
    return lines


def _sweep_metrics_rows(
    tp: List[int], fp: List[int], fn: List[int], tn: List[int], sweep_thrs: List[float]
) -> Tuple[List[Tuple[float, float, float, float, float, float]], int]:
    rows: List[Tuple[float, float, float, float, float, float]] = []
    best_i = 0
    best_dfg = -1.0
    for i, thr in enumerate(sweep_thrs):
        dfg, dbg, dmean, ifg, ibg, imean = _dice_iou_from_conf(tp[i], fp[i], fn[i], tn[i])
        prec_fg = tp[i] / (tp[i] + fp[i] + 1e-7)
        bg_fpr_all = fp[i] / (fp[i] + tn[i] + 1e-7)
        rows.append((thr, float(dfg), float(dmean), float(imean), float(prec_fg), float(bg_fpr_all)))
        if float(dfg) > best_dfg:
            best_dfg = float(dfg)
            best_i = i
    return rows, best_i


def _per_image_metrics(prob: torch.Tensor, gt: torch.Tensor, thr: float) -> dict:
    """
    prob/gt: [1,1,H,W] tensors.
    Returns per-image pixel-level metrics.
    """
    tp, fp, fn, tn = _confusion_from_prob(prob, gt, thr=thr)
    dice_fg, dice_bg, mdice, iou_fg, iou_bg, miou = _dice_iou_from_conf(tp, fp, fn, tn)
    prec_fg = tp / (tp + fp + 1e-7)
    bg_fpr_all = fp / (fp + tn + 1e-7)
    gt_area = int((gt > 0.5).sum().item())
    pred_area = int((prob > thr).sum().item())
    tp_area = int(((prob > thr) & (gt > 0.5)).sum().item())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice_fg": float(dice_fg),
        "dice_bg": float(dice_bg),
        "mdice": float(mdice),
        "iou_fg": float(iou_fg),
        "iou_bg": float(iou_bg),
        "miou": float(miou),
        "prec_fg": float(prec_fg),
        "bg_fpr_all": float(bg_fpr_all),
        "gt_area": gt_area,
        "pred_area": pred_area,
        "tp_area": tp_area,
    }


def _parse_thr_list(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    thrs = [float(p) for p in parts]
    thrs = [min(1.0, max(0.0, t)) for t in thrs]
    return sorted(set(thrs))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to model checkpoint (.pth)")
    parser.add_argument(
        "--model",
        type=str,
        default="sam3-sam-edl",
        help="model name used in this repo (e.g., sam3-sam-edl / sam3-edl / sam-edl / sam)",
    )
    parser.add_argument("--inp-size", type=int, default=1024)
    parser.add_argument("--thr", type=float, default=0.5, help="binarization threshold (used when not sweeping)")
    parser.add_argument(
        "--sweep-thr",
        type=str,
        default="",
        help="comma-separated thresholds to sweep, e.g. '0.25,0.3,0.35,0.4,0.45,0.5'. "
        "If provided, ignores --thr and reports best dice_fg threshold.",
    )
    parser.add_argument("--vdr-min-area", type=int, default=1, help="VDR min TP area (pixels) on positive frames")
    parser.add_argument("--use-gated", action="store_true", help="use prob_gated for metrics (default: raw prob_vessel)")
    parser.add_argument(
        "--mix-raw-gated",
        type=float,
        default=-1.0,
        help="if in [0,1] and model supports EDL (infer_prob_uncert), use prob = (1-a)*raw + a*gated. "
        "This sometimes improves dice_fg without retraining.",
    )
    parser.add_argument(
        "--post-close-k",
        type=int,
        default=0,
        help="optional: apply binary closing on (prob>thr) mask before computing confusion. "
        "Use odd k like 3/5/7. 0 disables.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="enable simple flip TTA (orig+hflip+vflip+hvflip average). Slower but may improve dice_fg.",
    )
    parser.add_argument(
        "--tta-mode",
        type=str,
        default="none",
        choices=["none", "h", "hv"],
        help="TTA mode: none | h (orig+hflip) | hv (orig+hflip+vflip+hvflip). "
        "If --tta is set, it is treated as --tta-mode hv.",
    )
    parser.add_argument("--per-image", action="store_true", help="print per-image dice_fg")
    parser.add_argument("--out-csv", type=str, default="", help="optional: save per-image metrics to CSV")
    parser.add_argument(
        "--cod-raw",
        action="store_true",
        help="also report raw COD metrics (no min-max). Default COD metrics use old normalized protocol.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Ensure project root on path when running from tools/
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    # Ensure we import the intended upstream `sam3` package (sam3-main) rather than any
    # local folders with the same top-level name.
    sam3_main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sam3-main"))
    if sam3_main_dir not in sys.path:
        sys.path.insert(0, sam3_main_dir)

    import datasets  # noqa
    import models  # noqa
    import train_vessel_esd as base  # noqa
    import datasets.vessel_dataset  # noqa: F401  (register 'vessel-esd-dataset')

    # --- build val split (prefix match) ---
    all_patients = base.get_patient_ids(base.Config.DATA_ROOTS)

    def _canon_pid(pid: str) -> str:
        return str(pid).split("_", 1)[0]

    val_prefixes = {_canon_pid(v) for v in base.Config.VAL_PATIENTS}
    val_patients = [p for p in all_patients if _canon_pid(p) in val_prefixes]

    val_raw = datasets.make(
        {
            "name": "vessel-esd-dataset",
            "args": {
                "data_root": base.Config.DATA_ROOTS,
                "patient_ids": val_patients,
                "split": "val",
                "cache": "none",
                "include_unlabeled_negatives": False,
                "neg_data_roots": None,
                "neg_ratio": None,
                "max_negatives": None,
                "seed": 0,
            },
        }
    )
    val_ds = datasets.make({"name": "val", "args": {"dataset": val_raw, "inp_size": int(args.inp_size), "augment": False}})
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- build model ---
    # NOTE: `models/sam.py` requires `encoder_mode` (and `loss`) to be provided.
    if str(args.model) == "sam":
        encoder_mode = {
            "name": "sam",
            "patch_size": 16,
            "prompt_embed_dim": 256,
            "embed_dim": 1024,
        }
        model_cfg = {"name": "sam", "args": {"inp_size": int(args.inp_size), "encoder_mode": encoder_mode, "loss": "bce"}}
    else:
        model_cfg = {"name": args.model, "args": {"inp_size": int(args.inp_size)}}
    model = models.make(model_cfg).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    # --- run eval ---
    sweep_thrs = _parse_thr_list(args.sweep_thr)
    if sweep_thrs and args.per_image:
        raise ValueError("--per-image 暂不支持与 --sweep-thr 同时使用；请先 sweep 找到最佳阈值，再用 --thr 复跑输出 per-image。")
    if sweep_thrs and args.out_csv:
        raise ValueError("--out-csv 暂不支持与 --sweep-thr 同时使用；请先 sweep 找到最佳阈值，再用 --thr 复跑保存 CSV。")

    if sweep_thrs:
        tp = [0 for _ in sweep_thrs]
        fp = [0 for _ in sweep_thrs]
        fn = [0 for _ in sweep_thrs]
        tn = [0 for _ in sweep_thrs]
        tp_g = [[0 for _ in sweep_thrs] for _ in range(_N_VAL_GROUPS)]
        fp_g = [[0 for _ in sweep_thrs] for _ in range(_N_VAL_GROUPS)]
        fn_g = [[0 for _ in sweep_thrs] for _ in range(_N_VAL_GROUPS)]
        tn_g = [[0 for _ in sweep_thrs] for _ in range(_N_VAL_GROUPS)]
    else:
        tp = fp = fn = tn = 0
        tp_g = [0] * _N_VAL_GROUPS
        fp_g = [0] * _N_VAL_GROUPS
        fn_g = [0] * _N_VAL_GROUPS
        tn_g = [0] * _N_VAL_GROUPS
    cod_s = cod_e = cod_m = 0.0  # normalized (old protocol)
    cod_s_raw = cod_e_raw = cod_m_raw = 0.0
    n = 0
    vdr_hit = 0
    vdr_tot = 0
    vdr_hit_g = [0] * _N_VAL_GROUPS
    vdr_tot_g = [0] * _N_VAL_GROUPS
    cod_s_g = [0.0] * _N_VAL_GROUPS
    cod_e_g = [0.0] * _N_VAL_GROUPS
    cod_m_g = [0.0] * _N_VAL_GROUPS
    cod_s_raw_g = [0.0] * _N_VAL_GROUPS
    cod_e_raw_g = [0.0] * _N_VAL_GROUPS
    cod_m_raw_g = [0.0] * _N_VAL_GROUPS
    n_g = [0] * _N_VAL_GROUPS
    per_rows = []

    pbar = tqdm(total=len(val_loader), desc="eval-val", leave=False)
    for batch in val_loader:
        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        # name
        name = ""
        if "name" in batch:
            name = batch["name"][0]
        if not name:
            name = f"Frame_{n:06d}"

        pid_raw = ""
        if "patient_id" in batch:
            pv = batch["patient_id"]
            if isinstance(pv, (list, tuple)):
                pid_raw = str(pv[0]) if len(pv) > 0 else ""
            else:
                pid_raw = str(pv)
        gid = _val_group_index(_canon_patient_prefix(pid_raw))

        tta_mode = "hv" if args.tta else str(args.tta_mode)
        mix_a = float(args.mix_raw_gated)
        if 0.0 <= mix_a <= 1.0 and hasattr(model, "infer_prob_uncert"):
            # compute raw and gated with the same TTA, then mix
            if tta_mode == "hv":
                prob_raw = _infer_prob_tta(model, inp, use_gated=False)
                prob_gated = _infer_prob_tta(model, inp, use_gated=True)
            elif tta_mode == "h":
                prob_raw = _infer_prob_tta_hflip(model, inp, use_gated=False)
                prob_gated = _infer_prob_tta_hflip(model, inp, use_gated=True)
            else:
                prob_raw = _infer_prob(model, inp, use_gated=False)
                prob_gated = _infer_prob(model, inp, use_gated=True)
            prob = (1.0 - mix_a) * prob_raw + mix_a * prob_gated
        else:
            if tta_mode == "hv":
                prob = _infer_prob_tta(model, inp, use_gated=args.use_gated)
            elif tta_mode == "h":
                prob = _infer_prob_tta_hflip(model, inp, use_gated=args.use_gated)
            else:
                prob = _infer_prob(model, inp, use_gated=args.use_gated)

        use_post = int(args.post_close_k) > 0

        if sweep_thrs:
            if use_post:
                for i, thr in enumerate(sweep_thrs):
                    pred_thr = (prob > float(thr)).to(prob.dtype)
                    pred_bin = _binary_closing(pred_thr, k=int(args.post_close_k))
                    gt1 = (gt > 0.5)
                    tpc = int((pred_bin & gt1).sum().item())
                    fpc = int((pred_bin & (~gt1)).sum().item())
                    fnc = int(((~pred_bin) & gt1).sum().item())
                    tnc = int(((~pred_bin) & (~gt1)).sum().item())
                    tp[i] += tpc
                    fp[i] += fpc
                    fn[i] += fnc
                    tn[i] += tnc
                    if gid is not None:
                        tp_g[gid][i] += tpc
                        fp_g[gid][i] += fpc
                        fn_g[gid][i] += fnc
                        tn_g[gid][i] += tnc
                pm_thr = 0.5
                pred_thr = (prob > float(pm_thr)).to(prob.dtype)
                pred_bin = _binary_closing(pred_thr, k=int(args.post_close_k))
                tp_area = int((pred_bin & (gt > 0.5)).sum().item())
                pm = {"tp_area": tp_area}
            else:
                thr_t = torch.tensor(sweep_thrs, device=prob.device, dtype=prob.dtype).view(-1, 1, 1, 1)
                pred = prob > thr_t
                gt1 = (gt > 0.5)
                tpi = (pred & gt1).sum(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.int64)
                fpi = (pred & (~gt1)).sum(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.int64)
                fni = ((~pred) & gt1).sum(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.int64)
                tni = ((~pred) & (~gt1)).sum(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.int64)
                for i in range(len(sweep_thrs)):
                    tp[i] += int(tpi[i])
                    fp[i] += int(fpi[i])
                    fn[i] += int(fni[i])
                    tn[i] += int(tni[i])
                if gid is not None:
                    for i in range(len(sweep_thrs)):
                        tp_g[gid][i] += int(tpi[i])
                        fp_g[gid][i] += int(fpi[i])
                        fn_g[gid][i] += int(fni[i])
                        tn_g[gid][i] += int(tni[i])
                pm = _per_image_metrics(prob, gt, thr=float(0.5))
        else:
            if use_post:
                pred_thr = (prob > float(args.thr)).to(prob.dtype)
                pred_bin = _binary_closing(pred_thr, k=int(args.post_close_k))
                gt1 = (gt > 0.5)
                tp1 = int((pred_bin & gt1).sum().item())
                fp1 = int((pred_bin & (~gt1)).sum().item())
                fn1 = int(((~pred_bin) & gt1).sum().item())
                tn1 = int(((~pred_bin) & (~gt1)).sum().item())
                pm = _per_image_metrics(prob, gt, thr=float(args.thr))
                pm["tp"], pm["fp"], pm["fn"], pm["tn"] = tp1, fp1, fn1, tn1
                pm["dice_fg"], pm["dice_bg"], pm["mdice"], pm["iou_fg"], pm["iou_bg"], pm["miou"] = _dice_iou_from_conf(tp1, fp1, fn1, tn1)
                pm["prec_fg"] = tp1 / (tp1 + fp1 + 1e-7)
                pm["bg_fpr_all"] = fp1 / (fp1 + tn1 + 1e-7)
                pm["pred_area"] = int(pred_bin.sum().item())
                pm["tp_area"] = int((pred_bin & (gt > 0.5)).sum().item())
            else:
                pm = _per_image_metrics(prob, gt, thr=float(args.thr))
            tp1, fp1, fn1, tn1 = pm["tp"], pm["fp"], pm["fn"], pm["tn"]
            tp += tp1
            fp += fp1
            fn += fn1
            tn += tn1
            if gid is not None:
                tp_g[gid] += tp1
                fp_g[gid] += fp1
                fn_g[gid] += fn1
                tn_g[gid] += tn1

        gt_pos = (gt > 0.5)
        if int(gt_pos.sum().item()) > 0:
            vdr_tot += 1
            if int(pm["tp_area"]) >= int(args.vdr_min_area):
                vdr_hit += 1
        if gid is not None:
            if int(gt_pos.sum().item()) > 0:
                vdr_tot_g[gid] += 1
                if int(pm["tp_area"]) >= int(args.vdr_min_area):
                    vdr_hit_g[gid] += 1

        p01 = prob.detach().float().cpu().numpy()[0, 0]
        g01 = (gt.detach().float().cpu().numpy()[0, 0] > 0.5).astype(np.float32)
        # default: old protocol (min-max normalize pred per-image)
        p01n = p01
        pmin = float(p01n.min())
        pmax = float(p01n.max())
        if pmax > pmin:
            p01n = (p01n - pmin) / (pmax - pmin)
        s_n, e_n, m_n = _cod_metrics(p01n, g01)
        cod_s += s_n
        cod_e += e_n
        cod_m += m_n
        if args.cod_raw:
            s_r, e_r, m_r = _cod_metrics(p01, g01)
            cod_s_raw += s_r
            cod_e_raw += e_r
            cod_m_raw += m_r
        n += 1

        if gid is not None:
            cod_s_g[gid] += s_n
            cod_e_g[gid] += e_n
            cod_m_g[gid] += m_n
            n_g[gid] += 1
            if args.cod_raw:
                cod_s_raw_g[gid] += s_r
                cod_e_raw_g[gid] += e_r
                cod_m_raw_g[gid] += m_r

        if args.per_image and not sweep_thrs:
            print(f"[{name}]\tDice_fg: {pm['dice_fg']:.4f}\tmIoU: {pm['miou']:.4f}\tGT_Pixels: {pm['gt_area']}")
        if args.out_csv and not sweep_thrs:
            g_label = _VAL_GROUP_DEFS[gid][0] if gid is not None else ""
            row = {
                "name": name,
                "patient_id": pid_raw,
                "val_group": g_label,
                **pm,
                "S_measure": float(s_n),
                "E_measure": float(e_n),
                "MAE": float(m_n),
            }
            if args.cod_raw:
                row.update({"S_measure_raw": float(s_r), "E_measure_raw": float(e_r), "MAE_raw": float(m_r)})
            per_rows.append(row)

        pbar.update(1)

    pbar.close()

    # --- optional: dump per-image metrics ---
    # NOTE: We intentionally only support this in non-sweep mode.
    if args.out_csv and not sweep_thrs:
        out_csv = str(args.out_csv)
        out_dir = os.path.dirname(os.path.abspath(out_csv))
        if out_dir and (not os.path.isdir(out_dir)):
            os.makedirs(out_dir, exist_ok=True)

        preferred = [
            "name",
            "patient_id",
            "val_group",
            "tp",
            "fp",
            "fn",
            "tn",
            "dice_fg",
            "dice_bg",
            "mdice",
            "iou_fg",
            "iou_bg",
            "miou",
            "prec_fg",
            "bg_fpr_all",
            "gt_area",
            "pred_area",
            "tp_area",
            "S_measure",
            "E_measure",
            "MAE",
        ]
        keys = set()
        for r in per_rows:
            keys.update(r.keys())
        extra = sorted([k for k in keys if k not in preferred])
        fieldnames = [k for k in preferred if k in keys] + extra

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_rows:
                w.writerow(r)
        print(f"[VAL] wrote per-image metrics CSV: {out_csv} (n={len(per_rows)})")

    if sweep_thrs:
        mode = "gated(prob_gated)" if args.use_gated else "raw(prob_vessel)"
        print(f"[VAL] mode={mode} inp_size={args.inp_size} model={args.model} (threshold sweep)")

        def _print_sweep_block(title: str, tp_r: List[int], fp_r: List[int], fn_r: List[int], tn_r: List[int]) -> None:
            rows_b, best_i_b = _sweep_metrics_rows(tp_r, fp_r, fn_r, tn_r, sweep_thrs)
            print(f"[VAL] --- {title} ---")
            print("thr\tdice_fg\tmDice\tmIoU\tprec_fg\tbg_fpr_all")
            for i, (thr, dfg, dmean, imean, prec, fpr) in enumerate(rows_b):
                mark = "*" if i == best_i_b else ""
                print(f"{thr:.3f}\t{dfg:.4f}\t{dmean:.4f}\t{imean:.4f}\t{prec:.4f}\t{fpr:.4f}{mark}")
            print(
                f"[VAL] [BEST {title}] thr={rows_b[best_i_b][0]:.3f} dice_fg={rows_b[best_i_b][1]:.4f} "
                f"mDice={rows_b[best_i_b][2]:.4f} mIoU={rows_b[best_i_b][3]:.4f}"
            )

        _print_sweep_block("TOTAL (all val)", tp, fp, fn, tn)

        vdr = vdr_hit / (vdr_tot + 1e-7)
        cod_s /= max(1, n)
        cod_e /= max(1, n)
        cod_m /= max(1, n)
        print(f"[VAL TOTAL] VDR(TP>={args.vdr_min_area})={vdr:.4f} (VDR @thr=0.5 during sweep)")
        print(f"[VAL TOTAL] S-measure={cod_s:.4f}, E-measure={cod_e:.4f}, MAE={cod_m:.4f}")
        if args.cod_raw:
            cod_s_raw /= max(1, n)
            cod_e_raw /= max(1, n)
            cod_m_raw /= max(1, n)
            print(f"[VAL TOTAL] S-measure(raw)={cod_s_raw:.4f}, E-measure(raw)={cod_e_raw:.4f}, MAE(raw)={cod_m_raw:.4f}")

        for gi in range(_N_VAL_GROUPS):
            gname, _prefs = _VAL_GROUP_DEFS[gi]
            if n_g[gi] <= 0:
                print(f"[VAL] --- {gname} --- n_imgs=0 (skipped)")
                continue
            _print_sweep_block(gname, tp_g[gi], fp_g[gi], fn_g[gi], tn_g[gi])
            vdr_g = vdr_hit_g[gi] / (vdr_tot_g[gi] + 1e-7)
            nn_g = max(1, n_g[gi])
            print(f"[VAL {gname}] VDR(TP>={args.vdr_min_area})={vdr_g:.4f} (VDR @thr=0.5 during sweep)")
            print(
                f"[VAL {gname}] S-measure={cod_s_g[gi] / nn_g:.4f}, E-measure={cod_e_g[gi] / nn_g:.4f}, "
                f"MAE={cod_m_g[gi] / nn_g:.4f}"
            )
            if args.cod_raw:
                print(
                    f"[VAL {gname}] S-measure(raw)={cod_s_raw_g[gi] / nn_g:.4f}, "
                    f"E-measure(raw)={cod_e_raw_g[gi] / nn_g:.4f}, MAE(raw)={cod_m_raw_g[gi] / nn_g:.4f}"
                )
        return

    cod_s /= max(1, n)
    cod_e /= max(1, n)
    cod_m /= max(1, n)
    if args.cod_raw:
        cod_s_raw /= max(1, n)
        cod_e_raw /= max(1, n)
        cod_m_raw /= max(1, n)

    mode = "gated(prob_gated)" if args.use_gated else "raw(prob_vessel)"
    print(f"[VAL] mode={mode} thr={args.thr} inp_size={args.inp_size} model={args.model}")

    for gi in range(_N_VAL_GROUPS):
        gname, _prefs = _VAL_GROUP_DEFS[gi]
        if n_g[gi] <= 0:
            print(f"[VAL {gname}] n_imgs=0 (no samples in this run)")
            continue
        raw_kw = {}
        if args.cod_raw:
            raw_kw = {
                "cod_s_raw": cod_s_raw_g[gi],
                "cod_e_raw": cod_e_raw_g[gi],
                "cod_m_raw": cod_m_raw_g[gi],
            }
        for line in _metrics_lines(
            gname,
            tp_g[gi],
            fp_g[gi],
            fn_g[gi],
            tn_g[gi],
            vdr_hit_g[gi],
            vdr_tot_g[gi],
            cod_s_g[gi],
            cod_e_g[gi],
            cod_m_g[gi],
            n_g[gi],
            int(args.vdr_min_area),
            **raw_kw,
        ):
            print(line)

    print("[VAL] ========== TOTAL (all val) ==========")
    nn_tot = max(1, n)
    total_raw_kw = {}
    if args.cod_raw:
        # cod_*_raw were normalized to means above; _metrics_lines expects per-image sums
        total_raw_kw = {
            "cod_s_raw": cod_s_raw * nn_tot,
            "cod_e_raw": cod_e_raw * nn_tot,
            "cod_m_raw": cod_m_raw * nn_tot,
        }
    for line in _metrics_lines(
        "TOTAL",
        tp,
        fp,
        fn,
        tn,
        vdr_hit,
        vdr_tot,
        cod_s * nn_tot,
        cod_e * nn_tot,
        cod_m * nn_tot,
        n,
        int(args.vdr_min_area),
        **total_raw_kw,
    ):
        print(line)


if __name__ == "__main__":
    main()


