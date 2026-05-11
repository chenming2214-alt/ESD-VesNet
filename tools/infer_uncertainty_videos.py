"""
Batch export EDL uncertainty maps from DeepFlux video inputs.

Expected layout:
  <data-root>/
    elec/input/*.mp4
    esd/input/*.mp4
    process/input/*.mp4

For each video, this script samples frames by a fixed step and saves
uncertainty maps for the selected frame indices only.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def _list_videos(input_dir: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(input_dir):
        return out
    for fn in sorted(os.listdir(input_dir)):
        low = fn.lower()
        if fn.startswith("._"):
            continue
        if low.endswith(".mp4"):
            out.append(os.path.join(input_dir, fn))
    return out


def _iter_sample_indices(start: int, stop_inclusive: int, step: int) -> Iterable[int]:
    cur = int(start)
    step = max(1, int(step))
    stop_inclusive = int(stop_inclusive)
    while cur <= stop_inclusive:
        yield cur
        cur += step


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
    cv2.imwrite(out_path, _make_heat_bgr(arr01))


def _save_overlay(frame_bgr: np.ndarray, heat01: np.ndarray, out_path: str, alpha: float) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    heat_bgr = _make_heat_bgr(heat01)
    if heat_bgr.shape[:2] != frame_bgr.shape[:2]:
        heat_bgr = cv2.resize(heat_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(heat_bgr, float(alpha), frame_bgr, 1.0 - float(alpha), 0.0)
    cv2.imwrite(out_path, overlay)


def _build_model(model_name: str, inp_size: int, device: torch.device):
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    sam3_main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sam3-main"))
    if sam3_main_dir not in sys.path:
        sys.path.insert(0, sam3_main_dir)

    import models  # noqa

    if str(model_name) == "sam":
        encoder_mode = {
            "name": "sam",
            "patch_size": 16,
            "prompt_embed_dim": 256,
            "embed_dim": 1024,
        }
        model_cfg = {"name": "sam", "args": {"inp_size": int(inp_size), "encoder_mode": encoder_mode, "loss": "bce"}}
    else:
        model_cfg = {"name": str(model_name), "args": {"inp_size": int(inp_size)}}
    return models.make(model_cfg).to(device)


@torch.no_grad()
def _infer_uncertainty_edl(model, inp: torch.Tensor) -> torch.Tensor:
    target = model.module if hasattr(model, "module") else model
    if not hasattr(target, "infer_prob_uncert"):
        raise RuntimeError("Current model does not support infer_prob_uncert(); true EDL uncertainty is unavailable.")
    _prob_v, u, _prob_g = target.infer_prob_uncert(inp)
    return u


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EDL uncertainty maps from DeepFlux input videos.")
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data-hdd/msc2025/chenming/esd/vessel_seg/DeepFlux-master/data",
        help="DeepFlux data root containing elec/esd/process.",
    )
    parser.add_argument("--sections", nargs="+", default=["elec", "esd", "process"], help="Sections to process.")
    parser.add_argument("--input-subdir", type=str, default="input", help="Video subdirectory under each section.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(
            default_project_dir,
            "save",
            "vessel_esd_sam3_fullsam_edl_hnm_0105",
            "best_fullsam_0105.pth",
        ),
        help="Checkpoint path.",
    )
    parser.add_argument("--model", type=str, default="sam3-sam-edl", help="models.register() name.")
    parser.add_argument("--inp-size", type=int, default=1024, help="Square inference input size.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Output root. Defaults to <data-root>/edl_uncertainty_videos_step5.",
    )
    parser.add_argument("--frame-start", type=int, default=0, help="First frame index to sample.")
    parser.add_argument("--frame-step", type=int, default=5, help="Sampling step.")
    parser.add_argument("--max-frame-idx", type=int, default=60, help="Maximum sampled frame index (inclusive).")
    parser.add_argument("--save-heat", action="store_true", help="Also save contrast-stretched heatmap PNG.")
    parser.add_argument("--overlay", action="store_true", help="Also save heatmap overlay on the raw frame.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Overlay alpha in [0,1].")
    parser.add_argument("--save-npy", action="store_true", help="Also save raw float32 uncertainty map as .npy.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if the gray PNG already exists.")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data root not found: {data_root}")

    out_root = str(args.out_root).strip()
    if not out_root:
        out_root = os.path.join(data_root, f"edl_uncertainty_videos_step{int(args.frame_step)}")
    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)

    device = torch.device(args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("device:", device, flush=True)
    print(f"data_root: {data_root}", flush=True)
    print(f"out_root: {out_root}", flush=True)
    print(
        f"sampling: start={int(args.frame_start)} step={int(args.frame_step)} max_idx={int(args.max_frame_idx)} (inclusive)",
        flush=True,
    )

    model = _build_model(str(args.model), int(args.inp_size), device=device)
    try:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    target = model.module if hasattr(model, "module") else model
    if not hasattr(target, "infer_prob_uncert"):
        raise RuntimeError(f"Model '{args.model}' does not expose infer_prob_uncert(); cannot export EDL uncertainty.")
    print("uncertainty mode: true EDL uncertainty", flush=True)

    tfm = transforms.Compose(
        [
            transforms.Resize((int(args.inp_size), int(args.inp_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    total_videos = 0
    total_frames_saved = 0

    for section in args.sections:
        input_dir = os.path.join(data_root, str(section), str(args.input_subdir))
        videos = _list_videos(input_dir)
        print(f"[{section}] videos={len(videos)} input_dir={input_dir}", flush=True)
        total_videos += len(videos)

        for vid_idx, video_path in enumerate(videos, start=1):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(out_root, str(section), video_name)
            gray_probe = os.path.join(out_dir, f"frame_{int(args.frame_start):04d}_uncertainty_gray.png")
            if args.skip_existing and os.path.exists(gray_probe):
                print(f"[{section}] skip existing video {vid_idx}/{len(videos)}: {video_name}", flush=True)
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[warn] failed to open video: {video_path}", flush=True)
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            sample_indices = [i for i in _iter_sample_indices(args.frame_start, args.max_frame_idx, args.frame_step) if i < frame_count]
            if not sample_indices:
                cap.release()
                print(f"[warn] no sampled frames for {video_name} (frame_count={frame_count})", flush=True)
                continue

            print(
                f"[{section}] video {vid_idx}/{len(videos)}: {video_name} frame_count={frame_count} sampled={sample_indices}",
                flush=True,
            )

            saved_this_video = 0
            for frame_idx in sample_indices:
                gray_path = os.path.join(out_dir, f"frame_{frame_idx:04d}_uncertainty_gray.png")
                heat_path = os.path.join(out_dir, f"frame_{frame_idx:04d}_uncertainty_heat.png")
                overlay_path = os.path.join(out_dir, f"frame_{frame_idx:04d}_uncertainty_overlay.png")
                npy_path = os.path.join(out_dir, f"frame_{frame_idx:04d}_uncertainty.npy")

                if args.skip_existing and os.path.exists(gray_path):
                    saved_this_video += 1
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    print(f"[warn] failed to read frame {frame_idx} from {video_path}", flush=True)
                    continue

                h0, w0 = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                inp = tfm(img).unsqueeze(0).to(device)

                uncert = _infer_uncertainty_edl(model, inp)
                uncert_up = F.interpolate(uncert, size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
                uncert_np = uncert_up.detach().float().cpu().numpy()
                uncert_np = np.clip(uncert_np, 0.0, 1.0)

                _save_gray_png(uncert_np, gray_path)

                if args.save_npy:
                    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                    np.save(npy_path, uncert_np.astype(np.float32))

                if args.save_heat or args.overlay:
                    uncert_vis = _normalize_for_vis(uncert_np, p_lo=1.0, p_hi=99.0)
                    if args.save_heat:
                        _save_heat_png(uncert_vis, heat_path)
                    if args.overlay:
                        _save_overlay(frame_bgr, uncert_vis, overlay_path, alpha=float(args.overlay_alpha))

                saved_this_video += 1
                total_frames_saved += 1

            cap.release()
            print(f"[{section}] done {video_name}: saved_frames={saved_this_video}", flush=True)

    print(f"done. total_videos={total_videos}, total_frames_saved={total_frames_saved}", flush=True)
    print(f"output_root: {out_root}", flush=True)


if __name__ == "__main__":
    main()
