"""
对每个 clip 子目录读取 summary.json 中的 video_path，对 local 帧 0..num_frames-1
（通常为 0-60）逐帧推理 EDL 不确定性，将 float32 图保存为 npy 到「uncertainty map」。

未指定 --selected-clip-root 时，依次尝试 NAS 与 DeepFlux HDD 路径（见
_DEFAULT_SELECTED_CLIP_CANDIDATES）；也可用环境变量 SELECTED_CLIP_ROOT。

默认写出根目录：{uncertainty_out_root}/{clip 名}/uncertainty map/

参考 tools/infer_uncertainty_videos.py 的模型加载与推理流程。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

UNCERTAINTY_SUBDIR = "uncertainty map"

# 读 labels / summary.json 的根目录（任一路径存在即可自动选用，也可用环境变量覆盖）
_DEFAULT_SELECTED_CLIP_CANDIDATES = (
    "/home/user/NAS/mengya/ESD_Bleeding/Vessel_seg/selected_clip",
    "/mnt/data-hdd/msc2025/chenming/esd/vessel_seg/DeepFlux-master/data/selected_clip",
)


def _resolve_selected_clip_root(cli_value: str | None) -> str:
    env = os.environ.get("SELECTED_CLIP_ROOT", "").strip()
    if env:
        root = os.path.abspath(env)
        if os.path.isdir(root):
            return root
        raise FileNotFoundError(
            f"SELECTED_CLIP_ROOT is set but not a directory: {root}\n"
            f"Unset it or point it to a folder that contains clip subdirs with summary.json."
        )
    if cli_value:
        root = os.path.abspath(str(cli_value).rstrip("/\\"))
        if os.path.isdir(root):
            return root
        raise FileNotFoundError(
            f"selected_clip root not found: {root}\n"
            f"Pass --selected-clip-root to your data, or set env SELECTED_CLIP_ROOT."
        )
    for cand in _DEFAULT_SELECTED_CLIP_CANDIDATES:
        root = os.path.abspath(cand)
        if os.path.isdir(root):
            if cand != _DEFAULT_SELECTED_CLIP_CANDIDATES[0]:
                print(f"[info] using selected_clip root: {root}", flush=True)
            return root
    raise FileNotFoundError(
        "Could not find selected_clip. Tried:\n  "
        + "\n  ".join(_DEFAULT_SELECTED_CLIP_CANDIDATES)
        + "\nPass --selected-clip-root or set SELECTED_CLIP_ROOT."
    )


def _list_clip_dirs(selected_root: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(selected_root)):
        if name.startswith("._"):
            continue
        p = os.path.join(selected_root, name)
        if os.path.isdir(p):
            out.append(p)
    return out


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


def _load_summary(clip_dir: str) -> Tuple[str, int]:
    path = os.path.join(clip_dir, "summary.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing summary.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    video_path = str(summary.get("video_path", "")).strip()
    if not video_path:
        raise ValueError(f"empty video_path in {path}")
    n = int(summary.get("num_frames", 61))
    if n <= 0:
        raise ValueError(f"invalid num_frames in {path}: {n}")
    return os.path.abspath(video_path), n


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-frame EDL uncertainty .npy under each selected_clip folder.")
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument(
        "--selected-clip-root",
        type=str,
        default="",
        help="Root with clip subfolders (summary.json). Empty = try NAS then DeepFlux HDD path. "
        "Override with env SELECTED_CLIP_ROOT (highest priority).",
    )
    parser.add_argument(
        "--uncertainty-out-root",
        type=str,
        default="/home/user/NAS/mengya/ESD_Bleeding/Vessel_seg/selected_clip",
        help="Mirror selected_clip layout here: <root>/<clip_name>/uncertainty map/*.npy",
    )
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
        "--max-local-idx",
        type=int,
        default=60,
        help="Last local frame index to write (inclusive). Frames beyond num_frames-1 are skipped.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip frame if its .npy already exists.")
    parser.add_argument("--max-clips", type=int, default=0, help="Process only first N clips (sorted). 0 = all.")
    args = parser.parse_args()

    cli_sel = str(args.selected_clip_root).strip() or None
    selected_root = _resolve_selected_clip_root(cli_sel)

    uncertainty_out_root = os.path.abspath(str(args.uncertainty_out_root).rstrip("/\\"))
    os.makedirs(uncertainty_out_root, exist_ok=True)

    device = torch.device(args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("device:", device, flush=True)
    print(f"selected_clip_root: {selected_root}", flush=True)
    print(f"uncertainty_out_root: {uncertainty_out_root}", flush=True)

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

    clip_dirs = _list_clip_dirs(selected_root)
    if int(args.max_clips) > 0:
        clip_dirs = clip_dirs[: int(args.max_clips)]

    max_local = int(args.max_local_idx)
    total_frames = 0
    skipped_clips = 0

    for ci, clip_dir in enumerate(clip_dirs, start=1):
        name = os.path.basename(clip_dir)
        try:
            video_path, num_frames = _load_summary(clip_dir)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"[skip] {name}: bad summary — {e}", flush=True)
            skipped_clips += 1
            continue

        if not os.path.isfile(video_path):
            print(f"[skip] {name}: video missing: {video_path}", flush=True)
            skipped_clips += 1
            continue

        n_end = min(num_frames, max_local + 1)
        if n_end <= 0:
            print(f"[skip] {name}: no frames in range", flush=True)
            skipped_clips += 1
            continue

        out_dir = os.path.join(uncertainty_out_root, name, UNCERTAINTY_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[skip] {name}: cannot open video {video_path}", flush=True)
            skipped_clips += 1
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        written_here = 0
        skipped_here = 0
        for local_idx in range(0, n_end):
            npy_path = os.path.join(out_dir, f"frame_{local_idx:04d}_uncertainty.npy")
            if args.skip_existing and os.path.isfile(npy_path):
                skipped_here += 1
                total_frames += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(local_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print(f"[warn] {name}: read failed local_idx={local_idx} (frame_count={frame_count})", flush=True)
                continue

            h0, w0 = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            inp = tfm(img).unsqueeze(0).to(device)

            uncert = _infer_uncertainty_edl(model, inp)
            uncert_up = F.interpolate(uncert, size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
            uncert_np = uncert_up.detach().float().cpu().numpy()
            uncert_np = np.clip(uncert_np, 0.0, 1.0).astype(np.float32)

            np.save(npy_path, uncert_np)
            written_here += 1
            total_frames += 1

        cap.release()
        print(
            f"[{ci}/{len(clip_dirs)}] {name}: local 0..{n_end - 1} -> {out_dir} "
            f"(written={written_here} skipped_existing={skipped_here})",
            flush=True,
        )

    print(f"done. clips={len(clip_dirs)} skipped_bad={skipped_clips} frames_handled={total_frames}", flush=True)


if __name__ == "__main__":
    main()
