"""
Run inference on images and save LabelMe JSONs.

默认将 JSON 写到 <项目根>/save/output/ 下，按子目录镜像数据源（见 --json-out-dir）。
可选 --json-next-to-image 仍写在图片同目录；--mp4 的 JSON 默认也进 --json-out-dir。

PNG 数据集示例（齐鲁 qilu）：.../vessel_data/qilu/<modality>/<Patient>/frameXXXXX.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, List, Tuple

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


def _mask_to_polygons(mask01: np.ndarray, min_area: int, poly_eps: float) -> List[List[List[float]]]:
    mask255 = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[List[float]]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            continue
        if poly_eps > 0:
            eps = poly_eps * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, eps, True)
        if cnt.shape[0] < 3:
            continue
        pts = cnt.reshape(-1, 2).astype(float).tolist()
        polys.append(pts)
    return polys


def _iter_images(root: str) -> Iterable[Tuple[str, str]]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                yield dirpath, fn


def _labelme_image_path_field(img_abs: str, json_path: str) -> str:
    """imagePath 相对 JSON 所在目录指向原图，便于 LabelMe 打开。"""
    ia = os.path.abspath(img_abs)
    jd = os.path.dirname(os.path.abspath(json_path))
    try:
        return os.path.relpath(ia, jd)
    except ValueError:
        return ia


def _write_labelme_json(
    json_path: str,
    image_path: str,
    image_height: int,
    image_width: int,
    polygons: List[List[List[float]]],
    label: str,
) -> None:
    shapes = []
    for pts in polygons:
        shapes.append(
            {
                "label": label,
                "points": pts,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        )
    payload = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
def _infer_and_write_labelme(
    model,
    tfm: transforms.Compose,
    device: torch.device,
    img_path: str,
    json_path: str,
    image_path_field: str,
    args: argparse.Namespace,
) -> bool:
    """Returns True if a new JSON was written."""
    if args.skip_existing and os.path.exists(json_path):
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
    bin01 = (prob_np > float(args.thr)).astype(np.uint8)

    polygons = _mask_to_polygons(bin01, min_area=int(args.min_area), poly_eps=float(args.poly_eps))
    _write_labelme_json(
        json_path=json_path,
        image_path=image_path_field,
        image_height=h0,
        image_width=w0,
        polygons=polygons,
        label=str(args.label),
    )
    return True


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="root folder; recursively processes .png/.jpg… (e.g. eomt-vessel/.../vessel_data/qilu)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="with --data-dir only: stop after N images (0 = no limit)",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="single image path (repeatable); JSON 默认写入 --json-out-dir",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    parser.add_argument("--model", type=str, default="sam3-sam-edl", help="models.register() name")
    parser.add_argument("--inp-size", type=int, default=1024)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--use-gated", action="store_true")
    parser.add_argument("--tta-mode", type=str, default="none", choices=["none", "h", "hv"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--label", type=str, default="vessel")
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--poly-eps", type=float, default=0.0, help="polygon approx epsilon ratio (0 disables)")
    parser.add_argument("--skip-existing", action="store_true", help="skip if .json already exists")
    parser.add_argument(
        "--json-out-dir",
        type=str,
        default="",
        help="LabelMe JSON 根目录；默认 <项目>/save/output（需配合默认布局时先创建）",
    )
    parser.add_argument(
        "--json-next-to-image",
        action="store_true",
        help="JSON 与图片同目录（忽略 --json-out-dir）",
    )
    parser.add_argument(
        "--mp4",
        action="append",
        default=[],
        help="video path (repeatable): extract frame PNG 到 --mp4-out-dir；JSON 默认到 --json-out-dir",
    )
    parser.add_argument(
        "--mp4-out-dir",
        type=str,
        default="",
        help="output folder for --mp4 mode (PNG basename matches JSON)",
    )
    parser.add_argument(
        "--mp4-frame-idx",
        type=int,
        default=30,
        help="local frame index in mp4 (default 30 = common 61-frame clip anchor)",
    )
    args = parser.parse_args()

    mp4_list = [os.path.abspath(p) for p in (args.mp4 or []) if str(p).strip()]
    image_list = [os.path.abspath(p) for p in (args.image or []) if str(p).strip()]
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

    if args.json_next_to_image:
        json_out_root: str | None = None
    else:
        j = str(args.json_out_dir).strip()
        json_out_root = os.path.abspath(j) if j else os.path.join(proj_root, "save", "output")
        os.makedirs(json_out_root, exist_ok=True)
        print(f"json_out_root: {json_out_root}", flush=True)

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
        model_cfg = {"name": args.model, "args": {"inp_size": int(args.inp_size)}}
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

    if str(args.data_dir).strip():
        data_dir = os.path.abspath(args.data_dir)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"data dir not found: {data_dir}")
        max_img = int(args.max_images)
        n_seen = 0
        for dirpath, fn in _iter_images(data_dir):
            n_seen += 1
            if max_img > 0 and n_seen > max_img:
                break
            total += 1
            base = os.path.splitext(fn)[0]
            img_path = os.path.join(dirpath, fn)
            if json_out_root is None:
                json_path = os.path.join(dirpath, f"{base}.json")
                ref = fn
            else:
                rel_dir = os.path.relpath(dirpath, data_dir)
                if rel_dir in (".", ""):
                    json_path = os.path.join(json_out_root, f"{base}.json")
                else:
                    json_path = os.path.join(json_out_root, rel_dir, f"{base}.json")
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                ref = _labelme_image_path_field(img_path, json_path)
            if _infer_and_write_labelme(model, tfm, device, img_path, json_path, ref, args):
                saved += 1

    for img_path in image_list:
        if not os.path.isfile(img_path):
            print(f"[skip] missing image: {img_path}", flush=True)
            continue
        low = img_path.lower()
        if not low.endswith(IMG_EXTS):
            print(f"[skip] not an image extension: {img_path}", flush=True)
            continue
        fn = os.path.basename(img_path)
        dirpath = os.path.dirname(img_path)
        base = os.path.splitext(fn)[0]
        if json_out_root is None:
            json_path = os.path.join(dirpath, f"{base}.json")
            ref = fn
        else:
            patient = os.path.basename(dirpath)
            modality = os.path.basename(os.path.dirname(dirpath))
            if patient and modality:
                json_path = os.path.join(json_out_root, modality, patient, f"{base}.json")
            else:
                json_path = os.path.join(json_out_root, f"{base}.json")
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            ref = _labelme_image_path_field(img_path, json_path)
        total += 1
        if _infer_and_write_labelme(model, tfm, device, img_path, json_path, ref, args):
            saved += 1
            print(f"[ok] {img_path} -> {json_path}", flush=True)

    mp4_out = os.path.abspath(str(args.mp4_out_dir).strip()) if mp4_list else ""
    fidx = int(args.mp4_frame_idx)
    for mp4_path in mp4_list:
        if not os.path.isfile(mp4_path):
            print(f"[skip] missing mp4: {mp4_path}", flush=True)
            continue
        stem = os.path.splitext(os.path.basename(mp4_path))[0]
        png_fn = f"{stem}_local{fidx:03d}.png"
        png_path = os.path.join(mp4_out, png_fn)
        json_name = f"{stem}_local{fidx:03d}.json"
        if json_out_root is None:
            json_path = os.path.join(mp4_out, json_name)
            ref = png_fn
        else:
            json_path = os.path.join(json_out_root, json_name)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            ref = _labelme_image_path_field(png_path, json_path)
        total += 1
        try:
            _extract_mp4_frame_png(mp4_path, fidx, png_path)
        except RuntimeError as e:
            print(f"[skip] {mp4_path}: {e}", flush=True)
            continue
        if _infer_and_write_labelme(model, tfm, device, png_path, json_path, ref, args):
            saved += 1
            print(f"[ok] {stem} -> {json_path}", flush=True)

    print(f"done. items={total}, json_saved={saved}")


if __name__ == "__main__":
    main()

