#!/usr/bin/env python3
"""
按“固定裁剪模板”批处理图片（参照 process_video.py 的思路）：

- 威海市立医院 / 聊城市人民医院 / 胜利油田中心医院：使用 P5 模板
- 山东第一医科大学附属省立医院：使用“当前模板”
- 注意：聊城市人民医院 仅处理 “食管ESD” 文件夹下的图片

输出默认保持原目录结构（相对 in-root）。
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
from tqdm import tqdm


def crop_black_border(img, up: int, bottom: int, left: int, right: int):
    h, w = img.shape[:2]
    up = max(0, int(up))
    bottom = max(0, int(bottom))
    left = max(0, int(left))
    right = max(0, int(right))
    y2 = h - bottom
    x2 = w - right
    if y2 <= up or x2 <= left:
        return None
    return img[up:y2, left:x2]


def iter_images(root: str, exts: Sequence[str]) -> List[str]:
    exts_l = {e.lower().lstrip(".") for e in exts}
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower().lstrip(".")
            if ext in exts_l:
                out.append(os.path.join(dirpath, fn))
    return out


def choose_template(abs_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    返回 (up, bottom, left, right) 或 None（表示跳过该文件）
    """
    # 医院名（路径包含即可）
    if "山东第一医科大学附属省立医院" in abs_path:
        # “现在的模版”：对应你之前常用的 left=658,right=20
        return (0, 0, 658, 20)

    # 聊城市人民医院：只处理 食管ESD 目录
    if "聊城市人民医院" in abs_path:
        if "食管ESD" not in abs_path:
            return None
        # P5 模板（process_video.py 注释里 “P39 P5 P21”）
        return (38, 38, 698, 60)

    # 其它指定医院：P5 模板
    if ("威海市立医院" in abs_path) or ("胜利油田中心医院" in abs_path):
        return (38, 38, 698, 60)

    return None


def process_one(
    in_path: str,
    in_root: str,
    out_root: str,
    overwrite: bool,
    out_ext: str,
    jpeg_quality: int,
    png_compression: int,
) -> Tuple[str, bool, str]:
    """
    Returns (in_path, ok, msg)
    """
    tpl = choose_template(in_path)
    if tpl is None:
        return in_path, True, "skip(no-template)"

    up, bottom, left, right = tpl
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        return in_path, False, "cv2.imread failed"

    cropped = crop_black_border(img, up, bottom, left, right)
    if cropped is None or cropped.size == 0:
        h, w = img.shape[:2]
        return in_path, False, f"invalid crop (h,w)={(h,w)} tpl={(up,bottom,left,right)}"

    rel = os.path.relpath(in_path, in_root)
    if rel.startswith(".."):
        rel = os.path.basename(in_path)
    rel_no_ext = os.path.splitext(rel)[0]
    out_rel = rel_no_ext + f".{out_ext.lstrip('.')}"
    out_path = os.path.join(out_root, out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if (not overwrite) and os.path.exists(out_path):
        return in_path, True, "skip(exists)"

    params: List[int] = []
    out_ext_l = out_ext.lower().lstrip(".")
    if out_ext_l in {"jpg", "jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif out_ext_l == "png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]

    ok = cv2.imwrite(out_path, cropped, params)
    if not ok:
        return in_path, False, f"cv2.imwrite failed -> {out_path}"
    return in_path, True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-root",
        type=str,
        default="./data",
        help="input root directory (scan recursively)",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="./data_processed_templates",
        help="output root directory (preserve relative structure)",
    )
    parser.add_argument("--exts", type=str, default="png,jpg,jpeg,bmp,tif,tiff,webp")
    parser.add_argument("--out-ext", type=str, default="png")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)
    exts = [e.strip() for e in str(args.exts).split(",") if e.strip()]
    out_ext = str(args.out_ext)

    paths = iter_images(in_root, exts=exts)
    # 只保留能匹配模板（以及聊城的食管ESD 子集）
    matched = [p for p in paths if choose_template(p) is not None]

    print(f"[scan] in_root={in_root}")
    print(f"[scan] total_images={len(paths)} matched_by_template={len(matched)}")
    print("[rule] P5 template for 威海市立医院/胜利油田中心医院/聊城市人民医院(仅食管ESD)")
    print("[rule] current template for 山东第一医科大学附属省立医院")

    if args.dry_run:
        return

    ok_cnt = 0
    fail_cnt = 0
    skip_cnt = 0
    errs: List[Tuple[str, str]] = []

    max_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                process_one,
                p,
                in_root,
                out_root,
                bool(args.overwrite),
                out_ext,
                int(args.jpeg_quality),
                int(args.png_compression),
            )
            for p in matched
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="process_images_templates"):
            _p, ok, msg = fut.result()
            if not ok:
                fail_cnt += 1
                errs.append((_p, msg))
            else:
                if msg.startswith("skip"):
                    skip_cnt += 1
                else:
                    ok_cnt += 1

    print(f"[done] ok={ok_cnt} skip={skip_cnt} fail={fail_cnt} out_root={out_root}")
    if errs:
        print("[errors] show first 20:")
        for p, e in errs[:20]:
            print(f" - {p} :: {e}")


if __name__ == "__main__":
    main()


