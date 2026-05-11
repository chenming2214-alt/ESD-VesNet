import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def crop_black_border(img, up: int, bottom: int, left: int, right: int):
    """Crop fixed borders from an image."""
    h, w = img.shape[:2]
    up = max(0, int(up))
    bottom = max(0, int(bottom))
    left = max(0, int(left))
    right = max(0, int(right))
    y2 = max(up, h - bottom)
    x2 = max(left, w - right)
    return img[up:y2, left:x2]


def auto_crop_black_border(
    img,
    thr: int = 10,
    pad: int = 0,
    pad4: Optional[Tuple[int, int, int, int]] = None,  # (top,bottom,left,right)
    min_keep_ratio: float = 0.2,
):
    """
    Auto-crop black borders by finding bounding box of non-black pixels.

    - thr: grayscale threshold; pixels > thr are treated as content
    - pad: keep extra margin around content bbox
    - min_keep_ratio: if detected content bbox area is too small vs full image, fall back to original
    """
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = g > int(thr)
    if not bool(m.any()):
        return img
    ys, xs = np.where(m)
    y1, y2 = int(ys.min()), int(ys.max() + 1)
    x1, x2 = int(xs.min()), int(xs.max() + 1)

    # pad and clamp
    if pad4 is None:
        p = max(0, int(pad))
        pt = pb = pl = pr = p
    else:
        pt, pb, pl, pr = [max(0, int(v)) for v in pad4]
    y1 = max(0, y1 - pt)
    x1 = max(0, x1 - pl)
    y2 = min(h, y2 + pb)
    x2 = min(w, x2 + pr)

    bbox_area = max(0, (y2 - y1)) * max(0, (x2 - x1))
    full_area = max(1, h * w)
    if float(bbox_area) / float(full_area) < float(min_keep_ratio):
        # likely mis-detection; keep original
        return img
    if (y2 - y1) <= 0 or (x2 - x1) <= 0:
        return img
    return img[y1:y2, x1:x2]


def _smooth_1d(x: np.ndarray, k: int = 7) -> np.ndarray:
    k = int(k)
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), ker, mode="same")


def detect_endoscope_boundaries(
    frame: np.ndarray,
    score_thr: float = 30.0,
    black_thr: float = 10.0,
    smooth_k: int = 7,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect endoscope content boundaries by scanning brightness+texture transition.
    Returns (left, right, top, bottom) inclusive bounds in pixel coordinates.

    This is a simplified, robust variant of your reference logic:
    - column score = 0.7 * mean + 0.3 * std in middle vertical ROI
    - row score    = 0.7 * mean + 0.3 * std in ROI between left/right
    - boundaries are found by scanning from edges for 'content' vs 'black' transition
    """
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- detect left/right on mid vertical ROI ---
    y0 = int(h * 0.25)
    y1 = int(h * 0.75)
    if y1 <= y0 + 10:
        y0, y1 = 0, h
    roi = gray[y0:y1, :]
    col_mean = roi.mean(axis=0)
    col_std = roi.std(axis=0)
    col_score = 0.7 * col_mean + 0.3 * col_std
    col_score = _smooth_1d(col_score, k=smooth_k)

    def _largest_true_segment(m: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return (start,end) inclusive of the longest contiguous True segment."""
        m = np.asarray(m).astype(bool)
        if m.size == 0 or not bool(m.any()):
            return None
        best = None
        best_len = -1
        start = None
        for i, v in enumerate(m.tolist()):
            if v and start is None:
                start = i
            if (not v) and (start is not None):
                end = i - 1
                ln = end - start + 1
                if ln > best_len:
                    best_len = ln
                    best = (start, end)
                start = None
        if start is not None:
            end = m.size - 1
            ln = end - start + 1
            if ln > best_len:
                best = (start, end)
        return best

    # Use adaptive threshold if fixed score_thr is too strict/too loose.
    hi = float(np.percentile(col_score, 90))
    lo = float(np.percentile(col_score, 10))
    thr_adapt = lo + 0.35 * (hi - lo)
    thr = float(max(min(float(score_thr), hi), thr_adapt))

    seg = _largest_true_segment(col_score > thr)
    if seg is None:
        return None
    left, right = seg

    left = max(0, int(left))
    right = min(w - 1, int(right))
    if right <= left + max(10, w // 20):
        return None

    # --- detect top/bottom on ROI between left/right ---
    x0 = max(0, left)
    x1b = min(w, right + 1)
    if x1b <= x0 + 10:
        return None
    roi2 = gray[:, x0:x1b]
    row_mean = roi2.mean(axis=1)
    row_std = roi2.std(axis=1)
    row_score = 0.7 * row_mean + 0.3 * row_std
    row_score = _smooth_1d(row_score, k=smooth_k)

    hi_r = float(np.percentile(row_score, 90))
    lo_r = float(np.percentile(row_score, 10))
    thr_adapt_r = lo_r + 0.35 * (hi_r - lo_r)
    thr_r = float(max(min(float(score_thr), hi_r), thr_adapt_r))
    seg_r = _largest_true_segment(row_score > thr_r)
    if seg_r is None:
        return None
    top, bottom = seg_r
    top = max(0, int(top))
    bottom = min(h - 1, int(bottom))
    if bottom <= top + max(10, h // 20):
        return None
    return int(left), int(right), int(top), int(bottom)


def auto_crop_endoscope(
    img,
    score_thr: float = 30.0,
    black_thr: float = 10.0,
    smooth_k: int = 7,
    pad: int = 0,
    pad4: Optional[Tuple[int, int, int, int]] = None,  # (top,bottom,left,right)
    min_keep_ratio: float = 0.2,
):
    h, w = img.shape[:2]
    out = detect_endoscope_boundaries(
        img,
        score_thr=float(score_thr),
        black_thr=float(black_thr),
        smooth_k=int(smooth_k),
    )
    if out is None:
        return img
    left, right, top, bottom = out
    if pad4 is None:
        p = max(0, int(pad))
        pt = pb = pl = pr = p
    else:
        pt, pb, pl, pr = [max(0, int(v)) for v in pad4]
    left = max(0, left - pl)
    right = min(w - 1, right + pr)
    top = max(0, top - pt)
    bottom = min(h - 1, bottom + pb)
    crop = img[top : bottom + 1, left : right + 1]
    if crop.size == 0:
        return img
    bbox_area = int(crop.shape[0] * crop.shape[1])
    full_area = max(1, int(h * w))
    if float(bbox_area) / float(full_area) < float(min_keep_ratio):
        return img
    return crop


def center_square_crop(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = max(0, (h - s) // 2)
    x0 = max(0, (w - s) // 2)
    return img[y0 : y0 + s, x0 : x0 + s]


def _iter_image_files(root: str, exts: Sequence[str]) -> Iterable[str]:
    exts_l = {e.lower().lstrip(".") for e in exts}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower().lstrip(".")
            if ext in exts_l:
                yield os.path.join(dirpath, fn)


def _safe_relpath(p: str, root: str) -> str:
    rp = os.path.relpath(p, root)
    # avoid weird relpath like ../../ outside root
    if rp.startswith(".."):
        return os.path.basename(p)
    return rp


def process_one(
    in_path: str,
    in_root: str,
    out_root: str,
    crop: Tuple[int, int, int, int],
    keep_structure: bool,
    out_ext: str,
    overwrite: bool,
    jpeg_quality: int,
    png_compression: int,
    auto_crop: bool,
    auto_thr: int,
    auto_pad: int,
    auto_min_keep_ratio: float,
    auto_mode: str,
    endo_score_thr: float,
    endo_black_thr: float,
    endo_smooth_k: int,
    square: bool,
    pad4: Optional[Tuple[int, int, int, int]],
    min_out_hw: Tuple[int, int],
) -> Tuple[str, bool, Optional[str]]:
    """
    Returns (in_path, ok, err_msg)
    """
    up, bottom, left, right = crop
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        return in_path, False, "cv2.imread failed"

    img2 = img
    if bool(auto_crop):
        mode = str(auto_mode or "bbox").lower()
        if mode == "bbox":
            img2 = auto_crop_black_border(
                img2,
                thr=int(auto_thr),
                pad=int(auto_pad),
                pad4=pad4,
                min_keep_ratio=float(auto_min_keep_ratio),
            )
        elif mode == "endoscope":
            img2 = auto_crop_endoscope(
                img2,
                score_thr=float(endo_score_thr),
                black_thr=float(endo_black_thr),
                smooth_k=int(endo_smooth_k),
                pad=int(auto_pad),
                pad4=pad4,
                min_keep_ratio=float(auto_min_keep_ratio),
            )
        else:
            # unknown mode -> no auto crop
            pass
    # optional manual crop after auto-crop (or alone)
    if any(int(v) != 0 for v in (up, bottom, left, right)):
        img2 = crop_black_border(img2, up=up, bottom=bottom, left=left, right=right)
    if bool(square):
        img2 = center_square_crop(img2)
    if img2 is None or img2.size == 0:
        return in_path, False, "empty after crop"
    mh, mw = int(min_out_hw[0]), int(min_out_hw[1])
    if mh > 0 and mw > 0:
        hh, ww = img2.shape[:2]
        if hh < mh or ww < mw:
            return in_path, False, f"too small after crop: {(hh, ww)} < min {(mh, mw)}"

    rel = _safe_relpath(in_path, in_root) if keep_structure else os.path.basename(in_path)
    rel_no_ext = os.path.splitext(rel)[0]
    out_rel = rel_no_ext + f".{out_ext.lstrip('.')}"
    out_path = os.path.join(out_root, out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if (not overwrite) and os.path.exists(out_path):
        return in_path, True, None

    params: List[int] = []
    out_ext_l = out_ext.lower().lstrip(".")
    if out_ext_l in {"jpg", "jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif out_ext_l == "png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]

    ok = cv2.imwrite(out_path, img2, params)
    if not ok:
        return in_path, False, f"cv2.imwrite failed -> {out_path}"
    return in_path, True, None


def main():
    parser = argparse.ArgumentParser(description="Process all images under a folder (crop borders, re-save).")
    parser.add_argument(
        "--in-root",
        type=str,
        default="./data",
        help="input root directory (will be scanned recursively)",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="./data_processed",
        help="output root directory",
    )
    parser.add_argument("--up", type=int, default=0, help="crop pixels from top")
    parser.add_argument("--bottom", type=int, default=0, help="crop pixels from bottom")
    parser.add_argument("--left", type=int, default=0, help="crop pixels from left")
    parser.add_argument("--right", type=int, default=0, help="crop pixels from right")
    parser.add_argument(
        "--auto-crop-black",
        action="store_true",
        help="auto-detect black borders and crop to the non-black content bbox (recommended for mixed sources)",
    )
    parser.add_argument(
        "--auto-mode",
        type=str,
        default="bbox",
        choices=["bbox", "endoscope"],
        help="auto crop mode: bbox (non-black bbox) | endoscope (detect scope boundaries by brightness/texture transition)",
    )
    parser.add_argument("--auto-thr", type=int, default=10, help="auto-crop grayscale threshold (default 10)")
    parser.add_argument("--auto-pad", type=int, default=0, help="auto-crop padding around detected bbox (pixels)")
    parser.add_argument("--auto-pad-top", type=int, default=0, help="auto-crop extra padding on top (pixels)")
    parser.add_argument("--auto-pad-bottom", type=int, default=0, help="auto-crop extra padding on bottom (pixels)")
    parser.add_argument("--auto-pad-left", type=int, default=0, help="auto-crop extra padding on left (pixels)")
    parser.add_argument("--auto-pad-right", type=int, default=0, help="auto-crop extra padding on right (pixels)")
    parser.add_argument(
        "--auto-min-keep-ratio",
        type=float,
        default=0.2,
        help="min bbox area ratio vs full image; below this fall back to original (default 0.2)",
    )
    parser.add_argument("--endo-score-thr", type=float, default=30.0, help="endoscope mode: content score threshold")
    parser.add_argument("--endo-black-thr", type=float, default=10.0, help="endoscope mode: black score threshold")
    parser.add_argument("--endo-smooth-k", type=int, default=7, help="endoscope mode: 1D smoothing kernel")
    parser.add_argument("--square", action="store_true", help="after cropping, center-crop to a square")
    parser.add_argument(
        "--include-substr",
        type=str,
        default="",
        help="only process files whose full path contains ANY of these substrings (comma-separated). "
        "Example: '威海市立医院' or '齐鲁医院青岛院区,胜利油田中心医院'. Empty means no filtering.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="for quick validation: if >0, only process first N matched images after filtering",
    )
    parser.add_argument(
        "--min-out-hw",
        type=str,
        default="64,64",
        help="min output height,width; too small outputs are treated as failures (default '64,64')",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default="png,jpg,jpeg,bmp,tif,tiff,webp",
        help="comma-separated input image extensions",
    )
    parser.add_argument(
        "--out-ext",
        type=str,
        default="png",
        help="output extension: png/jpg/jpeg/webp (default png)",
    )
    parser.add_argument("--keep-structure", action="store_true", help="preserve relative folder structure under out-root")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    parser.add_argument("--workers", type=int, default=20, help="number of threads for IO")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--png-compression", type=int, default=3, help="PNG compression (0-9)")
    parser.add_argument("--dry-run", action="store_true", help="only list how many images will be processed")
    args = parser.parse_args()

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)
    exts = [e.strip() for e in (args.exts or "").split(",") if e.strip()]
    if not exts:
        raise ValueError("--exts is empty")

    img_paths = list(_iter_image_files(in_root, exts=exts))
    inc = [s.strip() for s in str(args.include_substr).split(",") if s.strip()]
    if inc:
        img_paths = [p for p in img_paths if any(k in p for k in inc)]
    if int(args.max_files) > 0:
        img_paths = img_paths[: int(args.max_files)]
    print(f"[scan] in_root={in_root}")
    print(f"[scan] found_images={len(img_paths)} (exts={exts})")
    if args.dry_run:
        return

    crop = (int(args.up), int(args.bottom), int(args.left), int(args.right))
    keep_structure = bool(args.keep_structure)
    out_ext = str(args.out_ext)
    auto_crop = bool(args.auto_crop_black)
    auto_mode = str(args.auto_mode)
    pad4 = None
    if any(int(v) > 0 for v in (args.auto_pad_top, args.auto_pad_bottom, args.auto_pad_left, args.auto_pad_right)):
        pad4 = (int(args.auto_pad_top), int(args.auto_pad_bottom), int(args.auto_pad_left), int(args.auto_pad_right))
    try:
        mh_s, mw_s = [x.strip() for x in str(args.min_out_hw).split(",", 1)]
        min_out_hw = (int(mh_s), int(mw_s))
    except Exception:
        raise ValueError("--min-out-hw must be like '64,64'")

    ok_cnt = 0
    fail_cnt = 0
    errors: List[Tuple[str, str]] = []

    max_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                process_one,
                p,
                in_root,
                out_root,
                crop,
                keep_structure,
                out_ext,
                bool(args.overwrite),
                int(args.jpeg_quality),
                int(args.png_compression),
                auto_crop,
                int(args.auto_thr),
                int(args.auto_pad),
                float(args.auto_min_keep_ratio),
                auto_mode,
                float(args.endo_score_thr),
                float(args.endo_black_thr),
                int(args.endo_smooth_k),
                bool(args.square),
                pad4,
                min_out_hw,
            )
            for p in img_paths
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="process_images"):
            in_path, ok, err = fut.result()
            if ok:
                ok_cnt += 1
            else:
                fail_cnt += 1
                errors.append((in_path, err or "unknown"))

    print(f"[done] ok={ok_cnt} fail={fail_cnt} out_root={out_root}")
    if errors:
        print("[errors] show first 20:")
        for p, e in errors[:20]:
            print(f" - {p} :: {e}")


if __name__ == "__main__":
    main()


