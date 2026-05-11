"""
ESD-VesNet centerline visualization for ESD vessel predictions.

Paper:
ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for Endoscopic
Submucosal Dissection with Hard Negative Mining

This script is intentionally standalone and does NOT modify `tools/eval_val_metrics.py`.

Pipeline per image:
  image -> model -> prob -> (prob>thr) mask -> skeleton (morphological) -> overlay centerline -> save

Notes:
  - Centerline is obtained via morphological skeletonization (OpenCV erode/dilate loop).
  - Designed for quick qualitative inspection (not metric eval).
"""

from __future__ import annotations

import argparse
import os
import sys
import heapq
from typing import List, Tuple

import numpy as np
import torch


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "缺少依赖：无法 import cv2。请先在当前环境安装 OpenCV（pip install opencv-python）。"
        ) from e


def _list_pngs(root: str, recursive: bool = True) -> List[str]:
    out: List[str] = []
    exts = (".png", ".jpg", ".jpeg")
    if recursive:
        for dp, _dn, fnames in os.walk(root):
            for fn in fnames:
                if fn.lower().endswith(exts):
                    out.append(os.path.join(dp, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def _resize_with_pad(img: np.ndarray, out_size: int) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
    """
    Resize to square out_size with letterbox padding, similar to common segmentation preprocessing.
    Returns: (img_resized, (top, bottom, left, right), scale)
    """
    cv2 = _try_import_cv2()
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty image")
    scale = float(out_size) / float(max(h, w))
    nh = int(round(h * scale))
    nw = int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (out_size - nh) // 2
    bottom = out_size - nh - top
    left = (out_size - nw) // 2
    right = out_size - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return padded, (top, bottom, left, right), scale


def _unpad_and_resize_mask(mask: np.ndarray, pads: Tuple[int, int, int, int], orig_hw: Tuple[int, int]) -> np.ndarray:
    cv2 = _try_import_cv2()
    top, bottom, left, right = pads
    h, w = mask.shape[:2]
    m = mask[top : h - bottom, left : w - right]
    oh, ow = orig_hw
    m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return m


def _skeletonize_u8(bin_u8: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization.
    Input: uint8 mask with values {0,255}.
    Output: uint8 skeleton {0,255}.
    """
    cv2 = _try_import_cv2()
    img = (bin_u8 > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _neighbors8(y: int, x: int, h: int, w: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny = y + dy
            nx = x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def _chaikin_smooth(xy: np.ndarray, iters: int = 2) -> np.ndarray:
    """Chaikin corner-cutting to get a smooth polyline without scipy."""
    if xy is None or len(xy) < 3:
        return xy
    pts = xy.astype(np.float32)
    iters = int(max(0, iters))
    for _ in range(iters):
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p = pts[i]
            q = pts[i + 1]
            out.append(0.75 * p + 0.25 * q)
            out.append(0.25 * p + 0.75 * q)
        out.append(pts[-1])
        pts = np.stack(out, axis=0)
    return pts


def _resample_polyline(xy: np.ndarray, step: float = 2.0) -> np.ndarray:
    """Resample polyline by arc-length with roughly constant spacing."""
    if xy is None or len(xy) < 2:
        return xy
    step = float(step)
    if step <= 0:
        return xy
    pts = xy.astype(np.float32)
    seg = pts[1:] - pts[:-1]
    seglen = np.sqrt((seg * seg).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1])
    if total <= 1e-6:
        return xy
    n = int(max(2, np.ceil(total / step) + 1))
    t = np.linspace(0.0, total, n, dtype=np.float32)
    xs = np.interp(t, s, pts[:, 0])
    ys = np.interp(t, s, pts[:, 1])
    return np.stack([xs, ys], axis=1)


def _approx_diameter_endpoints(mask_u8: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Pick two far-apart boundary points (approx) using a 2-pass farthest heuristic."""
    cv2 = _try_import_cv2()
    m = (mask_u8 > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            return (0, 0), (0, 0)
        p = (int(ys[0]), int(xs[0]))
        return p, p
    c = max(contours, key=lambda x: x.shape[0])
    pts = c.reshape(-1, 2)  # (x,y)
    if len(pts) < 2:
        yx = (int(pts[0, 1]), int(pts[0, 0]))
        return yx, yx

    def _farthest_from(pxy: np.ndarray) -> np.ndarray:
        d = pts.astype(np.float32) - pxy.astype(np.float32)
        d2 = (d * d).sum(axis=1)
        return pts[int(np.argmax(d2))]

    p0 = pts[0]
    p1 = _farthest_from(p0)
    p2 = _farthest_from(p1)
    a = (int(p1[1]), int(p1[0]))  # (y,x)
    b = (int(p2[1]), int(p2[0]))
    return a, b


def _dijkstra_geodesic_centerline(mask_u8: np.ndarray, start_yx: Tuple[int, int], end_yx: Tuple[int, int]) -> np.ndarray:
    """
    Continuous centerline path inside mask using Dijkstra on a cost map derived from distance transform.
    Cost encourages going through the center (large distance-to-boundary).
    Returns polyline as (x,y) float array.
    """
    cv2 = _try_import_cv2()
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    dt = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    eps = 1e-3

    ys, xs = np.where(m > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = 6
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(m.shape[0] - 1, y1 + pad)
    x1 = min(m.shape[1] - 1, x1 + pad)

    m_roi = m[y0 : y1 + 1, x0 : x1 + 1]
    dt_roi = dt[y0 : y1 + 1, x0 : x1 + 1]
    H, W = m_roi.shape

    sy, sx = int(start_yx[0] - y0), int(start_yx[1] - x0)
    ty, tx = int(end_yx[0] - y0), int(end_yx[1] - x0)
    sy = min(max(sy, 0), H - 1)
    sx = min(max(sx, 0), W - 1)
    ty = min(max(ty, 0), H - 1)
    tx = min(max(tx, 0), W - 1)

    if m_roi[sy, sx] == 0 or m_roi[ty, tx] == 0:
        ys2, xs2 = np.where(m_roi > 0)
        if len(ys2) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        p = np.stack([ys2, xs2], axis=1).astype(np.int32)

        def _snap(yx):
            dy = p[:, 0] - yx[0]
            dx = p[:, 1] - yx[1]
            j = int(np.argmin(dy * dy + dx * dx))
            return int(p[j, 0]), int(p[j, 1])

        sy, sx = _snap((sy, sx))
        ty, tx = _snap((ty, tx))

    INF = 1e18
    dist = np.full((H, W), INF, dtype=np.float64)
    prev = np.full((H, W, 2), -1, dtype=np.int32)
    dist[sy, sx] = 0.0
    pq = [(0.0, sy, sx)]

    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, float(np.sqrt(2.0))),
        (-1, 1, float(np.sqrt(2.0))),
        (1, -1, float(np.sqrt(2.0))),
        (1, 1, float(np.sqrt(2.0))),
    ]

    while pq:
        dcur, y, x = heapq.heappop(pq)
        if dcur != dist[y, x]:
            continue
        if y == ty and x == tx:
            break
        for dy, dx, step_len in moves:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if m_roi[ny, nx] == 0:
                continue
            c = float(step_len) * float(1.0 / (dt_roi[ny, nx] + eps))
            nd = dcur + c
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny, nx, 0] = y
                prev[ny, nx, 1] = x
                heapq.heappush(pq, (nd, ny, nx))

    if prev[ty, tx, 0] < 0 and not (ty == sy and tx == sx):
        return np.zeros((0, 2), dtype=np.float32)

    path = []
    cy, cx = ty, tx
    path.append((cx + x0, cy + y0))
    while not (cy == sy and cx == sx):
        py, px = int(prev[cy, cx, 0]), int(prev[cy, cx, 1])
        if py < 0:
            break
        cy, cx = py, px
        path.append((cx + x0, cy + y0))
    path.reverse()
    return np.array(path, dtype=np.float32)


def _skeleton_endpoints(skel_u8: np.ndarray) -> List[Tuple[int, int]]:
    sk = (skel_u8 > 0).astype(np.uint8)
    if sk.sum() == 0:
        return []
    h, w = sk.shape
    ys, xs = np.where(sk > 0)
    pts = set(zip(ys.tolist(), xs.tolist()))
    endpoints: List[Tuple[int, int]] = []
    for (y, x) in pts:
        d = 0
        for ny, nx in _neighbors8(y, x, h, w):
            if (ny, nx) in pts:
                d += 1
        if d == 1:
            endpoints.append((y, x))
    return endpoints


def _filter_skel_components(skel_u8: np.ndarray, min_size: int) -> np.ndarray:
    cv2 = _try_import_cv2()
    m = (skel_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return skel_u8
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= int(min_size):
            keep.append(lab)
    if not keep:
        return np.zeros_like(skel_u8)
    out = np.zeros_like(m)
    for lab in keep:
        out[labels == lab] = 1
    return (out.astype(np.uint8) * 255)


def _bridge_close_endpoints(skel_u8: np.ndarray, max_dist: int, iters: int = 1) -> np.ndarray:
    cv2 = _try_import_cv2()
    if int(max_dist) <= 0:
        return skel_u8
    max_dist = int(max_dist)
    sk = (skel_u8 > 0).astype(np.uint8) * 255
    iters = int(max(1, iters))
    for _ in range(iters):
        endpoints = _skeleton_endpoints(sk)
        if len(endpoints) < 2:
            break
        used = set()
        for i, (y1, x1) in enumerate(endpoints):
            if i in used:
                continue
            best_j = -1
            best_d2 = None
            for j in range(i + 1, len(endpoints)):
                if j in used:
                    continue
                y2, x2 = endpoints[j]
                dy = y2 - y1
                dx = x2 - x1
                d2 = dy * dy + dx * dx
                if d2 <= max_dist * max_dist and (best_d2 is None or d2 < best_d2):
                    best_d2 = d2
                    best_j = j
            if best_j >= 0:
                y2, x2 = endpoints[best_j]
                cv2.line(sk, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_8)
                used.add(i)
                used.add(best_j)
    return sk


def _skel_close_and_thin(skel_u8: np.ndarray, k: int) -> np.ndarray:
    """
    Connect small gaps by doing a morphological closing on the skeleton, then re-skeletonize
    to keep it 1-pixel wide.
    """
    cv2 = _try_import_cv2()
    k = int(k)
    if k <= 0:
        return skel_u8
    if k % 2 == 0:
        k += 1
    sk = (skel_u8 > 0).astype(np.uint8) * 255
    if int(sk.sum()) == 0:
        return skel_u8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sk = cv2.morphologyEx(sk, cv2.MORPH_CLOSE, kernel)
    return _skeletonize_u8(sk)


def _skeleton_paths(skel_u8: np.ndarray) -> List[np.ndarray]:
    """
    Convert a skeleton mask (uint8 {0,255}) into a list of ordered paths.
    Each path is an array of shape [N,2] in (x,y) order.
    """
    sk = (skel_u8 > 0).astype(np.uint8)
    if sk.sum() == 0:
        return []

    h, w = sk.shape
    ys, xs = np.where(sk > 0)
    pts = set(zip(ys.tolist(), xs.tolist()))

    # degree map (only for skeleton pixels)
    deg = {}
    for (y, x) in pts:
        d = 0
        for ny, nx in _neighbors8(y, x, h, w):
            if (ny, nx) in pts:
                d += 1
        deg[(y, x)] = d

    visited = set()
    paths: List[np.ndarray] = []

    def _trace(start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        cur = start
        prev = None
        steps = 0
        max_steps = len(pts) + 5
        while True:
            visited.add(cur)
            path.append(cur)
            # choose next among neighbors not equal to prev
            nbs = [p for p in _neighbors8(cur[0], cur[1], h, w) if p in pts and p != prev]
            # prefer unvisited neighbor
            nbs2 = [p for p in nbs if p not in visited]
            cand = nbs2[0] if nbs2 else None
            if cand is None:
                break
            prev, cur = cur, cand
            # stop if we reached an endpoint (degree 1) and next would go backwards/visited
            if deg.get(cur, 0) <= 1 and cur in visited:
                break
            if deg.get(cur, 0) <= 1:
                # include endpoint and stop
                if cur not in visited:
                    visited.add(cur)
                    path.append(cur)
                break
            steps += 1
            if steps > max_steps:
                # safety guard: avoid infinite loops on cyclic skeletons
                break
        return path

    # 1) trace from endpoints
    endpoints = [p for p, d in deg.items() if d == 1]
    for ep in endpoints:
        if ep in visited:
            continue
        path = _trace(ep)
        if len(path) >= 2:
            arr = np.array([(x, y) for (y, x) in path], dtype=np.int32)
            paths.append(arr)

    # 2) handle loops / remaining pixels
    for p in list(pts):
        if p in visited:
            continue
        path = _trace(p)
        if len(path) >= 2:
            arr = np.array([(x, y) for (y, x) in path], dtype=np.int32)
            paths.append(arr)

    return paths


def _smooth_polyline_xy(xy: np.ndarray, win: int = 7) -> np.ndarray:
    """
    Simple moving-average smoothing on polyline coordinates.
    xy: [N,2] int/float
    """
    if xy is None or len(xy) < 3:
        return xy
    win = int(win)
    if win <= 1:
        return xy
    if win % 2 == 0:
        win += 1
    pad = win // 2
    x = xy[:, 0].astype(np.float32)
    y = xy[:, 1].astype(np.float32)
    # edge padding
    xpad = np.pad(x, (pad, pad), mode="edge")
    ypad = np.pad(y, (pad, pad), mode="edge")
    k = np.ones((win,), dtype=np.float32) / float(win)
    xs = np.convolve(xpad, k, mode="valid")
    ys = np.convolve(ypad, k, mode="valid")
    out = np.stack([xs, ys], axis=1)
    return out


def _simplify_polyline(xy: np.ndarray, eps_ratio: float = 0.01) -> np.ndarray:
    """
    Douglas-Peucker simplification using OpenCV approxPolyDP.
    eps_ratio is relative to arc length.
    """
    cv2 = _try_import_cv2()
    if xy is None or len(xy) < 4:
        return xy
    pts = xy.reshape(-1, 1, 2).astype(np.float32)
    arclen = float(cv2.arcLength(pts, False))
    eps = float(max(0.0, eps_ratio)) * max(1.0, arclen)
    if eps <= 0:
        return xy
    approx = cv2.approxPolyDP(pts, epsilon=eps, closed=False)
    return approx.reshape(-1, 2)


def _filter_components(
    mask_u8: np.ndarray,
    min_area: int = 0,
    min_aspect: float = 0.0,
    max_circularity: float = 1.1,
    keep_largest: bool = False,
) -> np.ndarray:
    """
    Filter connected components on a binary mask.

    - min_area: remove components with area < min_area.
    - min_aspect: keep only elongated components (max(w/h, h/w) >= min_aspect).
    - max_circularity: remove near-circular components. circularity = 4*pi*area/perimeter^2.
      Set to <=1.0 to enable filtering. (Circle ~ 1.0, elongated shapes << 1.0)
    - keep_largest: keep only the largest remaining component.
    """
    cv2 = _try_import_cv2()
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return mask_u8

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = []
    for lab in range(1, num):
        x, y, w, h, area = stats[lab].tolist()
        if int(min_area) > 0 and int(area) < int(min_area):
            continue
        ar = float(max(w / max(1, h), h / max(1, w)))
        if float(min_aspect) > 0 and ar < float(min_aspect):
            continue

        if float(max_circularity) <= 1.0:
            comp = (labels == lab).astype(np.uint8)
            # perimeter on contour; avoid div0
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            peri = float(sum(cv2.arcLength(c, True) for c in contours))
            if peri > 1e-6:
                circ = float(4.0 * np.pi * float(area) / (peri * peri))
                if circ > float(max_circularity):
                    continue

        keep.append((lab, int(area)))

    if not keep:
        return np.zeros_like(mask_u8)

    if bool(keep_largest):
        keep.sort(key=lambda t: t[1], reverse=True)
        keep = keep[:1]

    out = np.zeros_like(m)
    for lab, _a in keep:
        out[labels == lab] = 1
    return (out.astype(np.uint8) * 255)


@torch.no_grad()
def _infer_prob(model, inp: torch.Tensor, use_gated: bool) -> torch.Tensor:
    """
    Returns probability map in [0,1], shape [B,1,H,W].
    Mirrors `tools/eval_val_metrics.py` behavior for compatibility.
    """
    if hasattr(model, "infer_prob_uncert"):
        prob_v, _u, prob_g = model.infer_prob_uncert(inp)
        return prob_g if use_gated else prob_v
    if hasattr(model, "infer"):
        logits = model.infer(inp)
    else:
        logits = model(inp)
    return torch.sigmoid(logits)


def _build_model(model_name: str, inp_size: int, device: torch.device):
    # Ensure project root on path when running from tools/
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    # Ensure we import the intended upstream `sam3` package (sam3-main) rather than any local folders.
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
    model = models.make(model_cfg).to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(default_project_dir, "data"),
        help="待推理的目录（包含 png/jpg/jpeg）",
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
        help="模型 checkpoint 路径",
    )
    parser.add_argument("--model", type=str, default="sam3-sam-edl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inp-size", type=int, default=1024, help="推理输入尺寸（方形 letterbox）")
    parser.add_argument("--thr", type=float, default=0.43, help="二值化阈值")
    parser.add_argument("--use-gated", action="store_true", help="使用 prob_gated（如果模型支持）")
    parser.add_argument("--post-close-k", type=int, default=0, help="可选：对 mask 做一次 closing（奇数核 3/5/7；0 关闭）")
    parser.add_argument("--min-area", type=int, default=0, help="可选：过滤掉面积小于该阈值的连通域（0 关闭）")
    parser.add_argument(
        "--min-aspect",
        type=float,
        default=0.0,
        help="可选：过滤掉不够细长的连通域（max(w/h,h/w) < min_aspect 会被去掉；0 关闭）",
    )
    parser.add_argument(
        "--max-circularity",
        type=float,
        default=1.1,
        help="可选：过滤掉过于圆的连通域（circularity > max_circularity 去掉；<=1.0 才启用过滤）",
    )
    parser.add_argument("--keep-largest", action="store_true", help="可选：仅保留过滤后面积最大的一个连通域")
    parser.add_argument("--out-dir", type=str, default="", help="输出目录（默认：data-dir/centerline_vis）")
    parser.add_argument("--limit", type=int, default=12, help="只导出前 N 张（0 表示全部）")
    parser.add_argument("--recursive", action="store_true", help="递归搜索子目录图片")
    parser.add_argument("--alpha", type=float, default=1.0, help="中心线叠加强度")
    parser.add_argument("--line-thickness", type=int, default=2, help="画线粗细（像素）")
    parser.add_argument(
        "--line-color",
        type=str,
        default="0,80,255",
        help="中心线颜色，BGR 格式，逗号分隔（例如 0,0,255 为纯红）",
    )
    parser.add_argument(
        "--draw-mode",
        type=str,
        default="poly",
        choices=["dilate", "poly", "center"],
        help="中心线绘制方式：dilate=膨胀骨架后上色；poly=追踪骨架为折线并平滑后用 polylines 绘制（更平滑）；center=distance transform+geodesic 求连续中心曲线（更少断点）",
    )
    parser.add_argument("--skel-min-size", type=int, default=2, help="过滤骨架连通域的最小像素数（<=0 关闭）")
    parser.add_argument("--bridge-dist", type=int, default=6, help="连接临近线段的最大端点距离（像素，<=0 关闭）")
    parser.add_argument("--bridge-iters", type=int, default=2, help="端点连接迭代次数（越大越容易连成整条线）")
    parser.add_argument("--skel-close-k", type=int, default=0, help="对骨架做 closing 再细化（连接小断点；0 关闭；建议 3/5/7）")
    parser.add_argument("--smooth-win", type=int, default=9, help="poly 模式：平滑窗口大小（越大越平滑，建议 7~15）")
    parser.add_argument("--simplify-eps", type=float, default=0.01, help="poly 模式：折线简化强度（相对弧长比例，0 关闭）")
    parser.add_argument("--center-smooth-iters", type=int, default=2, help="center 模式：Chaikin 平滑迭代次数")
    parser.add_argument("--center-resample-step", type=float, default=2.0, help="center 模式：重采样步长（像素）")
    parser.add_argument("--log-every", type=int, default=5, help="每处理 N 张打印一次进度（避免看起来卡住）")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data-dir not found: {data_dir}")

    out_dir = args.out_dir.strip()
    if not out_dir:
        out_dir = os.path.join(data_dir, "centerline_vis")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cv2 = _try_import_cv2()
    pngs = _list_pngs(data_dir, recursive=bool(args.recursive))
    if not pngs:
        raise FileNotFoundError(f"未找到图片：{data_dir}")
    if int(args.limit) > 0:
        pngs = pngs[: int(args.limit)]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _build_model(args.model, int(args.inp_size), device=device)
    print(f"[centerline-vis] loading ckpt: {args.ckpt}", flush=True)
    # Prefer weights_only=True to avoid pickle warnings; fallback for older torch.
    try:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    print(f"[centerline-vis] data_dir={data_dir}", flush=True)
    print(f"[centerline-vis] out_dir={out_dir}", flush=True)
    print(
        f"[centerline-vis] n_images={len(pngs)} inp_size={int(args.inp_size)} thr={float(args.thr):.3f} "
        f"use_gated={bool(args.use_gated)} device={device} draw_mode={str(args.draw_mode)}",
        flush=True,
    )
    if int(args.post_close_k) > 0 or int(args.min_area) > 0 or float(args.min_aspect) > 0 or float(args.max_circularity) <= 1.0 or bool(args.keep_largest):
        print(
            f"[centerline-vis] post: close_k={int(args.post_close_k)} min_area={int(args.min_area)} "
            f"min_aspect={float(args.min_aspect):.3f} max_circularity={float(args.max_circularity):.3f} "
            f"keep_largest={bool(args.keep_largest)}"
        )

    log_every = max(1, int(args.log_every))
    # parse line color
    try:
        _b, _g, _r = [int(x) for x in str(args.line_color).split(",")]
        _b = max(0, min(255, _b))
        _g = max(0, min(255, _g))
        _r = max(0, min(255, _r))
        line_color = (_b, _g, _r)
    except Exception:
        line_color = (0, 80, 255)

    for idx, p in enumerate(pngs, start=1):
        if idx == 1 or (idx % log_every) == 0:
            print(f"[centerline-vis] processing {idx}/{len(pngs)}: {os.path.basename(p)}", flush=True)
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[warn] skip unreadable: {p}")
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        # preprocess: BGR -> RGB, resize+pad to square
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pad_img, pads, _scale = _resize_with_pad(img_rgb, int(args.inp_size))
        inp = torch.from_numpy(pad_img).float() / 255.0  # HWC
        inp = inp.permute(2, 0, 1).unsqueeze(0).contiguous()  # 1x3xHxW
        inp = inp.to(device)

        prob = _infer_prob(model, inp, use_gated=bool(args.use_gated))
        prob_np = prob.detach().float().cpu().numpy()[0, 0]  # HxW in [0,1]

        # binarize on padded space
        mask_u8 = (prob_np > float(args.thr)).astype(np.uint8) * 255
        # optional closing on padded mask (reduces small holes)
        if int(args.post_close_k) > 0:
            k = int(args.post_close_k)
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        # unpad/resize back to original resolution
        mask_u8 = _unpad_and_resize_mask(mask_u8, pads=pads, orig_hw=(orig_h, orig_w))
        # optional component filtering on original resolution
        if int(args.min_area) > 0 or float(args.min_aspect) > 0 or float(args.max_circularity) <= 1.0 or bool(args.keep_largest):
            mask_u8 = _filter_components(
                mask_u8,
                min_area=int(args.min_area),
                min_aspect=float(args.min_aspect),
                max_circularity=float(args.max_circularity),
                keep_largest=bool(args.keep_largest),
            )
        skel_u8 = _skeletonize_u8(mask_u8)
        if int(args.bridge_dist) > 0:
            skel_u8 = _bridge_close_endpoints(
                skel_u8, max_dist=int(args.bridge_dist), iters=int(args.bridge_iters)
            )
        if int(args.skel_close_k) > 0:
            skel_u8 = _skel_close_and_thin(skel_u8, k=int(args.skel_close_k))
        if int(args.skel_min_size) > 0:
            skel_u8 = _filter_skel_components(skel_u8, min_size=int(args.skel_min_size))

        # draw centerline on original
        overlay = img_bgr.copy()
        if str(args.draw_mode) == "center":
            # continuous center curve from mask (less broken than skeleton)
            a_yx, b_yx = _approx_diameter_endpoints(mask_u8)
            curve = _dijkstra_geodesic_centerline(mask_u8, a_yx, b_yx)
            if curve is not None and len(curve) >= 2:
                curve = _chaikin_smooth(curve, iters=int(args.center_smooth_iters))
                curve = _resample_polyline(curve, step=float(args.center_resample_step))
                pts = np.round(curve).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    overlay,
                    [pts],
                    isClosed=False,
                    color=line_color,
                    thickness=int(max(1, int(args.line_thickness))),
                    lineType=cv2.LINE_AA,
                )
        elif int(skel_u8.sum()) > 0:
            if str(args.draw_mode) == "poly":
                paths = _skeleton_paths(skel_u8)
                for xy in paths:
                    if xy is None or len(xy) < 2:
                        continue
                    # smooth + simplify
                    xy2 = _smooth_polyline_xy(xy, win=int(args.smooth_win))
                    xy2 = _simplify_polyline(xy2, eps_ratio=float(args.simplify_eps))
                    if xy2 is None or len(xy2) < 2:
                        continue
                    pts = np.round(xy2).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        overlay,
                        [pts],
                        isClosed=False,
                        color=line_color,
                        thickness=int(max(1, int(args.line_thickness))),
                        lineType=cv2.LINE_AA,
                    )
            else:
                # dilate mask overlay (fast, but less smooth)
                sk = (skel_u8 > 0).astype(np.uint8) * 255
                k = int(max(1, int(args.line_thickness)))
                if k > 1:
                    if k % 2 == 0:
                        k += 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    sk = cv2.dilate(sk, kernel, iterations=1)
                overlay[sk > 0] = line_color  # BGR

        alpha = float(args.alpha)
        vis = cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0.0)

        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, f"{base}_centerline.png")
        cv2.imwrite(out_path, vis)

    print("[centerline-vis] done.", flush=True)


if __name__ == "__main__":
    main()


