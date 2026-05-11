"""
ESD-VesNet real-time style centerline inference on a video file.

Paper:
ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for Endoscopic
Submucosal Dissection with Hard Negative Mining

Reads a video, runs the same vessel->centerline pipeline as
`infer_centerline_visualize.py`, and writes an overlay video.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import infer_centerline_visualize as icv


def _parse_line_color(s: str) -> tuple[int, int, int]:
    try:
        _b, _g, _r = [int(x) for x in str(s).split(",")]
        _b = max(0, min(255, _b))
        _g = max(0, min(255, _g))
        _r = max(0, min(255, _r))
        return (_b, _g, _r)
    except Exception:
        return (0, 80, 255)


def _warp_prob_with_flow(
    prev_prob: np.ndarray,
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    flow_scale: float,
) -> np.ndarray:
    cv2 = icv._try_import_cv2()
    if prev_prob is None or prev_gray is None or cur_gray is None:
        return prev_prob
    if prev_gray.shape != cur_gray.shape:
        prev_gray = cv2.resize(
            prev_gray, (cur_gray.shape[1], cur_gray.shape[0]), interpolation=cv2.INTER_AREA
        )

    h, w = cur_gray.shape[:2]
    flow_scale = float(flow_scale)
    if flow_scale <= 0.0 or flow_scale > 1.0:
        flow_scale = 1.0
    if flow_scale < 1.0:
        sw = max(8, int(w * flow_scale))
        sh = max(8, int(h * flow_scale))
        prev_small = cv2.resize(prev_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        cur_small = cv2.resize(cur_gray, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        prev_small = prev_gray
        cur_small = cur_gray

    flow = cv2.calcOpticalFlowFarneback(
        prev_small, cur_small, None, 0.5, 2, 15, 3, 5, 1.2, 0
    )
    if flow.shape[:2] != (h, w):
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[:, :, 0] *= float(w) / float(prev_small.shape[1])
        flow[:, :, 1] *= float(h) / float(prev_small.shape[0])

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(
        prev_prob, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return warped


def _snap_yx_to_mask(mask_u8: np.ndarray, yx: tuple[int, int]) -> tuple[int, int] | None:
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    y, x = int(yx[0]), int(yx[1])
    y = max(0, min(m.shape[0] - 1, y))
    x = max(0, min(m.shape[1] - 1, x))
    if m[y, x] > 0:
        return (y, x)
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return None
    dy = ys.astype(np.int32) - y
    dx = xs.astype(np.int32) - x
    j = int(np.argmin(dy * dy + dx * dx))
    return (int(ys[j]), int(xs[j]))


def _order_endpoints_consistent(
    a_yx: tuple[int, int],
    b_yx: tuple[int, int],
    prev_a_yx: tuple[int, int] | None,
    prev_b_yx: tuple[int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if prev_a_yx is None or prev_b_yx is None:
        return a_yx, b_yx
    a = np.array(a_yx, dtype=np.float32)
    b = np.array(b_yx, dtype=np.float32)
    pa = np.array(prev_a_yx, dtype=np.float32)
    pb = np.array(prev_b_yx, dtype=np.float32)
    d_keep = float(((a - pa) ** 2).sum() + ((b - pb) ** 2).sum())
    d_swap = float(((a - pb) ** 2).sum() + ((b - pa) ** 2).sum())
    return (a_yx, b_yx) if d_keep <= d_swap else (b_yx, a_yx)


def _apply_temporal_ema(
    prob_np: np.ndarray,
    cur_gray: np.ndarray,
    args: argparse.Namespace,
    state: dict | None,
) -> np.ndarray:
    if state is None:
        return prob_np
    ema = float(getattr(args, "temporal_ema", 0.0))
    if ema <= 0.0:
        return prob_np
    prev_prob = state.get("ema_prob")
    if prev_prob is None:
        state["ema_prob"] = prob_np.copy()
        state["prev_gray"] = cur_gray
        return prob_np

    if bool(getattr(args, "temporal_flow", False)):
        try:
            prev_gray = state.get("prev_gray")
            prev_prob = _warp_prob_with_flow(
                prev_prob, prev_gray, cur_gray, float(getattr(args, "flow_scale", 0.5))
            )
        except Exception:
            pass

    prob_sm = ema * prob_np + (1.0 - ema) * prev_prob
    state["ema_prob"] = prob_sm
    state["prev_gray"] = cur_gray
    return prob_sm


def _maybe_infer_or_propagate_prob(
    frame_bgr: np.ndarray,
    args: argparse.Namespace,
    model,
    device: torch.device,
    state: dict | None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], float]:
    """
    Returns:
      prob_np in padded inp-size space
      (orig_h, orig_w)
      pads
      scale
    """
    cv2 = icv._try_import_cv2()
    orig_h, orig_w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pad_img, pads, _scale = icv._resize_with_pad(img_rgb, int(args.inp_size))
    cur_gray = cv2.cvtColor(pad_img, cv2.COLOR_RGB2GRAY)

    infer_every = int(getattr(args, "infer_every", 1))
    if infer_every < 1:
        infer_every = 1
    idx = int(state.get("frame_idx", 0)) if state is not None else 0
    do_infer = (infer_every == 1) or ((idx % infer_every) == 0) or (state is None)

    prob_np: np.ndarray | None = None
    if not do_infer and state is not None:
        prev_prob = state.get("ema_prob")
        prev_gray = state.get("prev_gray")
        if prev_prob is not None and prev_gray is not None:
            try:
                prob_np = _warp_prob_with_flow(
                    prev_prob.astype(np.float32),
                    prev_gray.astype(np.uint8),
                    cur_gray.astype(np.uint8),
                    float(getattr(args, "flow_scale", 0.5)),
                ).astype(np.float32)
            except Exception:
                prob_np = None

    if prob_np is None:
        inp = torch.from_numpy(pad_img).float() / 255.0
        inp = inp.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        with torch.no_grad():
            if bool(args.fp16) and device.type == "cuda":
                # torch.cuda.amp.autocast is deprecated; use torch.amp.autocast
                with torch.amp.autocast("cuda"):
                    prob = icv._infer_prob(model, inp, use_gated=bool(args.use_gated))
            else:
                prob = icv._infer_prob(model, inp, use_gated=bool(args.use_gated))
        prob_np = prob.detach().float().cpu().numpy()[0, 0].astype(np.float32)

    if state is not None:
        state["frame_idx"] = idx + 1
        # apply EMA in padded space
        prob_np = _apply_temporal_ema(prob_np, cur_gray, args, state)
    return prob_np, (orig_h, orig_w), pads, float(_scale)


def _process_frame(
    frame_bgr: np.ndarray,
    args: argparse.Namespace,
    model,
    device: torch.device,
    line_color: tuple[int, int, int],
    state: dict | None = None,
) -> np.ndarray:
    cv2 = icv._try_import_cv2()
    prob_np, (orig_h, orig_w), pads, _scale = _maybe_infer_or_propagate_prob(
        frame_bgr, args, model, device, state
    )
    mask_u8 = (prob_np > float(args.thr)).astype(np.uint8) * 255

    if int(args.post_close_k) > 0:
        k = int(args.post_close_k)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    mask_u8 = icv._unpad_and_resize_mask(mask_u8, pads=pads, orig_hw=(orig_h, orig_w))

    if (
        int(args.min_area) > 0
        or float(args.min_aspect) > 0
        or float(args.max_circularity) <= 1.0
        or bool(args.keep_largest)
    ):
        mask_u8 = icv._filter_components(
            mask_u8,
            min_area=int(args.min_area),
            min_aspect=float(args.min_aspect),
            max_circularity=float(args.max_circularity),
            keep_largest=bool(args.keep_largest),
        )

    skel_u8 = icv._skeletonize_u8(mask_u8)
    if int(args.bridge_dist) > 0:
        skel_u8 = icv._bridge_close_endpoints(
            skel_u8, max_dist=int(args.bridge_dist), iters=int(args.bridge_iters)
        )
    if int(args.skel_close_k) > 0:
        skel_u8 = icv._skel_close_and_thin(skel_u8, k=int(args.skel_close_k))
    if int(args.skel_min_size) > 0:
        skel_u8 = icv._filter_skel_components(skel_u8, min_size=int(args.skel_min_size))

    overlay = frame_bgr.copy()
    if str(args.draw_mode) == "center":
        center_scale = float(getattr(args, "center_scale", 1.0))
        if 0.0 < center_scale < 1.0:
            ch = max(8, int(mask_u8.shape[0] * center_scale))
            cw = max(8, int(mask_u8.shape[1] * center_scale))
            mask_small = cv2.resize(mask_u8, (cw, ch), interpolation=cv2.INTER_AREA)
            # stable endpoints: prefer previous endpoints (snapped), else re-init
            prev_a = state.get("prev_a_yx_s") if state is not None else None
            prev_b = state.get("prev_b_yx_s") if state is not None else None
            a_yx = _snap_yx_to_mask(mask_small, prev_a) if isinstance(prev_a, tuple) else None
            b_yx = _snap_yx_to_mask(mask_small, prev_b) if isinstance(prev_b, tuple) else None
            if a_yx is None or b_yx is None:
                a_yx, b_yx = icv._approx_diameter_endpoints(mask_small)
            else:
                a_yx, b_yx = _order_endpoints_consistent(a_yx, b_yx, prev_a, prev_b)
            curve = icv._dijkstra_geodesic_centerline(mask_small, a_yx, b_yx)
            if curve is not None and len(curve) > 0:
                curve = curve / float(center_scale)
            if state is not None and a_yx is not None and b_yx is not None:
                state["prev_a_yx_s"] = a_yx
                state["prev_b_yx_s"] = b_yx
        else:
            prev_a = state.get("prev_a_yx") if state is not None else None
            prev_b = state.get("prev_b_yx") if state is not None else None
            a_yx = _snap_yx_to_mask(mask_u8, prev_a) if isinstance(prev_a, tuple) else None
            b_yx = _snap_yx_to_mask(mask_u8, prev_b) if isinstance(prev_b, tuple) else None
            if a_yx is None or b_yx is None:
                a_yx, b_yx = icv._approx_diameter_endpoints(mask_u8)
            else:
                a_yx, b_yx = _order_endpoints_consistent(a_yx, b_yx, prev_a, prev_b)
            curve = icv._dijkstra_geodesic_centerline(mask_u8, a_yx, b_yx)
            if state is not None and a_yx is not None and b_yx is not None:
                state["prev_a_yx"] = a_yx
                state["prev_b_yx"] = b_yx
        if curve is not None and len(curve) >= 2:
            curve = icv._chaikin_smooth(curve, iters=int(args.center_smooth_iters))
            curve = icv._resample_polyline(curve, step=float(args.center_resample_step))
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
            paths = icv._skeleton_paths(skel_u8)
            for xy in paths:
                if xy is None or len(xy) < 2:
                    continue
                xy2 = icv._smooth_polyline_xy(xy, win=int(args.smooth_win))
                xy2 = icv._simplify_polyline(xy2, eps_ratio=float(args.simplify_eps))
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
            sk = (skel_u8 > 0).astype(np.uint8) * 255
            k = int(max(1, int(args.line_thickness)))
            if k > 1:
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                sk = cv2.dilate(sk, kernel, iterations=1)
            overlay[sk > 0] = line_color

    alpha = float(args.alpha)
    vis = cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser()
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument("--video", type=str, required=True, help="输入视频文件路径")
    parser.add_argument("--out-video", type=str, default="", help="输出视频路径（默认：同目录 *_centerline.mp4）")
    parser.add_argument("--start-frame", type=int, default=0, help="从第 N 帧开始处理")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理 N 帧（0 表示全部）")
    parser.add_argument("--save-fps", type=float, default=0.0, help="输出视频 FPS（0 表示使用原视频 FPS）")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="输出视频编码（默认 mp4v）")

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
    parser.add_argument("--fp16", action="store_true", help="启用半精度推理（仅 CUDA）")
    parser.add_argument("--infer-every", type=int, default=1, help="每 N 帧跑一次模型（其余帧用光流传播；1 表示每帧推理）")

    parser.add_argument("--post-close-k", type=int, default=0, help="mask closing（奇数核 3/5/7；0 关闭）")
    parser.add_argument("--min-area", type=int, default=0, help="过滤面积过小连通域（0 关闭）")
    parser.add_argument("--min-aspect", type=float, default=0.0, help="过滤不够细长连通域（0 关闭）")
    parser.add_argument("--max-circularity", type=float, default=1.1, help="过滤过于圆的连通域（<=1.0 启用）")
    parser.add_argument("--keep-largest", action="store_true", help="仅保留最大连通域")

    parser.add_argument("--alpha", type=float, default=1.0, help="中心线叠加强度")
    parser.add_argument("--line-thickness", type=int, default=2, help="画线粗细（像素）")
    parser.add_argument("--line-color", type=str, default="0,80,255", help="中心线颜色 BGR（例如 0,0,255）")
    parser.add_argument(
        "--draw-mode",
        type=str,
        default="center",
        choices=["dilate", "poly", "center"],
        help="dilate / poly / center（推荐 center 更连续）",
    )
    parser.add_argument("--skel-min-size", type=int, default=2, help="过滤骨架连通域的最小像素数（<=0 关闭）")
    parser.add_argument("--bridge-dist", type=int, default=6, help="连接临近线段的最大端点距离（像素，<=0 关闭）")
    parser.add_argument("--bridge-iters", type=int, default=2, help="端点连接迭代次数")
    parser.add_argument("--skel-close-k", type=int, default=0, help="对骨架做 closing 再细化（0 关闭）")
    parser.add_argument("--smooth-win", type=int, default=9, help="poly 模式平滑窗口")
    parser.add_argument("--simplify-eps", type=float, default=0.01, help="poly 模式折线简化强度")
    parser.add_argument("--center-smooth-iters", type=int, default=2, help="center 模式 Chaikin 平滑迭代次数")
    parser.add_argument("--center-resample-step", type=float, default=2.0, help="center 模式重采样步长（像素）")
    parser.add_argument("--center-scale", type=float, default=1.0, help="center 模式下对 mask 下采样比例（<1 提速）")
    parser.add_argument("--temporal-ema", type=float, default=0.0, help="时间平滑 EMA 系数（0 关闭，0.3~0.7 较稳）")
    parser.add_argument("--temporal-flow", action="store_true", help="（仅 EMA 内部）用光流将上一帧概率对齐（更稳但更慢）")
    parser.add_argument("--flow-scale", type=float, default=0.35, help="光流计算下采样比例（0~1，越小越快）")
    parser.add_argument("--log-every", type=int, default=30, help="每 N 帧打印一次 FPS/进度")
    args = parser.parse_args()

    cv2 = icv._try_import_cv2()
    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")

    out_video = args.out_video.strip()
    if not out_video:
        base, _ext = os.path.splitext(video_path)
        out_video = f"{base}_centerline.mp4"
    out_video = os.path.abspath(out_video)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if in_fps <= 1e-3:
        in_fps = 30.0
    out_fps = float(args.save_fps) if float(args.save_fps) > 0 else in_fps

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*str(args.fourcc))
    writer = cv2.VideoWriter(out_video, fourcc, out_fps, (w, h))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = icv._build_model(args.model, int(args.inp_size), device=device)
    print(f"[centerline-video] loading ckpt: {args.ckpt}", flush=True)
    try:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    line_color = _parse_line_color(args.line_color)
    log_every = max(1, int(args.log_every))
    max_frames = int(args.max_frames)
    start_frame = int(max(0, int(args.start_frame)))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    t0 = time.time()
    last_log = t0

    print(
        f"[centerline-video] in_fps={in_fps:.2f} out_fps={out_fps:.2f} size={w}x{h} "
        f"start_frame={start_frame} max_frames={max_frames if max_frames > 0 else 'ALL'}",
        flush=True,
    )
    print(f"[centerline-video] out_video={out_video}", flush=True)
    if float(args.temporal_ema) > 0:
        print(
            f"[centerline-video] temporal_ema={float(args.temporal_ema):.2f} "
            f"temporal_flow={bool(args.temporal_flow)} flow_scale={float(args.flow_scale):.2f}",
            flush=True,
        )
    if 0.0 < float(args.center_scale) < 1.0:
        print(f"[centerline-video] center_scale={float(args.center_scale):.2f}", flush=True)

    state: dict = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = _process_frame(frame, args, model, device, line_color, state=state)
        writer.write(vis)

        processed += 1
        if max_frames > 0 and processed >= max_frames:
            break
        if processed == 1 or (processed % log_every) == 0:
            now = time.time()
            inst_fps = float(log_every) / max(1e-6, now - last_log) if processed > 1 else 0.0
            avg_fps = float(processed) / max(1e-6, now - t0)
            cur = start_frame + processed
            msg = f"[centerline-video] frame {cur}"
            if total > 0:
                msg += f"/{total}"
            msg += f" inst_fps={inst_fps:.2f} avg_fps={avg_fps:.2f}"
            print(msg, flush=True)
            last_log = now

    cap.release()
    writer.release()
    print(f"[centerline-video] done. saved: {out_video}", flush=True)


if __name__ == "__main__":
    main()

