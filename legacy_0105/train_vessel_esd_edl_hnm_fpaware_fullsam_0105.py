#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESD-VesNet training/evaluation entrypoint.

Paper:
ESD-VesNet: Uncertainty-Aware Vessel Segmentation Network for Endoscopic
Submucosal Dissection with Hard Negative Mining

This file was reconstructed because the original source script was deleted and only .pyc remained.
Goal: provide a runnable baseline that matches the old CLI used by
`scripts/train_edl_hnm_fullsam_0105.sh`, including:
  - DDP via torchrun
  - --eval-only/--ckpt
  - --batch-size/--grad-accum/--no-amp/--num-workers
  - simple FP-aware weighting for negative (empty GT) images
  - simple hard-negative mining (HNM): maintain a hard pool from negative samples

Note: This implementation focuses on robustness and reproducibility, not bitwise equivalence to the
original (which is not recoverable from .pyc alone).
"""

from __future__ import annotations

import argparse
import math
import atexit
import datetime as _dt
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

# Ensure we import the intended upstream `sam3` package from the bundled
# `sam3-main` dependency directory.
_SAM3_MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sam3-main"))
if _SAM3_MAIN_DIR not in sys.path:
    sys.path.insert(0, _SAM3_MAIN_DIR)

# Make sure project root is on sys.path so `import datasets/models` works
_PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)

import datasets
import models
from datasets.vessel_dataset import VesselESDDataset
from datasets.wrappers import TrainDataset, ValDataset
from sod_metric import Emeasure, MAE, Smeasure


class Config:
    """
    按 2026-01-08 日志格式：使用 class-level config（打印 Config.__dict__ 会包含 __module__/__dict__ 等字段）
    """

    # data
    DATA_ROOTS = [
        "/mnt/data-hdd/msc2025/chenming/esd/vessel_seg/eomt-vessel/datasets/vessel_data/qilu/esd",
        "/mnt/data-hdd/msc2025/chenming/esd/vessel_seg/eomt-vessel/datasets/vessel_data/qilu/elec",
    ]
    VAL_PATIENTS = ["P10", "P16", "P17", "P25_李翠兰", "P35", "P38"]
    NEG_DATA_ROOTS_TRAIN = [
        "/mnt/data-hdd/msc2025/chenming/esd/vessel_seg/eomt-vessel/datasets/vessel_data/qilu/elec"
    ]
    NEG_DATA_ROOTS_VAL = None

    INCLUDE_UNLABELED_NEG_TRAIN = True
    TRAIN_NEG_RATIO_IN_DATASET = None
    TRAIN_MAX_NEGATIVES = None
    NEG_SEED = 0

    INCLUDE_UNLABELED_NEG_VAL = False
    VAL_NEG_RATIO = 0.21
    VAL_MAX_NEGATIVES = 500

    # model + training
    SAM3_CHECKPOINT = "./checkpoints/sam3.pt"
    INP_SIZE = 1024
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    EPOCH_MAX = 100
    EPOCH_VAL = 1
    LR = 2e-4
    LR_MIN = 1e-7
    WEIGHT_DECAY = 0.0

    # fp-aware
    NEG_LOSS_WEIGHT = 1.0

    # batch sampler
    NEG_RATIO_IN_BATCH = 0.125

    # HNM
    HARD_PROB = 0.5
    HNM_UPDATE_EVERY = 1
    HNM_CANDIDATE_SCAN = 600
    HNM_TOPK = 500
    HNM_SCORE_THR = 0.5
    HNM_USE_UNCERT = True
    HNM_INIT_BEFORE_TRAIN = True
    # HNM init warm-up (only used when epoch==1 and HNM_INIT_BEFORE_TRAIN=True)
    # Motivation: if thr=0.5 and gated prob is conservative at init, scores can be all-zeros,
    # making the "hard pool" degenerate. Use a looser rule just for the init scan.
    HNM_INIT_THR = 0.3
    HNM_INIT_USE_UNCERT = False

    # EDL / gating
    EDL_BETA = 1.0
    EDL_LAMBDA_KL = 0.01
    EDL_ANNEAL_STEPS = 5000
    EDL_W_DICE = 5.0
    GATE_GAMMA = 1.0

    # finetune
    UNFREEZE_NECK = True
    # Small domain-adaptation tweak: unfreeze the last ViT block.
    # This keeps the SAM3 backbone mostly frozen while allowing limited adaptation.
    UNFREEZE_LAST_N_BLOCKS = 6

    # stability
    GRAD_CLIP_NORM = 1.0

    # misc
    VDR_THRESHOLD = 0.5
    VDR_MIN_AREA = 1
    VDR_REQUIRE_TP = True

    # runtime (filled at start)
    LOCAL_RANK = 0
    WORLD_SIZE = 1
    RANK = 0
    USE_AMP = True
    GRAD_ACCUM_STEPS = 1

    # logging/saving
    SAVE_PATH = "./save/vessel_esd_sam3_fullsam_edl_hnm_0105"
    VIS_SAVE_DIR = "./save/vessel_esd_sam3_fullsam_edl_hnm_0105/visualizations"


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_main() -> bool:
    return _get_rank() == 0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_ddp() -> Tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return rank, local_rank, device
    return 0, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _list_patients(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    pts = []
    for name in os.listdir(root):
        if name.startswith("P") and os.path.isdir(os.path.join(root, name)):
            pts.append(name)
    return sorted(pts)


def _build_patient_split(cfg: Config) -> Tuple[List[str], List[str]]:
    # union of patients across roots
    all_pts = set()
    for r in cfg.DATA_ROOTS:
        all_pts.update(_list_patients(r))
    val = sorted([p for p in cfg.VAL_PATIENTS if p in all_pts])
    train = sorted([p for p in all_pts if p not in set(val)])
    return train, val


def _dice_coeff(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # pred/gt: (N,1,H,W) in {0,1} or [0,1]
    pred = pred.float()
    gt = gt.float()
    inter = (pred * gt).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    return (2.0 * inter + eps) / (denom + eps)


def _soft_dice_loss_from_logits(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    return 1.0 - _dice_coeff(prob, gt).mean()


def _bce_from_logits(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, gt)


def _set_trainable_fullsam(model: torch.nn.Module, cfg: Config) -> None:
    """
    1/8 版本的默认策略（从 log 可见 UNFREEZE_NECK=True, UNFREEZE_LAST_N_BLOCKS=0）：
    - 默认冻结 backbone.trunk（ViT）的大部分 block
    - 解冻 detector neck（这里对应 vision_backbone.convs / position_encoding）
    - 始终训练 tracker 的 prompt_encoder / mask_decoder
    """
    for _n, p in model.named_parameters():
        p.requires_grad = False

    # tracker：始终可训练
    for _n, p in model.tracker.named_parameters():
        p.requires_grad = True

    # 解冻 neck（convs + position_encoding）
    if bool(cfg.UNFREEZE_NECK):
        for _n, p in model.image_encoder.vision_backbone.convs.named_parameters():
            p.requires_grad = True
        for _n, p in model.image_encoder.vision_backbone.position_encoding.named_parameters():
            p.requires_grad = True

    # 解冻 ViT 最后 N 个 block（N=0 就都不解冻）
    n_last = int(getattr(cfg, "UNFREEZE_LAST_N_BLOCKS", 0) or 0)
    if n_last > 0:
        blocks = model.image_encoder.vision_backbone.trunk.blocks
        for blk in list(blocks)[-n_last:]:
            for _n, p in blk.named_parameters():
                p.requires_grad = True


_LOG_FH = None


def _maybe_tee_stdout_to_log(cfg: Config) -> None:
    """Write rank0 stdout/stderr to both terminal and `${SAVE_PATH}/log.txt`.

    说明：很多 torchrun/NCCL/OOM/Traceback 会打到 stderr，不 tee 的话 log.txt 里会“突然断掉”而看不到真正原因。
    """
    global _LOG_FH
    if not _is_main():
        return
    try:
        os.makedirs(cfg.SAVE_PATH, exist_ok=True)
        log_path = os.path.join(cfg.SAVE_PATH, "log.txt")
        _LOG_FH = open(log_path, "a", buffering=1, encoding="utf-8")
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr

        class _Tee:
            def __init__(self, stream, fh):
                self._stream = stream
                self._fh = fh

            def write(self, s):
                self._stream.write(s)
                try:
                    if self._fh is not None and not self._fh.closed:
                        self._fh.write(s)
                except Exception:
                    pass

            def flush(self):
                self._stream.flush()
                try:
                    if self._fh is not None and not self._fh.closed:
                        self._fh.flush()
                except Exception:
                    pass

            def isatty(self):
                return bool(getattr(self._stream, "isatty", lambda: False)())

        sys.stdout = _Tee(sys.stdout, _LOG_FH)
        sys.stderr = _Tee(sys.stderr, _LOG_FH)

        def _close():
            try:
                # restore streams first to avoid shutdown-time flush hitting a closed file
                try:
                    sys.stdout = _orig_stdout
                except Exception:
                    pass
                try:
                    sys.stderr = _orig_stderr
                except Exception:
                    pass
                _LOG_FH.flush()
                _LOG_FH.close()
            except Exception:
                pass

        atexit.register(_close)
    except Exception:
        # Never crash training due to logging setup.
        return


def _cfg_serializable(cfg: Config) -> Dict[str, object]:
    """
    `cfg` 在本脚本里是 class-level Config，用于对齐 1/8 的打印（cfg.__dict__ 是 mappingproxy）。
    但 torch.save/pickle 不能序列化 mappingproxy / descriptor，所以这里导出一个可 pickle 的纯 dict。
    """
    out: Dict[str, object] = {}
    for k, v in dict(cfg.__dict__).items():
        if k.startswith("__"):
            continue
        # 过滤掉类 descriptor（例如 __dict__/__weakref__）
        if not isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
            continue
        out[k] = v
    return out



class BalancedHardNegBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        pos_indices: List[int],
        neg_indices: List[int],
        batch_size: int,
        neg_ratio_in_batch: float = 0.25,
        hard_prob: float = 0.5,
        hard_pool: Optional[List[int]] = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.pos = list(pos_indices)
        self.neg = list(neg_indices)
        self.bs = int(batch_size)
        self.neg_ratio = float(neg_ratio_in_batch)
        self.hard_prob = float(hard_prob)
        self.hard_pool = list(hard_pool or [])
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        if self.bs <= 0:
            raise ValueError("batch_size must be > 0")

        # per-batch counts
        n_neg = max(1, int(round(self.bs * self.neg_ratio))) if self.neg else 0
        n_neg = min(n_neg, self.bs)
        n_pos = self.bs - n_neg
        if n_pos <= 0:
            n_pos = max(1, self.bs - 1)
            n_neg = self.bs - n_pos
        self.n_pos = n_pos
        self.n_neg = n_neg

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def update_hard_pool(self, hard_pool: List[int]) -> None:
        self.hard_pool = list(hard_pool)

    def __len__(self) -> int:
        if not self.pos:
            return 0
        # rough epoch length: each batch consumes n_pos positives
        return max(1, len(self.pos) // max(1, self.n_pos) // max(1, self.world_size))

    def __iter__(self):
        if not self.pos:
            return iter(())

        rng = random.Random(self.seed + 1000 * self.epoch + 17 * self.rank)

        # shuffle indices
        pos = self.pos.copy()
        neg = self.neg.copy()
        rng.shuffle(pos)
        rng.shuffle(neg)

        # shard by rank (very simple sharding)
        pos = pos[self.rank :: self.world_size]
        neg = neg[self.rank :: self.world_size]

        neg_ptr = 0
        pos_ptr = 0
        for _ in range(len(self)):
            batch = []
            # positives
            for _k in range(self.n_pos):
                if pos_ptr >= len(pos):
                    pos_ptr = 0
                    rng.shuffle(pos)
                batch.append(pos[pos_ptr])
                pos_ptr += 1

            # negatives
            for _k in range(self.n_neg):
                use_hard = (len(self.hard_pool) > 0) and (rng.random() < self.hard_prob)
                if use_hard:
                    batch.append(rng.choice(self.hard_pool))
                else:
                    if not neg:
                        # no negatives available; duplicate a positive (won't crash)
                        batch.append(batch[-1])
                    else:
                        if neg_ptr >= len(neg):
                            neg_ptr = 0
                            rng.shuffle(neg)
                        batch.append(neg[neg_ptr])
                        neg_ptr += 1
            rng.shuffle(batch)
            yield batch


@torch.no_grad()
def _hnm_scan_negatives(
    model: torch.nn.Module,
    base_ds: VesselESDDataset,
    neg_indices: List[int],
    scan_k: int,
    device: torch.device,
    thr: float,
    use_uncert: bool,
    inp_size: int,
) -> List[Tuple[int, float]]:
    if scan_k <= 0 or len(neg_indices) == 0:
        return []
    scan_k = min(scan_k, len(neg_indices))
    # deterministic subset per scan
    rng = random.Random(0)
    cand = rng.sample(neg_indices, k=scan_k)

    # minimal val-style wrapper (no heavy aug)
    vwrap = ValDataset(base_ds, inp_size=int(inp_size), augment=False)

    # DDP parallelization: shard candidates across ranks, then all_gather results.
    # This avoids rank1/2 waiting while rank0 scans everything.
    if _is_dist():
        rank = dist.get_rank()
        world = dist.get_world_size()
        cand = cand[rank::world]

    scores: List[Tuple[int, float]] = []
    model.eval()
    for idx in cand:
        batch = vwrap[idx]
        inp = batch["inp"].unsqueeze(0).to(device, non_blocking=True)
        if hasattr(model, "module"):
            m = model.module
        else:
            m = model

        if hasattr(m, "infer_prob_uncert"):
            prob_vessel, _u, prob_gated = m.infer_prob_uncert(inp)
            prob_use = prob_gated if bool(use_uncert) else prob_vessel
        else:
            logits = m.infer(inp)
            prob_use = torch.sigmoid(logits)

        # 1/8 日志里的 score 量级 ~0.0x：使用 pos_frac（>thr 的像素占比）
        score = float((prob_use > thr).float().mean().item())
        scores.append((idx, score))

    if _is_dist():
        gathered: List[List[Tuple[int, float]]] = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
        dist.all_gather_object(gathered, scores)
        merged: List[Tuple[int, float]] = []
        for part in gathered:
            if part:
                merged.extend(part)
        return merged

    return scores


def _load_sam3_pretrained(model: torch.nn.Module, ckpt_path: str) -> int:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    ref = model.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}
    loaded = 0
    # debug counters (to verify we really loaded tracker.sam_prompt_encoder / tracker.sam_mask_decoder etc.)
    n_ckpt = 0
    n_ckpt_backbone = 0
    n_ckpt_tracker = 0
    n_ckpt_prompt = 0
    n_ckpt_maskdec = 0
    n_loaded_backbone = 0
    n_loaded_tracker = 0
    n_loaded_prompt = 0
    n_loaded_maskdec = 0
    n_skipped_shape = 0
    n_skipped_missing = 0
    skipped_tracker: List[Tuple[str, str]] = []  # (key, reason)
    for k, v in ckpt.items():
        n_ckpt += 1
        # 1/8 日志里 mapped params ≈ 550：不加载 RoPE 预计算表（freqs_cis）以及 ln_pre（不同实现下可选）
        if ".attn.freqs_cis" in k:
            continue
        if k.startswith("detector.backbone."):
            n_ckpt_backbone += 1
            new_k = k.replace("detector.backbone.", "image_encoder.")
        else:
            new_k = k
            if k.startswith("tracker."):
                n_ckpt_tracker += 1
                if k.startswith("tracker.sam_prompt_encoder."):
                    n_ckpt_prompt += 1
                if k.startswith("tracker.sam_mask_decoder."):
                    n_ckpt_maskdec += 1

        if new_k in ref and ref[new_k].shape == v.shape:
            new_sd[new_k] = v
            loaded += 1
            if k.startswith("detector.backbone."):
                n_loaded_backbone += 1
            elif k.startswith("tracker."):
                n_loaded_tracker += 1
                if k.startswith("tracker.sam_prompt_encoder."):
                    n_loaded_prompt += 1
                if k.startswith("tracker.sam_mask_decoder."):
                    n_loaded_maskdec += 1
        else:
            if new_k not in ref:
                n_skipped_missing += 1
                if k.startswith("tracker."):
                    skipped_tracker.append((k, "missing_in_model"))
            else:
                n_skipped_shape += 1
                if k.startswith("tracker."):
                    skipped_tracker.append((k, f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(ref[new_k].shape)}"))

    model.load_state_dict(new_sd, strict=False)
    if _is_main():
        print(
            f"[pretrain] ckpt_keys={n_ckpt} loaded={loaded} "
            f"(backbone: {n_loaded_backbone}/{n_ckpt_backbone}, tracker: {n_loaded_tracker}/{n_ckpt_tracker}) "
            f"skipped_missing={n_skipped_missing} skipped_shape={n_skipped_shape}"
        )
        if n_ckpt_tracker > 0:
            print(
                f"[pretrain] tracker detail: "
                f"sam_prompt_encoder {n_loaded_prompt}/{n_ckpt_prompt}, "
                f"sam_mask_decoder {n_loaded_maskdec}/{n_ckpt_maskdec}"
            )
        if n_ckpt_tracker > 0 and n_loaded_tracker == 0:
            print("[pretrain][warn] tracker.* weights were NOT loaded at all (likely key mismatch or architecture mismatch).")
        if (n_ckpt_prompt > 0 and n_loaded_prompt == 0) or (n_ckpt_maskdec > 0 and n_loaded_maskdec == 0):
            print(
                "[pretrain][warn] key tracker components were NOT loaded: "
                f"sam_prompt_encoder={n_loaded_prompt}/{n_ckpt_prompt}, sam_mask_decoder={n_loaded_maskdec}/{n_ckpt_maskdec}"
            )
        if skipped_tracker:
            print("[pretrain] tracker skipped examples:")
            for k, reason in skipped_tracker[:12]:
                print(f"  - {k}: {reason}")
    return loaded


def _build_model(cfg: Config) -> torch.nn.Module:
    model_spec = {
        "name": "sam3-sam-edl",
        "args": {
            "inp_size": int(cfg.INP_SIZE),
            "edl_beta": float(cfg.EDL_BETA),
            "edl_lambda_kl": float(cfg.EDL_LAMBDA_KL),
            "edl_anneal_steps": int(cfg.EDL_ANNEAL_STEPS),
            "edl_w_dice": float(cfg.EDL_W_DICE),
            "gate_gamma": float(cfg.GATE_GAMMA),
        },
    }
    return models.make(model_spec)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thr: float = 0.5,
    vdr_thr: float = 0.5,
) -> Dict[str, float]:
    def _metric_accum_init():
        return dict(
            dice_fg_sum=0.0,
            dice_bg_sum=0.0,
            n=0.0,
            tp=0.0,
            fp=0.0,
            fn=0.0,
            bg_all=0.0,
            fp_bg_all=0.0,
            bg_pos=0.0,
            fp_bg_pos=0.0,
            bg_empty=0.0,
            fp_bg_empty=0.0,
            n_pos_img=0.0,
            n_vdr_tp=0.0,
        )

    def _metric_accum_step(acc, prob_use: torch.Tensor, gt_bin: torch.Tensor):
        pred = (prob_use > thr).float()
        dice_fg = _dice_coeff(pred, gt_bin).detach()
        dice_bg = _dice_coeff(1.0 - pred, 1.0 - gt_bin).detach()
        acc["dice_fg_sum"] += float(dice_fg.sum().item())
        acc["dice_bg_sum"] += float(dice_bg.sum().item())
        acc["n"] += float(pred.shape[0])

        tp = float(((pred > 0.5) & (gt_bin > 0.5)).sum().item())
        fp = float(((pred > 0.5) & (gt_bin <= 0.5)).sum().item())
        fn = float(((pred <= 0.5) & (gt_bin > 0.5)).sum().item())
        acc["tp"] += tp
        acc["fp"] += fp
        acc["fn"] += fn

        # bg fpr (all / pos-gt-images / empty-gt-images)
        gt_bg = (gt_bin <= 0.5)
        pred_pos = (pred > 0.5)
        fp_bg = float((pred_pos & gt_bg).sum().item())
        bg = float(gt_bg.sum().item())
        acc["fp_bg_all"] += fp_bg
        acc["bg_all"] += bg

        gt_pos_img = (gt_bin.view(gt_bin.shape[0], -1).sum(dim=1) > 0)
        if gt_pos_img.any():
            gt_bg_pos = gt_bg[gt_pos_img]
            pred_pos_pos = pred_pos[gt_pos_img]
            acc["fp_bg_pos"] += float((pred_pos_pos & gt_bg_pos).sum().item())
            acc["bg_pos"] += float(gt_bg_pos.sum().item())
        gt_empty_img = ~gt_pos_img
        if gt_empty_img.any():
            gt_bg_empty = gt_bg[gt_empty_img]
            pred_pos_empty = pred_pos[gt_empty_img]
            acc["fp_bg_empty"] += float((pred_pos_empty & gt_bg_empty).sum().item())
            acc["bg_empty"] += float(gt_bg_empty.sum().item())

        # VDR
        if gt_pos_img.any():
            acc["n_pos_img"] += float(gt_pos_img.sum().item())
            pred_area = pred.view(pred.shape[0], -1).sum(dim=1)
            overlap = ((pred > 0.5) & (gt_bin > 0.5)).view(pred.shape[0], -1).sum(dim=1)
            # require_tp: overlap>=1；否则只要 pred_area>=min_area
            ok = (pred_area >= float(Config.VDR_MIN_AREA)) & (pred_area > 0)
            if bool(Config.VDR_REQUIRE_TP):
                ok = ok & (overlap >= 1)
            acc["n_vdr_tp"] += float((ok & gt_pos_img).sum().item())

    def _metric_finalize(acc):
        n = max(1.0, acc["n"])
        dice_fg = acc["dice_fg_sum"] / n
        dice_bg = acc["dice_bg_sum"] / n
        dice_mean = 0.5 * (dice_fg + dice_bg)
        prec = acc["tp"] / max(1.0, (acc["tp"] + acc["fp"]))
        bg_fpr_all = acc["fp_bg_all"] / max(1.0, acc["bg_all"])
        bg_fpr_pos = acc["fp_bg_pos"] / max(1.0, acc["bg_pos"])
        bg_fpr_empty = acc["fp_bg_empty"] / max(1.0, acc["bg_empty"])
        vdr = acc["n_vdr_tp"] / max(1.0, acc["n_pos_img"])
        return dict(
            dice_fg=dice_fg,
            dice_mean=dice_mean,
            prec_fg=prec,
            bg_fpr_all=bg_fpr_all,
            bg_fpr_pos=bg_fpr_pos,
            bg_fpr_empty=bg_fpr_empty,
            VDR=vdr,
        )

    def _cod_eval_step(sm: Smeasure, em: Emeasure, mae: MAE, prob_use: torch.Tensor, gt_bin: torch.Tensor):
        # 使用 SOD/COD 评估实现：输入 uint8 [0,255]
        prob_u8 = (prob_use.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        gt_u8 = (gt_bin.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        # Bx1xHxW -> loop
        for i in range(prob_u8.shape[0]):
            pred2d = prob_u8[i, 0]
            gt2d = gt_u8[i, 0]
            sm.step(pred2d, gt2d)
            em.step(pred2d, gt2d)
            mae.step(pred2d, gt2d)

    model.eval()

    # COD metrics（仅 rank0 统计即可，但为了 DDP 一致性，这里每 rank 都算，然后 all_reduce 平均）
    sm_raw, em_raw, mae_raw = Smeasure(), Emeasure(), MAE()
    sm_g, em_g, mae_g = Smeasure(), Emeasure(), MAE()

    acc_raw = _metric_accum_init()
    acc_g = _metric_accum_init()

    printed_once = False
    for batch in loader:
        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        gt_bin = (gt > 0.5).float()

        if hasattr(model, "module"):
            m = model.module
        else:
            m = model

        prob_vessel, _u, prob_gated = m.infer_prob_uncert(inp) if hasattr(m, "infer_prob_uncert") else (torch.sigmoid(m.infer(inp)), None, None)
        if prob_gated is None:
            prob_gated = prob_vessel

        # One-time debug stats to diagnose "dice_fg=0" cases (pred becomes empty under thr)
        if (not printed_once) and _is_main():
            try:
                pv = prob_vessel.detach()
                gt0 = gt_bin.detach()
                frac_pos = float((pv > thr).float().mean().item())
                pv_min = float(pv.min().item())
                pv_max = float(pv.max().item())
                pv_mean = float(pv.mean().item())
                gt_frac = float((gt0 > 0.5).float().mean().item())
                print(
                    f"[eval-debug] thr={thr:.2f} prob_vessel(min/mean/max)={pv_min:.4f}/{pv_mean:.4f}/{pv_max:.4f} "
                    f"pred_pos_frac={frac_pos:.6f} gt_pos_frac={gt_frac:.6f}"
                )
            except Exception:
                pass
            printed_once = True

        _metric_accum_step(acc_raw, prob_vessel, gt_bin)
        _metric_accum_step(acc_g, prob_gated, gt_bin)

        _cod_eval_step(sm_raw, em_raw, mae_raw, prob_vessel, gt_bin)
        _cod_eval_step(sm_g, em_g, mae_g, prob_gated, gt_bin)

    # reduce across ranks (scalar sums only)
    if _is_dist():
        def _all_reduce_acc(acc: dict):
            keys = [
                "dice_fg_sum","dice_bg_sum","n","tp","fp","fn",
                "bg_all","fp_bg_all","bg_pos","fp_bg_pos","bg_empty","fp_bg_empty","n_pos_img","n_vdr_tp",
            ]
            t = torch.tensor([float(acc[k]) for k in keys], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            for k, v in zip(keys, t.tolist()):
                acc[k] = float(v)

        _all_reduce_acc(acc_raw)
        _all_reduce_acc(acc_g)

    met_raw = _metric_finalize(acc_raw)
    met_g = _metric_finalize(acc_g)

    cod_raw = dict(
        S=float(sm_raw.get_results()["sm"]),
        E=float(em_raw.get_results()["em"]["adp"]),
        MAE=float(mae_raw.get_results()["mae"]),
    )
    cod_g = dict(
        S=float(sm_g.get_results()["sm"]),
        E=float(em_g.get_results()["em"]["adp"]),
        MAE=float(mae_g.get_results()["mae"]),
    )

    # 打印层面会在 main 里按 1/8 格式输出；这里返回结构化结果
    return dict(raw=met_raw, gated=met_g, cod_raw=cod_raw, cod_gated=cod_g)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--hnm-scan", type=int, default=None)
    parser.add_argument("--skip-hnm-init", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    rank, local_rank, device = _setup_ddp()

    cfg = Config
    cfg.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", str(local_rank)))
    cfg.WORLD_SIZE = int(os.environ.get("WORLD_SIZE", str(_get_world_size())))
    cfg.RANK = int(os.environ.get("RANK", str(rank)))
    cfg.USE_AMP = bool(not args.no_amp)
    cfg.GRAD_ACCUM_STEPS = int(max(1, int(args.grad_accum)))

    if args.batch_size is not None:
        cfg.BATCH_SIZE = int(args.batch_size)
    if args.hnm_scan is not None:
        cfg.HNM_CANDIDATE_SCAN = int(args.hnm_scan)
    if args.skip_hnm_init:
        cfg.HNM_INIT_BEFORE_TRAIN = False
    if args.num_workers is not None:
        cfg.NUM_WORKERS = int(args.num_workers)

    os.makedirs(cfg.SAVE_PATH, exist_ok=True)
    _maybe_tee_stdout_to_log(cfg)

    def _log(msg: str) -> None:
        if _is_main():
            print(msg)

    if _is_main():
        print(f"[ESD-VesNet] Training (uncertainty-aware vessel segmentation + HNM + FP-aware) started at {_dt.datetime.now()}")
        print(f"Device: {device}")
        print(f"World size: {_get_world_size()}, rank: {rank}, local_rank: {local_rank}")
        try:
            import sam3  # type: ignore

            print(f"[env] sam3 package resolved to: {getattr(sam3, '__file__', 'unknown')}")
        except Exception as _e:
            print(f"[env] sam3 import failed: {_e}")
        print(f"Config: {cfg.__dict__}")

    _seed_everything(2026 + rank)

    train_patients, val_patients = _build_patient_split(cfg)

    base_train = VesselESDDataset(
        data_root=cfg.DATA_ROOTS,
        patient_ids=train_patients,
        split="train",
        include_unlabeled_negatives=cfg.INCLUDE_UNLABELED_NEG_TRAIN,
        neg_data_roots=cfg.NEG_DATA_ROOTS_TRAIN,
        neg_ratio=cfg.TRAIN_NEG_RATIO_IN_DATASET,
        max_negatives=cfg.TRAIN_MAX_NEGATIVES,
        seed=cfg.NEG_SEED,
    )
    base_val = VesselESDDataset(
        data_root=cfg.DATA_ROOTS,
        patient_ids=val_patients,
        split="val",
        include_unlabeled_negatives=cfg.INCLUDE_UNLABELED_NEG_VAL,
        neg_data_roots=cfg.NEG_DATA_ROOTS_VAL,
        neg_ratio=cfg.VAL_NEG_RATIO,
        max_negatives=cfg.VAL_MAX_NEGATIVES,
        seed=cfg.NEG_SEED,
    )

    train_ds: Dataset = TrainDataset(base_train, inp_size=cfg.INP_SIZE, augment=True)
    val_ds: Dataset = ValDataset(base_val, inp_size=cfg.INP_SIZE, augment=False)

    # index split for sampler
    pos_idx = [i for i, s in enumerate(base_train.samples) if not s.get("is_negative", False)]
    neg_idx = [i for i, s in enumerate(base_train.samples) if s.get("is_negative", False)]

    if _is_main():
        pos_shard = len(pos_idx[_get_rank() :: _get_world_size()])
        neg_shard = len(neg_idx[_get_rank() :: _get_world_size()])
        print(f"Train samples(total ds): {len(base_train)} (pos_shard={pos_shard}, neg_shard={neg_shard})")
        print(f"Val samples: {len(base_val)}")

    batch_sampler = BalancedHardNegBatchSampler(
        pos_indices=pos_idx,
        neg_indices=neg_idx,
        batch_size=cfg.BATCH_SIZE,
        neg_ratio_in_batch=float(cfg.NEG_RATIO_IN_BATCH),
        hard_prob=cfg.HARD_PROB,
        hard_pool=[],
        seed=cfg.NEG_SEED,
        rank=rank,
        world_size=_get_world_size(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    # distributed shard for val
    if _is_dist():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    else:
        val_sampler = None
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.BATCH_SIZE),
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(0, min(4, cfg.NUM_WORKERS)),
        pin_memory=True,
    )

    model = _build_model(cfg).to(device)
    _set_trainable_fullsam(model, cfg)
    loaded = _load_sam3_pretrained(model, cfg.SAM3_CHECKPOINT)
    _log(f"Loaded SAM3 pretrained mapped params: {loaded}")
    if _is_main():
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        _log(
            f"[finetune] UNFREEZE_LAST_N_BLOCKS={int(getattr(cfg,'UNFREEZE_LAST_N_BLOCKS',0) or 0)}, "
            f"GRAD_CLIP_NORM={float(getattr(cfg,'GRAD_CLIP_NORM',0.0) or 0.0):.3f}, "
            f"trainable={n_trainable}/{n_total} ({(100.0*n_trainable/max(1,n_total)):.2f}%)"
        )

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        _log(f"[ckpt] loaded: {args.ckpt}")
        _log(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    if _is_dist():
        # Avoid per-iteration buffer broadcasts (can hang if ranks diverge); we handle sync via gradients.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    if args.eval_only:
        if val_sampler is not None:
            val_sampler.set_epoch(0)
        metrics = evaluate(model, val_loader, device=device, thr=0.5, vdr_thr=cfg.VDR_THRESHOLD)
        if _is_main():
            raw = metrics["raw"]
            gated = metrics["gated"]
            cod_raw = metrics["cod_raw"]
            cod_g = metrics["cod_gated"]
            print(
                f"[Eval] raw: dice_fg={raw['dice_fg']:.4f}, dice_mean={raw['dice_mean']:.4f}, "
                f"prec_fg={raw['prec_fg']:.4f}, bg_fpr_all={raw['bg_fpr_all']:.4f}, "
                f"bg_fpr_pos={raw['bg_fpr_pos']:.4f}, bg_fpr_empty={raw['bg_fpr_empty']:.4f}, VDR={raw['VDR']:.4f}"
            )
            print(f"[Eval] COD raw: S-measure={cod_raw['S']:.4f}, E-measure={cod_raw['E']:.4f}, MAE={cod_raw['MAE']:.4f}")
            print(
                f"[Eval] gated(gamma={cfg.GATE_GAMMA}): dice_fg={gated['dice_fg']:.4f}, dice_mean={gated['dice_mean']:.4f}, "
                f"prec_fg={gated['prec_fg']:.4f}, bg_fpr_all={gated['bg_fpr_all']:.4f}, "
                f"bg_fpr_pos={gated['bg_fpr_pos']:.4f}, bg_fpr_empty={gated['bg_fpr_empty']:.4f}, VDR={gated['VDR']:.4f}"
            )
            print(
                f"[Eval] COD gated(gamma={cfg.GATE_GAMMA}): S-measure={cod_g['S']:.4f}, "
                f"E-measure={cod_g['E']:.4f}, MAE={cod_g['MAE']:.4f}"
            )
        return

    # optimizer
    params = [p for p in (model.module.parameters() if hasattr(model, "module") else model.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    # Match 1/8 logs: LR decays smoothly over epochs (cosine to LR_MIN).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.EPOCH_MAX),
        eta_min=float(cfg.LR_MIN),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and bool(cfg.USE_AMP)))

    best_dice_mean = -1e9

    hard_pool: List[int] = []
    global_step = 0

    def _update_hnm(epoch: int) -> None:
        nonlocal hard_pool
        if len(neg_idx) == 0:
            hard_pool = []
            batch_sampler.update_hard_pool(hard_pool)
            return

        start = time.time()
        # IMPORTANT: all ranks must participate (DDP collective sequence)
        thr_eff = float(cfg.HNM_SCORE_THR)
        use_uncert_eff = bool(cfg.HNM_USE_UNCERT)

        # Init warm-up (epoch1 only):
        # We try a small ladder to avoid two failure modes:
        # - gated@0.5 can be overly conservative -> all-zero scores (degenerate hard pool)
        # - prob_vessel@0.5 can be overly aggressive at init -> extremely hard pool (hurts early learning)
        # Strategy (epoch1 only):
        #   (a) try the same settings as normal training (thr=0.5, use_uncert=HNM_USE_UNCERT)
        #   (b) if still all-zero, try non-uncert/prob_vessel at the same thr (use_uncert=False)
        #   (c) if still all-zero, fallback to a looser threshold (HNM_INIT_THR) with use_uncert=False
        init_mode = False
        init_fallback = False
        init_stage = "normal"
        if epoch == 1 and bool(cfg.HNM_INIT_BEFORE_TRAIN):
            init_mode = True

        def _scan(thr_val: float, use_uncert_val: bool) -> List[Tuple[int, float]]:
            return _hnm_scan_negatives(
                model,
                base_train,
                neg_idx,
                int(cfg.HNM_CANDIDATE_SCAN),
                device=device,
                thr=float(thr_val),
                use_uncert=bool(use_uncert_val),
                inp_size=int(cfg.INP_SIZE),
            )

        # default: one scan with (thr_eff, use_uncert_eff)
        scores = _scan(thr_eff, use_uncert_eff)
        if init_mode:
            # stage (a): same as normal training
            init_stage = f"try_a(thr={thr_eff:.3f},use_uncert={use_uncert_eff})"

            def _max_score(ss: List[Tuple[int, float]]) -> float:
                try:
                    return max((s for _i, s in ss), default=0.0)
                except Exception:
                    return 0.0

            if _max_score(scores) <= 0.0:
                # stage (b): non-uncert (prob_vessel) at same thr
                init_stage = f"try_b(thr={thr_eff:.3f},use_uncert=False)"
                scores = _scan(thr_eff, False)

            if _max_score(scores) <= 0.0:
                # stage (c): loosen thr + non-uncert
                thr_eff = float(getattr(cfg, "HNM_INIT_THR", thr_eff))
                init_fallback = True
                init_stage = f"try_c(thr={thr_eff:.3f},use_uncert=False)"
                scores = _scan(thr_eff, False)

        if _is_main():
            scores.sort(key=lambda x: x[1], reverse=True)
            hard_pool = [i for i, _s in scores[: int(cfg.HNM_TOPK)]]
        if _is_dist():
            obj = [hard_pool]
            dist.broadcast_object_list(obj, src=0)
            hard_pool = obj[0]
        batch_sampler.update_hard_pool(hard_pool)

        if _is_main():
            elapsed = time.time() - start
            merged_scan = len(scores)
            approx = min(int(cfg.HNM_CANDIDATE_SCAN), len(neg_idx))
            top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
            print(
                f"[HNM-DDP] merged_scan={merged_scan} (~{approx}), "
                f"hard_pool={len(hard_pool)}/{int(cfg.HNM_TOPK)}, time={elapsed:.1f}s"
            )
            if epoch == 1 and bool(cfg.HNM_INIT_BEFORE_TRAIN):
                print(
                    f"[HNM-DDP] init_warmup: thr={thr_eff:.3f}, use_uncert={use_uncert_eff}, "
                    f"fallback={init_fallback}, stage={init_stage}"
                )
            print(f"[HNM-DDP] top5={top5}")

    for epoch in range(1, cfg.EPOCH_MAX + 1):
        batch_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        if epoch == 1 and bool(cfg.HNM_INIT_BEFORE_TRAIN):
            _update_hnm(epoch)
        elif (epoch % max(1, int(cfg.HNM_UPDATE_EVERY))) == 0:
            _update_hnm(epoch)

        if _is_main():
            print(
                f"[Train] epoch={epoch} batches={len(train_loader)} "
                f"pos_per_batch={batch_sampler.n_pos} neg_per_batch={batch_sampler.n_neg} "
                f"hard_prob={float(cfg.HARD_PROB):.2f} hard_pool={len(hard_pool)}"
            )

        # train
        model.train()
        total_loss = 0.0
        nb = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            inp = batch["inp"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            gt_bin = (gt > 0.5).float()

            if hasattr(model, "module"):
                m = model.module
            else:
                m = model

            if hasattr(m, "set_global_step"):
                m.set_global_step(global_step)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and bool(cfg.USE_AMP))):
                loss_main = model(inp, gt_bin)
                # fp-aware: emphasize negatives (empty GT)
                is_empty = (gt_bin.view(gt_bin.shape[0], -1).sum(dim=1) == 0).float()
                w = 1.0 + is_empty.mean() * (float(cfg.NEG_LOSS_WEIGHT) - 1.0)
                loss = loss_main * w
                loss_scaled = loss / float(max(1, int(cfg.GRAD_ACCUM_STEPS)))

            # Guard against NaN/Inf.
            # IMPORTANT under DDP: the decision to "skip" must be consistent across all ranks,
            # otherwise collectives (allreduce/broadcast) will mismatch and hang.
            local_finite = bool(torch.isfinite(loss_scaled).all().item())
            all_finite = local_finite
            if _is_dist():
                flag = torch.tensor(1 if local_finite else 0, device=device, dtype=torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                all_finite = bool(flag.item())

            if not all_finite:
                bad_ranks = None
                if _is_dist():
                    # IMPORTANT: all ranks must participate in the same collectives.
                    flags = [torch.zeros_like(flag) for _ in range(int(_get_world_size()))]
                    dist.all_gather(flags, flag)
                    if _is_main():
                        bad_ranks = [i for i, f in enumerate(flags) if int(f.item()) == 0]
                if _is_main():
                    try:
                        lm = float(loss_main.detach().float().item())
                    except Exception:
                        lm = float("nan")
                    try:
                        ls = float(loss.detach().float().item())
                    except Exception:
                        ls = float("nan")
                    print(
                        f"[warn] non-finite loss (any-rank) at epoch={epoch} step={step} "
                        f"loss_main={lm} loss={ls} use_amp={bool(cfg.USE_AMP)} bad_ranks={bad_ranks}; "
                        f"skip step on all ranks"
                    )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            scaler.scale(loss_scaled).backward()

            if step % max(1, int(cfg.GRAD_ACCUM_STEPS)) == 0:
                # AMP-safe gradient clipping (unscale first)
                clip = float(getattr(cfg, "GRAD_CLIP_NORM", 0.0) or 0.0)
                if clip > 0:
                    try:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, max_norm=clip)
                    except Exception:
                        pass
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().item())
            nb += 1
            global_step += 1

        # reduce loss
        if _is_dist():
            t = torch.tensor([total_loss, float(nb)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss, nb = float(t[0].item()), float(t[1].item())

        if _is_main():
            print(f"Epoch {epoch}/{cfg.EPOCH_MAX} - Train Loss: {total_loss / max(1.0, nb):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Step scheduler after epoch (keeps epoch1 LR printed as initial 0.000200, similar to old logs)
        scheduler.step()

        if epoch % cfg.EPOCH_VAL == 0:
            metrics = evaluate(model, val_loader, device=device, thr=0.5, vdr_thr=cfg.VDR_THRESHOLD)
            if _is_main():
                raw = metrics["raw"]
                gated = metrics["gated"]
                cod_raw = metrics["cod_raw"]
                cod_g = metrics["cod_gated"]

                print(
                    f"Epoch {epoch} - Val raw: dice_fg={raw['dice_fg']:.4f}, dice_mean={raw['dice_mean']:.4f}, "
                    f"prec_fg={raw['prec_fg']:.4f}, bg_fpr_all={raw['bg_fpr_all']:.4f}, "
                    f"bg_fpr_pos={raw['bg_fpr_pos']:.4f}, bg_fpr_empty={raw['bg_fpr_empty']:.4f}, VDR={raw['VDR']:.4f}"
                )
                print(
                    f"Epoch {epoch} - Val COD raw: S-measure={cod_raw['S']:.4f}, "
                    f"E-measure={cod_raw['E']:.4f}, MAE={cod_raw['MAE']:.4f}"
                )
                print(
                    f"Epoch {epoch} - Val gated(gamma={cfg.GATE_GAMMA}): dice_fg={gated['dice_fg']:.4f}, "
                    f"dice_mean={gated['dice_mean']:.4f}, prec_fg={gated['prec_fg']:.4f}, "
                    f"bg_fpr_all={gated['bg_fpr_all']:.4f}, bg_fpr_pos={gated['bg_fpr_pos']:.4f}, "
                    f"bg_fpr_empty={gated['bg_fpr_empty']:.4f}, VDR={gated['VDR']:.4f}"
                )
                print(
                    f"Epoch {epoch} - Val COD gated(gamma={cfg.GATE_GAMMA}): S-measure={cod_g['S']:.4f}, "
                    f"E-measure={cod_g['E']:.4f}, MAE={cod_g['MAE']:.4f}"
                )

                # save best by dice_mean (matches old logs)
                if raw["dice_mean"] > best_dice_mean:
                    best_dice_mean = raw["dice_mean"]
                    save_path = os.path.join(cfg.SAVE_PATH, "model_epoch_best.pth")
                    sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(
                        {"model": sd, "epoch": epoch, "metrics": metrics, "config": _cfg_serializable(cfg)},
                        save_path,
                    )
                    _log(f"[ESD-VesNet] Saved best model with DiceMean: {best_dice_mean:.4f}")


if __name__ == "__main__":
    main()


