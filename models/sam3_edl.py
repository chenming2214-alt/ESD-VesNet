import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import register


def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


@register("sam3-edl")
class SAM3Evidential(nn.Module):
    """
    SAM3 ViTDet-neck backbone (from sam3-main) + light segmentation head + EDL uncertainty.

    目标：
    - 让 `sam3.pt` 中的 `detector.backbone.vision_backbone.*` 权重能对齐加载（恢复 1/5 那种“能加载大部分权重”的状态）
    - 提供与现有训练脚本一致的接口：
        - infer(x) -> vessel_logits [B,1,H,W]（logits，不做 sigmoid）
        - infer_prob_uncert(x) -> (prob_vessel, u, prob_gated)
    """

    def __init__(
        self,
        inp_size: int = 1024,
        # EDL
        edl_beta: float = 1.0,
        edl_lambda_kl: float = 0.01,
        edl_anneal_steps: int = 5000,
        edl_w_dice: float = 1.0,
        gate_gamma: float = 1.0,
        # head
        head_mid: int = 128,
    ):
        super().__init__()
        self.inp_size = int(inp_size)

        self.edl_beta = float(edl_beta)
        self.edl_lambda_kl = float(edl_lambda_kl)
        self.edl_anneal_steps = int(edl_anneal_steps)
        self.edl_w_dice = float(edl_w_dice)
        self.gate_gamma = float(gate_gamma)
        self._global_step = 0

        # ---- import sam3-main (kept inside ESD-VesNet) ----
        # `sam3-main` is bundled as a sibling dependency directory, so this fallback
        # sys.path injection keeps model loading reproducible without manual PYTHONPATH setup.
        sam3_main = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sam3-main"))
        if sam3_main not in os.sys.path:
            os.sys.path.insert(0, sam3_main)

        from sam3.model.necks import Sam3DualViTDetNeck
        from sam3.model.position_encoding import PositionEmbeddingSine
        from sam3.model.vitdet import ViT

        # 与 checkpoint 对齐的 ViT 规格（对应 `detector.backbone.vision_backbone.trunk.*`）
        vit = ViT(
            img_size=1008,
            pretrain_img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=32,
            num_heads=16,
            mlp_ratio=4.625,
            norm_layer="LayerNorm",
            drop_path_rate=0.1,
            qkv_bias=True,
            use_abs_pos=True,
            tile_abs_pos=True,
            global_att_blocks=(7, 15, 23, 31),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=24,
            pretrain_use_cls_token=True,
            retain_cls_token=False,
            ln_pre=True,
            ln_post=False,
            return_interm_layers=False,
            bias_patch_embed=False,
            compile_mode=None,
            use_act_checkpoint=True,
        )
        pos_enc = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=None,
        )

        # 关键：命名为 image_encoder.vision_backbone，使其 state_dict key 与 ckpt 的
        # detector.backbone.vision_backbone.* 通过简单替换即可对齐。
        self.image_encoder = nn.Module()
        self.image_encoder.vision_backbone = Sam3DualViTDetNeck(
            trunk=vit,
            position_encoding=pos_enc,
            d_model=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            add_sam2_neck=False,
        )

        # 轻量分割 head：取最高分辨率 FPN 特征 -> 1ch logits -> resize 回输入尺寸
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, int(head_mid), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(head_mid), 1, kernel_size=1),
        )

    def set_global_step(self, step: int):
        self._global_step = int(step)

    def _alpha_from_binary_logits(self, vessel_logits: torch.Tensor) -> torch.Tensor:
        logits2 = torch.cat([-vessel_logits, vessel_logits], dim=1)  # [B,2,H,W]
        evidence = F.softplus(self.edl_beta * logits2)
        return evidence + 1.0

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        # SAM3 的 ViT 使用 patch_size=14 + RoPE，内部预计算表按 1008(=72*14) 对齐。
        # 训练脚本里 INP_SIZE=1024（不改用户配置），这里在模型内部做一次 resize 以保证 RoPE 形状一致，
        # 最终 logits 会 resize 回原始尺寸。
        orig_hw = (int(x.shape[-2]), int(x.shape[-1]))
        if orig_hw != (1008, 1008):
            x_in = F.interpolate(x, size=(1008, 1008), mode="bilinear", align_corners=False)
        else:
            x_in = x

        sam3_feats, _sam3_pos, _sam2_feats, _sam2_pos = self.image_encoder.vision_backbone(x_in)
        # highest resolution after scalp=1 in sam3-main 的默认设置通常是 sam3_feats[0]
        feat = sam3_feats[0]
        logits = self.seg_head(feat)
        logits = F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)
        return logits

    @torch.no_grad()
    def infer_prob_uncert(self, x: torch.Tensor):
        vessel_logits = self.infer(x)
        alpha = self._alpha_from_binary_logits(vessel_logits)
        S = alpha.sum(dim=1, keepdim=True)
        prob = alpha / (S + 1e-8)
        prob_vessel = prob[:, 1:2]
        u = 2.0 / (S + 1e-8)
        gate = (1.0 - u).clamp(0.0, 1.0) ** self.gate_gamma
        prob_gated = (prob_vessel * gate).clamp(0.0, 1.0)
        return prob_vessel, u, prob_gated

    def infer_gated_logits(self, x: torch.Tensor) -> torch.Tensor:
        _p, _u, pg = self.infer_prob_uncert(x)
        return _safe_logit(pg)


