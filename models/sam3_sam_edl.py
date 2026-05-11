import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import register


def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def _dirichlet_kl_to_uniform(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # KL( Dir(alpha) || Dir(1) ) per-pixel
    K = alpha.shape[1]
    prior = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_prior = float(K)
    lnB_alpha = torch.lgamma(sum_alpha + eps) - torch.lgamma(alpha + eps).sum(dim=1, keepdim=True)
    lnB_prior = torch.lgamma(torch.tensor(sum_prior, device=alpha.device, dtype=alpha.dtype)) - torch.lgamma(
        prior + eps
    ).sum(dim=1, keepdim=True)
    digamma_sum = torch.digamma(sum_alpha + eps)
    digamma_alpha = torch.digamma(alpha + eps)
    kl = (lnB_alpha - lnB_prior) + ((alpha - prior) * (digamma_alpha - digamma_sum)).sum(dim=1, keepdim=True)
    return kl


def _soft_dice_loss(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = prob.float()
    gt = gt.float()
    inter = (prob * gt).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + gt.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()


@register("sam3-sam-edl")
class SAM3SAMEDL(nn.Module):
    """
    目标：做一个“能加载 sam3.pt 里 tracker.sam_prompt_encoder + tracker.sam_mask_decoder”的训练模型，
    以便加载时映射参数数量回到 ~554（而不仅是 vision_backbone 的 410）。

    - backbone: sam3-main 的 Sam3DualViTDetNeck（输出 288/144/72/36 的特征）
    - prompt encoder: sam3-main 的 PromptEncoder（含 no_mask_embed / pe_layer）
    - mask decoder: sam3-main 的 MaskDecoder（含 iou_token/mask_tokens/obj_score_token 等）
    - segmentation: prompt-free（无点/框/文本），用 empty prompts + no_mask_embed 走一遍 sam_mask_decoder
    - EDL: 在最终 vessel_logits 上做 evidence->Dirichlet，输出不确定性 u，并提供 gated prob
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
    ):
        super().__init__()
        self.inp_size = int(inp_size)
        self._core_size = 1008  # sam3-main RoPE/patch=14 对齐

        self.edl_beta = float(edl_beta)
        self.edl_lambda_kl = float(edl_lambda_kl)
        self.edl_anneal_steps = int(edl_anneal_steps)
        self.edl_w_dice = float(edl_w_dice)
        self.gate_gamma = float(gate_gamma)
        self._global_step = 0

        # ---- import sam3-main (kept inside repo) ----
        sam3_main = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sam3-main"))
        if sam3_main not in os.sys.path:
            os.sys.path.insert(0, sam3_main)

        from sam3.model.necks import Sam3DualViTDetNeck
        from sam3.model.position_encoding import PositionEmbeddingSine
        from sam3.model.vitdet import ViT
        from sam3.sam.prompt_encoder import PromptEncoder
        from sam3.sam.mask_decoder import MaskDecoder
        from sam3.sam.transformer import TwoWayTransformer

        vit = ViT(
            img_size=self._core_size,
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

        self.image_encoder = nn.Module()
        self.image_encoder.vision_backbone = Sam3DualViTDetNeck(
            trunk=vit,
            position_encoding=pos_enc,
            d_model=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            add_sam2_neck=False,
        )

        # ---- tracker submodule: 命名对齐 sam3.pt 的 key（tracker.sam_prompt_encoder / tracker.sam_mask_decoder）----
        self.tracker = nn.Module()
        self.tracker.sam_prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(72, 72),
            input_image_size=(self._core_size, self._core_size),
            mask_in_chans=16,
        )
        self.tracker.sam_mask_decoder = MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048),
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            pred_obj_scores=True,
        )

    def set_global_step(self, step: int):
        self._global_step = int(step)

    def _alpha_from_binary_logits(self, vessel_logits: torch.Tensor) -> torch.Tensor:
        # NOTE: digamma/lgamma in EDL loss can be numerically unstable under AMP(fp16),
        # so we keep alpha computation in fp32.
        vessel_logits = vessel_logits.float()
        # Numerical guards:
        # - Keep fp32 and sanitize NaN/Inf.
        # - Do NOT clamp finite logits here; clamping logits can create an artificial probability "floor"
        #   (e.g. prob_vessel ~= 1/32 when many logits <= -30) and make outputs look constant.
        # - We instead cap evidence/alpha below to prevent lgamma/digamma overflow.
        vessel_logits = torch.nan_to_num(vessel_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logits2 = torch.cat([-vessel_logits, vessel_logits], dim=1)  # [B,2,H,W]
        evidence = F.softplus(self.edl_beta * logits2)
        # Cap evidence/alpha to avoid lgamma overflow in KL term.
        evidence = evidence.clamp(min=0.0, max=50.0)
        alpha = evidence + 1.0
        alpha = alpha.clamp(min=1.0 + 1e-6, max=1000.0)
        return alpha

    def _prep_empty_prompts(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = torch.zeros((batch_size, 0, 2), device=device, dtype=torch.float32)
        labels = torch.zeros((batch_size, 0), device=device, dtype=torch.int64)
        return coords, labels

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        orig_hw = (int(x.shape[-2]), int(x.shape[-1]))
        if orig_hw != (self._core_size, self._core_size):
            x_in = F.interpolate(x, size=(self._core_size, self._core_size), mode="bilinear", align_corners=False)
        else:
            x_in = x

        sam3_feats, _sam3_pos, _sam2_feats, _sam2_pos = self.image_encoder.vision_backbone(x_in)
        # 72x72 的 embedding 作为 SAM decoder 的 image_embeddings
        img_embed = sam3_feats[2]  # [B,256,72,72]

        # prompt-free：empty points + no_mask_embed
        B = int(img_embed.shape[0])
        coords, labels = self._prep_empty_prompts(B, device=img_embed.device)
        sparse_embeddings, dense_embeddings = self.tracker.sam_prompt_encoder(
            points=(coords, labels), boxes=None, masks=None
        )

        # high-res features（288/144）用于更好的上采样
        feat_s0 = self.tracker.sam_mask_decoder.conv_s0(sam3_feats[0])  # [B,32,288,288]
        feat_s1 = self.tracker.sam_mask_decoder.conv_s1(sam3_feats[1])  # [B,64,144,144]

        masks, _iou, _tok, _obj = self.tracker.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.tracker.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[feat_s0, feat_s1],
        )  # masks: [B,1,288,288] logits

        # 288 -> 1008 -> orig
        logits = F.interpolate(masks, size=(self._core_size, self._core_size), mode="bilinear", align_corners=False)
        if orig_hw != (self._core_size, self._core_size):
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

    def forward(self, input: torch.Tensor, gt_mask: torch.Tensor | None = None):
        if gt_mask is None:
            return self.infer(input)

        vessel_logits = self.infer(input)
        # IMPORTANT: keep EDL loss terms in fp32 to avoid NaN/Inf under AMP.
        with torch.cuda.amp.autocast(enabled=False):
            gt = (gt_mask.float() > 0.5).float()
            alpha = self._alpha_from_binary_logits(vessel_logits)  # fp32
            S = alpha.sum(dim=1, keepdim=True)
            prob = alpha / (S + 1e-8)
            prob_vessel = prob[:, 1:2]

            # ACE (digamma)
            y = torch.cat([1.0 - gt, gt], dim=1)
            ace = (y * (torch.digamma(S + 1e-8) - torch.digamma(alpha + 1e-8))).sum(dim=1, keepdim=True).mean()

            # KL (annealed)
            alp = (alpha - 1.0) * (1.0 - y) + 1.0
            kl = _dirichlet_kl_to_uniform(alp).mean()
            anneal = min(1.0, float(self._global_step) / float(max(1, self.edl_anneal_steps)))
            loss = ace + (anneal * self.edl_lambda_kl * kl)

            if self.edl_w_dice > 0:
                loss = loss + (self.edl_w_dice * _soft_dice_loss(prob_vessel, gt))
            return loss

