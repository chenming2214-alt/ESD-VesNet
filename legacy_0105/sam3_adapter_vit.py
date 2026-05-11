import math
import os
import sys
from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

_sam3_main = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sam3-main"))
if _sam3_main not in sys.path:
    sys.path.insert(0, _sam3_main)

from sam3.model.vitdet import ViT, get_abs_pos  # type: ignore


class AdapterPromptGenerator(nn.Module):
    """
    0105 baseline 思路的“简化版 PromptGenerator”：
    - 只保留 embedding_tune + adaptor（不引入 FFT/handcrafted 分支，依赖更少、召回更稳）
    - 在每个 ViT block 前做一次 residual prompt 注入：x <- x + Proj(MLP(ProjDown(x)))
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        tuning_stage: str = "1234",
        scale_factor: int = 32,
        adaptor: str = "adaptor",
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.scale_factor = int(scale_factor)
        self.tuning_stage = str(tuning_stage)
        self.adaptor = str(adaptor)

        # 将 depth 均分成 4 个 stage（与 baseline 一致）
        depth_per_stage = depth // 4
        remainder = depth % 4
        self.depths: List[int] = [depth_per_stage] * 4
        if remainder > 0:
            self.depths[-1] += remainder

        low_dim = max(1, self.embed_dim // self.scale_factor)

        # stage-specific: down-proj + up-proj（baseline 的 embedding_generator/shared_mlp）
        self.down = nn.ModuleList([nn.Linear(self.embed_dim, low_dim) for _ in range(4)])
        self.up = nn.ModuleList([nn.Linear(low_dim, self.embed_dim) for _ in range(4)])

        # per-block lightweight mlp（baseline 的 lightweight_mlp{stage}_{idx}）
        self.block_mlps = nn.ModuleList()
        for s in range(4):
            n = self.depths[s] + 1
            stage_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(low_dim, low_dim),
                        nn.GELU(),
                    )
                    for _ in range(n)
                ]
            )
            self.block_mlps.append(stage_mlps)  # type: ignore[arg-type]

    def _stage_rel_idx(self, block_idx: int) -> tuple[int, int]:
        d0, d1, d2, _d3 = self.depths
        if block_idx < d0:
            return 1, block_idx
        if block_idx < d0 + d1:
            return 2, block_idx - d0
        if block_idx < d0 + d1 + d2:
            return 3, block_idx - d0 - d1
        return 4, block_idx - d0 - d1 - d2

    def inject(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        """
        x: [B,H,W,C] 或 [B,N,C]
        """
        stage_idx, rel_idx = self._stage_rel_idx(block_idx)
        if str(stage_idx) not in self.tuning_stage:
            return x

        s = stage_idx - 1
        # channel-last 的 Linear：直接对最后一维做投影
        x_low = self.down[s](x)
        mlp = self.block_mlps[s][min(rel_idx, len(self.block_mlps[s]) - 1)]
        x_low = mlp(x_low)
        delta = self.up[s](x_low)
        return x + delta


class ViTAdapter(ViT):
    """
    在 sam3-main 的 ViT 上插入 AdapterPromptGenerator（prompt-free 视觉提示）。
    """

    def __init__(self, *args, tuning_stage: str = "1234", adaptor_scale: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapter_pg = AdapterPromptGenerator(
            embed_dim=int(self.channel_list[-1]),
            depth=len(self.blocks),
            tuning_stage=tuning_stage,
            scale_factor=adaptor_scale,
        )

    @property
    def prompt_generator(self) -> nn.Module:
        # 保持命名一致，便于训练脚本“freeze_image_encoder(..., prompt_generator 不冻结)”沿用
        return self._adapter_pg

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            x = torch.cat([self.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (h, w),
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )

        x = self.ln_pre(x)

        outputs = []
        for i, blk in enumerate(self.blocks):
            # ---- Adapter injection (before each ViT block) ----
            x = self._adapter_pg.inject(x, i)

            if self.use_act_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

            if (i == self.full_attn_ids[-1]) or (
                self.return_interm_layers and i in self.full_attn_ids
            ):
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)

                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)
                else:
                    h2 = w2 = math.sqrt(feats.shape[1])
                    feats = feats.reshape(
                        feats.shape[0], h2, w2, feats.shape[-1]
                    ).permute(0, 3, 1, 2)

                outputs.append(feats)

        return outputs


