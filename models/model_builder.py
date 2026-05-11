# Copyright (c) Meta Platforms, Inc. and affiliates.
# All Rights Reserved.

"""models/model_builder.py

这个仓库里我们主要用到 **SAM3 的图像编码器(Trunk+Neck)** 来给 SAM 解码器提供特征。
原始 upstream 的 `model_builder.py` 还会导入大量 video / tracker / dataset 相关模块，
在没有安装 `mmcv` / `decord` / `huggingface_hub` 等依赖时会导致 import 直接失败。

这里保留一个 **最小可用** 的 builder：
- 支持 `_create_vision_backbone()`（供 `models/sam3/build_sam3.py` 使用）
- 其它高阶构建函数保留占位，避免误用时静默失败
"""

from __future__ import annotations

from typing import Optional

import torch


def _setup_tf32() -> None:
    """Enable TF32 on Ampere+ GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution: Optional[int] = None):
    from sam3.model.position_encoding import PositionEmbeddingSine

    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None):
    # ViTDet backbone
    from sam3.model.vitdet import ViT

    return ViT(
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
        compile_mode=compile_mode,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity: bool = False):
    from sam3.model.necks import Sam3DualViTDetNeck

    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vision_backbone(compile_mode=None, enable_inst_interactivity: bool = False):
    """Create SAM3 visual backbone (ViT + neck)."""
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    return vit_neck


def build_sam3_image_model(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "当前环境使用的是最小化 model_builder（只保证 image encoder 能跑）。\n"
        "如果你确实需要 build_sam3_image_model/full pipeline，请安装完整依赖并恢复 upstream 版本。"
    )


def build_sam3_video_model(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "当前环境使用的是最小化 model_builder（未包含 video/tracker 构建）。"
    )
