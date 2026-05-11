"""
Lightweight mmseg models init.

本工程只用到了 `mmseg.models.sam` 里的 SAM 组件（不依赖 mmcv Registry）。
如果环境里没有安装 mmcv，这里就跳过 builder/losses 的导入，避免 ImportError。
"""

try:
    from .builder import build_loss  # type: ignore
    from .losses import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    build_loss = None  # type: ignore

__all__ = ['build_loss'] if build_loss is not None else []
