"""
兼容入口（shim）

历史上仓库里既存在 `legacy_0105/train_vessel_esd_edl_hnm_fpaware_fullsam_0105.py` 这样的训练脚本，
也有工具脚本会 `import train_vessel_esd as base` 来读取：
  - Config.DATA_ROOTS
  - Config.VAL_PATIENTS
  - get_patient_ids(...)

因此这里保留最小 API 面，避免破坏 `tools/eval_val_metrics.py` 等依赖。
若你直接执行 `python train_vessel_esd.py ...`，则会转调 legacy_0105 的训练入口。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Config:
    # NOTE: 这里用 class-level 默认值，方便被其他脚本 `import` 后直接访问。
    DATA_ROOTS: List[str] = None  # type: ignore[assignment]
    VAL_PATIENTS: List[str] = None  # type: ignore[assignment]


# Provide sane defaults matching the environment used earlier in this project.
Config.DATA_ROOTS = [
    "/mnt/data-hdd/msc2025/chenming/esd/vessel_seg/eomt-vessel/datasets/vessel_data/qilu/esd",
    "/mnt/data-hdd/msc2025/chenming/esd/vessel_seg/eomt-vessel/datasets/vessel_data/qilu/elec",
]
Config.VAL_PATIENTS = ["P10", "P16", "P17", "P25", "P35", "P38"]


def get_patient_ids(data_roots: Iterable[str]) -> List[str]:
    """Collect patient folder names from one or more roots."""
    ids = set()
    for root in data_roots:
        root = os.path.abspath(str(root))
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p) and not name.startswith("."):
                ids.add(name)
    return sorted(ids)


def _run_legacy_entrypoint() -> None:
    # 允许用户仍然用 `python train_vessel_esd.py ...` 启动训练/评测（内部转调 legacy 脚本）
    from legacy_0105.train_vessel_esd_edl_hnm_fpaware_fullsam_0105 import main as legacy_main

    legacy_main()


if __name__ == "__main__":
    # 作为脚本运行时，把 argv 原封不动交给 legacy 脚本解析
    _run_legacy_entrypoint()


