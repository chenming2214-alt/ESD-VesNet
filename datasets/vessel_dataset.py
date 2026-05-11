import os
import json
import glob
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset
from datasets import register


@register('vessel-esd-dataset')
class VesselESDDataset(Dataset):
    """
    内窥镜血管分割数据集
    从 JSON 标注文件中读取多边形并转换为 mask
    """
    def __init__(
        self,
        data_root,
        patient_ids,
        split='train',
        cache='none',
        include_unlabeled_negatives: bool = False,
        # 只允许从指定 data_root 收集“无 json 的 png”作为负样本（用于多源训练时控制负样本来源）
        # - None: 所有 data_root 都允许（默认行为，兼容旧代码）
        # - list[str]: 仅当 sample['data_root'] 属于该列表时，才把 png_no_json 收进 neg_pool
        neg_data_roots: list[str] | None = None,
        neg_ratio: float | None = None,
        max_negatives: int | None = None,
        seed: int = 0,
    ):
        """
        Args:
            data_root: 数据集根目录，例如 '/path/to/vessel_data/esd'
            patient_ids: 病人ID列表，例如 ['P1', 'P2', ...]
            split: 'train' 或 'val'
            cache: 'none' 或 'in_memory'
        """
        # 兼容单路径或多路径
        if isinstance(data_root, (list, tuple)):
            self.data_roots = list(data_root)
        else:
            self.data_roots = [data_root]

        # 规范化 neg_data_roots
        if neg_data_roots is None:
            self.neg_data_roots = None
        else:
            self.neg_data_roots = [os.path.normpath(str(p)) for p in neg_data_roots]
        self.patient_ids = patient_ids
        self.split = split
        self.cache = cache
        
        def _resolve_patient_dir(data_root_: str, patient_id_: str):
            """
            兼容病人目录重命名：
            - 优先使用原 patient_id
            - 若不存在，尝试从类似 'P3_韩树林2' 映射到 'P3_2'
            - 若仍不存在，尝试从类似 'P7_李小英' 映射到 'P7'
            返回 (resolved_patient_id, patient_dir) 或 (None, None)
            """
            pdir = os.path.join(data_root_, patient_id_)
            if os.path.exists(pdir):
                return patient_id_, pdir

            # 常见：P{n}_中文{m} -> P{n}_{m}
            import re
            m = re.match(r'^(P\\d+)_.*?(\\d+)$', patient_id_)
            if m:
                cand = f"{m.group(1)}_{m.group(2)}"
                pdir2 = os.path.join(data_root_, cand)
                if os.path.exists(pdir2):
                    return cand, pdir2

            # 常见：P{n}_中文 -> P{n}
            m2 = re.match(r'^(P\\d+)_.*$', patient_id_)
            if m2:
                cand = m2.group(1)
                pdir3 = os.path.join(data_root_, cand)
                if os.path.exists(pdir3):
                    return cand, pdir3

            return None, None

        # 收集所有图像和标注文件
        # - 正样本：存在 json（里面可能也会是空 shapes，但我们仍视为“有标注”）
        # - 负样本：png 存在但 json 不存在（可选）
        pos_samples = []
        neg_samples = []
        missing_image = 0
        missing_json_for_pos = 0
        missing_patient_dir = 0
        renamed_patient_dirs = {}  # old -> new
        for data_root in self.data_roots:
            root_norm = os.path.normpath(str(data_root))
            allow_neg_from_this_root = (self.neg_data_roots is None) or (root_norm in self.neg_data_roots)
            for patient_id in patient_ids:
                resolved_id, patient_dir = _resolve_patient_dir(data_root, patient_id)
                if patient_dir is None:
                    missing_patient_dir += 1
                    continue
                if resolved_id != patient_id:
                    renamed_patient_dirs[patient_id] = resolved_id
                
                # 获取所有 PNG 图像文件
                png_files = sorted(glob.glob(os.path.join(patient_dir, '*.png')))
                
                for png_file in png_files:
                    if not os.path.exists(png_file):
                        # 极少数情况下目录/文件在训练过程中被移动/删除；这里直接跳过避免中途崩溃
                        missing_image += 1
                        continue
                    json_file = png_file.replace('.png', '.json')
                    # 如果存在 JSON 文件，说明有标注
                    if os.path.exists(json_file):
                        pos_samples.append({
                            'image': png_file,
                            'annotation': json_file,
                            'patient_id': patient_id,
                            'data_root': data_root,
                            'is_negative': False,
                        })
                    elif include_unlabeled_negatives and allow_neg_from_this_root:
                        # 没有 json 的 png：按“纯背景/空 mask”负样本处理
                        neg_samples.append({
                            'image': png_file,
                            'annotation': None,
                            'patient_id': patient_id,
                            'data_root': data_root,
                            'is_negative': True,
                        })
                    else:
                        # 这种样本既不是正样本也不会进入 neg_pool（例如：该 root 不允许收无标注负样本）
                        missing_json_for_pos += 1

        # 根据 neg_ratio / max_negatives 对负样本做子采样或重复采样
        import random as _random
        rng = _random.Random(seed)

        if neg_ratio is not None:
            if not (0.0 <= neg_ratio < 1.0):
                raise ValueError(f"neg_ratio must be in [0,1), got {neg_ratio}")
            n_pos = len(pos_samples)
            if n_pos == 0:
                n_neg_target = 0
            else:
                n_neg_target = int(round((neg_ratio / (1.0 - neg_ratio)) * n_pos))
        else:
            n_neg_target = len(neg_samples)

        if max_negatives is not None:
            n_neg_target = min(n_neg_target, int(max_negatives))

        if n_neg_target <= 0:
            neg_selected = []
        elif len(neg_samples) == 0:
            neg_selected = []
        elif len(neg_samples) >= n_neg_target:
            neg_selected = rng.sample(neg_samples, k=n_neg_target)
        else:
            # 负样本不足：允许重复采样（对 FP 优先的场景通常是 OK 的）
            neg_selected = [rng.choice(neg_samples) for _ in range(n_neg_target)]

        self.samples = pos_samples + neg_selected
        rng.shuffle(self.samples)

        if missing_patient_dir > 0 or missing_image > 0 or missing_json_for_pos > 0 or len(renamed_patient_dirs) > 0:
            print(
                f"[{split}] Dataset warnings: "
                f"missing_patient_dir_skipped={missing_patient_dir}, "
                f"missing_image_skipped={missing_image}, "
                f"png_without_json_skipped={missing_json_for_pos}"
            )
            if len(renamed_patient_dirs) > 0:
                # 只打印前若干条，避免刷屏
                items = list(renamed_patient_dirs.items())[:20]
                preview = ", ".join([f"{a}->{b}" for a, b in items])
                more = "" if len(renamed_patient_dirs) <= 20 else f" ...(+{len(renamed_patient_dirs)-20})"
                print(f"[{split}] Patient dir rename mapping: {preview}{more}")

        print(
            f"[{split}] Loaded {len(self.samples)} samples "
            f"(pos={len(pos_samples)}, neg={len(neg_selected)}; "
            f"neg_pool={len(neg_samples)}, include_unlabeled_negatives={include_unlabeled_negatives}, "
            f"neg_data_roots={self.neg_data_roots}, "
            f"neg_ratio={neg_ratio}, max_negatives={max_negatives}) "
            f"from patients: {patient_ids}"
        )
    
    def __len__(self):
        return len(self.samples)
    
    def _json_to_mask(self, json_file, img_size):
        """从 LabelMe JSON 文件生成 mask"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 创建空白 mask (L 模式，值范围 0-255)
            mask = Image.new('L', img_size, 0)
            draw = ImageDraw.Draw(mask)
            
            # 绘制所有多边形
            if 'shapes' in data:
                for shape in data['shapes']:
                    # 支持多种标签名称
                    label = shape.get('label', '').lower()
                    if ('vessel' in label or 'segmentation' in label) and 'points' in shape:
                        points = [(int(p[0]), int(p[1])) for p in shape['points']]
                        if len(points) >= 3:  # 至少需要3个点才能形成多边形
                            draw.polygon(points, fill=255)
            
            return mask
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return Image.new('L', img_size, 0)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        img = Image.open(sample['image']).convert('RGB')
        img_size = img.size  # (W, H)
        
        # 加载 mask
        if sample.get('annotation', None) is None:
            mask = Image.new('L', img_size, 0)
        else:
            mask = self._json_to_mask(sample['annotation'], img_size)
        
        return img, mask
    
    def get_patient_id(self, idx):
        """获取样本对应的病人ID"""
        return self.samples[idx]['patient_id']
