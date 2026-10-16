from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import platform
import sys

import numpy as np
from PIL import Image
import torch

from .backbone import FrozenClipBackbone, load_openclip
from .cache import FeatureCache, FeatureCacheMetadata
from .config import ExperimentConfig
from .data import SampleRecord, scan_mvtec_category, select_limited_records, split_records
from .preprocessing import downsample_mask, preprocess_image
from .prompts import build_text_prototypes


def collect_selected_records(config: ExperimentConfig) -> list[SampleRecord]:
    records = []
    for root in config.dataset_roots:
        records.extend(scan_mvtec_category(root))
    records = split_records(records, config.seed, float(config.split["abnormal_train_fraction"]))
    return select_limited_records(
        records,
        config.seed,
        int(config.split["max_train_normal_per_category"]),
        int(config.split["max_train_abnormal_per_category"]),
        int(config.split["max_val_normal_per_category"]),
        int(config.split["max_val_abnormal_per_category"]),
    )


def record_to_dict(record: SampleRecord, project_root: Path) -> dict[str, object]:
    return {
        "category": record.category,
        "defect_type": record.defect_type,
        "image_path": record.image_path.relative_to(project_root).as_posix(),
        "mask_path": record.mask_path.relative_to(project_root).as_posix() if record.mask_path else "",
        "label": record.label,
        "source_partition": record.source_partition,
        "split": record.split,
        "mask_area_ratio": record.mask_area_ratio,
    }


def write_split_csv(records: list[SampleRecord], config: ExperimentConfig) -> Path:
    output = config.output_root / "experiments" / "source_split.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [record_to_dict(record, config.project_root) for record in records]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_environment_report(config: ExperimentConfig) -> Path:
    import onnx
    import onnxruntime
    import torchvision

    output = config.output_root / "experiments" / "environment.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 实验环境审计",
        "",
        "结果类型：源域小样本探索性验证",
        "",
        f"- 操作系统：{platform.platform()}",
        f"- Python：{sys.version.split()[0]}",
        f"- PyTorch：{torch.__version__}",
        f"- torchvision：{torchvision.__version__}",
        f"- ONNX：{onnx.__version__}",
        f"- ONNX Runtime：{onnxruntime.__version__}",
        f"- CUDA 可用：{torch.cuda.is_available()}",
        f"- CPU 线程数：{torch.get_num_threads()}",
        f"- 模型：{config.model['name']} / {config.model['pretrained']}",
        f"- 随机种子：{config.seed}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def feature_cache(config: ExperimentConfig) -> FeatureCache:
    metadata = FeatureCacheMetadata(
        str(config.model["name"]),
        str(config.model["pretrained"]),
        int(config.model["input_size"]),
        int(config.model["patch_grid"]),
        "clip-rgb-bicubic-maxpool-mask-v1",
    )
    path = config.output_root / "experiments" / "feature_cache" / "midterm_cpu_features.pt"
    return FeatureCache(path, metadata)


def _load_mask(record: SampleRecord, size: int) -> np.ndarray:
    if record.mask_path is None:
        return np.zeros((size, size), dtype=np.uint8)
    with Image.open(record.mask_path) as mask:
        resized = mask.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return (np.asarray(resized) > 0).astype(np.uint8)


def build_feature_cache(config: ExperimentConfig, batch_size: int = 4) -> Path:
    records = collect_selected_records(config)
    write_split_csv(records, config)
    clip_model, tokenizer = load_openclip(str(config.model["name"]), str(config.model["pretrained"]))
    backbone = FrozenClipBackbone(clip_model)
    prototypes = build_text_prototypes(clip_model, tokenizer)
    images: list[torch.Tensor] = []
    patch_masks: list[torch.Tensor] = []
    full_masks: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    size = int(config.model["input_size"])
    grid = int(config.model["patch_grid"])
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        images.clear()
        for record in batch_records:
            with Image.open(record.image_path) as image:
                images.append(preprocess_image(image, size))
            full_mask = _load_mask(record, size)
            full_masks.append(torch.from_numpy(full_mask))
            patch_masks.append(downsample_mask(full_mask, grid))
        features.append(backbone(torch.stack(images)).half().cpu())
        print(f"cached {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    payload = {
        "features": torch.cat(features),
        "masks": torch.stack(patch_masks),
        "full_masks": torch.stack(full_masks),
        "labels": torch.tensor([record.label for record in records], dtype=torch.long),
        "prototypes": prototypes,
        "records": [record_to_dict(record, config.project_root) for record in records],
    }
    cache = feature_cache(config)
    cache.save(payload)
    return cache.path
