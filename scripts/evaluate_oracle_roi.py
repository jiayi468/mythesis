from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from clip_anomaly.backbone import FrozenClipBackbone, load_openclip
from clip_anomaly.config import load_config
from clip_anomaly.experiment import experiment_root, feature_cache, supplement_output
from clip_anomaly.preprocessing import preprocess_image
from clip_anomaly.roi import expand_bbox, fuse_maps, mask_bbox, paste_roi_map
from clip_anomaly.training import load_trained_head


def _scaled_box(box, source_width: int, source_height: int, target: int = 224):
    x0, y0, x1, y1 = box
    scaled = (
        max(0, min(target - 1, round(x0 * target / source_width))),
        max(0, min(target - 1, round(y0 * target / source_height))),
        max(1, min(target, round(x1 * target / source_width))),
        max(1, min(target, round(y1 * target / source_height))),
    )
    return scaled[0], scaled[1], max(scaled[0] + 1, scaled[2]), max(scaled[1] + 1, scaled[3])


def _random_box(width: int, height: int, box, seed: int):
    x0, y0, x1, y1 = box
    box_width, box_height = x1 - x0, y1 - y0
    rng = np.random.default_rng(seed)
    left = int(rng.integers(0, max(1, width - box_width + 1)))
    top = int(rng.integers(0, max(1, height - box_height + 1)))
    return left, top, left + box_width, top + box_height


def _extract_features(config, records, context_ratio: float):
    oracle_tensors, random_tensors, metadata = [], [], []
    for index, record in enumerate(records):
        image_path = config.project_root / record["image_path"]
        mask_path = config.project_root / record["mask_path"]
        with Image.open(mask_path) as mask_image:
            original_mask = np.asarray(mask_image.convert("L")) > 0
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            box = expand_bbox(mask_bbox(original_mask), image.width, image.height, context_ratio)
            random_box = _random_box(image.width, image.height, box, 42 + index)
            oracle_tensors.append(preprocess_image(image.crop(box), 224))
            random_tensors.append(preprocess_image(image.crop(random_box), 224))
            metadata.append({
                "box": box, "random_box": random_box, "source_width": image.width, "source_height": image.height,
            })
    clip_model, _ = load_openclip(str(config.model["name"]), str(config.model["pretrained"]))
    backbone = FrozenClipBackbone(clip_model)

    def encode(tensors):
        batches = []
        for start in range(0, len(tensors), 4):
            batches.append(backbone(torch.stack(tensors[start:start + 4])).half().cpu())
            print(f"ROI cached {min(start + 4, len(tensors))}/{len(tensors)}", flush=True)
        return torch.cat(batches)

    return encode(oracle_tensors), encode(random_tensors), metadata


def _pixel_ap(mask: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(mask.reshape(-1).astype(np.uint8), score.reshape(-1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds or [int(value) for value in config.experiment["training_seeds"]]
    payload = feature_cache(config).load()
    anomaly_indices = [
        index for index, record in enumerate(payload["records"])
        if record["split"] == "val" and int(record["label"]) == 1
    ]
    records = [payload["records"][index] for index in anomaly_indices]
    oracle_features, random_features, metadata = _extract_features(
        config, records, float(config.experiment.get("roi_context_ratio", 0.25))
    )
    masks = payload["full_masks"][anomaly_indices].numpy().astype(np.uint8)
    rows = []
    for seed in seeds:
        checkpoint = supplement_output(config.output_root, "b2_text_prototype", seed) / "checkpoints" / "clip_b16_patch_head.pt"
        model = load_trained_head(checkpoint, payload["prototypes"])
        with torch.inference_mode():
            full_maps, _ = model(payload["features"][anomaly_indices].float())
            oracle_maps, oracle_scores = model(oracle_features.float())
            random_maps, random_scores = model(random_features.float())
            full_224 = F.interpolate(full_maps[:, None], (224, 224), mode="bilinear", align_corners=False)[:, 0].numpy()
        for index, (record, meta) in enumerate(zip(records, metadata)):
            box_224 = _scaled_box(meta["box"], meta["source_width"], meta["source_height"])
            random_box_224 = _scaled_box(meta["random_box"], meta["source_width"], meta["source_height"])
            oracle_projected = paste_roi_map(np.zeros((224, 224), dtype=np.float32), oracle_maps[index].numpy(), box_224)
            random_projected = paste_roi_map(np.zeros((224, 224), dtype=np.float32), random_maps[index].numpy(), random_box_224)
            fused = fuse_maps(full_224[index], oracle_projected)
            conditions = {
                "full": (full_224[index], float(np.quantile(full_224[index], 0.995))),
                "oracle": (oracle_projected, float(oracle_scores[index])),
                "full_oracle": (fused, float(np.quantile(fused, 0.995))),
                "random": (random_projected, float(random_scores[index])),
            }
            for condition, (score_map, image_score) in conditions.items():
                rows.append({
                    "seed": seed, "category": record["category"], "defect_type": record["defect_type"],
                    "image_path": record["image_path"], "condition": condition,
                    "pixel_ap": _pixel_ap(masks[index], score_map), "image_score": image_score,
                    "mask_area_ratio": record["mask_area_ratio"], "box": ",".join(map(str, meta["box"])),
                })
    root = experiment_root(config) / "oracle_roi"
    root.mkdir(parents=True, exist_ok=True)
    per_image = root / "per_image.csv"
    with per_image.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["category"], row["condition"])].append(float(row["pixel_ap"]))
        grouped[(row["seed"], "combined", row["condition"])].append(float(row["pixel_ap"]))
    summary_rows = [
        {"seed": key[0], "scope": key[1], "condition": key[2], "samples": len(values),
         "mean_per_image_pixel_ap": float(np.mean(values)), "std_per_image_pixel_ap": float(np.std(values, ddof=1))}
        for key, values in sorted(grouped.items())
    ]
    summary = root / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)

    gains = defaultdict(dict)
    for row in summary_rows:
        gains[(int(row["seed"]), str(row["scope"]))][str(row["condition"])] = float(row["mean_per_image_pixel_ap"])
    positive_by_category = {
        category: sum(gains[(seed, category)]["full_oracle"] > gains[(seed, category)]["full"] for seed in seeds)
        for category in sorted({record["category"] for record in records})
    }
    combined_gain = float(np.mean([
        gains[(seed, "combined")]["full_oracle"] - gains[(seed, "combined")]["full"] for seed in seeds
    ]))
    if any(count >= 2 for count in positive_by_category.values()) and combined_gain >= 0.02:
        decision = "GO"
        explanation = "Oracle ROI 达到预注册门槛；下一阶段可以实现自动 Top-K 与 Region Recall@K。"
    elif combined_gain > 0:
        decision = "HOLD"
        explanation = "Oracle ROI 有局部正向现象但未达到稳定门槛；暂不启动动态 K。"
    else:
        decision = "NO-GO"
        explanation = "Oracle ROI 未获得正向平均收益；暂停粗到细路线，优先检查特征与监督。"
    decision_path = root / "decision.md"
    decision_path.write_text(
        "# Oracle ROI 路线决策\n\n"
        f"- 决策：**{decision}**\n"
        f"- 合并平均 Pixel AP 增量：`{combined_gain:.6f}`\n"
        f"- 各类别正向种子数：`{positive_by_category}`\n"
        f"- 说明：{explanation}\n\n"
        "该实验使用 GT 掩膜定位 ROI，仅是方法上界诊断，不是可部署结果；统计量为异常图像逐图 Pixel AP 的均值。\n",
        encoding="utf-8",
    )
    print(per_image); print(summary); print(decision_path); print(decision, combined_gain, positive_by_category)


if __name__ == "__main__":
    main()
