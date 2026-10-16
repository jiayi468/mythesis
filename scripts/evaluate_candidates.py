from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from clip_anomaly.candidates import (
    candidate_union_area_ratio, connected_components, generate_candidates, generate_random_candidates,
    region_recall_components,
)
from clip_anomaly.config import load_config
from clip_anomaly.training import load_trained_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "source_holdout"], default="dev")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_name = "train_dev_features.pt" if args.split == "dev" else "source_holdout_features.pt"
    payload = torch.load(config.output_root / f"feature_cache/{cache_name}", map_location="cpu")
    indices = [
        index for index, record in enumerate(payload["records"])
        if record["split"] == args.split and int(record["label"]) == 1
    ]
    candidate_config = dict(config.experiment["candidate"])
    random_seeds = list(map(int, config.experiment["random_seeds"]))
    sample_data = {}
    for source_index in indices:
        record = payload["records"][source_index]
        with Image.open(config.project_root / record["image_path"]) as image:
            image_size = image.size
        with Image.open(config.project_root / record["mask_path"]) as mask_image:
            mask = np.asarray(mask_image.convert("L")) > 0
        sample_data[source_index] = (image_size, mask.shape, connected_components(mask))
    rows = []
    for model_seed in map(int, config.experiment["training_seeds"]):
        checkpoint = config.output_root / f"e2/dev/b1_classifier/seed_{model_seed}/checkpoints/clip_b16_patch_head.pt"
        model = load_trained_head(checkpoint, payload["prototypes"])
        with torch.inference_mode():
            maps, _ = model(payload["features"][indices].float())
        for local_index, source_index in enumerate(indices):
            record = payload["records"][source_index]
            image_size, mask_shape, components = sample_data[source_index]
            subset = (
                "tiny" if float(record["mask_area_ratio"]) <= float(candidate_config["tiny_mask_area_ratio_max"])
                else "small" if float(record["mask_area_ratio"]) <= float(candidate_config["small_mask_area_ratio_max"])
                else "regular"
            )
            for k in map(int, config.experiment["k_values"]):
                auto = generate_candidates(maps[local_index].numpy(), image_size, k, candidate_config)
                recall = region_recall_components(
                    auto, components, mask_shape, float(candidate_config["min_instance_coverage"])
                )
                rows.append({
                    "model_seed": model_seed, "random_seed": "", "category": record["category"],
                    "defect_type": record["defect_type"], "image_path": record["image_path"],
                    "subset": subset, "method": "auto", "k": k,
                    "recalled_instances": recall.recalled_instances, "total_instances": recall.total_instances,
                    "gt_pixel_coverage": float(np.mean(recall.coverages)) if recall.coverages else 0.0,
                    "union_area_ratio": candidate_union_area_ratio(auto, image_size),
                    "duplicate": 0,
                    "candidate_count": len(auto),
                })
                for random_seed in random_seeds:
                    random_candidates = generate_random_candidates(
                        image_size, k, candidate_config, seed=random_seed + source_index * 1009
                    )
                    random_recall = region_recall_components(
                        random_candidates, components, mask_shape,
                        float(candidate_config["min_instance_coverage"]),
                    )
                    rows.append({
                        "model_seed": model_seed, "random_seed": random_seed, "category": record["category"],
                        "defect_type": record["defect_type"], "image_path": record["image_path"],
                        "subset": subset, "method": "random", "k": k,
                        "recalled_instances": random_recall.recalled_instances,
                        "total_instances": random_recall.total_instances,
                        "gt_pixel_coverage": float(np.mean(random_recall.coverages)) if random_recall.coverages else 0.0,
                        "union_area_ratio": candidate_union_area_ratio(random_candidates, image_size),
                        "duplicate": 0, "candidate_count": len(random_candidates),
                    })
    output = config.output_root / "e3" / args.split
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "candidate_recall.csv"
    with raw_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for model_seed in map(int, config.experiment["training_seeds"]):
        for method in ("auto", "random"):
            for k in map(int, config.experiment["k_values"]):
                group = [row for row in rows if row["model_seed"] == model_seed and row["method"] == method and row["k"] == k]
                if method == "random":
                    recalls = []
                    for random_seed in random_seeds:
                        seed_group = [row for row in group if row["random_seed"] == random_seed]
                        recalls.append(sum(row["recalled_instances"] for row in seed_group) / sum(row["total_instances"] for row in seed_group))
                    recall_mean = float(np.mean(recalls))
                    recall_low, recall_high = map(float, np.quantile(recalls, [0.025, 0.975]))
                else:
                    recall_mean = sum(row["recalled_instances"] for row in group) / sum(row["total_instances"] for row in group)
                    recall_low = recall_high = recall_mean
                summary.append({
                    "model_seed": model_seed, "method": method, "k": k,
                    "region_recall": recall_mean, "recall_interval_low": recall_low,
                    "recall_interval_high": recall_high,
                    "mean_gt_pixel_coverage": float(np.mean([row["gt_pixel_coverage"] for row in group])),
                    "mean_union_area_ratio": float(np.mean([row["union_area_ratio"] for row in group])),
                    "candidate_failures": sum(row["candidate_count"] != k for row in group),
                })
    summary_path = output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
