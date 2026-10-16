from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clip_anomaly.config import load_config
from clip_anomaly.experiment import feature_cache
from clip_anomaly.metrics import compute_detection_metrics
from clip_anomaly.training import load_trained_head
from clip_anomaly.visualization import save_anomaly_figure


def _metrics_for(indices, labels, scores, masks, pixel_scores):
    idx = np.asarray(indices, dtype=np.int64)
    return compute_detection_metrics(labels[idx], scores[idx], masks[idx], pixel_scores[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    payload = feature_cache(config).load()
    checkpoint = config.output_root / "experiments" / "checkpoints" / "clip_b16_patch_head.pt"
    model = load_trained_head(checkpoint, payload["prototypes"])
    val_indices = [index for index, record in enumerate(payload["records"]) if record["split"] == "val"]
    with torch.inference_mode():
        maps, scores = model(payload["features"][val_indices].float())
        upsampled = F.interpolate(maps[:, None], size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
    labels = payload["labels"][val_indices].numpy()
    masks = payload["full_masks"][val_indices].numpy().astype(np.uint8)
    score_values = scores.numpy()
    pixel_values = upsampled.numpy()
    records = [payload["records"][index] for index in val_indices]

    rows = []
    for category in sorted({record["category"] for record in records}):
        indices = [index for index, record in enumerate(records) if record["category"] == category]
        rows.append({"scope": category, **_metrics_for(indices, labels, score_values, masks, pixel_values), "samples": len(indices)})
    rows.append({"scope": "combined", **_metrics_for(range(len(records)), labels, score_values, masks, pixel_values), "samples": len(records)})
    result_path = config.output_root / "experiments" / "results.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "samples", "image_auroc", "image_ap", "pixel_auroc", "pixel_ap"])
        writer.writeheader()
        writer.writerows(rows)

    prediction_path = config.output_root / "experiments" / "predictions.csv"
    prediction_rows = []
    for record, label, score in zip(records, labels, score_values):
        prediction_rows.append({**record, "image_score": float(score), "result_type": "源域小样本探索性验证"})
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    normal_indices = [index for index, label in enumerate(labels) if label == 0]
    anomaly_indices = [index for index, label in enumerate(labels) if label == 1]
    chosen = {
        "normal_high_score": max(normal_indices, key=lambda index: score_values[index]),
        "anomaly_high_score": max(anomaly_indices, key=lambda index: score_values[index]),
        "anomaly_low_score": min(anomaly_indices, key=lambda index: score_values[index]),
        "tiny_anomaly": min(anomaly_indices, key=lambda index: records[index]["mask_area_ratio"]),
    }
    figure_dir = config.output_root / "experiments" / "figures"
    for name, index in chosen.items():
        record = records[index]
        save_anomaly_figure(
            config.project_root / record["image_path"],
            masks[index],
            pixel_values[index],
            figure_dir / f"{name}.png",
            f"{name} | {record['category']}/{record['defect_type']} | score={score_values[index]:.4f} | source-domain development",
        )
    for row in rows:
        print(row)
    print(result_path)


if __name__ == "__main__":
    main()
