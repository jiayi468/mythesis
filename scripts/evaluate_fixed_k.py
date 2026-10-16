from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from clip_anomaly.config import load_config
from clip_anomaly.metrics import compute_detection_metrics
from clip_anomaly.roi import fuse_maps, paste_roi_map
from clip_anomaly.training import load_trained_head


def _scaled_box(candidate: dict, width: int, height: int, target: int = 224):
    x1 = max(0, min(target - 1, round(candidate["x1"] * target / width)))
    y1 = max(0, min(target - 1, round(candidate["y1"] * target / height)))
    x2 = max(x1 + 1, min(target, round(candidate["x2"] * target / width)))
    y2 = max(y1 + 1, min(target, round(candidate["y2"] * target / height)))
    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "source_holdout"], default="dev")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_name = "train_dev_features.pt" if args.split == "dev" else "source_holdout_features.pt"
    roi_name = "dev_roi_features.pt" if args.split == "dev" else "source_holdout_roi_features.pt"
    payload = torch.load(config.output_root / f"feature_cache/{cache_name}", map_location="cpu")
    roi_payload = torch.load(config.output_root / f"feature_cache/{roi_name}", map_location="cpu")
    if roi_payload["split"] != args.split:
        raise ValueError("ROI cache split mismatch")
    indices = [index for index, record in enumerate(payload["records"]) if record["split"] == args.split]
    thresholds = json.loads((config.output_root / "e2/dev/thresholds.json").read_text(encoding="utf-8"))
    roi_index = {item["key"]: index for index, item in enumerate(roi_payload["metadata"])}
    random_seed = int(config.experiment["e4_random_seed"])
    prediction_rows, metric_rows = [], []
    for model_seed in map(int, config.experiment["training_seeds"]):
        model = load_trained_head(
            config.output_root / f"e2/dev/b1_classifier/seed_{model_seed}/checkpoints/clip_b16_patch_head.pt",
            payload["prototypes"],
        )
        with torch.inference_mode():
            full_maps, _ = model(payload["features"][indices].float())
            roi_maps, _ = model(roi_payload["features"].float())
            full_224 = F.interpolate(
                full_maps[:, None], (224, 224), mode="bilinear", align_corners=False
            )[:, 0].numpy()
        records = [payload["records"][index] for index in indices]
        labels = payload["labels"][indices].numpy()
        masks = payload["full_masks"][indices].numpy().astype(np.uint8)
        conditions = [("full", 0)] + [
            (method, k) for method in ("auto", "random", "oracle")
            for k in map(int, config.experiment["k_values"])
        ]
        for method, k in conditions:
            maps, scores = [], []
            for local_index, (source_index, record) in enumerate(zip(indices, records)):
                fused = full_224[local_index].copy()
                if method != "full":
                    width = int(record.get("image_width", 0))
                    height = int(record.get("image_height", 0))
                    if not width or not height:
                        from PIL import Image
                        with Image.open(config.project_root / record["image_path"]) as image:
                            width, height = image.size
                    for rank in range(1, k + 1):
                        if method == "auto":
                            key = f"auto:{model_seed}:{source_index}:{rank}"
                        elif method == "random":
                            key = f"random:{random_seed}:{source_index}:{rank}"
                        else:
                            key = f"oracle:{source_index}:{rank}"
                        if key not in roi_index:
                            continue
                        roi_position = roi_index[key]
                        candidate = roi_payload["metadata"][roi_position]["candidate"]
                        projected = paste_roi_map(
                            np.zeros((224, 224), dtype=np.float32),
                            roi_maps[roi_position].numpy(),
                            _scaled_box(candidate, width, height),
                        )
                        fused = fuse_maps(fused, projected)
                maps.append(fused)
                scores.append(float(np.quantile(fused, 0.995)))
            maps_array, scores_array = np.stack(maps), np.asarray(scores)
            for local_index, record in enumerate(records):
                pixel_ap = (
                    float(average_precision_score(masks[local_index].reshape(-1), maps_array[local_index].reshape(-1)))
                    if labels[local_index] == 1 else ""
                )
                prediction_rows.append({
                    "model_seed": model_seed, "method": method, "k": k,
                    "category": record["category"], "defect_type": record["defect_type"],
                    "image_path": record["image_path"], "label": int(labels[local_index]),
                    "image_score": float(scores_array[local_index]), "per_image_pixel_ap": pixel_ap,
                })
            for scope in sorted({record["category"] for record in records}) + ["combined"]:
                chosen = np.arange(len(records)) if scope == "combined" else np.asarray([
                    index for index, record in enumerate(records) if record["category"] == scope
                ])
                metrics = compute_detection_metrics(
                    labels[chosen], scores_array[chosen], masks[chosen], maps_array[chosen]
                )
                normal = chosen[labels[chosen] == 0]
                fpr = float(np.mean(scores_array[normal] > float(thresholds[str(model_seed)])))
                anomaly = chosen[labels[chosen] == 1]
                per_image_ap = float(np.mean([
                    average_precision_score(masks[index].reshape(-1), maps_array[index].reshape(-1))
                    for index in anomaly
                ]))
                metric_rows.append({
                    "model_seed": model_seed, "method": method, "k": k, "scope": scope,
                    "samples": len(chosen), **metrics, "fpr": fpr,
                    "mean_per_image_pixel_ap": per_image_ap,
                })
    output = config.output_root / "e4" / args.split
    output.mkdir(parents=True, exist_ok=True)
    for path, rows in ((output / "per_image.csv", prediction_rows), (output / "metrics.csv", metric_rows)):
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    combined = [row for row in metric_rows if row["scope"] == "combined"]
    eligible = []
    for k in map(int, config.experiment["k_values"]):
        auto = [row for row in combined if row["method"] == "auto" and row["k"] == k]
        full = {row["model_seed"]: row for row in combined if row["method"] == "full"}
        fpr_ok = all(row["fpr"] - full[row["model_seed"]]["fpr"] <= 0.02 for row in auto)
        eligible.append({
            "k": k, "fpr_ok": fpr_ok,
            "mean_pixel_ap": float(np.mean([row["pixel_ap"] for row in auto])),
            "mean_fpr_increase": float(np.mean([row["fpr"] - full[row["model_seed"]]["fpr"] for row in auto])),
        })
    valid = [item for item in eligible if item["fpr_ok"]]
    recommended = sorted(valid, key=lambda item: (-item["mean_pixel_ap"], item["k"]))[0]["k"] if valid else None
    recommendation = {
        "recommended_k": recommended,
        "selection_rule": "highest dev Pixel AP among K=1/2 that satisfies per-seed FPR increase <=0.02; tie chooses lower K",
        "candidates": eligible,
        "random_seed_for_e4_precision": random_seed,
    }
    (output / "recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
