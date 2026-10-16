from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clip_anomaly.config import load_config
from clip_anomaly.decisive import sha256_file
from clip_anomaly.metrics import compute_detection_metrics
from clip_anomaly.model import RawClipPatchHead
from clip_anomaly.training import load_trained_head, train_head


def _evaluate(model, features, payload, indices):
    with torch.inference_mode():
        maps, scores = model(features[indices].float())
        pixels = F.interpolate(maps[:, None], (224, 224), mode="bilinear", align_corners=False)[:, 0]
    labels = payload["labels"][indices].numpy()
    masks = payload["full_masks"][indices].numpy().astype(np.uint8)
    records = [payload["records"][index] for index in indices]
    score_values, pixel_values = scores.numpy(), pixels.numpy()
    rows = []
    for scope in sorted({record["category"] for record in records}) + ["combined"]:
        chosen = np.arange(len(records)) if scope == "combined" else np.asarray([
            index for index, record in enumerate(records) if record["category"] == scope
        ])
        rows.append({"scope": scope, "samples": len(chosen), **compute_detection_metrics(
            labels[chosen], score_values[chosen], masks[chosen], pixel_values[chosen]
        )})
    normal_scores = score_values[labels == 0]
    threshold = float(np.quantile(normal_scores, 0.95, method="higher"))
    return rows, threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    cache_path = config.output_root / "feature_cache" / "train_dev_features.pt"
    payload = torch.load(cache_path, map_location="cpu")
    if set(payload["included_splits"]) != {"dev", "train"}:
        raise ValueError("decisive dev cache must contain only train and dev")
    splits = [record["split"] for record in payload["records"]]
    if "source_holdout" in splits:
        raise ValueError("source_holdout contamination detected")
    dev_indices = [index for index, split in enumerate(splits) if split == "dev"]
    output = config.output_root / "e2" / "dev"
    output.mkdir(parents=True, exist_ok=True)
    rows, thresholds, manifest = [], {}, []
    baseline = RawClipPatchHead(payload["prototypes"])
    metrics, threshold = _evaluate(baseline, payload["projected_features"], payload, dev_indices)
    thresholds["b0_raw_clip"] = threshold
    for item in metrics:
        rows.append({"head_type": "b0_raw_clip", "seed": "", **item})
    settings = config.training
    for seed in map(int, config.experiment["training_seeds"]):
        run_root = output / "b1_classifier" / f"seed_{seed}"
        result = train_head(
            payload["features"], payload["masks"], splits, payload["prototypes"],
            run_root / "checkpoints", int(config.model["hidden_dim"]), float(config.model["dropout"]),
            int(settings["epochs"]), int(settings["batch_size"]), float(settings["learning_rate"]),
            float(settings["weight_decay"]), int(settings["patience"]), float(settings["focal_gamma"]),
            seed, head_type="b1_classifier",
        )
        model = load_trained_head(result.checkpoint_path, payload["prototypes"])
        metrics, threshold = _evaluate(model, payload["features"], payload, dev_indices)
        thresholds[str(seed)] = threshold
        for item in metrics:
            rows.append({"head_type": "b1_classifier", "seed": seed, **item})
        manifest.append({
            "seed": seed,
            "path": result.checkpoint_path.relative_to(config.project_root).as_posix(),
            "sha256": sha256_file(result.checkpoint_path),
            "best_epoch": result.best_epoch,
            "best_dev_loss": result.best_val_loss,
        })
        print(f"trained seed={seed} best_epoch={result.best_epoch} best_dev_loss={result.best_val_loss:.6f}", flush=True)
    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    (config.output_root / "model_manifest.json").write_text(
        json.dumps({"models": manifest, "feature_cache_sha256": sha256_file(cache_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics_path)


if __name__ == "__main__":
    main()
