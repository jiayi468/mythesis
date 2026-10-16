from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from clip_anomaly.config import load_config
from clip_anomaly.experiment import experiment_root, feature_cache, supplement_output
from clip_anomaly.model import aggregate_image_scores
from clip_anomaly.training import load_trained_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    payload = feature_cache(config).load()
    val_indices = [index for index, record in enumerate(payload["records"]) if record["split"] == "val"]
    labels = payload["labels"][val_indices].numpy()
    records = [payload["records"][index] for index in val_indices]
    methods = [("max", {}), ("quantile_0_995", {"method": "quantile", "quantile": 0.995}),
               ("top5_mean", {"method": "topk_mean", "top_k": 5})]
    rows = []
    for seed in [int(value) for value in config.experiment["training_seeds"]]:
        checkpoint = supplement_output(config.output_root, "b2_text_prototype", seed) / "checkpoints" / "clip_b16_patch_head.pt"
        model = load_trained_head(checkpoint, payload["prototypes"])
        with torch.inference_mode():
            maps, _ = model(payload["features"][val_indices].float())
        for display_name, options in methods:
            method = options.pop("method", display_name)
            scores = aggregate_image_scores(maps, method, **options).numpy()
            options["method"] = method
            for scope in sorted({record["category"] for record in records}) + ["combined"]:
                idx = np.asarray(
                    list(range(len(records))) if scope == "combined" else
                    [index for index, record in enumerate(records) if record["category"] == scope],
                    dtype=np.int64,
                )
                rows.append({
                    "seed": seed, "scope": scope, "method": display_name, "samples": len(idx),
                    "image_auroc": float(roc_auc_score(labels[idx], scores[idx])),
                    "image_ap": float(average_precision_score(labels[idx], scores[idx])),
                })
    output = experiment_root(config) / "score_aggregation.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
