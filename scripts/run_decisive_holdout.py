from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from clip_anomaly.config import load_config
from clip_anomaly.decisive import build_result, sha256_file, write_result
from clip_anomaly.metrics import compute_detection_metrics
from clip_anomaly.model import RawClipPatchHead
from clip_anomaly.training import load_trained_head


def _now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def _metrics(model, features, payload, indices):
    with torch.inference_mode():
        maps, scores = model(features[indices].float())
        upsampled = F.interpolate(maps[:, None], (224, 224), mode="bilinear", align_corners=False)[:, 0]
    labels = payload["labels"][indices].numpy(); masks = payload["full_masks"][indices].numpy().astype(np.uint8)
    return compute_detection_metrics(labels, scores.numpy(), masks, upsampled.numpy())


def _bootstrap_pixel_ap(b0_maps, b1_maps, masks, labels, categories, seed, samples=2000):
    rng = np.random.default_rng(seed)
    groups = {}
    for index, (label, category) in enumerate(zip(labels, categories)):
        groups.setdefault((int(label), category), []).append(index)
    deltas = []
    for _ in range(samples):
        chosen = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in groups.values()])
        target = masks[chosen].reshape(-1)
        b0 = b0_maps[chosen].reshape(-1); b1 = b1_maps[chosen].reshape(-1)
        deltas.append(float(average_precision_score(target, b1)) - float(average_precision_score(target, b0)))
    values = np.asarray(deltas)
    return {"samples": samples, "mean": float(values.mean()), "low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975)), "positive_fraction": float(np.mean(values > 0)), "resolution": "14x14 patch map"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); started = _now(); root = config.output_root
    payload = torch.load(root / "feature_cache/source_holdout_features.pt", map_location="cpu")
    indices = list(range(len(payload["records"])))
    b0 = RawClipPatchHead(payload["prototypes"])
    rows = []; b0_maps = b0_scores = None
    with torch.inference_mode():
        b0_maps, b0_scores = b0(payload["projected_features"].float())
    labels = payload["labels"].numpy(); masks = payload["masks"].numpy().astype(np.uint8)
    categories = np.asarray([record["category"] for record in payload["records"]])
    b0_metrics = _metrics(b0, payload["projected_features"], payload, indices)
    rows.append({"head_type": "b0_raw_clip", "seed": "", "scope": "combined", "samples": len(indices), **b0_metrics})
    for category in sorted(set(categories)):
        chosen = np.where(categories == category)[0].tolist()
        rows.append({"head_type": "b0_raw_clip", "seed": "", "scope": category, "samples": len(chosen), **_metrics(b0, payload["projected_features"], payload, chosen)})
    bootstrap = {}
    for seed in map(int, config.experiment["training_seeds"]):
        model = load_trained_head(root / f"e2/dev/b1_classifier/seed_{seed}/checkpoints/clip_b16_patch_head.pt", payload["prototypes"])
        metrics = _metrics(model, payload["features"], payload, indices)
        rows.append({"head_type": "b1_classifier", "seed": seed, "scope": "combined", "samples": len(indices), **metrics})
        with torch.inference_mode():
            maps, scores = model(payload["features"].float())
        bootstrap[str(seed)] = _bootstrap_pixel_ap(
            b0_maps.numpy(), maps.numpy(), masks, labels, categories, seed=seed,
            samples=int(config.experiment.get("bootstrap_samples", 2000)),
        )
        for category in sorted(set(categories)):
            chosen = np.where(categories == category)[0].tolist()
            rows.append({"head_type": "b1_classifier", "seed": seed, "scope": category, "samples": len(chosen), **_metrics(model, payload["features"], payload, chosen)})
    metrics_path = root / "e2/holdout/metrics.csv"; metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root / "e2/holdout/bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
    b0_row = b0_metrics
    b1_rows = [row for row in rows if row["head_type"] == "b1_classifier" and row["scope"] == "combined"]
    pixel_gains = [float(row["pixel_ap"]) - b0_row["pixel_ap"] for row in b1_rows]
    image_gains = [float(row["image_auroc"]) - b0_row["image_auroc"] for row in b1_rows]
    category_gains = {}
    for category in sorted(set(categories)):
        b0_category = next(r for r in rows if r["head_type"] == "b0_raw_clip" and r["scope"] == category)
        category_gains[category] = [
            float(next(r for r in rows if r["head_type"] == "b1_classifier" and r["seed"] == row["seed"] and r["scope"] == category)["pixel_ap"]) - float(b0_category["pixel_ap"])
            for row in b1_rows
        ]
    criteria = {
        "all_b1_pixel_ap_above_b0": all(gain > 0 for gain in pixel_gains),
        "mean_pixel_ap_gain_ge_0.02": float(np.mean(pixel_gains)) >= 0.02,
        "mean_image_auroc_gain_ge_0.02": float(np.mean(image_gains)) >= 0.02,
        "category_pixel_ap_gains_positive": all(all(gain > 0 for gain in gains) for gains in category_gains.values()),
        "bootstrap_positive_fraction_ge_0.90": all(item["positive_fraction"] >= 0.90 for item in bootstrap.values()),
    }
    result = build_result(
        criteria=criteria,
        observed={"b0": b0_row, "pixel_ap_gains": pixel_gains, "image_auroc_gains": image_gains, "bootstrap": bootstrap},
        artifact_paths=[metrics_path.relative_to(config.project_root).as_posix(), (root / "e2/holdout/bootstrap.json").relative_to(config.project_root).as_posix()],
        protocol_hash=sha256_file(root / "protocol.json"), git_commit="working-tree", started_at=started, completed_at=_now(),
        planned=4, succeeded=4, failed=0, skipped=0, failure_reasons=[], criteria_passed=all(criteria.values()),
    )
    write_result(root / "e2/result.json", result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
