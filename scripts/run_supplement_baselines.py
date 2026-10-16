from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clip_anomaly.config import load_config
from clip_anomaly.experiment import build_run_matrix, experiment_root, feature_cache, supplement_output
from clip_anomaly.metrics import compute_detection_metrics
from clip_anomaly.model import RawClipPatchHead
from clip_anomaly.training import load_trained_head, train_head


FIELDS = [
    "head_type", "seed", "scope", "samples", "trainable_parameters", "best_epoch",
    "best_val_loss", "image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "status", "error",
]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate(model, features, payload) -> list[dict[str, object]]:
    val_indices = [index for index, record in enumerate(payload["records"]) if record["split"] == "val"]
    with torch.inference_mode():
        maps, scores = model(features[val_indices].float())
        pixels = F.interpolate(maps[:, None], size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
    labels = payload["labels"][val_indices].numpy()
    masks = payload["full_masks"][val_indices].numpy().astype(np.uint8)
    records = [payload["records"][index] for index in val_indices]
    output = []
    scopes = sorted({record["category"] for record in records}) + ["combined"]
    for scope in scopes:
        indices = list(range(len(records))) if scope == "combined" else [
            index for index, record in enumerate(records) if record["category"] == scope
        ]
        idx = np.asarray(indices, dtype=np.int64)
        metrics = compute_detection_metrics(labels[idx], scores.numpy()[idx], masks[idx], pixels.numpy()[idx])
        output.append({"scope": scope, "samples": len(indices), **metrics})
    return output


def _summary(rows: list[dict[str, object]], path: Path) -> None:
    completed = [row for row in rows if row["status"] == "complete"]
    metrics = ["image_auroc", "image_ap", "pixel_auroc", "pixel_ap"]
    summary_rows = []
    for key in sorted({(str(row["head_type"]), str(row["scope"])) for row in completed}):
        group = [row for row in completed if (str(row["head_type"]), str(row["scope"])) == key]
        result = {"head_type": key[0], "scope": key[1], "runs": len(group)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(result)
    if not summary_rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = [int(seed) for seed in config.experiment.get("training_seeds", [config.seed])]
    matrix = build_run_matrix(seeds)
    cache_path = feature_cache(config).path
    print(f"split={experiment_root(config) / 'source_split.csv'}")
    print(f"cache={cache_path}")
    for head_type, seed in matrix:
        print(f"run={head_type} seed={seed}")
    if args.dry_run:
        return

    payload = feature_cache(config).load()
    baseline_root = experiment_root(config) / "baselines"
    runs_path = baseline_root / "runs.csv"
    rows: list[dict[str, object]] = []
    settings = config.training
    for head_type, seed in matrix:
        try:
            if head_type == "b0_raw_clip":
                model = RawClipPatchHead(payload["prototypes"])
                features = payload["projected_features"]
                best_epoch, best_loss = "", ""
                parameters = 0
            else:
                run_root = supplement_output(config.output_root, head_type, int(seed))
                result = train_head(
                    payload["features"], payload["masks"],
                    [record["split"] for record in payload["records"]], payload["prototypes"],
                    run_root / "checkpoints", int(config.model["hidden_dim"]), float(config.model["dropout"]),
                    int(settings["epochs"]), int(settings["batch_size"]), float(settings["learning_rate"]),
                    float(settings["weight_decay"]), int(settings["patience"]), float(settings["focal_gamma"]),
                    int(seed), head_type=head_type,
                )
                model = load_trained_head(result.checkpoint_path, payload["prototypes"])
                features = payload["features"]
                best_epoch, best_loss = result.best_epoch, result.best_val_loss
                parameters = sum(parameter.numel() for parameter in model.parameters())
            for metrics in _evaluate(model, features, payload):
                rows.append({
                    "head_type": head_type, "seed": "" if seed is None else seed,
                    "trainable_parameters": parameters, "best_epoch": best_epoch, "best_val_loss": best_loss,
                    **metrics, "status": "complete", "error": "",
                })
        except Exception as error:
            rows.append({
                "head_type": head_type, "seed": "" if seed is None else seed, "scope": "all", "samples": 0,
                "trainable_parameters": "", "best_epoch": "", "best_val_loss": "", "image_auroc": "",
                "image_ap": "", "pixel_auroc": "", "pixel_ap": "", "status": "failed", "error": repr(error),
            })
        _write_rows(runs_path, rows)
        _summary(rows, baseline_root / "summary.csv")
        print(f"finished {head_type} seed={seed} status={rows[-1]['status']}", flush=True)
    print(runs_path)


if __name__ == "__main__":
    main()
