from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from clip_anomaly.backbone import FrozenClipBackbone, load_openclip
from clip_anomaly.config import load_config
from clip_anomaly.preprocessing import downsample_mask, preprocess_image
from clip_anomaly.prompts import build_text_prototypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    protocol = json.loads((config.output_root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("holdout_locked", True):
        raise RuntimeError("source_holdout remains locked; run freeze_protocol.py first")
    manifest = json.loads((config.output_root / "data_manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest if record["split"] == "source_holdout"]
    size, grid = int(config.model["input_size"]), int(config.model["patch_grid"])
    clip_model, tokenizer = load_openclip(str(config.model["name"]), str(config.model["pretrained"]))
    backbone = FrozenClipBackbone(clip_model)
    prototypes = build_text_prototypes(clip_model, tokenizer)
    features, projected, masks, full_masks = [], [], [], []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        images = []
        for record in batch:
            with Image.open(config.project_root / record["image_path"]) as image:
                images.append(preprocess_image(image, size))
            if record["mask_path"]:
                with Image.open(config.project_root / record["mask_path"]) as mask_image:
                    full = (np.asarray(mask_image.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0).astype(np.uint8)
            else:
                full = np.zeros((size, size), dtype=np.uint8)
            full_masks.append(torch.from_numpy(full)); masks.append(downsample_mask(full, grid))
        encoded = backbone(torch.stack(images))
        features.append(encoded.half().cpu()); projected.append(backbone.project_tokens(encoded).half().cpu())
        print(f"cached holdout {min(start + args.batch_size, len(records))}/{len(records)}", flush=True)
    output = config.output_root / "feature_cache/source_holdout_features.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": torch.cat(features), "projected_features": torch.cat(projected),
        "masks": torch.stack(masks), "full_masks": torch.stack(full_masks),
        "labels": torch.tensor([record["label"] for record in records], dtype=torch.long),
        "prototypes": prototypes.cpu(), "records": records, "included_splits": ["source_holdout"],
    }, output)
    print(output)


if __name__ == "__main__":
    main()
