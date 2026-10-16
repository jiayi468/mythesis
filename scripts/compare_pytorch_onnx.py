from __future__ import annotations

import argparse
import csv

from PIL import Image
import torch

from clip_anomaly.backbone import FrozenClipBackbone, load_openclip
from clip_anomaly.config import load_config
from clip_anomaly.experiment import collect_selected_records, feature_cache
from clip_anomaly.export import FullAnomalyModel, compare_onnx
from clip_anomaly.preprocessing import preprocess_image
from clip_anomaly.training import load_trained_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    payload = feature_cache(config).load()
    clip_model, _ = load_openclip(str(config.model["name"]), str(config.model["pretrained"]))
    head = load_trained_head(config.output_root / "experiments/checkpoints/clip_b16_patch_head.pt", payload["prototypes"])
    model = FullAnomalyModel(FrozenClipBackbone(clip_model), head)
    records = [record for record in collect_selected_records(config) if record.split == "val"][: args.samples]
    images = []
    for record in records:
        with Image.open(record.image_path) as image:
            images.append(preprocess_image(image, int(config.model["input_size"])))
    rows = compare_onnx(model, config.output_root / "deployment/clip_b16_patch_head_b1.onnx", torch.stack(images))
    output = config.output_root / "deployment/pytorch_onnx_comparison.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    print(rows)


if __name__ == "__main__":
    main()
