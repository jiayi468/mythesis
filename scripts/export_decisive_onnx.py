from __future__ import annotations

import argparse
import json

import onnx
import torch

from clip_anomaly.backbone import FrozenClipBackbone, load_openclip
from clip_anomaly.config import load_config
from clip_anomaly.decisive import sha256_file
from clip_anomaly.export import FullAnomalyModel, export_static_onnx
from clip_anomaly.training import load_trained_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    cache = torch.load(config.output_root / "feature_cache/train_dev_features.pt", map_location="cpu")
    checkpoint = config.output_root / f"e2/dev/b1_classifier/seed_{args.seed}/checkpoints/clip_b16_patch_head.pt"
    clip_model, _ = load_openclip(str(config.model["name"]), str(config.model["pretrained"]))
    head = load_trained_head(checkpoint, cache["prototypes"])
    model = FullAnomalyModel(FrozenClipBackbone(clip_model), head)
    output = config.output_root / "deployment" / f"clip_b16_patch_head_b1_seed_{args.seed}.onnx"
    export_static_onnx(model, output)
    onnx.checker.check_model(onnx.load(str(output)))
    metadata = {
        "seed": args.seed, "path": output.relative_to(config.project_root).as_posix(),
        "sha256": sha256_file(output), "checkpoint_sha256": sha256_file(checkpoint),
        "input": "image float32 [1,3,224,224]",
        "outputs": ["anomaly_map float32 [1,14,14]", "image_score float32 [1]"],
    }
    (output.parent / "onnx_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
