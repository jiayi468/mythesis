from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image
import torch

from clip_anomaly.backbone import FrozenClipBackbone, load_openclip
from clip_anomaly.candidates import generate_candidates, generate_oracle_candidates, generate_random_candidates
from clip_anomaly.config import load_config
from clip_anomaly.preprocessing import preprocess_image
from clip_anomaly.training import load_trained_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "source_holdout"], default="dev")
    parser.add_argument("--unlock", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.split == "source_holdout":
        protocol = json.loads((config.output_root / "protocol.json").read_text(encoding="utf-8"))
        if not args.unlock or protocol.get("holdout_locked", True):
            raise RuntimeError("source_holdout ROI extraction requires unlocked protocol and --unlock")
        cache_name = "source_holdout_features.pt"
    else:
        cache_name = "train_dev_features.pt"
    payload = torch.load(config.output_root / f"feature_cache/{cache_name}", map_location="cpu")
    indices = [index for index, record in enumerate(payload["records"]) if record["split"] == args.split]
    candidate_config = dict(config.experiment["candidate"])
    max_k = max(map(int, config.experiment["k_values"]))
    random_seed = int(config.experiment["e4_random_seed"])
    tasks = []
    image_sizes, masks = {}, {}
    for source_index in indices:
        record = payload["records"][source_index]
        with Image.open(config.project_root / record["image_path"]) as image:
            image_sizes[source_index] = image.size
        if record["mask_path"]:
            with Image.open(config.project_root / record["mask_path"]) as mask:
                masks[source_index] = np.asarray(mask.convert("L")) > 0
    for model_seed in map(int, config.experiment["training_seeds"]):
        checkpoint = config.output_root / f"e2/dev/b1_classifier/seed_{model_seed}/checkpoints/clip_b16_patch_head.pt"
        model = load_trained_head(checkpoint, payload["prototypes"])
        with torch.inference_mode():
            maps, _ = model(payload["features"][indices].float())
        for local_index, source_index in enumerate(indices):
            for candidate in generate_candidates(
                maps[local_index].numpy(), image_sizes[source_index], max_k, candidate_config
            ):
                tasks.append((f"auto:{model_seed}:{source_index}:{candidate.rank}", source_index, candidate))
    for source_index in indices:
        if source_index in masks:
            for candidate in generate_oracle_candidates(
                masks[source_index], image_sizes[source_index], max_k, candidate_config
            ):
                tasks.append((f"oracle:{source_index}:{candidate.rank}", source_index, candidate))
        for candidate in generate_random_candidates(
            image_sizes[source_index], max_k, candidate_config, random_seed + source_index * 1009
        ):
            tasks.append((f"random:{random_seed}:{source_index}:{candidate.rank}", source_index, candidate))
    clip_model, _ = load_openclip(str(config.model["name"]), str(config.model["pretrained"]))
    backbone = FrozenClipBackbone(clip_model)
    feature_batches = []
    metadata = []
    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start:start + args.batch_size]
        tensors = []
        for key, source_index, candidate in batch:
            record = payload["records"][source_index]
            with Image.open(config.project_root / record["image_path"]) as image:
                crop = image.convert("RGB").crop((candidate.x1, candidate.y1, candidate.x2, candidate.y2))
                tensors.append(preprocess_image(crop, int(config.model["input_size"])))
            metadata.append({"key": key, "source_index": source_index, "candidate": candidate.__dict__})
        feature_batches.append(backbone(torch.stack(tensors)).half().cpu())
        print(f"cached ROI {min(start + args.batch_size, len(tasks))}/{len(tasks)}", flush=True)
    output = config.output_root / f"feature_cache/{args.split}_roi_features.pt"
    torch.save({"features": torch.cat(feature_batches), "metadata": metadata, "split": args.split}, output)
    tasks_path = config.output_root / f"e4/{args.split}_roi_tasks.json"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
