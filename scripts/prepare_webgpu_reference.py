from __future__ import annotations

import argparse
import json

import numpy as np
import onnxruntime as ort
from PIL import Image

from clip_anomaly.config import load_config
from clip_anomaly.preprocessing import preprocess_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    image_path = (config.project_root / args.image).resolve()
    with Image.open(image_path) as image:
        tensor = preprocess_image(image, int(config.model["input_size"]))[None].numpy().astype(np.float32)
    deployment = config.output_root / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    tensor_path = deployment / "webgpu_reference_input.bin"
    tensor.tofile(tensor_path)
    model_path = deployment / "clip_b16_patch_head_b1.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    anomaly_map, image_score = session.run(None, {"image": tensor})
    metadata = {
        "source_image": args.image,
        "tensor_shape": [1, 3, 224, 224],
        "tensor_dtype": "float32 little-endian",
        "onnx_cpu_image_score": float(image_score[0]),
        "onnx_cpu_map_min": float(anomaly_map.min()),
        "onnx_cpu_map_max": float(anomaly_map.max()),
    }
    metadata_path = deployment / "webgpu_reference.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(tensor_path)
    print(metadata_path)
    print(metadata)


if __name__ == "__main__":
    main()
