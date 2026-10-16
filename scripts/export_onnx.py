from __future__ import annotations

import argparse
import traceback

import onnx

from clip_anomaly.backbone import FrozenClipBackbone, load_openclip
from clip_anomaly.config import load_config
from clip_anomaly.experiment import feature_cache
from clip_anomaly.export import FullAnomalyModel, export_static_onnx
from clip_anomaly.training import load_trained_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    log_path = config.output_root / "deployment" / "export_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = feature_cache(config).load()
        clip_model, _ = load_openclip(str(config.model["name"]), str(config.model["pretrained"]))
        head = load_trained_head(config.output_root / "experiments/checkpoints/clip_b16_patch_head.pt", payload["prototypes"])
        model = FullAnomalyModel(FrozenClipBackbone(clip_model), head)
        output = export_static_onnx(model, config.output_root / "deployment/clip_b16_patch_head_b1.onnx")
        onnx.checker.check_model(onnx.load(str(output)))
        message = f"SUCCESS\nmodel={output}\ninput=image float32 [1,3,224,224]\noutputs=anomaly_map [1,14,14], image_score [1]\n"
        log_path.write_text(message, encoding="utf-8")
        print(message)
    except Exception:
        message = "FAILED\n" + traceback.format_exc()
        log_path.write_text(message, encoding="utf-8")
        print(message)
        raise


if __name__ == "__main__":
    main()
