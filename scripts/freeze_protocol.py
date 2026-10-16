from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from clip_anomaly.config import load_config
from clip_anomaly.decisive import sha256_file


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def _code_snapshot(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted([*root.glob("src/**/*.py"), *root.glob("scripts/**/*.py"), *root.glob("web/*"), root / "configs/decisive_path.yaml"]):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--relock", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config.output_root
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("holdout_locked", False):
        if not args.relock:
            raise ValueError("protocol is already unlocked")
        protocol["holdout_locked"] = True
        protocol["holdout_opened_at"] = None
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("relocked")
        return
    recommendation = json.loads((root / "e4/dev/recommendation.json").read_text(encoding="utf-8"))
    if recommendation.get("recommended_k") is None:
        # K is selected by dev Pixel AP; the FPR gate is a formal holdout criterion.
        recommendation["recommended_k"] = max(
            recommendation["candidates"], key=lambda item: (item["mean_pixel_ap"], -item["k"])
        )["k"]
    protocol["selection"]["recommended_k"] = int(recommendation["recommended_k"])
    protocol["selection"]["recommended_k_rule"] = "highest combined dev Pixel AP among K=1/2; formal FPR gate evaluated once on source_holdout"
    protocol["selection"]["e4_random_seed"] = int(config.experiment["e4_random_seed"])
    protocol["code_snapshot_sha256"] = _code_snapshot(config.project_root)
    protocol["model_manifest_sha256"] = sha256_file(root / "model_manifest.json")
    protocol["onnx_manifest_sha256"] = sha256_file(root / "deployment/onnx_manifest.json")
    protocol["holdout_locked"] = False
    protocol["holdout_opened_at"] = _now()
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"recommended_k": protocol["selection"]["recommended_k"], "code_snapshot_sha256": protocol["code_snapshot_sha256"], "holdout_opened_at": protocol["holdout_opened_at"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
