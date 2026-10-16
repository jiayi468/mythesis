from __future__ import annotations

import json
import platform
from pathlib import Path
import subprocess
import sys

import onnxruntime
import torch

from clip_anomaly.config import load_config
from clip_anomaly.data import decisive_split_records, scan_mvtec_category
from clip_anomaly.decisive import sha256_file, write_data_manifest


def _git(project_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()


def _chrome_version(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return str(path.stat()) if False else str(path)
    except OSError:
        return None


def _select_e1(records) -> list:
    selected = []
    for category in sorted({record.category for record in records}):
        dev = [record for record in records if record.category == category and record.split == "dev"]
        normal = sorted((record for record in dev if record.label == 0), key=lambda item: item.image_path.name)
        abnormal = sorted((record for record in dev if record.label == 1), key=lambda item: item.mask_area_ratio)
        if len(normal) < 2 or len(abnormal) < 3:
            raise ValueError(f"{category} lacks the required dev samples for E1")
        selected.extend([normal[0], normal[-1], abnormal[0], abnormal[len(abnormal) // 2], abnormal[-1]])
    return selected


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    records = []
    for root in config.dataset_roots:
        records.extend(scan_mvtec_category(root))
    records = decisive_split_records(
        records,
        config.seed,
        float(config.split["abnormal_train_fraction"]),
        float(config.split["dev_fraction"]),
    )
    output = config.output_root
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = write_data_manifest(records, config.project_root, output / "data_manifest.json")
    e1_manifest_path = write_data_manifest(_select_e1(records), config.project_root, output / "e1_samples.json")
    commit = _git(config.project_root, "rev-parse", "HEAD")
    dirty = bool(_git(config.project_root, "status", "--porcelain"))
    protocol = {
        "version": 1,
        "created_at": "2026-10-16",
        "config_path": config_path.relative_to(config.project_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "data_manifest": manifest_path.relative_to(config.project_root).as_posix(),
        "data_manifest_sha256": sha256_file(manifest_path),
        "e1_samples": e1_manifest_path.relative_to(config.project_root).as_posix(),
        "e1_samples_sha256": sha256_file(e1_manifest_path),
        "git_commit": commit,
        "git_dirty_at_freeze": dirty,
        "holdout_locked": True,
        "holdout_opened_at": None,
        "thresholds": {
            "e1_input_mean_abs": 1e-3,
            "e1_backend_score_and_map_abs": 1e-4,
            "e1_end_to_end_score_and_map_abs": 1e-3,
            "e2_mean_pixel_ap_gain": 0.02,
            "e2_mean_image_auroc_gain": 0.02,
            "e3_oracle_pixel_ap_gain": 0.02,
            "e3_recall_at_2": 0.60,
            "e3_random_recall_gain": 0.15,
            "e4_pixel_ap_gain": 0.02,
            "e4_oracle_gain_fraction": 0.40,
            "e4_fpr_absolute_increase": 0.02,
        },
        "selection": {
            "main_model": "b1_classifier",
            "auto_method": "peak_nms",
            "recommended_k": None,
            "recommended_k_rule": "highest dev Pixel AP among K=1/2 that satisfies FPR; tie chooses lower K",
        },
        "config": {
            "split": config.split,
            "model": config.model,
            "training": config.training,
            "experiment": config.experiment,
        },
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    environment = {
        "os": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "onnxruntime_providers": onnxruntime.get_available_providers(),
        "chrome_path": str(chrome),
        "chrome_version": str(chrome.stat().st_mtime_ns) if chrome.is_file() else None,
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    split_counts = {}
    for record in records:
        key = f"{record.category}:{record.split}:{record.label}"
        split_counts[key] = split_counts.get(key, 0) + 1
    (output / "split_counts.json").write_text(
        json.dumps(split_counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(protocol_path)
    print(json.dumps(split_counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
