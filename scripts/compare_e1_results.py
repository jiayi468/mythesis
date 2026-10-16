from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

from clip_anomaly.backbone import FrozenClipBackbone, load_openclip
from clip_anomaly.config import load_config
from clip_anomaly.decisive import build_result, compare_arrays, sha256_file, write_result
from clip_anomaly.export import FullAnomalyModel
from clip_anomaly.training import load_trained_head


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def _read(path: Path, dtype, shape=None):
    values = np.fromfile(path, dtype=dtype)
    return values.reshape(shape) if shape else values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    started = _now()
    root = config.output_root
    records = json.loads((root / "e1_samples.json").read_text(encoding="utf-8"))
    model_path = root / "deployment/clip_b16_patch_head_b1_seed_42.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    cache = torch.load(root / "feature_cache/train_dev_features.pt", map_location="cpu")
    clip_model, _ = load_openclip(str(config.model["name"]), str(config.model["pretrained"]))
    head = load_trained_head(root / "e2/dev/b1_classifier/seed_42/checkpoints/clip_b16_patch_head.pt", cache["prototypes"])
    pytorch_model = FullAnomalyModel(FrozenClipBackbone(clip_model), head).eval()
    rows, failures = [], []
    python_scores, real_scores = [], []
    fixed_cpu_scores, fixed_webgpu_scores, fixed_wasm_scores = [], [], []
    for index, record in enumerate(records):
        sample_id = f"{index:02d}_{record['category']}_{record['defect_type']}_{Path(record['image_path']).stem}"
        sample_root = root / "e1/captures"
        python_dir = sample_root / "python" / sample_id
        real_dir = sample_root / "real/webgpu" / sample_id
        fixed_gpu_dir = sample_root / "fixed/webgpu" / sample_id
        fixed_wasm_dir = sample_root / "fixed/wasm" / sample_id
        try:
            reference = _read(python_dir / "normalized_chw.bin", "<f4", (1, 3, 224, 224))
            python_rgba = _read(python_dir / "decoded_rgba.bin", "u1")
            browser_rgba = _read(real_dir / "decoded_rgba.bin", "u1")
            rgba_error = compare_arrays(python_rgba, browser_rgba)
            python_rgb = _read(python_dir / "resized_rgb.bin", "u1")
            browser_rgb = _read(real_dir / "resized_rgb.bin", "u1")
            rgb_error = compare_arrays(python_rgb, browser_rgb)
            browser_input = _read(real_dir / "normalized_chw.bin", "<f4", (1, 3, 224, 224))
            input_error = compare_arrays(reference, browser_input)
            fixed_input = _read(fixed_gpu_dir / "normalized_chw.bin", "<f4", (1, 3, 224, 224))
            fixed_input_error = compare_arrays(reference, fixed_input)
            cpu_map, cpu_score = session.run(None, {"image": reference})
            real_map = _read(real_dir / "anomaly_map.bin", "<f4", (1, 14, 14))
            real_score = float(_read(real_dir / "image_score.bin", "<f4")[0])
            gpu_map = _read(fixed_gpu_dir / "anomaly_map.bin", "<f4", (1, 14, 14))
            gpu_score = float(_read(fixed_gpu_dir / "image_score.bin", "<f4")[0])
            wasm_map = _read(fixed_wasm_dir / "anomaly_map.bin", "<f4", (1, 14, 14))
            wasm_score = float(_read(fixed_wasm_dir / "image_score.bin", "<f4")[0])
            with torch.inference_mode():
                torch_map, torch_score = pytorch_model(torch.from_numpy(reference))
            torch_map = torch_map.numpy()
            torch_score = float(torch_score.numpy()[0])
            torch_onnx_map = compare_arrays(torch_map, cpu_map)
            torch_onnx_score = abs(torch_score - float(cpu_score[0]))
            fixed_gpu_map = compare_arrays(cpu_map, gpu_map)
            fixed_gpu_score = abs(float(cpu_score[0]) - gpu_score)
            fixed_wasm_map = compare_arrays(cpu_map, wasm_map)
            fixed_wasm_score = abs(float(cpu_score[0]) - wasm_score)
            real_map_error = compare_arrays(torch_map, real_map)
            real_score_error = abs(torch_score - real_score)
            rows.append({
                "sample_id": sample_id, "category": record["category"], "label": record["label"],
                "rgba_mean_abs": rgba_error["mean_abs_error"], "rgba_max_abs": rgba_error["max_abs_error"],
                "resized_rgb_mean_abs": rgb_error["mean_abs_error"], "resized_rgb_max_abs": rgb_error["max_abs_error"],
                "normalized_mean_abs": input_error["mean_abs_error"], "normalized_max_abs": input_error["max_abs_error"],
                "torch_onnx_map_mean_abs": torch_onnx_map["mean_abs_error"],
                "torch_onnx_score_abs": torch_onnx_score,
                "cpu_webgpu_map_mean_abs": fixed_gpu_map["mean_abs_error"],
                "cpu_webgpu_score_abs": fixed_gpu_score,
                "cpu_wasm_map_mean_abs": fixed_wasm_map["mean_abs_error"],
                "cpu_wasm_score_abs": fixed_wasm_score,
                "real_input_mean_abs": input_error["mean_abs_error"],
                "real_webgpu_map_mean_abs": real_map_error["mean_abs_error"],
                "real_webgpu_score_abs": real_score_error,
                "python_score": torch_score, "real_webgpu_score": real_score,
                "fixed_cpu_score": float(cpu_score[0]), "fixed_webgpu_score": gpu_score,
                "fixed_wasm_score": wasm_score,
            })
            python_scores.append(torch_score); real_scores.append(real_score)
            fixed_cpu_scores.append(float(cpu_score[0])); fixed_webgpu_scores.append(gpu_score); fixed_wasm_scores.append(wasm_score)
        except Exception as error:
            failures.append(f"{sample_id}: {error!r}")
    rows_path = root / "e1/per_sample.csv"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with rows_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    thresholds = {
        "input_mean_abs": 1e-3, "backend_score_abs": 1e-4, "backend_map_mean_abs": 1e-4,
        "end_to_end_score_abs": 1e-3, "end_to_end_map_mean_abs": 1e-3,
    }
    complete = len(rows) == len(records) and not failures
    fixed_ok = complete and all(
        max(float(row["normalized_mean_abs"]) for row in rows) <= thresholds["input_mean_abs"]
        and max(float(row["torch_onnx_score_abs"]) for row in rows) <= thresholds["backend_score_abs"]
        and max(float(row["torch_onnx_map_mean_abs"]) for row in rows) <= thresholds["backend_map_mean_abs"]
        and max(float(row["cpu_webgpu_score_abs"]) for row in rows) <= thresholds["backend_score_abs"]
        and max(float(row["cpu_webgpu_map_mean_abs"]) for row in rows) <= thresholds["backend_map_mean_abs"]
        for _ in [0]
    )
    real_ok = fixed_ok and max(float(row["real_webgpu_score_abs"]) for row in rows) <= thresholds["end_to_end_score_abs"] and max(float(row["real_webgpu_map_mean_abs"]) for row in rows) <= thresholds["end_to_end_map_mean_abs"]
    ranking_ok = sorted(range(len(python_scores)), key=lambda i: python_scores[i]) == sorted(range(len(real_scores)), key=lambda i: real_scores[i])
    resolution = "FIXED" if real_ok and ranking_ok else "ISOLATED" if fixed_ok else "INCOMPLETE" if not complete else "ISOLATED"
    criteria_passed = resolution == "FIXED" and ranking_ok
    result = build_result(
        criteria={"thresholds": thresholds, "resolution": resolution, "ranking_preserved": ranking_ok},
        observed={"samples": len(rows), "fixed_ok": fixed_ok, "real_ok": real_ok,
                  "max_normalized_mean_abs": max((float(row["normalized_mean_abs"]) for row in rows), default=None),
                  "max_cpu_webgpu_score_abs": max((float(row["cpu_webgpu_score_abs"]) for row in rows), default=None),
                  "max_cpu_webgpu_map_mean_abs": max((float(row["cpu_webgpu_map_mean_abs"]) for row in rows), default=None),
                  "max_real_webgpu_score_abs": max((float(row["real_webgpu_score_abs"]) for row in rows), default=None),
                  "max_real_webgpu_map_mean_abs": max((float(row["real_webgpu_map_mean_abs"]) for row in rows), default=None)},
        artifact_paths=[rows_path.relative_to(config.project_root).as_posix(), (root / "e1/capture_summary.json").relative_to(config.project_root).as_posix()],
        protocol_hash=sha256_file(root / "protocol.json"), git_commit="working-tree",
        started_at=started, completed_at=_now(), planned=len(records), succeeded=len(rows),
        failed=len(failures), skipped=0, failure_reasons=failures, criteria_passed=criteria_passed,
    )
    result["resolution"] = resolution
    write_result(root / "e1/result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
