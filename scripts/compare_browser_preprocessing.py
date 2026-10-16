from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort

from clip_anomaly.config import load_config
from clip_anomaly.decisive import compare_arrays


def compare_tensors(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return compare_arrays(reference, candidate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--browser-tensor", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    deployment = config.output_root / "deployment"
    reference_path = deployment / "webgpu_reference_input.bin"
    browser_path = Path(args.browser_tensor)
    reference = np.fromfile(reference_path, dtype="<f4").reshape(1, 3, 224, 224)
    browser = np.fromfile(browser_path, dtype="<f4").reshape(1, 3, 224, 224)
    tensor_errors = compare_tensors(reference, browser)

    model_path = deployment / "clip_b16_patch_head_b1.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    _, reference_score = session.run(None, {"image": reference})
    _, browser_score = session.run(None, {"image": browser})
    score_error = abs(float(reference_score[0]) - float(browser_score[0]))
    fixed_webgpu_score = 0.167000
    fixed_backend_error = abs(float(reference_score[0]) - fixed_webgpu_score)
    rows = [
        {"comparison": "python_vs_canvas_input", **tensor_errors, "score_abs_error": "",
         "threshold": 0.001, "passed": tensor_errors["mean_abs_error"] <= 0.001},
        {"comparison": "python_vs_canvas_onnx_cpu_score", "max_abs_error": "", "mean_abs_error": "",
         "score_abs_error": score_error, "threshold": 0.001, "passed": score_error <= 0.001},
        {"comparison": "onnx_cpu_vs_webgpu_fixed_score", "max_abs_error": "", "mean_abs_error": "",
         "score_abs_error": fixed_backend_error, "threshold": 0.0001, "passed": fixed_backend_error <= 0.0001},
    ]
    output = deployment / "preprocessing_consistency.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"reference_score={float(reference_score[0]):.9f}")
    print(f"browser_score={float(browser_score[0]):.9f}")
    print(output)


if __name__ == "__main__":
    main()
