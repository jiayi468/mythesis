from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np
from PIL import Image

from clip_anomaly.preprocessing import preprocess_image


STAGES = {"decoded_rgba", "resized_rgb", "normalized_chw", "anomaly_map", "image_score", "timings"}


def _capture_one(project_root: Path, chrome: Path, record: dict, sample_index: int, backend: str,
                 mode: str, model_url: str, output_root: Path, port: int, timeout: int) -> dict:
    sample_id = f"{sample_index:02d}_{record['category']}_{record['defect_type']}_{Path(record['image_path']).stem}"
    output = output_root / mode / backend / sample_id
    output.mkdir(parents=True, exist_ok=True)
    received, metadata, errors = set(), {}, []
    completed = threading.Event()

    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/capture":
                self.send_error(404); return
            stage = parse_qs(parsed.query).get("stage", [""])[0]
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            suffix = ".json" if stage == "timings" else ".txt" if stage == "error" else ".bin"
            (output / f"{stage}{suffix}").write_bytes(body)
            metadata[stage] = {
                "bytes": len(body), "width": self.headers.get("X-Width"),
                "height": self.headers.get("X-Height"), "dtype": self.headers.get("X-Dtype"),
                "sample_id": self.headers.get("X-Sample-Id"),
            }
            received.add(stage)
            if stage == "error":
                errors.append(body.decode("utf-8", errors="replace")); completed.set()
            elif STAGES.issubset(received):
                completed.set()
            self.send_response(204); self.end_headers()

        def log_message(self, format, *values):
            print(f"[{sample_id} {mode} {backend}] " + format % values, flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    query = {
        "autorun": "1", "sample": "/" + record["image_path"], "capture": "/capture",
        "captureMode": "full", "sampleId": sample_id, "backend": backend, "model": model_url,
        "runs": "1",
    }
    if mode == "fixed":
        tensor_path = output_root / "python" / sample_id / "normalized_chw.bin"
        query["tensor"] = "/" + tensor_path.relative_to(project_root).as_posix()
    url = f"http://127.0.0.1:{port}/web/?{urlencode(query)}"
    with tempfile.TemporaryDirectory(prefix="decisive-browser-") as profile:
        process = subprocess.Popen([
            str(chrome), "--headless=new", "--enable-unsafe-webgpu", "--disable-gpu-sandbox",
            f"--user-data-dir={profile}", url,
        ])
        try:
            if not completed.wait(timeout=timeout):
                errors.append(f"timeout after {timeout}s; received={sorted(received)}")
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill()
            server.shutdown(); server.server_close()
    result = {
        "sample_id": sample_id, "image_path": record["image_path"], "backend": backend, "mode": mode,
        "received": sorted(received), "missing": sorted(STAGES - received), "metadata": metadata,
        "errors": errors, "success": not errors and STAGES.issubset(received),
    }
    (output / "capture.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="midterm/experiments/decisive_path")
    parser.add_argument("--chrome", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    parser.add_argument("--model-url", default="/midterm/experiments/decisive_path/deployment/clip_b16_patch_head_b1_seed_42.onnx")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    root = (project_root / args.root).resolve()
    records = json.loads((root / "e1_samples.json").read_text(encoding="utf-8"))
    for index, record in enumerate(records):
        sample_id = f"{index:02d}_{record['category']}_{record['defect_type']}_{Path(record['image_path']).stem}"
        python_root = root / "e1/captures/python" / sample_id
        python_root.mkdir(parents=True, exist_ok=True)
        with Image.open(project_root / record["image_path"]) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            resized = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC), dtype=np.uint8)
            normalized = preprocess_image(image, 224).numpy().astype("<f4")
        rgba.tofile(python_root / "decoded_rgba.bin")
        resized.tofile(python_root / "resized_rgb.bin")
        normalized.tofile(python_root / "normalized_chw.bin")
        (python_root / "metadata.json").write_text(json.dumps({
            "sample_id": sample_id, "image_path": record["image_path"],
            "decoded_shape": list(rgba.shape), "resized_shape": list(resized.shape),
            "normalized_shape": list(normalized.shape),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results = []
    captures = root / "e1/captures"
    for mode, backend in (("real", "webgpu"), ("fixed", "webgpu"), ("fixed", "wasm")):
        for index, record in enumerate(records):
            result = _capture_one(
                project_root, Path(args.chrome), record, index, backend, mode, args.model_url,
                captures, args.port, args.timeout,
            )
            results.append(result)
            print(json.dumps({key: result[key] for key in ("sample_id", "mode", "backend", "success", "missing")}), flush=True)
    (root / "e1/capture_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
