from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading
from urllib.parse import quote


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", default="midterm/deployment/browser_input.bin")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    captured = threading.Event()

    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/capture":
                self.send_error(404); return
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            if len(payload) != 3 * 224 * 224 * 4:
                self.send_error(400, "unexpected float32 tensor length"); return
            output.write_bytes(payload)
            self.send_response(204); self.end_headers(); captured.set()

        def log_message(self, format, *values):
            print(format % values, flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    sample = "/" + args.sample.replace("\\", "/").lstrip("/")
    url = f"http://127.0.0.1:{args.port}/web/?autorun&sample={quote(sample)}&capture=/capture"
    with tempfile.TemporaryDirectory(prefix="clip-browser-capture-") as profile:
        process = subprocess.Popen([
            args.chrome, "--headless=new", "--enable-unsafe-webgpu", "--disable-gpu-sandbox",
            f"--user-data-dir={profile}", url,
        ])
        try:
            if not captured.wait(timeout=120):
                raise TimeoutError("Chrome did not upload the Canvas tensor within 120 seconds")
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill()
            server.shutdown(); server.server_close()
    print(output)


if __name__ == "__main__":
    main()
