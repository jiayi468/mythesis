import json
from pathlib import Path

from PIL import Image

from clip_anomaly.data import SampleRecord
from clip_anomaly.decisive import write_data_manifest


def test_data_manifest_contains_relative_paths_and_sha256(tmp_path: Path) -> None:
    image = tmp_path / "dataset/pill/test/good/000.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(image)
    record = SampleRecord("pill", "good", image, None, 0, "test", "dev")

    output = write_data_manifest([record], tmp_path, tmp_path / "manifest.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload[0]["image_path"] == "dataset/pill/test/good/000.png"
    assert len(payload[0]["image_sha256"]) == 64
    assert payload[0]["mask_path"] is None
    assert payload[0]["mask_sha256"] is None
