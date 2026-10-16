from pathlib import Path

import pytest
import torch

from clip_anomaly.cache import FeatureCache, FeatureCacheMetadata


def test_cache_round_trip(tmp_path: Path) -> None:
    metadata = FeatureCacheMetadata("ViT-B-16", "openai", 224, 14, "v1")
    cache = FeatureCache(tmp_path / "features.pt", metadata)
    payload = {
        "features": torch.randn(2, 196, 8),
        "masks": torch.zeros(2, 14, 14),
        "labels": torch.tensor([0, 1]),
        "records": [{"image_path": "dataset/pill/a.png"}, {"image_path": "dataset/pill/b.png"}],
    }

    cache.save(payload)
    loaded = cache.load()

    assert torch.equal(loaded["features"], payload["features"])
    assert loaded["records"] == payload["records"]


def test_cache_rejects_metadata_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "features.pt"
    FeatureCache(path, FeatureCacheMetadata("ViT-B-16", "openai", 224, 14, "v1")).save(
        {"features": torch.zeros(1, 196, 8)}
    )

    incompatible = FeatureCache(path, FeatureCacheMetadata("ViT-B-32", "openai", 224, 7, "v1"))
    with pytest.raises(ValueError, match="metadata"):
        incompatible.load()
