from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class FeatureCacheMetadata:
    model_name: str
    pretrained: str
    input_size: int
    patch_grid: int
    preprocessing_version: str


class FeatureCache:
    def __init__(self, path: str | Path, metadata: FeatureCacheMetadata) -> None:
        self.path = Path(path)
        self.metadata = metadata

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": asdict(self.metadata), "payload": payload}, self.path)

    def load(self) -> dict[str, Any]:
        stored = torch.load(self.path, map_location="cpu")
        if stored.get("metadata") != asdict(self.metadata):
            raise ValueError("Feature cache metadata does not match the requested model or preprocessing")
        payload = stored.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Feature cache payload is invalid")
        return payload
