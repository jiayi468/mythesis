from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def image_scores_from_map(anomaly_map: torch.Tensor, quantile: float = 0.995) -> torch.Tensor:
    if anomaly_map.ndim != 3:
        raise ValueError("anomaly_map must have shape [B,H,W]")
    flat = anomaly_map.flatten(1)
    position = (flat.shape[1] - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    top_count = flat.shape[1] - lower_index
    top_values = torch.topk(flat, k=top_count, dim=1, largest=True, sorted=True).values
    lower = top_values[:, -1]
    if upper_index == lower_index:
        return lower
    upper = top_values[:, -2]
    weight = position - lower_index
    return lower + (upper - lower) * weight


class PatchAnomalyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        prototypes: torch.Tensor,
        dropout: float = 0.1,
        logit_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if prototypes.shape != (2, embedding_dim):
            raise ValueError(f"prototypes must have shape [2,{embedding_dim}]")
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.register_buffer("text_prototypes", F.normalize(prototypes.float(), dim=-1))
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale)))

    def forward(self, patch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_features.ndim != 3:
            raise ValueError("patch_features must have shape [B,N,D]")
        grid = math.isqrt(patch_features.shape[1])
        if grid * grid != patch_features.shape[1]:
            raise ValueError("number of patch tokens must form a square grid")
        embeddings = F.normalize(self.adapter(patch_features), dim=-1)
        logits = self.logit_scale * embeddings @ self.text_prototypes.T
        anomaly_probability = logits.softmax(dim=-1)[..., 1]
        anomaly_map = anomaly_probability.reshape(-1, grid, grid)
        return anomaly_map, image_scores_from_map(anomaly_map)
