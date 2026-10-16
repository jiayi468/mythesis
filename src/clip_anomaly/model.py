from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def image_scores_from_map(anomaly_map: torch.Tensor, quantile: float = 0.995) -> torch.Tensor:
    return aggregate_image_scores(anomaly_map, "quantile", quantile=quantile)


def aggregate_image_scores(
    anomaly_map: torch.Tensor,
    method: str,
    top_k: int = 5,
    quantile: float = 0.995,
) -> torch.Tensor:
    if anomaly_map.ndim != 3:
        raise ValueError("anomaly_map must have shape [B,H,W]")
    flat = anomaly_map.flatten(1)
    if method == "max":
        return flat.max(dim=1).values
    if method == "topk_mean":
        if not 1 <= top_k <= flat.shape[1]:
            raise ValueError(f"top_k must be between 1 and {flat.shape[1]}")
        return torch.topk(flat, k=top_k, dim=1).values.mean(dim=1)
    if method != "quantile":
        raise ValueError(f"Unknown score aggregation method: {method}")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
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


def _probabilities_from_embeddings(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim != 3:
        raise ValueError("patch features must have shape [B,N,D]")
    grid = math.isqrt(embeddings.shape[1])
    if grid * grid != embeddings.shape[1]:
        raise ValueError("number of patch tokens must form a square grid")
    normalized = F.normalize(embeddings, dim=-1)
    normalized_prototypes = F.normalize(prototypes, dim=-1)
    logits = logit_scale * normalized @ normalized_prototypes.T
    anomaly_map = logits.softmax(dim=-1)[..., 1].reshape(-1, grid, grid)
    return anomaly_map, image_scores_from_map(anomaly_map)


class RawClipPatchHead(nn.Module):
    def __init__(self, prototypes: torch.Tensor, logit_scale: float = 10.0) -> None:
        super().__init__()
        if prototypes.ndim != 2 or prototypes.shape[0] != 2:
            raise ValueError("prototypes must have shape [2,D]")
        self.register_buffer("text_prototypes", F.normalize(prototypes.float(), dim=-1))
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale)))

    def forward(self, projected_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if projected_features.shape[-1] != self.text_prototypes.shape[-1]:
            raise ValueError("projected Patch features and text prototypes must share a dimension")
        return _probabilities_from_embeddings(projected_features, self.text_prototypes, self.logit_scale)


class LearnedPrototypePatchHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        logit_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.class_prototypes = nn.Parameter(torch.empty(2, embedding_dim))
        nn.init.normal_(self.class_prototypes, std=embedding_dim**-0.5)
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale)))

    def forward(self, patch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _probabilities_from_embeddings(
            self.adapter(patch_features), self.class_prototypes, self.logit_scale
        )


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
        return _probabilities_from_embeddings(
            self.adapter(patch_features), self.text_prototypes, self.logit_scale
        )
