from __future__ import annotations

import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.clamp(self.eps, 1.0 - self.eps)
        target = target.float()
        probability = prediction * target + (1.0 - prediction) * (1.0 - target)
        return (-((1.0 - probability) ** self.gamma) * torch.log(probability)).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.float().flatten(1)
        target = target.float().flatten(1)
        intersection = (prediction * target).sum(dim=1)
        denominator = prediction.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()
