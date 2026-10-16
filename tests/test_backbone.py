import torch
from torch import nn

from clip_anomaly.backbone import FrozenClipBackbone


class _FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.output_tokens = False

    def forward(self, images: torch.Tensor):
        tokens = images.mean(dim=(2, 3))[:, None, :].repeat(1, 196, 1) * self.weight
        return tokens.mean(dim=1), tokens


class _FakeClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _FakeVisual()


def test_frozen_backbone_output_can_feed_trainable_head() -> None:
    backbone = FrozenClipBackbone(_FakeClip())
    head = nn.Linear(3, 2)

    output = head(backbone(torch.randn(1, 3, 4, 4))).sum()
    output.backward()

    assert head.weight.grad is not None
    assert backbone.visual.weight.grad is None
