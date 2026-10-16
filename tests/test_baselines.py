import torch

from clip_anomaly.backbone import project_patch_tokens
from clip_anomaly.model import LearnedPrototypePatchHead, PatchAnomalyHead, RawClipPatchHead


def test_project_patch_tokens_uses_visual_projection() -> None:
    tokens = torch.tensor([[[1.0, 2.0, 3.0]]])
    projection = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    projected = project_patch_tokens(tokens, projection)

    assert torch.equal(projected, torch.tensor([[[4.0, 5.0]]]))


def test_raw_clip_head_has_no_trainable_parameters_and_returns_maps() -> None:
    prototypes = torch.eye(2, 4)
    head = RawClipPatchHead(prototypes)

    anomaly_map, scores = head(torch.randn(2, 196, 4))

    assert anomaly_map.shape == (2, 14, 14)
    assert scores.shape == (2,)
    assert sum(parameter.numel() for parameter in head.parameters()) == 0


def test_learned_and_fixed_prototype_heads_share_adapter_capacity() -> None:
    prototypes = torch.eye(2, 4)
    learned = LearnedPrototypePatchHead(6, 4, 8, dropout=0.0)
    fixed = PatchAnomalyHead(6, 4, 8, prototypes, dropout=0.0)

    learned_parameters = sum(parameter.numel() for parameter in learned.parameters())
    fixed_parameters = sum(parameter.numel() for parameter in fixed.parameters())

    assert learned(torch.randn(2, 196, 6))[0].shape == (2, 14, 14)
    assert learned_parameters - fixed_parameters == 2 * 4
