from __future__ import annotations

import torch
from torch import nn


def project_patch_tokens(tokens: torch.Tensor, projection: torch.Tensor | nn.Module) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B,N,D]")
    if isinstance(projection, torch.Tensor):
        return tokens @ projection
    return projection(tokens)


class FrozenClipBackbone(nn.Module):
    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.visual = clip_model.visual
        self.visual.output_tokens = True
        self.visual.eval()
        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.visual(images)
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("OpenCLIP visual encoder did not return pooled and token features")
        _, tokens = output
        if tokens.ndim != 3 or tokens.shape[1] != 196:
            raise RuntimeError(f"Expected [B,196,D] Patch tokens, got {tuple(tokens.shape)}")
        return tokens.float()

    @torch.no_grad()
    def project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        projection = getattr(self.visual, "proj", None)
        if projection is None:
            raise RuntimeError("OpenCLIP visual encoder does not expose a Patch projection")
        return project_patch_tokens(tokens, projection).float()


def load_openclip(model_name: str = "ViT-B-16", pretrained: str = "openai", device: str = "cpu"):
    import open_clip

    model = open_clip.create_model(model_name, pretrained=pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer
