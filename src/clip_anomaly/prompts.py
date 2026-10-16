from __future__ import annotations

import torch
import torch.nn.functional as F


NORMAL_PROMPTS = (
    "an intact industrial object",
    "a normal industrial object",
    "an industrial object without defects",
)
ANOMALY_PROMPTS = (
    "a damaged industrial object",
    "a defective industrial object",
    "an anomalous industrial object",
)


@torch.inference_mode()
def build_text_prototypes(model, tokenizer, device: str | torch.device = "cpu") -> torch.Tensor:
    prototypes = []
    for prompts in (NORMAL_PROMPTS, ANOMALY_PROMPTS):
        tokens = tokenizer(list(prompts)).to(device)
        features = F.normalize(model.encode_text(tokens), dim=-1)
        prototypes.append(F.normalize(features.mean(dim=0), dim=-1))
    return torch.stack(prototypes).cpu()
