from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .backbone import FrozenClipBackbone


class FullAnomalyModel(nn.Module):
    def __init__(self, backbone: FrozenClipBackbone, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.backbone(image))


def export_static_onnx(model: nn.Module, path: str | Path, input_size: int = 224) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        output,
        input_names=["image"],
        output_names=["anomaly_map", "image_score"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    return output


def compare_onnx(model: nn.Module, onnx_path: str | Path, inputs: torch.Tensor) -> list[dict[str, float]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rows = []
    model.eval()
    with torch.inference_mode():
        for index, image in enumerate(inputs):
            pt_map, pt_score = model(image[None].float())
            outputs = session.run(None, {"image": image[None].numpy().astype(np.float32)})
            ort_map, ort_score = outputs
            map_diff = np.abs(pt_map.numpy() - ort_map)
            score_diff = np.abs(pt_score.numpy() - ort_score)
            rows.append(
                {
                    "sample_index": index,
                    "map_max_abs_error": float(map_diff.max()),
                    "map_mean_abs_error": float(map_diff.mean()),
                    "score_max_abs_error": float(score_diff.max()),
                }
            )
    return rows
