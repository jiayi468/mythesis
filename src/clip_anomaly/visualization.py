from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_anomaly_figure(
    image_path: str | Path,
    mask: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[3].imshow(rgb)
    axes[3].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[3].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
