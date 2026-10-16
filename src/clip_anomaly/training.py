from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import DiceLoss, FocalLoss
from .model import LearnedPrototypePatchHead, PatchAnomalyHead


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path
    history_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loss(prediction: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    return FocalLoss(gamma=gamma)(prediction, target) + DiceLoss()(prediction, target)


def train_head(
    features: torch.Tensor,
    masks: torch.Tensor,
    splits: list[str],
    prototypes: torch.Tensor,
    output_dir: str | Path,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    gamma: float,
    seed: int,
    head_type: str = "b2_text_prototype",
) -> TrainingResult:
    set_seed(seed)
    train_idx = torch.tensor([index for index, split in enumerate(splits) if split == "train"])
    val_idx = torch.tensor([index for index, split in enumerate(splits) if split in ("val", "dev")])
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise ValueError("Feature cache must contain train and dev/val records")
    if head_type == "b2_text_prototype":
        model = PatchAnomalyHead(
            input_dim=features.shape[-1],
            embedding_dim=prototypes.shape[-1],
            hidden_dim=hidden_dim,
            prototypes=prototypes,
            dropout=dropout,
        )
    elif head_type == "b1_classifier":
        model = LearnedPrototypePatchHead(
            input_dim=features.shape[-1],
            embedding_dim=prototypes.shape[-1],
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported trainable head type: {head_type}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features[train_idx].float(), masks[train_idx].float()),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "clip_b16_patch_head.pt"
    history_path = output_path.parent / "logs" / "training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_features, batch_masks in loader:
            optimizer.zero_grad(set_to_none=True)
            anomaly_map, _ = model(batch_features)
            loss = _loss(anomaly_map, batch_masks, gamma)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            val_map, _ = model(features[val_idx].float())
            val_loss = _loss(val_map, masks[val_idx].float(), gamma).item()
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": features.shape[-1],
                    "embedding_dim": prototypes.shape[-1],
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                    "head_type": head_type,
                    "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    return TrainingResult(best_epoch, best_loss, checkpoint_path, history_path)


def load_trained_head(checkpoint_path: str | Path, prototypes: torch.Tensor) -> torch.nn.Module:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    head_type = checkpoint.get("head_type", "b2_text_prototype")
    if head_type == "b2_text_prototype":
        model = PatchAnomalyHead(
            checkpoint["input_dim"], checkpoint["embedding_dim"], checkpoint["hidden_dim"],
            prototypes, checkpoint["dropout"],
        )
    elif head_type == "b1_classifier":
        model = LearnedPrototypePatchHead(
            checkpoint["input_dim"], checkpoint["embedding_dim"], checkpoint["hidden_dim"],
            checkpoint["dropout"],
        )
    else:
        raise ValueError(f"Unsupported checkpoint head type: {head_type}")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
