from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    dataset_roots: tuple[Path, ...]
    output_root: Path
    seed: int
    split: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = ("project_root", "dataset_roots", "output_root", "seed", "split", "model")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")

    fraction = float(raw["split"].get("abnormal_train_fraction", 0.0))
    if not 0.0 < fraction < 1.0:
        raise ValueError("abnormal_train_fraction must be strictly between 0 and 1")

    project_root = (config_path.parent / raw["project_root"]).resolve()
    dataset_roots = tuple((project_root / value).resolve() for value in raw["dataset_roots"])
    output_root = (project_root / raw["output_root"]).resolve()
    return ExperimentConfig(
        project_root=project_root,
        dataset_roots=dataset_roots,
        output_root=output_root,
        seed=int(raw["seed"]),
        split=dict(raw["split"]),
        model=dict(raw["model"]),
        training=dict(raw.get("training", {})),
    )
