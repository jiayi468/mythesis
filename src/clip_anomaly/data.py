from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SampleRecord:
    category: str
    defect_type: str
    image_path: Path
    mask_path: Path | None
    label: int
    source_partition: str
    split: str = "unassigned"
    mask_area_ratio: float = 0.0


def _is_category_root(path: Path) -> bool:
    return (path / "train" / "good").is_dir() and (path / "test").is_dir()


def discover_category_root(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    if _is_category_root(candidate):
        return candidate
    if candidate.is_dir():
        children = sorted(child for child in candidate.iterdir() if child.is_dir())
        matches = [child for child in children if _is_category_root(child)]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"Cannot locate an MVTec category root below: {candidate}")


def _mask_area_ratio(path: Path) -> float:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    return float(np.count_nonzero(mask) / mask.size)


def scan_mvtec_category(path: str | Path) -> list[SampleRecord]:
    root = discover_category_root(path)
    records: list[SampleRecord] = []
    for image_path in sorted((root / "train" / "good").glob("*.png")):
        records.append(SampleRecord(root.name, "good", image_path, None, 0, "train"))

    for defect_dir in sorted((root / "test").iterdir()):
        if not defect_dir.is_dir():
            continue
        for image_path in sorted(defect_dir.glob("*.png")):
            if defect_dir.name == "good":
                records.append(SampleRecord(root.name, "good", image_path, None, 0, "test"))
                continue
            mask_path = root / "ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                raise ValueError(f"Missing anomaly mask for {image_path}: expected {mask_path}")
            records.append(
                SampleRecord(
                    root.name,
                    defect_dir.name,
                    image_path,
                    mask_path,
                    1,
                    "test",
                    mask_area_ratio=_mask_area_ratio(mask_path),
                )
            )
    return records


def split_records(
    records: list[SampleRecord], seed: int, abnormal_train_fraction: float
) -> list[SampleRecord]:
    if not 0.0 < abnormal_train_fraction < 1.0:
        raise ValueError("abnormal_train_fraction must be strictly between 0 and 1")
    output: list[SampleRecord] = []
    abnormal_groups: dict[tuple[str, str], list[SampleRecord]] = {}
    for record in records:
        if record.label == 0:
            split = "train" if record.source_partition == "train" else "val"
            output.append(replace(record, split=split))
        else:
            abnormal_groups.setdefault((record.category, record.defect_type), []).append(record)

    for group_index, key in enumerate(sorted(abnormal_groups)):
        group = sorted(abnormal_groups[key], key=lambda item: item.image_path.as_posix())
        random.Random(seed + group_index).shuffle(group)
        if len(group) == 1:
            train_count = 0
        else:
            train_count = round(len(group) * abnormal_train_fraction)
            train_count = min(max(1, train_count), len(group) - 1)
        output.extend(replace(record, split="train" if index < train_count else "val") for index, record in enumerate(group))

    return sorted(output, key=lambda item: (item.category, item.split, item.defect_type, item.image_path.name))


def decisive_split_records(
    records: list[SampleRecord],
    seed: int,
    abnormal_train_fraction: float,
    dev_fraction: float,
) -> list[SampleRecord]:
    """Create deterministic train/dev/source_holdout partitions.

    Original MVTec train/good images remain in train. Test normals are split
    between dev and source_holdout. Each anomaly stratum is split three ways;
    strata with at least three samples retain at least one dev and holdout item.
    """
    holdout_fraction = 1.0 - abnormal_train_fraction - dev_fraction
    if abnormal_train_fraction <= 0 or dev_fraction <= 0 or holdout_fraction <= 0:
        raise ValueError("train and dev fractions must leave a positive source_holdout fraction")

    output: list[SampleRecord] = []
    grouped: dict[tuple[str, str, int], list[SampleRecord]] = defaultdict(list)
    for record in records:
        if record.source_partition == "train":
            output.append(replace(record, split="train"))
        else:
            key = (record.category, record.defect_type, record.label)
            grouped[key].append(record)

    for group_index, key in enumerate(sorted(grouped)):
        group = sorted(grouped[key], key=lambda item: item.image_path.as_posix())
        random.Random(seed + group_index).shuffle(group)
        if key[2] == 0:
            dev_count = max(1, round(len(group) * 0.5)) if len(group) > 1 else 0
            assignments = ["dev"] * dev_count + ["source_holdout"] * (len(group) - dev_count)
        else:
            if len(group) < 3:
                raise ValueError(f"Anomaly stratum {key} needs at least 3 samples for three-way split")
            train_count = max(1, round(len(group) * abnormal_train_fraction))
            dev_count = max(1, round(len(group) * dev_fraction))
            if train_count + dev_count >= len(group):
                overflow = train_count + dev_count - len(group) + 1
                train_count = max(1, train_count - overflow)
            holdout_count = len(group) - train_count - dev_count
            if holdout_count < 1:
                raise ValueError(f"Anomaly stratum {key} has no source_holdout sample")
            assignments = (
                ["train"] * train_count
                + ["dev"] * dev_count
                + ["source_holdout"] * holdout_count
            )
        output.extend(replace(record, split=split) for record, split in zip(group, assignments))

    return sorted(
        output,
        key=lambda item: (item.category, item.split, item.defect_type, item.image_path.name),
    )


def select_limited_records(
    records: list[SampleRecord],
    seed: int,
    max_train_normal_per_category: int,
    max_train_abnormal_per_category: int,
    max_val_normal_per_category: int,
    max_val_abnormal_per_category: int,
) -> list[SampleRecord]:
    limits = {
        ("train", 0): max_train_normal_per_category,
        ("train", 1): max_train_abnormal_per_category,
        ("val", 0): max_val_normal_per_category,
        ("val", 1): max_val_abnormal_per_category,
    }
    grouped: dict[tuple[str, str, int], list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.category, record.split, record.label)].append(record)

    selected: list[SampleRecord] = []
    for group_index, key in enumerate(sorted(grouped)):
        group = sorted(grouped[key], key=lambda item: (item.defect_type, item.image_path.name))
        random.Random(seed + 1000 + group_index).shuffle(group)
        limit = limits[(key[1], key[2])]
        if key[2] == 0 or len(group) <= limit:
            chosen = group[:limit]
        else:
            by_defect: dict[str, list[SampleRecord]] = defaultdict(list)
            for record in group:
                by_defect[record.defect_type].append(record)
            chosen = [items.pop() for _, items in sorted(by_defect.items()) if items]
            remaining = [record for items in by_defect.values() for record in items]
            random.Random(seed + 2000 + group_index).shuffle(remaining)
            chosen.extend(remaining[: max(0, limit - len(chosen))])
        selected.extend(chosen)
    return sorted(selected, key=lambda item: (item.category, item.split, item.label, item.defect_type, item.image_path.name))
