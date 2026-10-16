from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_RESULT_FIELDS = (
    "status", "criteria", "observed", "artifact_paths", "protocol_hash", "git_commit",
    "started_at", "completed_at", "planned", "succeeded", "failed", "skipped", "failure_reasons",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_data_manifest(records, project_root: str | Path, path: str | Path) -> Path:
    root = Path(project_root).resolve()
    rows = []
    for record in records:
        image = Path(record.image_path).resolve()
        mask = Path(record.mask_path).resolve() if record.mask_path else None
        rows.append({
            "category": record.category,
            "defect_type": record.defect_type,
            "image_path": image.relative_to(root).as_posix(),
            "image_sha256": sha256_file(image),
            "mask_path": mask.relative_to(root).as_posix() if mask else None,
            "mask_sha256": sha256_file(mask) if mask else None,
            "label": record.label,
            "source_partition": record.source_partition,
            "split": record.split,
            "mask_area_ratio": record.mask_area_ratio,
        })
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def build_result(
    *, criteria: dict[str, Any], observed: dict[str, Any], artifact_paths: list[str],
    protocol_hash: str, git_commit: str, started_at: str, completed_at: str,
    planned: int, succeeded: int, failed: int, skipped: int,
    failure_reasons: list[str], criteria_passed: bool,
) -> dict[str, Any]:
    counts = (planned, succeeded, failed, skipped)
    if any(value < 0 for value in counts) or succeeded + failed + skipped != planned:
        raise ValueError("result counts must be non-negative and sum to planned")
    complete = failed == 0 and skipped == 0 and succeeded == planned
    status = "PASS" if complete and criteria_passed else "FAIL" if complete else "INCOMPLETE"
    return {
        "status": status, "criteria": criteria, "observed": observed,
        "artifact_paths": artifact_paths, "protocol_hash": protocol_hash, "git_commit": git_commit,
        "started_at": started_at, "completed_at": completed_at,
        "planned": planned, "succeeded": succeeded, "failed": failed, "skipped": skipped,
        "failure_reasons": failure_reasons,
    }


def write_result(path: str | Path, result: dict[str, Any]) -> Path:
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in result]
    if missing:
        raise ValueError(f"result is missing required fields: {', '.join(missing)}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def compare_arrays(reference, candidate) -> dict[str, float]:
    import numpy as np

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"Array shapes differ: {reference.shape} != {candidate.shape}")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {"max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean())}
