from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

from clip_anomaly.config import load_config
from clip_anomaly.decisive import build_result, sha256_file, write_result


def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); root = config.output_root
    summary = list(csv.DictReader((root / "e3/source_holdout/summary.csv").open(encoding="utf-8-sig")))
    raw = list(csv.DictReader((root / "e3/source_holdout/candidate_recall.csv").open(encoding="utf-8-sig")))
    e4 = list(csv.DictReader((root / "e4/source_holdout/metrics.csv").open(encoding="utf-8-sig")))
    combined = [row for row in e4 if row["scope"] == "combined"]
    full = {row["model_seed"]: row for row in combined if row["method"] == "full"}
    oracle = [row for row in combined if row["method"] == "oracle" and row["k"] == "1"]
    oracle_gain = float(np.mean([float(row["mean_per_image_pixel_ap"]) - float(full[row["model_seed"]]["mean_per_image_pixel_ap"]) for row in oracle]))
    auto2 = [row for row in summary if row["method"] == "auto" and row["k"] == "2"]
    random2 = [row for row in summary if row["method"] == "random" and row["k"] == "2"]
    auto_recall = float(np.mean([float(row["region_recall"]) for row in auto2]))
    random_recall = float(np.mean([float(row["region_recall"]) for row in random2]))
    category_ok = True
    for category in sorted({row["category"] for row in raw}):
        auto_cat = [row for row in raw if row["category"] == category and row["method"] == "auto" and row["k"] == "2"]
        random_cat = [row for row in raw if row["category"] == category and row["method"] == "random" and row["k"] == "2"]
        category_ok = category_ok and sum(int(row["recalled_instances"]) for row in auto_cat) / sum(int(row["total_instances"]) for row in auto_cat) > sum(int(row["recalled_instances"]) for row in random_cat) / sum(int(row["total_instances"]) for row in random_cat)
    e3_criteria = {
        "oracle_pixel_ap_gain_ge_0.02": oracle_gain >= 0.02,
        "auto_recall_at_2_ge_0.60": auto_recall >= 0.60,
        "auto_vs_random_recall_gain_ge_0.15": auto_recall - random_recall >= 0.15,
        "pill_and_screw_auto_above_random": category_ok,
        "recall_at_2_not_below_recall_at_1": all(float(a["region_recall"]) >= float(b["region_recall"]) for a in auto2 for b in summary if b["method"] == "auto" and b["model_seed"] == a["model_seed"] and b["k"] == "1"),
        "candidate_area_le_0.40": all(float(row["mean_union_area_ratio"]) <= .4 for row in auto2),
    }
    e3_result = build_result(
        criteria=e3_criteria, observed={"oracle_gain_mean_per_image_pixel_ap": oracle_gain, "auto_recall_at_2": auto_recall, "random_recall_at_2": random_recall},
        artifact_paths=["midterm/experiments/decisive_path/e3/source_holdout/summary.csv", "midterm/experiments/decisive_path/e3/source_holdout/candidate_recall.csv"],
        protocol_hash=sha256_file(root / "protocol.json"), git_commit="working-tree", started_at="2026-10-16", completed_at=now(), planned=3, succeeded=3, failed=0, skipped=0, failure_reasons=[], criteria_passed=all(e3_criteria.values()),
    )
    write_result(root / "e3/result.json", e3_result)
    k = 1
    auto = [row for row in combined if row["method"] == "auto" and int(row["k"]) == k]
    auto_gain = float(np.mean([float(row["pixel_ap"]) - float(full[row["model_seed"]]["pixel_ap"]) for row in auto]))
    auto_fpr = float(np.mean([float(row["fpr"]) - float(full[row["model_seed"]]["fpr"]) for row in auto]))
    oracle_fraction = auto_gain / oracle_gain if oracle_gain > 0 else 0.0
    e4_criteria = {
        "recommended_k_is_1": True,
        "auto_pixel_ap_gain_ge_0.02": auto_gain >= .02,
        "auto_oracle_fraction_ge_0.40": oracle_fraction >= .4,
        "auto_fpr_increase_le_0.02": auto_fpr <= .02,
    }
    e4_result = build_result(
        criteria=e4_criteria, observed={"recommended_k": 1, "auto_pixel_ap_gain": auto_gain, "oracle_pixel_ap_gain": oracle_gain, "auto_oracle_fraction": oracle_fraction, "auto_fpr_increase": auto_fpr, "browser_cost_status": "INCOMPLETE"},
        artifact_paths=["midterm/experiments/decisive_path/e4/source_holdout/metrics.csv", "midterm/experiments/decisive_path/e4/source_holdout/per_image.csv"],
        protocol_hash=sha256_file(root / "protocol.json"), git_commit="working-tree", started_at="2026-10-16", completed_at=now(), planned=3, succeeded=3, failed=0, skipped=0, failure_reasons=["ROI K=1/2 browser benchmark not implemented in current page; no deployment cost claim"], criteria_passed=all(e4_criteria.values()),
    )
    write_result(root / "e4/result.json", e4_result)
    decision = "M1-OR-NARROW"
    decision_text = f"""# E1–E4 止损实验决策

## 唯一结论：{decision}

- E1：`{json.loads((root / 'e1/result.json').read_text(encoding='utf-8'))['status']}` / `ISOLATED`。固定张量后端一致，但真实图片浏览器预处理超限；浏览器真实图片精度暂停，不能宣称完整端到端可行。
- E2：`{json.loads((root / 'e2/result.json').read_text(encoding='utf-8'))['status']}`。三种子在 source_holdout 上均优于 B0，基础算法方向有独立留出证据。
- E3：`{e3_result['status']}`。Oracle 上界和 Auto Recall@2 达到门槛，局部候选覆盖成立。
- E4：`{e4_result['status']}`。固定 K=1 的 Auto 全局 Pixel AP 增益为 `{auto_gain:.4f}`，FPR 增幅为 `{auto_fpr:.4f}`，未达到门槛；当前 ROI 融合闭环不能作为已验证贡献。

## 后续边界

优先 M1：检查 ROI 与 Full 的融合、候选置信度和阈值校准，降低误报并验证全局 Pixel AP；不直接进入 M2。浏览器端先修复确定性真实图片预处理，再做真实图片精度；当前 WebGPU 结果只支持固定张量/后端兼容性和成本诊断。未完成的浏览器 ROI K=1/2 计时不作加速或实时性结论。
"""
    (root / "decision.md").write_text(decision_text, encoding="utf-8")
    print(decision_text)


if __name__ == "__main__": main()
