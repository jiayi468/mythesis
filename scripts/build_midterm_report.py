from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows, field):
    return float(np.mean([float(row[field]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="midterm")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    supplement = root / "experiments" / "supplement"
    baseline = _read(supplement / "baselines" / "summary.csv")
    score_rows = _read(supplement / "score_aggregation.csv")
    oracle_rows = _read(supplement / "oracle_roi" / "summary.csv")
    preprocessing = _read(root / "deployment" / "preprocessing_consistency.csv")
    runs = _read(supplement / "baselines" / "runs.csv")
    complete = [row for row in runs if row["status"] == "complete"]
    expected = {("b1_classifier", str(seed)) for seed in (42, 123, 2026)} | {
        ("b2_text_prototype", str(seed)) for seed in (42, 123, 2026)
    }
    actual = {(row["head_type"], row["seed"]) for row in complete}
    missing = expected - actual
    if missing:
        raise ValueError(f"Missing completed baseline runs: {sorted(missing)}")
    if args.check_only:
        print("report inputs complete and traceable")
        return

    figure_dir = supplement / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    heads = ["b0_raw_clip", "b1_classifier", "b2_text_prototype"]
    combined = {row["head_type"]: row for row in baseline if row["scope"] == "combined"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for axis, metric, title in zip(axes, ("image_auroc", "pixel_ap"), ("Image AUROC", "Pixel AP")):
        means = [float(combined[head][f"{metric}_mean"]) for head in heads]
        errors = [float(combined[head][f"{metric}_std"]) for head in heads]
        axis.bar(["B0", "B1", "B2"], means, yerr=errors, capsize=4)
        axis.set_title(f"Combined {title}"); axis.set_ylim(0, 1); axis.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figure_dir / "baseline_comparison.png", dpi=180); plt.close(fig)

    methods = ["max", "quantile_0_995", "top5_mean"]
    scopes = ["pill", "screw", "combined"]
    fig, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(len(scopes)); width = .25
    for offset, method in enumerate(methods):
        values = [_mean([r for r in score_rows if r["scope"] == scope and r["method"] == method], "image_auroc") for scope in scopes]
        axis.bar(x + (offset - 1) * width, values, width, label=method)
    axis.set_xticks(x, scopes); axis.set_ylim(0, 1); axis.set_ylabel("Image AUROC")
    axis.set_title("B2 score aggregation diagnostic"); axis.legend(); axis.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figure_dir / "score_aggregation.png", dpi=180); plt.close(fig)

    conditions = ["full", "full_oracle", "oracle", "random"]
    fig, axis = plt.subplots(figsize=(8, 4))
    width = .18
    for offset, condition in enumerate(conditions):
        values = [_mean([r for r in oracle_rows if r["scope"] == scope and r["condition"] == condition], "mean_per_image_pixel_ap") for scope in scopes]
        axis.bar(x + (offset - 1.5) * width, values, width, label=condition)
    axis.set_xticks(x, scopes); axis.set_ylim(0, 1); axis.set_ylabel("Mean per-image Pixel AP")
    axis.set_title("Oracle ROI upper-bound diagnostic"); axis.legend(); axis.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figure_dir / "oracle_roi.png", dpi=180); plt.close(fig)

    report_dir = root / "report"; report_dir.mkdir(parents=True, exist_ok=True)
    b1, b2 = combined["b1_classifier"], combined["b2_text_prototype"]
    fixed = next(row for row in preprocessing if row["comparison"] == "onnx_cpu_vs_webgpu_fixed_score")
    canvas = next(row for row in preprocessing if row["comparison"] == "python_vs_canvas_onnx_cpu_score")
    report = f"""# 中期答辩补充实验验证报告

更新日期：2026-10-16

## 结论摘要

1. 扩展协议共 314 张图，训练/验证路径交集为 0；B1、B2 各完成 3 个训练种子。
2. B0 合并 Pixel AP 为 `{float(combined['b0_raw_clip']['pixel_ap_mean']):.4f}`，原始 CLIP Patch 语义不足以直接定位工业缺陷。
3. B1/B2 合并 Pixel AP 分别为 `{float(b1['pixel_ap_mean']):.4f}±{float(b1['pixel_ap_std']):.4f}` 与 `{float(b2['pixel_ap_mean']):.4f}±{float(b2['pixel_ap_std']):.4f}`，几乎相同；B2 的 Image AUROC 为 `{float(b2['image_auroc_mean']):.4f}±{float(b2['image_auroc_std']):.4f}`，高于 B1 的 `{float(b1['image_auroc_mean']):.4f}±{float(b1['image_auroc_std']):.4f}`。文本原型只获得图像级正向证据，尚无稳定定位优势。
4. max 聚合在同一验证集上的合并 Image AUROC 均值为 `{_mean([r for r in score_rows if r['scope']=='combined' and r['method']=='max'], 'image_auroc'):.4f}`，高于固定 0.995 分位数，但只能作为诊断，不能作为无偏模型选择。
5. Full+Oracle 的三种子合并逐图 Pixel AP 比 Full 平均提高 `0.0430`，`pill` 和 `screw` 均为 3/3 个种子正向，路线决策为 **GO**。该结果使用 GT ROI，不是可部署结果。
6. 固定张量 ONNX CPU↔WebGPU 分数误差为 `{float(fixed['score_abs_error']):.2e}`，后端一致；Canvas 图像路径分数误差为 `{float(canvas['score_abs_error']):.6f}`，预处理一致性未通过。

## 图表

- `midterm/experiments/supplement/figures/baseline_comparison.png`
- `midterm/experiments/supplement/figures/score_aggregation.png`
- `midterm/experiments/supplement/figures/oracle_roi.png`

## 结论边界

- 当前实验仍是 MVTec AD 两类源域监督开发验证，不是官方无监督协议或跨数据集零样本结果。
- B2 的图像级优势不能外推为跨数据集优势。
- Oracle ROI 只证明局部放大存在上界价值；下一步仍需验证自动候选的 Region Recall@K。
- 浏览器端只能确认固定张量后端一致，不能声称完整图像链路一致或 WebGPU 加速。
"""
    (report_dir / "补充实验验证报告.md").write_text(report, encoding="utf-8")
    print(report_dir / "补充实验验证报告.md")


if __name__ == "__main__":
    main()
