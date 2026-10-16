# 中期最低线验收状态

更新时间：2026-10-16

| 项目 | 状态 | 证据与说明 |
|---|---|---|
| 环境审计 | 已完成 | `environment.md`；当前为 CPU 版 PyTorch，无 CUDA |
| 数据审计 | 已完成 | `source_split.csv`；两类原始数据完整，本轮固定选择 96 张，训练/验证不重叠 |
| PyTorch Patch 前向 | 已完成 | 真实 CLIP ViT-B/16 输出 `[B,196,768]`，缓存元数据固定 |
| 最小训练 | 已完成 | 最佳 Epoch 29，验证组合 Loss 0.930274 |
| 基础指标 | 已完成 | `results.csv`；验证集 32 张，属于源域小样本探索性验证 |
| 热力图和失败案例 | 已完成 | `figures/`；包含正常高分、异常高分、异常低分和微小异常 |
| ONNX 导出 | 已完成 | 静态 B=1；`export_log.txt`，ONNX checker 通过 |
| ONNX CPU 一致性 | 已完成 | `pytorch_onnx_comparison.csv`；5 张样本最大热力图绝对误差 `3.5137e-5` |
| WebGPU 前向 | 已完成 | Chrome 155 无头模式仅指定 `webgpu`，固定 Python float32 输入得到异常分数 `0.167000`；截图见 `deployment/webgpu_reference_evidence.png` |
| WebGPU 时延 | 已完成 | 固定输入实测：加载 5939.6 ms、首次 8755.6 ms、预热后 P50 3728.7 ms、P95 3853.6 ms，共 10 轮 |
| Oracle ROI | 未开展 | 不属于本轮最低线 |
| PPT/讲稿 | 未开展 | 本轮任务聚焦实验代码与证据，待根据真实结果制作 |

## 补充实验状态（2026-10-16）

| 项目 | 状态 | 证据与说明 |
|---|---|---|
| 扩展数据协议 | 已完成 | `supplement/source_split.csv`；314 张，训练/验证交集为 0 |
| B0/B1/B2 公平基线 | 已完成 | `supplement/baselines/runs.csv`、`summary.csv`；B1/B2 各 3 个训练种子 |
| 图像分数聚合诊断 | 已完成 | `supplement/score_aggregation.csv`；max 在当前验证集更好，但不作为无偏选择 |
| Oracle ROI | 已完成 | `supplement/oracle_roi/`；决策为 GO，合并逐图 Pixel AP 平均增量 0.043036 |
| 浏览器预处理一致性 | 验证失败 | 固定张量后端误差 8.69e-7；Canvas/Python 分数误差 0.018327 |
| 补充实验报告 | 已完成 | `midterm/report/补充实验验证报告.md` |

## 当前结果摘要

| 范围 | 样本数 | Image AUROC | Image AP | Pixel AUROC | Pixel AP |
|---|---:|---:|---:|---:|---:|
| pill | 16 | 0.921875 | 0.929365 | 0.992669 | 0.813968 |
| screw | 16 | 0.531250 | 0.711905 | 0.963915 | 0.155535 |
| 合并 | 32 | 0.746094 | 0.807794 | 0.992049 | 0.769578 |

## 可信边界

- `pill` 已显示明显异常信号；`screw` 的图像级 AUROC 接近随机，基础头对该类仍不稳定。
- 合并 Pixel 指标受类别与异常像素数量影响，不能替代逐类别 Pixel AP。
- 验证样本少，所有指标只用于中期可行性证据，不作为正式论文性能结论。
- WebGPU Session 明确使用 `executionProviders: ["webgpu"]`，页面未配置 WASM 回退。当前时延很高，只能证明兼容性，不能声称获得加速。
- 同一 Python float32 输入的 ONNX CPU 分数为 `0.166999131`，WebGPU 页面显示为 `0.167000`；受页面当时仅显示 6 位小数限制，只能确认分数绝对差小于 `1.4e-6`。
- 浏览器 Canvas 直接预处理同图时分数为 `0.185323`，与固定张量结果存在约 `0.0183` 差异，说明浏览器图像缩放/插值必须继续与 Python 严格对齐。
