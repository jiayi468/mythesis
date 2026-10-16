# 中期答辩最低线实验

本目录保存“冻结 CLIP ViT-B/16 + 轻量 Patch 异常头”的源域小样本探索性验证证据。当前结果不是 MVTec 官方无监督协议成绩，也不是正式跨数据集零样本结论。

## 当前实验范围

- 类别：`pill`、`screw`。
- 训练：每类正常 16 张、异常 16 张。
- 验证：每类正常 8 张、异常 8 张，共 32 张。
- 主干：OpenAI CLIP ViT-B/16，冻结。
- 输出：`14×14` 异常图与 `99.5%` 分位图像分数。
- Pixel 指标：异常图双线性上采样到 `224×224`，与最近邻缩放的二值 GT 比较。
- 随机种子：42。

## 复现

在项目根目录执行：

```powershell
python -m pip install -e ".[dev,model,onnx]"
python scripts/audit_environment.py --config configs/midterm_cpu.yaml
python scripts/audit_dataset.py --config configs/midterm_cpu.yaml
python scripts/cache_features.py --config configs/midterm_cpu.yaml --batch-size 4
python scripts/train_baseline.py --config configs/midterm_cpu.yaml
python scripts/evaluate_baseline.py --config configs/midterm_cpu.yaml
python scripts/export_onnx.py --config configs/midterm_cpu.yaml
python scripts/compare_pytorch_onnx.py --config configs/midterm_cpu.yaml --samples 5
```

启动 WebGPU 手工验证页面：

```powershell
python -m http.server 8000
```

随后访问 `http://localhost:8000/web/`。页面明确指定 `executionProviders: ["webgpu"]`，创建失败时不会自动改用 WASM。

## 结果文件

- `experiments/environment.md`：环境审计。
- `experiments/source_split.csv`：固定样本清单。
- `experiments/results.csv`：按类别及合并结果。
- `experiments/predictions.csv`：逐样本分数。
- `experiments/figures/`：真实热力图及失败案例。
- `deployment/export_log.txt`：静态 ONNX 导出结果。
- `deployment/pytorch_onnx_comparison.csv`：5 张输入的一致性误差。
- `deployment/webgpu_performance.csv`：真实 WebGPU 模型加载、首次与预热时延。
- `deployment/webgpu_console_log.txt`：WebGPU Session 与固定张量前向日志。
- `deployment/webgpu_reference_evidence.png`：无头 Chrome 的真实输出截图。

大体积数据、特征缓存、checkpoint 和 ONNX 模型不提交 Git。

## WebGPU 结果说明

Chrome 155 无头模式中仅指定 `executionProviders: ["webgpu"]`，模型 Session 和真实前向均成功。固定 Python 预处理张量的模型加载时间为 5939.6 ms，首次推理为 8755.6 ms，10 轮预热后 P50/P95 分别为 3728.7/3853.6 ms。该结果证明当前设备上的 WebGPU 兼容性，但性能较差，不能表述为“WebGPU 加速”。

浏览器 Canvas 直接缩放同一图像时分数为 0.185323，而 Python 固定张量的 WebGPU 分数显示为 0.167000，后者与 ONNX CPU 的 0.166999131 基本一致。这说明正式浏览器评测前必须进一步统一 PIL 与 Canvas 的缩放插值，不能把前处理差异当作后端数值误差。
