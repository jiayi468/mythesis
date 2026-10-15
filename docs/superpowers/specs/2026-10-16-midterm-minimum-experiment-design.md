# 中期答辩最低线实验设计

日期：2026-10-16

## 1. 目标与范围

本设计用于在当前 CPU 环境中建立一条真实、可复现、可核查的中期答辩最低实验链路。实验使用 MVTec AD 的 `pill` 和 `screw` 两个类别，完成源域开发数据审计、CLIP Patch 特征提取、轻量异常头训练、基础评价、可视化和静态 ONNX 验证。

本阶段的结论仅表述为“源域小样本探索性验证”。本阶段不声称已经完成正式跨数据集零样本测试、M1/M2、WebGPU 加速或完整浏览器系统。

最低链路如下：

```text
MVTec 数据审计与固定划分
        ↓
CLIP ViT-B/16 提取并缓存 Patch 特征
        ↓
训练轻量 Patch 异常头
        ↓
输出 14×14 异常图和图像级分数
        ↓
计算基础指标并生成可视化
        ↓
导出静态 B=1 ONNX
        ↓
PyTorch 与 ONNX CPU 一致性验证
        ↓
条件允许时进行 WebGPU B=1 前向
```

## 2. 当前环境与数据事实

- 工作环境：Windows，Python 3.10.5，PyTorch 1.13.0 CPU 版。
- 当前无 CUDA，未安装 `open_clip`、`onnx` 和 `onnxruntime`。
- Node.js、Chrome 和 Edge 已安装，可用于后续浏览器兼容性验证。
- `pill` 数据根目录为 `dataset/pill/pill`：267 张训练正常图、26 张测试正常图、141 张异常图。
- `screw` 数据根目录为 `dataset/screw/screw`：320 张训练正常图、41 张测试正常图、119 张异常图。
- 两类异常图与 `_mask.png` 掩膜一一对应，未发现孤立掩膜、缺失掩膜或不可读 PNG。
- 数据采用 CC BY-NC-SA 4.0，实验与材料保留原始引用和许可说明。

实现必须兼容用户提供路径比真实类别根目录多一层同名目录的情况。

## 3. 模型定义

### 3.1 模型身份

本阶段模型命名为 **CLIP Patch Baseline**。它不是 WinCLIP，也不是 AnomalyCLIP。

模型由以下部分组成：

1. 预训练 CLIP ViT-B/16 视觉编码器；
2. 一个轻量 Patch Adapter/异常头；
3. 离线计算并固化的对象无关正常/异常文本原型。

WinCLIP 和 AnomalyCLIP 在后续正式研究中作为对照方法或可选教师，不进入中期最低线的实现依赖。

### 3.2 输入、特征和输出

- 输入：RGB、`float32 [B,3,224,224]`。
- 归一化：CLIP 标准 mean/std。
- 主干：CLIP ViT-B/16，全部参数冻结。
- Patch 网格：移除 CLS Token 后得到 `14×14` Patch 特征。
- 异常头：两层轻量 MLP，将 Patch 特征映射到文本原型所在空间。
- Patch 概率：归一化 Patch 表征分别与正常、异常原型计算相似度，经二类 softmax 得到异常概率。
- 异常图：`anomaly_map float32 [B,14,14]`。
- 图像分数：每张异常图的 `99.5%` 分位数，输出 `image_score float32 [B]`。

### 3.3 文本原型

正常 Prompt：

- `an intact industrial object`
- `a normal industrial object`
- `an industrial object without defects`

异常 Prompt：

- `a damaged industrial object`
- `a defective industrial object`
- `an anomalous industrial object`

同组文本特征先逐个 L2 归一化，再取均值并再次归一化。训练和部署使用相同的固化原型，不使用 `pill`、`screw` 或未来目标域类别名。浏览器部署不包含文本编码器。

### 3.4 训练目标

训练只更新轻量异常头，损失固定为：

```text
L = FocalLoss(anomaly_map, patch_mask) + DiceLoss(anomaly_map, patch_mask)
```

Patch 掩膜采用缺陷保持下采样。只要原始掩膜包含异常像素，对应覆盖 Patch 不得因普通插值而全部变为零。

## 4. 数据协议

### 4.1 协议定位

MVTec AD 官方训练集只含正常图像，但本研究设定允许源域异常监督。因此本阶段会重新组织源域开发划分，不把结果表述为 MVTec 官方无监督协议成绩。

### 4.2 划分原则

- 原始 `train/good` 用于正常训练样本。
- 原始 `test/good` 固定用于正常验证样本。
- 原始 `test/<defect>` 按类别、缺陷类型和掩膜面积进行确定性分层拆分，分别形成异常训练和异常验证样本。
- 默认随机种子固定为 `42`。
- 单个原始图像只能属于一个划分。
- 划分清单记录类别、缺陷类型、图像路径、掩膜路径、二值标签、掩膜面积比例和用途。
- 所有路径以项目根目录为基准保存为相对路径，保证项目移动后仍可复现。

配置文件必须显式记录异常训练/验证比例和正常训练采样上限。具体数值在实施计划中固定，不通过查看验证指标反复搜索。

### 4.3 CPU 执行策略

冻结主干仅对每张参与实验的图像执行一次前向。Patch 特征、文本原型引用、Patch 掩膜、样本标签和源文件元数据写入版本化缓存。训练和验证阶段只读取缓存并运行轻量异常头。

先用固定小样本执行冒烟实验，链路通过后使用两类完整开发划分。冒烟结果不得写入正式结果表。

## 5. 软件结构

```text
configs/
└── midterm_cpu.yaml

src/clip_anomaly/
├── __init__.py
├── config.py
├── data.py
├── preprocessing.py
├── prompts.py
├── backbone.py
├── cache.py
├── model.py
├── losses.py
├── metrics.py
├── visualization.py
├── training.py
└── export.py

scripts/
├── audit_environment.py
├── audit_dataset.py
├── cache_features.py
├── train_baseline.py
├── evaluate_baseline.py
├── export_onnx.py
└── compare_pytorch_onnx.py

web/
├── index.html
├── app.js
└── styles.css

tests/
├── test_config.py
├── test_data.py
├── test_preprocessing.py
├── test_cache.py
├── test_model.py
├── test_losses.py
├── test_metrics.py
└── test_export.py
```

`clip_anomaly` 表示“基于 CLIP 的异常检测代码”，避免使用容易与 AnomalyCLIP 方法混淆的 `anomaly_clip` 名称。

训练、评测、导出和浏览器预处理必须共享同一组常量和可核对的测试向量，禁止分别实现互不验证的归一化或评分规则。

## 6. 实验产物

```text
midterm/
├── README.md
├── experiments/
│   ├── environment.md
│   ├── source_split.csv
│   ├── baseline_config.yaml
│   ├── status.md
│   ├── logs/
│   ├── checkpoints/
│   ├── feature_cache/
│   ├── figures/
│   └── results.csv
└── deployment/
    ├── clip_b16_patch_head_b1.onnx
    ├── export_log.txt
    ├── pytorch_onnx_comparison.csv
    └── webgpu_console_log.txt
```

数据集、特征缓存、checkpoint 和 ONNX 大文件默认不提交 Git。Git 提交配置、代码、测试、划分清单、结果 CSV、日志摘要和体积合理的真实可视化证据。

## 7. 评价与可视化

正式结果至少包含：

- 验证 Loss；
- Image AUROC；
- Image AP；
- Pixel AUROC；
- Pixel AP；
- 正常样本、明显异常、微小异常和至少一个误报或漏检案例的热力图。

指标按 `pill`、`screw` 和合并集合分别报告。无法计算的单类指标明确记录为不可计算，不使用零值代替。所有图表注明数据类别、缺陷类型、源文件、模型 checkpoint 和“源域开发验证”。

## 8. ONNX 与 WebGPU 边界

ONNX 最小接口固定为：

```text
input:
  image         float32 [1,3,224,224]

output:
  anomaly_map   float32 [1,14,14]
  image_score   float32 [1]
```

先完成静态 B=1 导出、ONNX checker 和 ONNX Runtime CPU 前向，再比较 PyTorch 与 ONNX 的异常图、图像分数和多样本排序。只有 CPU 一致性通过后才尝试浏览器 WebGPU。

浏览器阶段只承担图像读取、同等预处理、ONNX Runtime WebGPU 前向、热力图上采样与叠加，以及设备和时间展示。必须确认实际 execution provider；WASM 或 CPU 回退不能写成 WebGPU 成功。

## 9. 测试策略

实现采用测试先行。每项生产行为先由失败测试定义，再写最小实现使其通过。

### 9.1 数据与配置

- 自动识别同名双层类别目录。
- 异常图与掩膜一一配对，正常样本产生全零掩膜。
- 固定种子得到完全相同的划分。
- 训练与验证文件集合不相交。
- 缺失路径、孤立掩膜和非法比例给出明确错误。

### 9.2 预处理与缓存

- 图像输出为 `float32 [3,224,224]`，RGB 顺序和 CLIP mean/std 正确。
- Patch 掩膜输出为 `[14,14]`，微小非空缺陷不会在缺陷保持下采样后消失。
- 缓存记录模型标识、预处理版本、配置哈希和源文件相对路径；元数据不匹配时拒绝静默复用。

### 9.3 模型、损失与指标

- 批量输出 shape 正确，异常概率位于 `[0,1]`。
- 图像分数严格按 `99.5%` 分位数计算。
- CLIP 主干参数无梯度，异常头参数有梯度。
- 完美预测的 Dice Loss 接近零，明显错误预测的组合损失更高。
- 使用人工构造数据验证 AUROC/AP 的准确值和异常输入行为。

### 9.4 导出与一致性

- ONNX 输入输出名称、dtype 和静态 shape 符合约定。
- ONNX checker 通过，ONNX Runtime CPU 能加载并前向。
- 固定输入的 PyTorch/ONNX 最大绝对误差和平均绝对误差被实际计算并写入对照结果。

## 10. 执行阶段与验收

1. 环境、数据审计和固定划分；
2. 小样本 CLIP 前向和特征缓存；
3. 异常头冒烟训练和验证；
4. 两类完整开发实验；
5. ONNX 导出和 CPU 一致性；
6. WebGPU 条件性验证；
7. 状态表、结果图和复现说明收口。

每阶段验收通过后进入下一阶段。失败时保存配置、命令、日志和唯一下一步，不创建虚假成功产物。

中期最低验收要求：

1. 环境记录和固定数据划分可复现；
2. 至少一次真实 CLIP Patch 特征前向；
3. 可复现的训练日志、配置和 checkpoint；
4. 四项基础检测指标及真实热力图；
5. 静态 B=1 ONNX 成功证据或可复现失败证据；
6. PyTorch/ONNX CPU 对照或明确失败记录；
7. 所有项目使用“已完成、部分完成、验证失败、未开展、不适用”之一标注状态。

WebGPU 成功是加分项，不是 CPU 最低线合格的前置条件。

## 11. 止损与回退

- ViT-B/16 无法在当前 CPU 环境运行：保存错误和资源记录后，使用 ViT-B/32 建立兼容性基线；不得将其结果混写为 B/16。
- 固定文本原型匹配没有有效异常信号：使用同等容量的普通二分类 Patch 头作为回退，并把文本原型有效性标为未通过。
- ONNX 导出失败：保存最小复现、算子和错误日志，保留 PyTorch 最低线，不直接开始 WebGPU 页面包装。
- WebGPU 失败：保存浏览器版本、适配器信息、执行入口和控制台日志，不用 WASM 回退替代成功证据。
- CPU 完整实验耗时不可接受：先完成固定小样本真实链路并记录样本范围，不降低真实性要求，也不填写估计指标。

## 12. 明确排除

本阶段不实现：

- M1 完整消融和全图/ROI 联合训练；
- M2 动态 K、设备成本表和学习式区域排序；
- 正式 MVTec AD ↔ VisA 双向迁移；
- 教师蒸馏、Prompt Learning、复杂多层特征融合；
- 动态 Batch、Graph Capture、FP16/INT8 量化；
- Oracle/Top-K ROI，除非全部最低验收项已经完成；
- 复杂前端、PPT 自动生成和未经测量的性能结论。

