# 中期答辩补充实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development`（推荐）或 `executing-plans` 按任务执行。未通过当前任务验收前，不进入下一项高成本实验。

**Goal:** 在 2～3 天内，以当前真实基线为起点，补齐多种子可信度、三组公平基线、图像分数诊断、Oracle ROI 上界和浏览器预处理一致性证据，并形成可直接用于中期答辩的可追溯材料。

**Architecture:** 保留现有“冻结 CLIP ViT-B/16 + 轻量 Patch 异常头”链路，以一份冻结的数据清单和共享特征缓存驱动所有内部对照。按“协议冻结 → 多种子与公平基线 → Oracle 路线判定 → 部署一致性 → 材料收口”推进；补充结果写入独立目录，不覆盖当前最低线证据。

**Tech Stack:** Python 3.10、PyTorch 1.13、OpenCLIP、NumPy、scikit-learn、Pillow、ONNX Runtime、pytest、JavaScript、ONNX Runtime WebGPU、Chrome。

**Spec:** `docs/研究内容v3.md`；基础实验协议见 `docs/superpowers/specs/2026-10-16-midterm-minimum-experiment-design.md`。

## Global Constraints

- 当前环境为 CPU 版 PyTorch，无 CUDA；时间预算按 CPU 实测安排。
- 当前结果只能称为“源域小样本探索性验证”，不得写成 MVTec AD 官方无监督成绩或正式跨数据集零样本结论。
- 后续“零样本”严格指每个迁移方向的目标域零训练、零调参、零阈值校准；本轮不宣称已经完成。
- 固定 CLIP ViT-B/16、输入 `224×224`、Patch 网格 `14×14`；仅训练轻量异常头。
- 固定损失为 `Focal Loss + Dice Loss`；B1/B2 共用训练预算，不单独调参。
- 数据划分种子固定为 `42`；训练种子为 `42、123、2026`，三次训练必须共用数据清单和特征缓存。
- 扩展上限：每类训练正常 48、训练异常 48、验证异常 32；`pill` 验证正常使用全部 26 张，`screw` 验证正常固定 32 张。
- 现有 96 张实验以及 `midterm/experiments/`、`midterm/deployment/` 中的证据不得覆盖、改名或删除。
- 所有答辩数字必须回溯到固定配置、CSV、日志、checkpoint 或图像；未运行实验不得填写估计值。
- WebGPU 只报告兼容性和实测时延；预处理、多设备和稳定性证据完成前不使用“加速”表述。

---

## 1. 进展快照与待回答问题

### 1.1 已完成证据

| 项目 | 当前事实 | 证据 |
|---|---|---|
| 数据与模型 | `pill`、`screw` 共 96 张；冻结 CLIP ViT-B/16；Adapter + 固定文本原型 | `configs/midterm_cpu.yaml`、`midterm/experiments/source_split.csv` |
| 自动化验证 | 18 项测试已通过 | `tests/` |
| 训练 | 种子 42；最佳 Epoch 29；验证组合 Loss 0.930274 | `midterm/experiments/logs/training_history.csv` |
| `pill` | Image AUROC/AP 0.921875/0.929365；Pixel AUROC/AP 0.992669/0.813968 | `midterm/experiments/results.csv` |
| `screw` | Image AUROC/AP 0.531250/0.711905；Pixel AUROC/AP 0.963915/0.155535 | 同上 |
| ONNX | 静态 `B=1` 成功；5 张样本最大热力图绝对误差 `3.5137e-5` | `midterm/deployment/pytorch_onnx_comparison.csv` |
| WebGPU | Chrome 155、AMD Radeon 集显真实前向；预热 P50/P95 3728.7/3853.6 ms | `midterm/deployment/webgpu_performance.csv` |
| 后端一致性 | 固定 Python 张量下 ONNX CPU 分数 0.166999131，WebGPU 显示 0.167000 | `midterm/deployment/webgpu_reference.json` |
| 预处理缺口 | Canvas 同图分数 0.185323，与固定张量相差约 0.0183 | `midterm/experiments/status.md` |

### 1.2 本轮必须回答

1. 换训练种子后是否稳定，特别是 `screw` 的失败是否可重复？
2. 固定文本原型头 B2 是否优于同训练预算的普通分类头 B1；原始 CLIP Patch 语义 B0 能提供多少能力？
3. 真实缺陷 ROI 放大后是否存在稳定上界收益，是否值得继续 Top-K、M1 和动态 K？
4. 浏览器端差异来自图像预处理还是 WebGPU 后端？

### 1.3 本轮不实现

- MVTec AD ↔ VisA 正式双向迁移；
- 自动 Top-K、NMS、Region Recall@K 和 M2 动态 K；
- Prompt Learning、蒸馏、多层融合、量化、Graph Capture；
- AnomalyCLIP 完整训练；
- 第三个类别，除非所有必做项提前完成且仍有一个完整工作日。

---

## 2. 文件结构

### 2.1 代码与配置

| 文件 | 责任 |
|---|---|
| `configs/midterm_supplement.yaml` | 固定样本上限、划分种子、训练种子、三类头、聚合规则、ROI 比例 |
| `src/clip_anomaly/config.py` | 向后兼容地解析 `experiment` 配置，分离划分种子与训练种子 |
| `src/clip_anomaly/backbone.py` | 将 OpenCLIP Patch token 通过视觉投影映射到文本空间，供 B0 使用 |
| `src/clip_anomaly/model.py` | 三类头与统一图像分数聚合 |
| `src/clip_anomaly/training.py` | 按 `head_type` 训练 B1/B2，并记录种子、类型、参数量 |
| `src/clip_anomaly/experiment.py` | 冻结清单、共享缓存、互不覆盖的运行目录 |
| `src/clip_anomaly/roi.py` | Oracle 外接框、上下文扩展、回投和融合 |
| `scripts/run_supplement_baselines.py` | 执行 B0/B1/B2 × 三种子并汇总 |
| `scripts/diagnose_score_aggregation.py` | 比较 max、0.995 quantile、top-5 mean |
| `scripts/evaluate_oracle_roi.py` | 执行 Full、Oracle-1、Full+Oracle-1、Random ROI |
| `scripts/compare_browser_preprocessing.py` | 对比 Python 与浏览器输入张量和输出 |
| `scripts/build_midterm_report.py` | 从真实 CSV 生成图表和状态表 |
| `web/app.js` | 增加张量导出和预处理元数据，不改模型接口 |

### 2.2 新增测试

- `tests/test_baselines.py`：三类头、投影、参数口径。
- `tests/test_score_aggregation.py`：三种图像分数的精确数值。
- `tests/test_experiment_matrix.py`：固定清单、共享缓存、输出隔离。
- `tests/test_roi.py`：外接框、边界、回投、融合。
- `tests/test_preprocessing_parity.py`：张量格式和误差统计。

### 2.3 输出目录

```text
midterm/
├── experiments/supplement/
│   ├── protocol.yaml
│   ├── source_split.csv
│   ├── feature_cache/
│   ├── baselines/{runs.csv,summary.csv,<head_type>/seed_<seed>/}
│   ├── score_aggregation.csv
│   ├── oracle_roi/{per_image.csv,summary.csv,decision.md,figures/}
│   └── figures/
├── deployment/preprocessing_consistency.csv
├── report/{中期答辩讲稿.md,答辩问答.md,方法与协议对照.md,实验状态表.md}
└── slides/{中期答辩.pptx,中期答辩.pdf}
```

---

## 3. 决策门

| 顺序 | 决策门 | 通过条件 | 未通过时动作 |
|---:|---|---|---|
| 1 | 协议冻结 | 清单无交集；三种子共用清单；旧结果未覆盖 | 只修协议，不训练 |
| 2 | B0/B1/B2 可比 | 相同特征来源、训练预算、损失和指标；参数量已记录 | 修正公平性后重跑 |
| 3 | 多种子可信度 | 三种子均有日志、checkpoint、逐类结果 | 缺失即标“部分完成” |
| 4 | Oracle Go/No-Go | 至少一类在多数种子中 Pixel AP 提升，合并平均提升 ≥ 0.02 | 否则停止 Top-K/M2 |
| 5 | 预处理一致性 | 固定张量后端分数误差 ≤ `1e-4`；图像路径分数误差 ≤ `1e-3` | 只保留固定张量证据 |
| 6 | 材料闭环 | 每个 PPT 数字可追溯，已完成与拟开展明确分区 | 删除无来源数字 |

文本原型结论预先冻结：B2 在 `pill`、`screw` 上均至少有 2/3 个种子的 Pixel AP 不低于 B1，且合并三种子平均 Pixel AP 高出至少 0.01，才写“当前协议下有稳定正向证据”；否则写“相近”“不稳定”或“弱于普通分类头”。

---

## 4. 实施任务

### Task 1：冻结补充协议并保护既有证据

**Files:** Create `configs/midterm_supplement.yaml`; modify `src/clip_anomaly/config.py`, `src/clip_anomaly/experiment.py`; test `tests/test_config.py`, `tests/test_experiment_matrix.py`。

**Interfaces:** `ExperimentConfig.experiment: dict[str, Any]`; `supplement_output(config, head_type, seed) -> Path`；唯一的补充实验 `source_split.csv`。

- [ ] 写失败测试：划分种子为 42、训练种子为 `[42,123,2026]`，B1/B2 六个运行目录互不相同。

```python
def test_supplement_matrix_separates_runs(config):
    assert config.experiment["split_seed"] == 42
    assert config.experiment["training_seeds"] == [42, 123, 2026]
    paths = {supplement_output(config, h, s) for h in ("b1_classifier", "b2_text_prototype") for s in (42, 123, 2026)}
    assert len(paths) == 6
```

- [ ] 运行 `python -m pytest tests/test_config.py tests/test_experiment_matrix.py -q`，确认因接口未定义而失败。
- [ ] 新配置固定 `split_seed: 42`、`training_seeds: [42,123,2026]`、三类头、三种聚合、`roi_context_ratio: 0.25` 及 Global Constraints 中的样本上限；旧配置继续可加载。
- [ ] 运行 `python -m pytest -q`，预期全部通过。
- [ ] 运行 `python scripts/audit_dataset.py --config configs/midterm_supplement.yaml`；验收：训练/验证路径交集为 0，清单不随训练种子变化。

### Task 2：实现三组公平基线

**Files:** Modify `src/clip_anomaly/backbone.py`, `model.py`, `training.py`, `experiment.py`; create `tests/test_baselines.py`。

**Interfaces:**
- `project_patch_tokens(tokens, projection) -> Tensor`
- `RawClipPatchHead(prototypes)`：无可训练参数，输入已投影 Patch 特征。
- `LearnedPrototypePatchHead(input_dim, embedding_dim, hidden_dim, dropout)`：与 B2 使用相同 Adapter。
- 现有 `PatchAnomalyHead` 作为 B2。

- [ ] 先写形状与参数测试。

```python
def test_baseline_heads_share_contract():
    prototypes = torch.eye(2, 4)
    b0 = RawClipPatchHead(prototypes)
    assert sum(p.numel() for p in b0.parameters()) == 0
    assert b0(torch.randn(2, 196, 4))[0].shape == (2, 14, 14)
    b1 = LearnedPrototypePatchHead(6, 4, 8, dropout=0.0)
    b2 = PatchAnomalyHead(6, 4, 8, prototypes, dropout=0.0)
    assert sum(p.numel() for p in b1.parameters()) - sum(p.numel() for p in b2.parameters()) == 2 * 4
```

- [ ] 运行 `python -m pytest tests/test_baselines.py -q`，确认失败。
- [ ] 实现 B0：必须使用视觉投影矩阵将 768 维 Patch token 映射到文本空间；禁止截断或随机映射。实现 B1：相同两层 Adapter + 两个可学习归一化类别原型。B1 比 B2 固定多出 `2 × embedding_dim` 个可训练参数，结果表必须显式报告该差异，不将其描述为完全等参数。缓存同时保存原始 `features` 与 `projected_features`，并升级缓存版本。
- [ ] 运行 `python -m pytest tests/test_baselines.py tests/test_cache.py tests/test_model.py -q` 及 `python -m pytest -q`，预期全部通过。

### Task 3：执行扩展样本、多种子和三基线矩阵

**Files:** Create `scripts/run_supplement_baselines.py`, `tests/test_experiment_matrix.py`; produce `baselines/runs.csv`, `baselines/summary.csv`。

**Output schema:** `head_type,seed,scope,samples,trainable_parameters,best_epoch,best_val_loss,image_auroc,image_ap,pixel_auroc,pixel_ap,status`。

- [ ] 写失败测试，固定 7 次运行：B0 一次，B1/B2 各三个种子。
- [ ] 实现 `build_run_matrix([42,123,2026])` 和原子化状态记录；开始写 `running`，成功写 `complete`，异常写 `failed` 并保留日志。
- [ ] 运行 `python scripts/run_supplement_baselines.py --config configs/midterm_supplement.yaml --dry-run`；验收：打印 7 次运行、唯一清单、唯一缓存，不生成伪结果。
- [ ] 运行 `python scripts/cache_features.py --config configs/midterm_supplement.yaml --batch-size 4`；验收：缓存覆盖冻结清单且可重新加载。
- [ ] 运行 `python scripts/run_supplement_baselines.py --config configs/midterm_supplement.yaml`。
- [ ] 验收：B1/B2 每个种子有独立日志和 checkpoint；`runs.csv` 报告逐类与合并结果；`summary.csv` 报告均值和样本标准差。缺失种子不得用其他结果替代。

### Task 4：诊断图像级分数聚合

**Files:** Modify `src/clip_anomaly/model.py`; create `scripts/diagnose_score_aggregation.py`, `tests/test_score_aggregation.py`; produce `score_aggregation.csv`。

**Interface:** `aggregate_image_scores(anomaly_map, method, top_k=5, quantile=0.995) -> Tensor`。

- [ ] 先写精确值测试。

```python
def test_score_aggregations_are_exact():
    x = torch.arange(196, dtype=torch.float32).reshape(1, 14, 14)
    assert aggregate_image_scores(x, "max").item() == 195.0
    assert aggregate_image_scores(x, "topk_mean", top_k=5).item() == 193.0
    assert torch.allclose(aggregate_image_scores(x, "quantile", quantile=.995), torch.quantile(x.flatten(1), .995, dim=1))
```

- [ ] 保持现有 `image_scores_from_map()` 等价于 0.995 quantile，避免改变旧 ONNX 口径。
- [ ] 运行 `python scripts/diagnose_score_aggregation.py --config configs/midterm_supplement.yaml`；只对 B2 三种子已有异常图重算，不重新训练、不搜索 k。
- [ ] 输出 `seed,scope,method,image_auroc,image_ap`；同一验证集上的最优项只能作为诊断，不能写成无偏最终选择。

### Task 5：执行 Oracle ROI 并决策路线

**Files:** Create `src/clip_anomaly/roi.py`, `tests/test_roi.py`, `scripts/evaluate_oracle_roi.py`; produce `oracle_roi/per_image.csv`, `summary.csv`, `decision.md`。

**Interfaces:** `mask_bbox(mask)`、`expand_bbox(box,width,height,ratio=.25)`、`paste_roi_map(canvas,roi_map,box)`、`fuse_maps(full_map,roi_map)`；融合固定为逐像素最大值。

- [ ] 写边缘缺陷测试。

```python
def test_bbox_expands_and_clips():
    mask = np.zeros((100, 120), dtype=np.uint8); mask[:10, 110:] = 1
    box = expand_bbox(mask_bbox(mask), 120, 100, .25)
    assert 0 <= box[0] < box[2] <= 120
    assert 0 <= box[1] < box[3] <= 100
```

- [ ] 实现 Oracle-1：最小外接框扩展 25% 上下文，裁剪原图并缩放到 224，使用同一 B2 权重，回投原图；对照为 Full、Oracle-1、Full+Oracle-1、固定种子 42 的同尺寸 Random ROI。
- [ ] 运行 `python -m pytest tests/test_roi.py -q` 和 `python -m pytest -q`，预期全部通过。
- [ ] 运行 `python scripts/evaluate_oracle_roi.py --config configs/midterm_supplement.yaml --seeds 42 123 2026`。
- [ ] 每个异常样本记录全图/ROI/融合分数、Pixel AP、缺陷面积、框坐标和种子。Oracle 使用 GT，Image AUROC 只能标为上界诊断。
- [ ] `decision.md` 只写一种结论：`GO`（满足决策门）、`HOLD`（个例改善或跨种子不一致）、`NO-GO`（无改善或退化）。只有 `GO` 才在后续实现 Top-K。

### Task 6：量化浏览器预处理一致性

**Files:** Modify `web/app.js`; create `scripts/compare_browser_preprocessing.py`, `tests/test_preprocessing_parity.py`; produce `midterm/deployment/preprocessing_consistency.csv`。

- [ ] 写 `compare_tensors()` 失败测试，输入 `[0,1]` 与 `[0,1.25]` 时最大误差 0.25、平均误差 0.125。
- [ ] 浏览器增加 little-endian `float32 [1,3,224,224]` 张量导出；元数据记录源图、浏览器版本、Canvas 尺寸、RGBA→RGB、mean/std、shape 和 dtype。
- [ ] 从 `source_split.csv` 固定首张验证图，运行 `prepare_webgpu_reference.py`，再导出 Canvas 张量。
- [ ] 运行 `python scripts/compare_browser_preprocessing.py --config configs/midterm_cpu.yaml --browser-tensor midterm/deployment/browser_input.bin`。
- [ ] CSV 分开记录 Python↔Canvas 输入误差、ONNX CPU↔WebGPU 固定输入输出误差、图像路径分数误差。固定输入后端分数误差 ≤ `1e-4` 且图像路径分数误差 ≤ `1e-3` 才标“通过”；否则保留固定张量结论并记录 Canvas/PIL 插值限制。

### Task 7：生成图表与可追溯摘要

**Files:** Create `scripts/build_midterm_report.py`; produce `loss_curves.png`, `score_distribution.png`, `pr_roc_curves.png`, `failure_gallery.png`, `midterm/report/实验状态表.md`。

- [ ] 启动时校验：每个 `complete` 运行都有配置和日志，B1/B2 各有三种子；缺失时只生成“部分完成”，不画误导性均值。
- [ ] 运行 `python scripts/build_midterm_report.py --root midterm`。
- [ ] 图表标题包含类别、方法、种子范围和“源域开发验证”；Pixel AUROC/AP 同时展示；失败案例含 `screw` 漏检、正常高分、微小异常。
- [ ] 从候选 PPT 表随机抽 5 个数字，逐个定位到 CSV 行、配置和日志；无法追溯的数字不得使用。

### Task 8：完成答辩材料和最终审查

**Files:** Create `midterm/report/中期答辩讲稿.md`, `答辩问答.md`, `方法与协议对照.md`, `midterm/slides/中期答辩.pptx`, `中期答辩.pdf`; modify `midterm/experiments/status.md`, `midterm/README.md`。

- [ ] PPT 固定 11 页：封面；背景；相关工作/协议；目标/边界；路线；数据协议；当前模型；工程链路；三基线多种子结果；失败/Oracle/预处理；决策与后续。
- [ ] 写 5～8 分钟讲稿和 10 个问答，覆盖 CLIP 选择、文本原型、异常监督与零样本口径、Pixel AUROC/AP、`screw` 失败、Oracle 边界、WebGPU 未加速、不可与排行榜直比、Oracle 回退、目标域隔离。
- [ ] `status.md` 只使用“已完成、部分完成、验证失败、未开展、不适用”；`README.md` 仅列实际可运行命令。
- [ ] 运行 `python -m pytest -q`，预期全部通过。
- [ ] 运行 `python scripts/build_midterm_report.py --root midterm --check-only`，预期无缺失、重复运行键和不可追溯数字。
- [ ] 断网打开 PDF、截图、日志和录屏，完整彩排；超过 8 分钟优先删背景，不删限制与失败结果。

---

## 5. 2～3 天排期

| 时间 | 必做任务 | 验收产物 |
|---|---|---|
| Day 1 上午 | Task 1、2；同步建 PPT 骨架 | 冻结协议、测试通过、三类头、11 页标题 |
| Day 1 下午/晚上 | Task 3；缓存只提取一次 | 7 次运行状态、日志、checkpoint、CSV |
| Day 2 上午 | 完成 Task 3、4 | 三基线汇总、分数诊断 |
| Day 2 下午 | Task 5 | Oracle 明细、汇总、唯一决策 |
| Day 2 晚上 | Task 7 初版、Task 8 文稿 | 真实图表、状态表、讲稿、问答 |
| Day 3 上午 | Task 6 | 预处理误差报告或失败边界 |
| Day 3 下午 | Task 7、8 收口 | PPT、PDF、说明、测试与追溯检查 |

若只有两天：保留 Task 1～5、7～8；Task 6 只报告固定张量一致性和当前约 0.0183 的差异。官方 WinCLIP、第三类别和界面美化不得挤占必做项。

## 6. 风险与止损

| 风险 | 止损线 | 回退 |
|---|---|---|
| CPU 缓存过慢 | 超过 4 小时 | 缩减训练样本但不减少验证类别，记录实际样本数 |
| 三种子未跑完 | 材料收口前仍未结束 | 已完成结果标“部分完成”，不算虚假均值 |
| B0 维度不匹配 | Patch token 未经视觉投影 | 正确投影后再跑，禁止截断/随机映射 |
| B1 优于 B2 | 公平性复核后仍成立 | 收缩文本贡献，以 B1 为强基线 |
| `screw` 仍接近随机 | 三种子一致 | 如实报告，优先检查聚合和特征层 |
| Oracle 无收益 | `NO-GO` 成立 | 停止 Top-K/M2，转向特征与缺陷保持监督 |
| Canvas 半天未对齐 | 图像路径分数误差 > `1e-3` | 只保留固定张量后端一致性 |
| Day 2 晚上无材料骨架 | 时间不足 | 停止条件性实验，优先 PPT、讲稿、问答、PDF |

## 7. 最终验收

- [ ] 原 96 张实验、ONNX 和 WebGPU 证据未覆盖；
- [ ] 清单固定、训练/验证无交集、三种子共享清单；
- [ ] B0/B1/B2 特征空间正确，B1/B2 预算相同；
- [ ] 三种子单次结果、均值、标准差、参数量齐全或明确标“部分完成”；
- [ ] `pill`、`screw`、合并指标分别报告，Pixel AUROC/AP 同时出现；
- [ ] 聚合诊断未包装成无偏模型选择；
- [ ] Oracle 有逐样本数据、图像和唯一 `GO/HOLD/NO-GO`；
- [ ] 固定张量与图像路径的 WebGPU 误差分开报告；
- [ ] PPT、PDF、讲稿、问答、状态表、离线证据齐全；
- [ ] 论文数字、官方参考运行、本项目结果分区；
- [ ] PPT 数字全部可追溯；
- [ ] 全量测试通过，或失败项已如实记录。

本轮结束后只作三项决策：是否保留文本原型头、是否继续粗到细路线、浏览器端是否具备多设备性能实验前提。正式双向跨数据集实验另立实施计划。
