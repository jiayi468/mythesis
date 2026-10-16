# Oracle ROI 路线决策

- 决策：**GO**
- 合并平均 Pixel AP 增量：`0.043036`
- 各类别正向种子数：`{'pill': 3, 'screw': 3}`
- 说明：Oracle ROI 达到预注册门槛；下一阶段可以实现自动 Top-K 与 Region Recall@K。

该实验使用 GT 掩膜定位 ROI，仅是方法上界诊断，不是可部署结果；统计量为异常图像逐图 Pixel AP 的均值。
