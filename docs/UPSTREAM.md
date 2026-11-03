# 上游来源与本发布快照

主要上游是Microsoft TimeCraft，提交9a81eeef4f8487353a4aedb0b80b8a534e765720。原神经网络/扩散代码位于DiGA/diffusion，版权与MIT许可保留。其文件头进一步注明基于lucidrains/denoising-diffusion-pytorch，本仓库一并保留相关MIT许可和链接。

| 文件 | 来源与变化 |
|---|---|
| code/diffusion/cont_ctrl_net.py | 复用TimeCraft连续条件1D U-Net；本发布快照未修改网络 |
| code/diffusion/disc_ctrl_net.py | 复用TimeCraft离散条件1D U-Net；本发布快照未修改网络 |
| code/diffusion/ddpm.py | 复用扩散核心；增加正数逐通道x0边界及非持久buffer以支持设备迁移/旧权重加载 |
| code/prepare.py | 日频连续窗口、训练期归一化和分箱，增加明确日期校验 |
| code/train.py / evaluate.py | 独立PyTorch训练及验证期CFG评估；复用既有日频实现 |
| code/generate.py / fidelity.py | 独立本地采样与四指标CLI，避免研究工作区/服务器硬编码 |
| code/smoke.py | 自动生成合成面板，覆盖训练/采样/评估，不依赖私有行情 |
| docs/figures/render_showcase.py | 仅用历史汇总统计生成精选图 |
| docs/figures/render_report_figures.py | 从本地保存样本重绘真实/生成密度、量价路径、五档控制与裁剪对照；不内嵌数据 |

上游原始Git blob记录见third_party/provenance.json；fidelity指标参考函数置于third_party中，作为来源记录，正常运行不导入其全局环境。

本仓库固定范围为日频扩散生成模块；没有复制后续OHLCV联合模型、强化学习策略、服务器管理、私有配置或训练产物。新默认边界来自本地裁剪消融，不是论文原始参数。目录中的代码调整旨在可移植运行和明确采样配置，不宣称提出新的U-Net或扩散算法。

原先的标量x0_clip在此成为不写入state_dict的逐通道buffer，所以同结构旧checkpoint参数键保持兼容。持久化配置仍记录实际采样边界。原训练合同、预处理hash和权重不因演示图而修改。
