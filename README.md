# DigMA-Diffusion-Daily

**基于 DigMA 扩散控制器的日频量价情景生成。** 学习历史股票量价分布，生成 **60 个交易日的收益率与成交量序列**，支持无条件、离散条件和连续条件三种方式。

*A daily-frequency adaptation of DigMA / TimeCraft's diffusion controller for controllable return–volume scenario generation.*

[![Synthetic smoke](https://github.com/adrianelo0107/DigMA-Diffusion-Daily/actions/workflows/smoke.yml/badge.svg)](https://github.com/adrianelo0107/DigMA-Diffusion-Daily/actions/workflows/smoke.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> 代码快照 **v0.1.0**。默认使用 **对数成交量 + 收益裁剪 ±6 / 成交量裁剪 ±3**。仓库仅含源码、配置、文档及由汇总统计绘制的结果图，**不包含真实行情、生成路径、模型权重或服务器配置**。

## 主要参考

Yu-Hao Huang, Chang Xu, Yang Liu, Weiqing Liu, Wu-Jun Li, Jiang Bian. **Controllable Financial Market Generation with Diffusion Guided Meta Agent**, AAAI 2026.

- [论文链接的扩展版本](https://arxiv.org/abs/2408.12991)
- [官方代码 Microsoft / TimeCraft](https://github.com/microsoft/TimeCraft)
- 参考提交：[`9a81eeef`](https://github.com/microsoft/TimeCraft/tree/9a81eeef4f8487353a4aedb0b80b8a534e765720/DiGA)

本项目复用其 **1D U-Net、条件编码、噪声预测目标与 CFG**，重点适配扩散模块。原论文生成日内市场状态并通过 Meta Agent 形成订单流；这里直接生成日频对数收益与 log1p 成交量，覆盖预处理、训练、采样和生成质量评估。未包含 Meta Agent、交易所或强化学习模块。

## 主要调整

| 原论文 / 原实现 | 本项目 |
|---|---|
| 日内市场状态、分钟收益与订单到达率 | 日频对数收益与 log1p 成交量，双通道60日窗口 |
| 原始数据与 Lightning 包装 | 独立 PyTorch 训练器、配置管理、断点与哈希核验 |
| 行情控制目标 | 保留收益、振幅、波动率；各按训练集五分位分为五档 |
| 条件采样 | 支持离散档位与连续数值，验证期选择 CFG 强度 |
| 单一采样边界 | 标准化收益与成交量分通道设定，默认 `[6, 3]` |
| 日内市场指标 | 日收益 DayR、收益相关 RetAC、波动聚集 VolC，新增成交量 LogV |

数据按日期划分，窗口不跨集合边界；通道归一化、档位边界与生成目标只用训练集拟合。三个目标分别训练，每条路径接受其中一个目标的控制。

```mermaid
flowchart LR
    A[用户日频量价面板] --> B[连续60日窗口与训练期标准化]
    B --> C[1D U-Net / DDPM噪声预测]
    D[收益 / 振幅 / 波动率条件] --> C
    C --> E[DDIM采样 + CFG]
    E --> F[分通道干净状态裁剪]
    F --> G[收益率与成交量路径]
```

**训练与采样：** 默认采用 DDPM 噪声预测训练，扩散步数200；生成时采用 **DDIM 20步**。文件名 `ddpm.py` 不意味着默认运行200步DDPM采样。

## 精选结果：收紧成交量采样边界

固定三个已训练无条件模型，不重新训练。每种子生成5,000条60日路径，调整前后使用相同权重、初始噪声与批次划分。下表为三个种子的均值 ± 样本标准差。

| 指标 | 调整前：收益6 / 成交量6 | 选定方案：收益6 / 成交量3 |
|---|---:|---:|
| LogV KL ↓ | 0.6990 ± 0.2918 | **0.2655 ± 0.1665** |
| DayR KL ↓ | 0.1120 ± 0.0601 | **0.0947 ± 0.0590** |
| 成交量 P99 / 真实 P99，理想值1 | 33.68 ± 11.27 | **1.95 ± 0.19** |

**LogV KL 降低约62%**，显著缓解原始成交量上尾过大；同时保持收益通道的 ±6 支持范围。这里调整的是每一步去噪预测的标准化干净状态，不是对生成后的原始成交量直接作最终截断。

![采样边界改进](docs/figures/clipping_improvement.png)

## 精选结果：五档连续波动率控制

下面展示选定 `[6,3]` 配置的波动率控制案例。包含三个扩散种子，每个档位、每个种子16,196条路径，目标取训练期相应五分位箱内中位数；纵轴为日对数收益标准差。

![连续波动率控制](docs/figures/volatility_control.png)

这是三个支持目标中的一个精选展示。收益和振幅也支持五档控制，完整使用方式见[控制与指标定义](docs/METHOD.md)。历史共同 ±6 条件评估中，连续控制在15个“目标×档位”组合中的14个取得低于离散控制的MSE；该历史对照与上面的新裁剪实验分开标注，详见[结果口径](docs/RESULTS.md)。

这些图聚焦报告中已验证的改善，未声称所有分布、时序指标或所有目标均已最优。裁剪阈值来自开发阶段的探索诊断，会限制尾部支持；本仓库的结果不构成整体生成质量排名或收益能力证据。

## 快速开始

推荐 Python 3.11。依赖范围保留 NumPy 1.26 与 PyArrow 15—19 的兼容组合。先按硬件安装合适的 [PyTorch](https://pytorch.org/get-started/locally/)，再安装依赖：

```bash
python -m pip install -r requirements.txt
python code/smoke.py
```

烟测只在本机自动创建**合成**量价面板，完成预处理、七种模型组合的小规模训练、五档评估、采样与四项KL计算；不需要任何私有数据。临时输出位于被 Git 忽略的 `outputs/`。它验证软件流程，不用于证明真实金融生成质量。

### 1. 提供自己的数据

将有权使用的 Parquet 文件放在 `data/market.parquet`，字段为：

| 字段 | 类型 | 含义 |
|---|---|---|
| `order_book_id` | string | 资产标识 |
| `date` | datetime | 交易日期 |
| `close` | positive numeric | 收盘价格，复权口径由使用者明确 |
| `volume` | positive numeric | 日成交量，保持源数据单位 |

默认区间2015—2024，按市场交易日70%/15%/15%切分，构造60日窗口、步长20；这些是日频适配设定，不是论文指定的日频协议。可在[默认配置](configs/default.json)调整。

```bash
python code/prepare.py
```

缺失、零成交量和非连续交易日会打断窗口，不跨股票或训练/验证/测试边界拼接。归一化参数与五档目标仅由训练窗口确定。

### 2. 训练

单个连续波动率模型：

```bash
python code/train.py --method continuous --target volatility --seed 0 --device cuda
```

运行三个种子、每种子七个模型及评估：

```bash
python code/run_suite.py --device cuda
```

默认三个种子为0/1/2、训练10轮。这是本适配实验的配置，不应当作论文对所有数据的推荐值。

### 3. 生成指定档位

```bash
python code/generate.py --checkpoint outputs/digma_daily60/seed_0/volatility_continuous/last.pt --bin Higher --n 1000 --guidance 4 --device cuda --output outputs/volatility_higher.npz
```

连续条件还可直接指定数值。对波动率模型，`0.03` 表示日对数收益标准差3%：

```bash
python code/generate.py --checkpoint outputs/digma_daily60/seed_0/volatility_continuous/last.pt --target-value 0.03 --n 1000 --device cuda --output outputs/volatility_3pct.npz
```

`--n` 可自由设置；5,000只是展示实验的抽样数量，不是生成器上限。模型权重需自行训练，本仓库不提供权重下载。

### 4. 检验生成分布

```bash
python code/fidelity.py --real data/processed/test.npy --generated outputs/volatility_higher.npz --output outputs/fidelity.json
```

输出 DayR/RetAC/VolC/LogV 的真实→生成KL、范围外比例及成交量分位数比值。LogV是成交量维度的日频适配，**不等同于订单不平衡率OIR**。

## 输出格式

`states` 为 `(N, 2, 60)`：

- 通道0：日对数收益；
- 通道1：`log1p(日成交量)`，原始成交量可用 `numpy.expm1` 还原。

每次独立采样附带 JSON 记录，保存条件、裁剪范围、采样种子和文件哈希。数据、样本、检查点与运行日志均默认被 `.gitignore` 排除。

## 复现裁剪对照与展示图

[原边界配置](configs/baseline_r6_v6.json)保留 `[6,6]` 作为采样对照。使用同一检查点、元数据、样本数、批次与随机种子，仅改变 `--volume-clip`，即可在自己的数据上比较采样边界，不需要重训模型。

```bash
python code/generate.py --checkpoint outputs/digma_daily60/seed_0/no_control/last.pt --n 5000 --seed 70000 --volume-clip 6 --device cuda --output outputs/before.npz
python code/generate.py --checkpoint outputs/digma_daily60/seed_0/no_control/last.pt --n 5000 --seed 70000 --volume-clip 3 --device cuda --output outputs/after.npz
python docs/figures/render_showcase.py
```

展示图脚本内仅保存报告的**汇总统计**，没有真实逐日记录或生成路径。[结果口径](docs/RESULTS.md)记录样本量与适用范围；[源码来源与发布改动](docs/UPSTREAM.md)记录复用关系。

## 引用与许可

研究引用优先参考上述 DigMA 论文；本代码是独立的日频适配实现。软件引用见 [CITATION.cff](CITATION.cff)。MIT许可证及上游版权说明保留于 [LICENSE](LICENSE)、[NOTICE.md](NOTICE.md) 和 `third_party/licenses/`。
