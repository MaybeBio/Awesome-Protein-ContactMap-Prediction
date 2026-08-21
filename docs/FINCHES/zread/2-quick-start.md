---
slug:2-quick-start
blog_type:normal
---


在五分钟内从零开始完成你的首次相互作用预测。本指南将带你安装 FINCHES、初始化前端对象，并计算你的第一个 **epsilon** 相互作用值——这是量化两个本质无序区域（IDR）之间化学特异性的核心指标。

## 前提条件

FINCHES 需要 **Python 3.9+**（为获得最佳性能，推荐使用 3.11 或 3.12），并依赖于 `numpy`、`scipy`、`cython`、`pytorch`、`mdtraj` 和 `metapredict`。下表汇总了支持的安装方式：

| 方式 | 适用对象 | 命令 |
|--------|----------|---------|
| 通过 GitHub 使用 pip | 大多数用户 | `pip install git+https://git@github.com/idptools/finches.git` |
| 本地可编辑安装 | 贡献者 / 开发者 | `git clone git@github.com:idptools/finches.git` → `cd finches` → `pip install -e .` |
| Google Colab | 免安装探索 | [finches-colab notebooks](https://github.com/idptools/finches-colab) |
| Web 服务器 | 快速查询，无需写代码 | [finches-online.com](https://www.finches-online.com/) |

来源: [README.md](/README.md#L75-L155), [pyproject.toml](/pyproject.toml#L13-L31)

## 安装

### 步骤 1 — 创建并激活 conda 环境

```bash
conda create -n finches python=3.12 -y
conda activate finches
```

### 步骤 2 — 通过 conda 安装核心依赖

这可以确保 NumPy/PyTorch 构建版本的一致性（在 macOS 上这一点至关重要，混用 conda 和 PyPI 构建会导致失败）：

```bash
conda install numpy pytorch scipy cython matplotlib jupyter -c pytorch
conda install mdtraj
pip install metapredict
```

### 步骤 3 — 安装 FINCHES

```bash
pip install git+https://git@github.com/idptools/finches.git
```

### 步骤 4 — 验证安装

```bash
python -c "from finches.frontend.mpipi_frontend import Mpipi_frontend; mf = Mpipi_frontend(); print('Success')"
```

> **注意：** 首次导入需要 60–90 秒，因为需要加载 Cython 编译的模块。后续导入将会很快。

来源: [README.md](/README.md#L83-L158), [docs/getting_started.rst](/docs/getting_started.rst#L22-L40)

## 你的首次计算

下图展示了从序列到预测的最简工作流。FINCHES 遵循**前端模式**：你实例化一个特定力场的前端对象，然后传入氨基酸序列来调用其方法。

```mermaid
flowchart LR
    A["导入前端"] --> B["实例化对象<br/>(选择力场)"]
    B --> C["定义序列<br/>为 Python 字符串"]
    C --> D["调用 .epsilon()"]
    D --> E["获取相互作用值<br/>(float)"]
    style A fill:#e8f5e9,stroke:#2e7d32
    style B fill:#e3f2fd,stroke:#1565c0
    style C fill:#fff3e0,stroke:#ef6c00
    style D fill:#f3e5f5,stroke:#7b1fa2
    style E fill:#fce4ec,stroke:#c62828
```

### 初始化前端

FINCHES 暴露了两个前端类——**`Mpipi_frontend`** 和 **`CALVADOS_frontend`**——它们提供了相同的接口，但底层由不同的力场参数化方案提供支持：

```python
from finches import Mpipi_frontend, CALVADOS_frontend

# Mpipi-GG 力场 (支持蛋白质和 RNA，通过 'U' 残基实现)
mf = Mpipi_frontend(salt=0.150, dielectric=80.0)

# CALVADOS2 力场 (仅支持蛋白质；对 pH 和温度敏感)
cf = CALVADOS_frontend(salt=0.150, pH=7.4, temp=288)
```

这两个构造函数默认采用近似生理条件。关键区别在于：`Mpipi_frontend` 接受 `salt` 和 `dielectric` 参数，而 `CALVADOS_frontend` 接受 `salt`、`pH` 和 `temp`（以开尔文为单位）参数。

来源: [finches/__init__.py](/finches/__init__.py#L16-L17), [finches/frontend/mpipi_frontend.py](/finches/frontend/mpipi_frontend.py#L12-L21), [finches/frontend/calvados_frontend.py](/finches/frontend/calvados_frontend.py#L34-L43)

### 计算 epsilon 值

**epsilon** 值是两个 IDR 序列之间相互作用自由能的标量代理——越负表示越具有吸引力，越正表示越具有排斥力，零则对应于中性的 GS-linker 参考基准：

```python
# DDX4 N 端结构域 (一个被广泛研究的相分离 IDR)
ddx4_ntd = 'MGDEDWEAEINPHMSSYVPIFEKDRYSGENGDNFNRTPASSSEMDDGPSRRDHFMKSGFASGRNFGNRDAGECNKRDNTSTMGGFGVGKSFGNRGFSNSRFEDGDSSGFWRESSNDCEDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGRGSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFGSRRPVLSGTGNGDTSQSRSGSGSERGGYKGLNEEVITGSGKNSWKSEAEGGES'

# 同型相互作用 (DDX4 与自身)
print(mf.epsilon(ddx4_ntd, ddx4_ntd))
```

对于**异型**相互作用，只需传入两个不同的序列：

```python
seq_A = 'GGGGGRRRRR'   # 一段富含精氨酸的短基序
seq_B = 'GGGGGYYYYY'   # 一段富含酪氨酸的短基序

print(mf.epsilon(seq_A, seq_B))  # 如果 π-阳离子接触占主导，则表现出吸引作用
```

<CgxTip>epsilon 值是**依赖顺序的**（外在的）：通常 `epsilon(A, B) ≠ epsilon(B, A)`。你可以将第一个序列想象为“浸润”在第二个序列中。要获得内在的（与顺序无关的）值，请除以第一个序列的长度：`epsilon(A, B) / len(A) == epsilon(B, A) / len(B)`。</CgxTip>

来源: [finches/frontend/frontend_base.py](/finches/frontend/frontend_base.py#L206-L241), [README.md](/README.md#L36-L43), [demo/overview_uses/basic_uses.ipynb](/demo/overview_uses/basic_uses.ipynb#L140-L175)

### 生成相互作用图

超越单一标量，**相互作用图** 会生成一个 2D 热图，展示两条序列中位置解析的相互作用强度——这非常适合用于识别哪些子区域驱动了结合：

```python
mf.interaction_figure(ddx4_ntd, ddx4_ntd)
```

这会计算一个滑动窗口相互作用矩阵，并使用并行的无序预测轨道对其进行渲染。返回的元组包含 matplotlib 的 figure 和 axes 对象，以便进行进一步的自定义。

来源: [finches/frontend/frontend_base.py](/finches/frontend/frontend_base.py#L288-L416), [finches/frontend/mpipi_frontend.py](/finches/frontend/mpipi_frontend.py#L128-L149)

## 核心前端 API 速览

`Mpipi_frontend` 和 `CALVADOS_frontend` 都暴露了相同的一组高级方法。下表汇总了你最常使用的主要入口点：

| 方法 | 返回值 | 用途 |
|--------|---------|---------|
| `epsilon(seq1, seq2)` | `float` | 两条完整序列之间的标量相互作用能 |
| `epsilon_vectors(seq1, seq2)` | 数组元组 | 每个残基的吸引和排斥相互作用向量 |
| `intermolecular_idr_matrix(seq1, seq2)` | `(matrix_tuple, disorder_1, disorder_2)` | 原始滑动窗口相互作用矩阵 + 无序分布 |
| `interaction_figure(seq1, seq2)` | `(fig, im, ax_main, ...)` | 渲染出的带有无序轨道的 2D 相互作用热图 |
| `per_residue_attractive_vector(seq1, seq2)` | `(indices, values)` | 每个残基的平均吸引相互作用（用于识别“贴纸”） |
| `per_residue_repulsive_vector(seq1, seq2)` | `(indices, values)` | 每个残基的平均排斥相互作用（用于识别“间隔物”） |

所有方法都接受 `use_aliphatic_weighting` 和 `use_charge_weighting` 标志（默认为 `True`），这些标志会应用针对疏水和电荷环境效应的校正项。

来源: [finches/frontend/frontend_base.py](/finches/frontend/frontend_base.py#L206-L282), [finches/frontend/frontend_base.py](/finches/frontend/frontend_base.py#L570-L696)

## 选择力场

| 特性 | `Mpipi_frontend` | `CALVADOS_frontend` |
|---------|-------------------|----------------------|
| 底层模型 | Mpipi-GG v1 | CALVADOS2 |
| 构造函数参数 | `salt`, `dielectric` | `salt`, `pH`, `temp` |
| RNA 支持（`U` 残基） | ✅ 是 | ❌ 否 (引发 `ValueError`) |
| 最适用场景 | 蛋白质-蛋白质和蛋白质-RNA | 对 pH/温度敏感的蛋白质-蛋白质 |
| 主要引用文献 | Joseph et al. (2021) + Lotthammer et al. (2024) | Tesei et al. (2022) |

对于大多数仅涉及蛋白质的用例，两种力场都表现良好。处理含 RNA 的序列时请使用 `Mpipi_frontend`（用 `U` 表示尿嘧啶）；需要依赖 pH 或温度的预测时请使用 `CALVADOS_frontend`。

来源: [finches/frontend/mpipi_frontend.py](/finches/frontend/mpipi_frontend.py#L12-L21), [finches/frontend/calvados_frontend.py](/finches/frontend/calvados_frontend.py#L16-L25), [finches/frontend/calvados_frontend.py](/finches/frontend/calvados_frontend.py#L28-L43)

## 故障排除

| 症状 | 原因 | 解决方案 |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'finches'` | 未安装 FINCHES | 运行 `pip install git+https://git@github.com/idptools/finches.git` |
| 首次导入耗时 >60s | 首次加载时进行 Cython 编译 | 正常现象——后续导入会很快 |
| `ValueError: CALVADOS2 cannot handle RNA ('U')` | CALVADOS 输入中包含 `U` 残基 | 切换至支持 RNA 的 `Mpipi_frontend` |
| `metapredict` 的 `ImportError` | 缺少仅限 pip 的依赖项 | 运行 `pip install metapredict` |
| macOS NumPy/PyTorch 崩溃 | 混用 conda + PyPI 构建 | 通过 conda 安装两者：`conda install numpy pytorch -c pytorch` |

来源: [README.md](/README.md#L94-L109), [finches/frontend/calvados_frontend.py](/finches/frontend/calvados_frontend.py#L16-L25)

## 接下来的去向

现在你已经能够计算相互作用能并生成相互作用图了，接下来可以探索 FINCHES 更深层的能力：

1. **[安装指南](3-installation-guide)** — 高级设置：可编辑安装、Cython 编译细节、`uv` 包管理器
2. **[Mpipi 和 CALVADOS 前端](5-mpipi-and-calvados-frontends)** — 每个前端方法的完整参数参考
3. **[相互作用图和图形](6-interaction-maps-and-figures)** — 自定义热图外观、结构域叠加和零洗牌归一化
4. **[Epsilon 计算与加权](9-epsilon-calculation-and-weighting)** — 理解脂肪族和电荷加权校正
5. **[Flory-Huggins 相图](14-flory-huggins-phase-diagrams)** — 从 epsilon 值推导相行为预测