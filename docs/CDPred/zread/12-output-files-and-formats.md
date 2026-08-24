---
slug:12-output-files-and-formats
blog_type:normal
---


CDPred 会为每个复合体生成一组结构化的输出文件，这些文件被组织在两个主目录中：**`feature/`**（被神经网络使用的中间特征产物）和 **`predmap/`**（最终预测结果）。理解这些格式对于下游分析、与真实值进行基准测试，以及将预测结果输入到 GDFold 等对接工具中至关重要。

## 输出目录结构

对于每个输入复合体（例如 `H1017A_H1017B`），CDPred 会创建一个专属的子目录，其中包含中间和最终输出：

```
<complex_name>/
├── feature/                    # 中间特征文件
│   ├── <complex>.a3m           # 过滤后的 MSA（A3M 格式）
│   ├── <complex>.aln           # 原始 MSA 比对
│   ├── <complex>.fasta         # 拼接的复合体序列
│   ├── <complex>.mat           # 共进化矩阵
│   ├── <complex>.dist          # 距离矩阵（特征阶段）
│   ├── <complex>.npy           # NumPy 特征数组
│   ├── <complex>_pssm.txt      # PSSM 特征谱（组合）
│   └── pssm/                   # PSI-BLAST PSSM 产物
│       ├── <complex>.fasta
│       ├── <complex>.psiblast.output
│       └── <complex>.pssm
└── predmap/                    # 最终预测输出
    ├── <complex>.htxt          # 链间接触概率图
    ├── <complex>.dist          # 预测的链间距离图
    ├── <complex>_con.rr        # 接触预测（RR 格式）
    └── <complex>_dist.rr       # 距离预测（RR 格式）
```

来源: [expection_output](/example/expection_output)

## 特征目录文件

`feature/` 目录存储在 [特征生成](5-feature-generation) 流水线期间生成的中间计算产物。这些文件会被神经网络集成模型使用，但也可以独立检查用于调试或分析。

### FASTA 序列文件 (`.fasta`)

一个标准的 FASTA 文件，包含二聚体复合体的**拼接氨基酸序列**。对于像 H1017A_H1017B 这样的异二聚体，链 A 的残基直接后接链 B 的残基，形成单个序列条目：

```
>H1017A_H1017B
LLLNDKQYNELCEAAEGRNLGAVFSYSEPEEPPPLNFSFEERKKIFLWVLTRLLKEGRIKLAKH...
```

首行使用复合体名称（`ChainA_ChainB`）作为序列标识符。序列长度等于两条链长度之和，这定义了所有后续矩阵输出的维度（N × N，其中 N = len(chain_A) + len(chain_B)）。

来源: [H1017A_H1017B.fasta](/example/expection_output/H1017A_H1017B/feature/H1017A_H1017B.fasta#L1-L3)

### A3M 比对文件 (`.a3m`)

一个 **A3M 格式的多序列比对**——一种紧凑的 MSA 表示，其中相对于查询序列的插入已被移除（小写残基被删除）。第一行是查询序列，随后是与它比对的同源序列。此文件由 [ZComplexMSA](10-zcomplexmsa-for-msa-generation) 生成，并作为共进化分析的输入。

来源: [H1017A_H1017B.a3m](/example/expection_output/H1017A_H1017B/feature/H1017A_H1017B.a3m)

### 原始比对文件 (`.aln`)

一个**逐行的 MSA**，其中每一行代表多序列比对中的一个比对序列，无头信息。所有序列共享相同的长度（包括间隙字符 `-`）。这种格式更容易通过编程方式解析，用于计算序列权重或有效序列计数（Meff）等统计信息。

来源: [H1017A_H1017B.aln](/example/expection_output/H1017A_H1017B/feature/H1017A_H1017B.aln#L1-L61)

### 共进化矩阵 (`.mat`)

一个**对称的 N × N 共进化得分矩阵**，以制表符分隔的科学计数法浮点数存储。每个条目 `mat[i][j]` 表示残基 i 和 j 之间的共进化耦合强度，源自通过互信息或直接耦合分析等方法从 MSA 中推导的结果。对角线为零（`0.00000000000000000000e+00`）。

| 属性 | 值 |
|---|---|
| 格式 | 制表符分隔的科学计数法浮点数 |
| 维度 | N × N（N = 两条链中的残基总数） |
| 对称性 | `mat[i][j] == mat[j][i]` |
| 对角线 | `0.0` |

来源: [H1017A_H1017B.mat](/example/expection_output/H1017A_H1017B/feature/H1017A_H1017B.mat#L1-L10)

### 特征阶段距离矩阵 (`.dist`)

在特征生成阶段的**预测残基间距离矩阵**，以空格分隔的浮点数存储。这与 `predmap/` 中的最终预测距离图不同——它代表从共进化数据导出的早期估计或输入特征。值以**埃（Å）**为单位，矩阵维度为 N × N。

来源: [H1017A_H1017B.dist](/example/expection_output/H1017A_H1017B/feature/H1017A_H1017B.dist)

### NumPy 特征数组 (`.npy`)

一个**序列化的 NumPy 二进制数组**，包含所有残基对的完整特征向量堆叠。这是直接输入到 [神经网络模型](6-neural-network-model-design) 中进行预测的数据。该数组将所有计算出的特征（共进化、PSSM、基于序列的、基于距离的）整合到一个适合批量推理的单一张量中。

来源: [H1017A_H1017B.npy](/example/expection_output/H1017A_H1017B/feature/H1017A_H1017B.npy)

### 组合 PSSM 特征谱 (`_pssm.txt`)

一个**位置特异性得分矩阵**，采用归一化文本格式，以 `# PSSM` 头行作为前缀。随后的每一行包含每个残基位置的 42 个空格分隔的浮点值，代表 20 种标准氨基酸及附加特征的归一化进化谱（概率）。值的范围通常从接近零（不可能的残基）到接近一（高度保守的残基）。

来源: [H1017A_H1017B_pssm.txt](/example/expection_output/H1017A_H1017B/feature/H1017A_H1017B_pssm.txt#L1-L10)

### PSI-BLAST PSSM 产物 (`pssm/`)

一个子目录，包含用于构建组合 PSSM 的**原始 PSI-BLAST 输出**：

| 文件 | 描述 |
|---|---|
| `<complex>.fasta` | PSI-BLAST 搜索的输入 FASTA |
| `<complex>.psiblast.output` | 原始 PSI-BLAST 标准输出（迭代日志，E 值） |
| `<complex>.pssm` | NCBI 格式的原始 PSSM（位置特异性得分） |

来源: [pssm/](/example/expection_output/H1017A_H1017B/feature/pssm)

## 预测输出文件 (predmap/)

`predmap/` 目录包含**最终预测结果**——这是用户用于生物学解释、下游对接和评估的主要输出。这些文件由 [神经网络模型](6-neural-network-model-design) 集成生成。

### 链间接触概率图 (`.htxt`)

**CDPred 的核心输出**：一个 N × N 的**链间接触概率**矩阵，以空格分隔的浮点数存储（保留 4 位小数）。每个条目 `P[i][j]` 表示残基 i（来自一条链）与残基 j（来自另一条链）接触的预测概率。这是由 CDPred 神经网络直接预测的矩阵。

主要特征：

- **值范围从 0.0 到约 0.42**，值越高表示预测的链间接触越强
- **链内块接近零**：i 和 j 同属一条链的条目表现为接近 0.0000 的值，因为 CDPred 专注于链间相互作用
- **链间块包含信号**：非对角块（链 A 残基 × 链 B 残基）包含有意义的预测
- **非对称性**：完整矩阵同时包含 `P[i_A][j_B]` 和 `P[i_B][j_A]` 区域，它们不一定是对称的

`example/ground_truth/` 中的真实 `.htxt` 文件遵循相同格式，但包含**实际的链间距离**（以埃为单位）而非概率，以便通过 [预测评估指标](13-prediction-evaluation-metrics) 进行评估。

来源: [H1017A_H1017B.htxt](/example/expection_output/H1017A_H1017B/predmap/H1017A_H1017B.htxt), [ground_truth H1017A_H1017B.htxt](/example/ground_truth/H1017A_H1017B.htxt)

### 预测距离图 (`.dist`)

一个 N × N 的**预测残基间距离**矩阵，以埃为单位，存储为空格分隔的浮点数（保留 4 位小数）。这是接触概率图在距离空间的对应物。对于链间区域，值通常在约 12.7 到约 22.0 Å 之间，较小的距离对应于预测在二聚体界面两侧距离接近的残基。

矩阵结构镜像 `.htxt` 格式：链内区域包含链内距离，而链间块携带对接关键的链间距离预测。

来源: [H1017A_H1017B.dist](/example/expection_output/H1017A_H1017B/predmap/H1017A_H1017B.dist)

### RR 格式的接触预测 (`_con.rr`)

一个**标准 RR（残基-残基）格式**文件，按置信度降序列出预测的链间接触。此格式广泛用于 CASP 竞赛，并兼容许多结构预测工具。每行包含 5 个空格分隔的字段：

```
i  j  segID  minDist  probability
```

| 字段 | 描述 | 示例 |
|---|---|---|
| `i` | 第一个残基的残基索引（从 1 开始） | `109` |
| `j` | 第二个残基的残基索引（从 1 开始） | `92` |
| `segID` | 片段标识符（始终为 `0`） | `0` |
| `minDist` | 最小接触距离阈值（始终为 `8.0` Å） | `8.0` |
| `probability` | 预测的接触概率 | `0.575` |

接触被定义为 8.0 Å 内的残基对，文件按概率降序排列。顶部的高置信度接触（例如 `109 92 0 8.0 0.575`）代表最可靠的链间接触预测——这些作为 [GDFold 结构对接](11-gdfold-for-structure-docking) 的约束条件尤为有价值。

来源: [H1017A_H1017B_con.rr](/example/expection_output/H1017A_H1017B/predmap/H1017A_H1017B_con.rr#L1-L200)

### RR 格式的距离预测 (`_dist.rr`)

一个**RR 格式文件，列出预测的链间距离**，按距离升序排列。每行包含 3 个空格分隔的字段：

```
i  j  predicted_distance
```

| 字段 | 描述 | 示例 |
|---|---|---|
| `i` | 第一个残基的残基索引（从 1 开始） | `17` |
| `j` | 第二个残基的残基索引（从 1 开始） | `121` |
| `predicted_distance` | 预测的 Cβ–Cβ 距离（以埃为单位） | `7.744` |

文件按距离递增排序，意味着**预测最近的链间残基对排在最前面**。最短的预测距离（例如残基 17 和 121 之间的 `7.744` Å）标识了预测的接触界面。此格式通过保留实际距离大小而非仅仅是二元的接触/非接触分类，为 `_con.rr` 提供了补充信息。

<CgxTip>当将预测结果输入 GDFold 进行结构对接时，请使用 `_con.rr` 文件作为接触约束。排名最高的接触（最高概率）可作为最可靠的链间约束。`.htxt` 概率矩阵也可以在用户定义的截断值（例如 P > 0.3）处进行阈值处理，以生成自定义接触列表。</CgxTip>

来源: [H1017A_H1017B_dist.rr](/example/expection_output/H1017A_H1017B/predmap/H1017A_H1017B_dist.rr#L1-L200)

## 真实值文件

`example/ground_truth/` 目录提供**用于评估的参考数据**。真实值文件遵循与预测输出相同的格式，但包含实验确定的值：

| 文件 | 格式 | 内容 |
|---|---|---|
| `H1017A.fasta` | FASTA | 链 A 单体序列 |
| `H1017B.fasta` | FASTA | 链 B 单体序列 |
| `H1017A_H1017B.htxt` | 空格分隔矩阵 | 真实链间距离矩阵 (Å) |

真实值 `.htxt` 包含**实际的 Cβ–Cβ 距离**（以埃为单位，而非概率），以便如 [预测评估指标](13-prediction-evaluation-metrics) 中所述进行定量评估。单体 FASTA 文件定义了拆分拼接复合体序列所需的链边界。

来源: [ground_truth/](/example/ground_truth/H1017A_H1017B.htxt), [H1017A.fasta](/example/ground_truth/H1017A.fasta)

## 输出格式比较

| 文件 | 目录 | 域 | 值类型 | 范围 | 主要用途 |
|---|---|---|---|---|---|
| `.htxt` | predmap/ | 链间接触 | 概率 | [0, ~0.5] | 核心预测；评估 |
| `.dist` | predmap/ | 所有残基对 | 距离 (Å) | [~7, ~42] | 距离空间预测 |
| `_con.rr` | predmap/ | 链间接触 | 概率 | [0, 1] | GDFold 约束；CASP 提交 |
| `_dist.rr` | predmap/ | 链间距离 | 距离 (Å) | [~7, ~22] | 基于距离的分析 |
| `.mat` | feature/ | 所有残基对 | 共进化得分 | [0, ~0.05] | 特征检查；调试 |
| `_pssm.txt` | feature/ | 逐残基特征谱 | 归一化概率 | [0, 1] | 特征检查；调试 |
| `.npy` | feature/ | 所有残基对 | 多特征向量 | 变化 | 神经网络输入 |

来源: [expection_output](/example/expection_output), [ground_truth](/example/ground_truth)

## 下一步

理解输出格式后，请参阅 [预测评估指标](13-prediction-evaluation-metrics) 了解如何定量比较 `.htxt` 预测与真实值，并参阅 [GDFold 结构对接](11-gdfold-for-structure-docking) 了解 `_con.rr` 接触预测如何驱动 3D 结构组装。