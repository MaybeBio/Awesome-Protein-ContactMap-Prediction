---
slug:5-using-esmfold
blog_type:normal
---


ESMFold 是 ColabFold 中提供的一种快速高效的蛋白质结构预测方法，它使用语言模型方法直接从氨基酸序列预测蛋白质结构。本指南将带领您了解如何在 ColabFold 生态系统中使用 ESMFold，从基本结构预测到高级选项。

## ESMFold 简介

ESMFold 基于 Meta AI 的进化尺度建模方法，该方法将语言建模与结构预测相结合。与 AlphaFold2 不同，ESMFold 可以在不生成多序列比对（MSA）的情况下进行预测，使得其在许多预测任务中显著更快。

ESMFold 特别适用于：
- **快速初步预测**，当您需要快速获得结果时
- **中小型蛋白质**（尤其是使用 API 选项时）
- **新型蛋白质**，具有有限的进化信息

```mermaid
flowchart LR
    A[蛋白质序列] --> B[ESM 语言模型]
    B --> C[结构模块]
    C --> D[3D 结构预测]
```

来源：[ESMFold.ipynb](ESMFold.ipynb)

## 开始使用 ESMFold

ColabFold 提供三种使用 ESMFold 的方式：

| 方法 | 适用于 | 最大长度 | 需要GPU | 链接 |
|------|--------|----------|---------|------|
| ESMFold API | 小型蛋白质的快速预测 | 400 aa | 否 | [esmatlas.com](https://esmatlas.com/resources?action=fold) |
| ESMFold Colab | 中型蛋白质 | ~900 aa（在 T4 GPU 上） | 是 | [标准笔记本](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb) |
| ESMFold 高级 | 复杂预测，实验性功能 | ~900 aa（在 T4 GPU 上） | 是 | [高级笔记本](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/ESMFold_advanced.ipynb) |

来源：[ESMFold.ipynb](ESMFold.ipynb)，[beta/ESMFold_advanced.ipynb](beta/ESMFold_advanced.ipynb)，[beta/ESMFold_api.ipynb](beta/ESMFold_api.ipynb)

## 基本用法

### 使用标准 ESMFold 笔记本

1. 打开 [ESMFold Colab 笔记本](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb)
2. 运行安装单元格（大约需要 2-3 分钟）
3. 在“运行 ESMFold”单元格中输入您的蛋白质序列
4. 配置基本参数：
   - `jobname`：预测任务的标识符
   - `sequence`：氨基酸序列 - 使用“/”表示链断裂
   - `copies`：同源寡聚体预测的副本数
   - `num_recycles`：回收次数（越高 = 结果可能更好，但速度更慢）
5. 运行单元格以执行预测
6. 在可视化单元格中查看 3D 结构

```python
# 示例基本配置
jobname = "my_protein"  
sequence = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
copies = 1
num_recycles = 3
```

<CgxTip>
要在序列中包含链断裂，请使用“/”作为分隔符（例如，“ABCDEF/GHIJKL”）。这告诉 ESMFold 这些是复合物中的独立链。
</CgxTip>

来源：[ESMFold.ipynb](ESMFold.ipynb)

### 理解 ESMFold 输出

运行 ESMFold 后，您将获得几个关键输出：

1. **3D 结构可视化**：预测蛋白质结构的交互式 3D 模型
2. **评分指标**：
   - **pLDDT**（每个残基的置信度）：指示局部置信度（0-100，越高越好）
   - **PTM**（预测 TM 分数）：全局折叠质量指标（0-1，越高越好）
3. **PDB 文件**：包含预测原子坐标的标准格式文件

预测文件保存在以您的任务命名的本地目录中，格式为：
`{ID}/ptm{ptm_score}_r{num_recycles}_default.pdb`

来源：[ESMFold.ipynb](ESMFold.ipynb)

## 高级选项

为了更精细地控制您的预测，请使用 [ESMFold 高级笔记本](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/ESMFold_advanced.ipynb)。

### 多聚体预测选项

在预测蛋白质复合物或寡聚体时：

```python
# 同源二聚体预测示例
sequence = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQ"
copies = 2  # 创建一个同源二聚体
chain_linker = 25  # 链之间的残基间距
```

来源：[beta/ESMFold_advanced.ipynb](beta/ESMFold_advanced.ipynb)

### 采样和随机性

高级笔记本允许您生成多个具有可控随机性的结构预测：

```python
# 生成多个不同预测的配置
samples = 8  # 要生成的不同模型数量
masking_rate = 0.15  # 随机掩码的序列比例
stochastic_mode = "LM"  # 选项："LM"（语言模型），"SM"（结构模块），"LM_SM"（两者）
```

此功能在以下情况下很有用：
- 您想探索替代构象
- 您对真实结构不确定
- 您需要集合预测以进行下游分析

来源：[beta/ESMFold_advanced.ipynb](beta/ESMFold_advanced.ipynb)

### 其他高级功能

- **语言模型接触**：启用从语言模型直接提取接触预测
- **自定义链连接器**：调整多聚体预测中链之间的间距
- **GPU 内存优化**：根据序列长度自动调整块大小

来源：[beta/ESMFold_advanced.ipynb](beta/ESMFold_advanced.ipynb)

## 使用 ESMFold API

对于少于 400 个氨基酸的蛋白质，ESMFold API 提供了一种无需 GPU 即可快速获得预测的最快方式：

1. 打开 [ESMFold API 笔记本](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/ESMFold_api.ipynb)
2. 输入您的蛋白质序列
3. 运行单元格以将您的序列提交到 ESMFold API
4. 查看并下载结果

```python
sequence = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQ"
# 笔记本将自动调用 API 并显示结果
```

<CgxTip>
API 选项非常适合教学环境或快速初步筛选多个蛋白质，因为它不需要 GPU，并且几乎可以立即返回小型蛋白质的结果。
</CgxTip>

来源：[beta/ESMFold_api.ipynb](beta/ESMFold_api.ipynb)

## 限制与故障排除

### 大小限制

- **ESMFold API**：最大序列长度为 400 个氨基酸
- **ESMFold Colab**（在典型 T4 GPU 上）：最大总长度约为 900 个氨基酸
- 对于更长的蛋白质，考虑：
  1. 将您的蛋白质分解为域
  2. 使用 AlphaFold2
  3. 升级到更强大的 GPU

### 内存问题

如果遇到内存错误：
1. 重启运行时并再次运行
2. 尝试减少序列长度
3. 确保笔记本中没有其他内存密集型进程在运行

### 准确性考虑

- ESMFold 对大多数蛋白质的准确性通常低于 AlphaFold2
- 将 ESMFold 视为快速初步预测工具
- 对于需要最高准确性的预测，尤其是对于难以预测的目标，请使用 AlphaFold2

来源：[ESMFold.ipynb](ESMFold.ipynb)，[beta/ESMFold_advanced.ipynb](beta/ESMFold_advanced.ipynb)

## 结论

ESMFold 在 ColabFold 生态系统中提供了一种快速高效的蛋白质结构预测方法。其速度使其成为初步结构预测的理想选择，尤其是对于较小蛋白质或当您需要快速结果时。对于更具挑战性的目标或更高准确性，您可以基于从 ESMFold 获得的结构洞察转向 AlphaFold2。

通过了解标准和高级笔记本中的可用选项，您可以根据特定需求定制 ESMFold，无论是预测简单的单体蛋白质还是探索复杂的多链组装体。