---
slug:4-using-alphafold2
blog_type:normal
---


AlphaFold2 是由 DeepMind 开发的一种突破性的深度学习系统，用于蛋白质结构预测。ColabFold 通过一个优化的界面使这项技术变得易于使用和高效。本指南介绍了如何在 ColabFold 框架内有效使用 AlphaFold2。

## ColabFold 中的 AlphaFold2 是什么？

ColabFold 提供了一个用户友好的 AlphaFold2 实现，具有以下特点：

- 使用 MMseqs2 进行更快、更高效的多序列比对（MSA）生成
- 支持蛋白质单体和复合体（多聚体）结构预测
- 提供多种配置选项，以控制预测的准确性和速度
- 可以完全在 Google Colab 中运行，并支持 GPU 加速

来源：[AlphaFold2.ipynb](AlphaFold2.ipynb), [README.md](README.md)

## 开始使用 AlphaFold2

### 前提条件

要在 ColabFold 中使用 AlphaFold2，您需要：

- 一个 Google 账户（用于在 Colab 中运行）
- 一个蛋白质序列或一组序列
- 对蛋白质结构概念的基本理解

对于本地安装，还有其他要求（Python 环境，GPU 访问）。

### 基本流程

典型的工作流程包括以下关键步骤：

1. **输入您的蛋白质序列**
2. **配置预测参数**
3. **运行预测**
4. **可视化和分析结果**
5. **下载预测文件**

让我们详细了解一下每个步骤。

来源：[AlphaFold2.ipynb#L57-L130](AlphaFold2.ipynb#L57-L130)

## 步骤 1：提供蛋白质序列

以 FASTA 格式或纯文本形式输入您的蛋白质序列。对于蛋白质复合体（多个链），使用冒号字符（`:`）作为序列之间的分隔符。

```python
query_sequence = 'PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK'  # 单个蛋白质
# 或者对于复合体：
# query_sequence = 'PIAQIHILEGRSDEQKETLIRE:MAKGHFGIGGELASK'  # 两个蛋白质链
```

<CgxTip>
对于蛋白质复合体，ColabFold 支持同源寡聚体（同一序列的多个副本）和异源寡聚体（不同蛋白质序列）。使用冒号分隔符表示链的边界。
</CgxTip>

您还需要提供一个作业名称，用于组织输出文件：

```python
jobname = 'my_protein'
```

来源：[AlphaFold2.ipynb#L64-L81](AlphaFold2.ipynb#L64-L81)

## 步骤 2：配置 MSA 生成

多序列比对（MSA）是蛋白质结构预测的关键步骤。ColabFold 提供了多种选项：

```python
msa_mode = "mmseqs2_uniref_env"  # 默认选项，搜索 UniRef + 环境序列
# 备选选项：
# "mmseqs2_uniref"     - 仅使用 UniRef 数据库
# "single_sequence"    - 完全跳过 MSA 生成
# "custom"             - 使用用户提供的 MSA 文件（A3M 格式）
```

对于蛋白质复合体，您还可以指定配对模式：

```python
pair_mode = "unpaired_paired"  # 默认：使用配对和非配对序列
# 备选选项：
# "paired"    - 仅使用可以配对的序列（同种）
# "unpaired"  - 保持每个链的 MSA 分开
```

来源：[AlphaFold2.ipynb#L188-L220](AlphaFold2.ipynb#L188-L220), [batch.py#L561-L599](colabfold/batch.py#L561-L599)

## 步骤 3：配置高级设置

AlphaFold2 提供了众多参数来控制预测过程：

### 模型类型选择

```python
model_type = "auto"  # 自动选择适当的模型
# 备选选项：
# "alphafold2_ptm"           - 用于单个蛋白质
# "alphafold2_multimer_v3"   - 用于蛋白质复合体（最新版本）
# 其他特定模型类型也可用
```

### 回收设置

回收通过将模型输出作为输入反馈来提高预测准确性：

```python
num_recycles = "3"  # 默认：3 次回收迭代
# 对于复杂问题，更高的值（6, 12）可能会改善结果
```

### 模板使用

已知结构的模板可以提高预测准确性：

```python
template_mode = "none"  # 默认：不使用模板
# 备选选项：
# "pdb100"   - 在 PDB100 数据库中搜索模板
# "custom"   - 使用用户提供的自定义模板
```

来源：[AlphaFold2.ipynb#L232-L277](AlphaFold2.ipynb#L232-L277)

## 步骤 4：运行预测

所有设置配置完成后，运行预测：

```python
results = run(
    queries=queries,
    result_dir=result_dir,
    use_templates=use_templates,
    custom_template_path=custom_template_path,
    num_relax=num_relax,
    msa_mode=msa_mode,
    model_type=model_type,
    num_models=5,
    num_recycles=num_recycles
    # 以及其他配置选项
)
```

在 Colab 笔记本中，只需选择“运行时”→“全部运行”来执行整个预测流程。

预测过程包括以下几个步骤：

1. **MSA 生成**：使用 MMseqs2 查找相关序列
2. **模板搜索**：（如果启用）在 PDB 中查找相关结构
3. **特征处理**：为神经网络准备输入
4. **结构预测**：运行 AlphaFold2 模型
5. **结构松弛**：（如果启用）使用 AMBER 精化预测结构

来源：[AlphaFold2.ipynb#L288-L387](AlphaFold2.ipynb#L288-L387), [batch.py#L329-L559](colabfold/batch.py#L329-L559)

## 步骤 5：可视化和分析结果

ColabFold 自动生成预测结构的可视化：

```python
# 显示 3D 结构
show_pdb(rank_num=1, color="lDDT")

# 显示置信度图（pLDDT 和 PAE）
# 这些图在 Colab 笔记本中会自动显示
```

评估预测质量的关键指标：

- **pLDDT**（预测 LDDT）：每个残基的置信度评分（0-100，越高越好）
  - <50：低置信度
  - 50-70：中等置信度
  - 70-90：高置信度
  - >90：非常高置信度

- **pTM-score**：整体折叠置信度（0-1，越高越好）
  - >0.8 通常表示置信度较高的预测

- **PAE**（预测对齐误差）：指示残基间相对位置的置信度

来源：[AlphaFold2.ipynb#L402-L504](AlphaFold2.ipynb#L402-L504), [AlphaFold2.ipynb#L614-L617](AlphaFold2.ipynb#L614-L617)

## 步骤 6：下载结果

预测结果被打包成一个 zip 文件，包括：

```
my_protein.result.zip/
├── my_protein/
│   ├── my_protein_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb  # 最高排名模型
│   ├── my_protein_unrelaxed_rank_002_*.pdb  # 第二高排名模型
│   ├── ...（其他模型）
│   ├── my_protein_relaxed_rank_001_*.pdb  # 松弛模型（如果启用）
│   ├── my_protein_pae.png  # 预测对齐误差图
│   ├── my_protein_plddt.png  # 每个残基置信度图
│   ├── my_protein_coverage.png  # MSA 覆盖率可视化
│   ├── my_protein.a3m  # 多序列比对文件
│   ├── my_protein_scores_rank_001_*.json  # 详细评分的 JSON 格式文件
│   └── my_protein.citations.bibtex  # 使用工具的引用
```

在 Colab 中，zip 文件会自动提供下载。您也可以通过启用 `save_to_google_drive` 选项将其保存到 Google Drive。

来源：[AlphaFold2.ipynb#L511-L524](AlphaFold2.ipynb#L511-L524), [AlphaFold2.ipynb#L550-L557](AlphaFold2.ipynb#L550-L557)

## 高级功能

### 结构松弛

结构松弛使用 AMBER 力场提高预测模型的物理真实性：

```python
num_relax = 1  # 松弛最高排名模型
# 选项：0（不松弛），1，或 5（松弛前 5 个模型）
```

松弛计算密集，但可以改善结构质量，特别是对于高置信度预测。

### 自定义 MSA 输入

对于希望提供自己的多序列比对的用户：

1. 设置 `msa_mode = "custom"`
2. 在提示时上传您的 A3M 格式 MSA 文件
3. MSA 中的第一个序列必须与您的查询序列匹配

当您有专门的 MSA 或希望使用不同的比对方法时，这很有用。

### 自定义模板

要使用特定结构作为模板：

1. 设置 `template_mode = "custom"`
2. 在提示时上传 PDB 或 mmCIF 文件
3. 模板必须遵循 PDB 命名约定

当存在密切相关结构时，模板可以显著提高预测准确性。

来源：[AlphaFold2.ipynb#L570-L593](AlphaFold2.ipynb#L570-L593)

### 采样预测

为了采样多样化的预测（适用于灵活区域）：

```python
num_seeds = 4  # 使用 4 个不同的随机种子
use_dropout = True  # 启用 dropout 以增加多样性
```

这生成多个具有结构多样性的预测，这对于分析灵活区域或构象集合很有用。

来源：[AlphaFold2.ipynb#L247-L252](AlphaFold2.ipynb#L247-L252)

## 故障排除

常见问题及解决方案：

1. **内存不足错误**
   - 尝试较小的蛋白质或复合体
   - 使用 `max_msa` 参数减少 MSA 深度
   - Google Colab 分配的 GPU 内存容量不同

2. **预测质量差**
   - 检查您的输入序列（正确的蛋白质序列，无异常字符）
   - 尝试增加 `num_recycles`（3 → 6 → 12）
   - 如果可用，启用模板
   - 对于复合体，尝试不同的 `pair_mode` 设置

3. **下载问题**
   - 禁用可能干扰下载的广告拦截器
   - 使用 Colab 文件浏览器中的手动下载选项
   - 保存到 Google Drive

来源：[AlphaFold2.ipynb#L600-L607](AlphaFold2.ipynb#L600-L607)

## 限制

- Google Colab 资源各异，可能不支持非常大的蛋白质/复合体
- MSA 生成受 MMseqs2 API 容量限制（每天约 20-50k 请求）
- 对于关键研究，考虑额外使用完整的 AlphaFold2 流程进行验证

来源：[AlphaFold2.ipynb#L609-L613](AlphaFold2.ipynb#L609-L613)

## 更多资源

- **ColabFold GitHub**：[https://github.com/sokrypton/ColabFold](https://github.com/sokrypton/ColabFold)
- **ColabFold 论文**：Mirdita M, et al. *Nature Methods*, 2022
- **AlphaFold2 论文**：Jumper J, et al. *Nature*, 2021
- **Nature Protocols 指南**：关于有效使用 ColabFold 的综合指南

通过遵循本指南，您现在应该能够在 ColabFold 框架内有效使用 AlphaFold2 来预测蛋白质结构，并充满信心。