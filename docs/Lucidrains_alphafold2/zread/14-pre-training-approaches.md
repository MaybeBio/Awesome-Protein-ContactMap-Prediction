---
slug:14-pre-training-approaches
blog_type:normal
---


蛋白质结构预测模型如AlphaFold2，在针对特定任务进行微调之前，通过在大数据集上进行预训练能够获得显著收益。本文档探讨了该AlphaFold2仓库中实现的预训练方法，包括训练目标、数据准备和嵌入策略。

## AlphaFold2预训练概述

预训练帮助蛋白质结构预测模型在处理特定预测任务之前，学习蛋白质序列和结构的一般关系。该仓库中的实现支持多种预训练目标和嵌入策略，以促进有效学习。

来源：[alphafold2.py#L469-L597](alphafold2_pytorch/alphafold2.py#L469-L597)

## 预训练目标

### 距离图预测

主要的预训练目标是**距离图预测**——预测氨基酸残基之间的离散距离矩阵。这有助于模型在学习完整3D坐标之前，了解蛋白质结构内的空间关系。

```python
# 距离图预测目标
distogram = model(seq, mask=mask)
loss = F.cross_entropy(
    distogram,
    discretized_distances,
    ignore_index=IGNORE_INDEX
)
```

模型预测残基之间的距离，并将这些距离离散化为桶状，训练使用交叉熵损失来优化这些预测，与真实距离矩阵进行对比。

来源：[train_pre.py#L76-L89](train_pre.py#L76-L89), [alphafold2.py#L594-L597](alphafold2_pytorch/alphafold2.py#L594-L597)

### 掩码语言模型（MLM）

次要的预训练目标是**掩码语言模型（MLM）**，用于多序列比对（MSA）。该方法包括：

1. 随机掩码输入序列中的标记
2. 随机用其他氨基酸替换一些标记
3. 尽管被选中掩码，仍保留一些标记不变
4. 训练模型预测掩码位置的原始氨基酸

```python
# MLM噪声函数
noised_seq, mlm_mask = mlm.noise(seq, mask)
```

该技术帮助模型学习氨基酸序列及其在MSA中的进化模式的有意义表示。

来源：[mlm.py#L27-L46](alphafold2_pytorch/mlm.py#L27-L46), [alphafold2.py#L582-L590](alphafold2_pytorch/alphafold2.py#L582-L590)

## 数据来源和准备

### SidechainNet数据集

预训练脚本使用**SidechainNet**数据集，该数据集提供包含侧链在内的完整原子细节的蛋白质结构。

```python
data = scn.load(
    casp_version=12,
    thinning=30,
    with_pytorch='dataloaders',
    batch_size=1,
    dynamic_batching=False
)
```

数据集配置为使用CASP12数据，并进行稀疏化以减少冗余。应用250个残基的长度阈值，以便在预训练期间专注于可管理的蛋白质大小。

来源：[train_pre.py#L37-L47](train_pre.py#L37-L47)

### TrRosetta数据集

仓库还包括在训练脚本目录中对**TrRosetta数据集**的支持，该数据集可以作为预训练的替代数据源。

来源：[trrosetta.py](training_scripts/datasets/trrosetta.py)

## 嵌入策略

仓库提供了多种在预训练期间可使用的蛋白质嵌入策略：

| 嵌入类型 | 描述 | 输入类型 |
|----------|------|---------|
| MSA Transformer | 使用MSA Transformer进行上下文嵌入 | 序列 + MSA |
| ESM | 使用Facebook的ESM模型进行嵌入 | 序列或序列 + MSA |
| ProtTran/ProtBERT | 使用ProtBERT进行蛋白质嵌入 | 序列 + MSA |

这些嵌入策略可以作为预训练的起点，为模型提供丰富的蛋白质序列初始表示。

```python
# ESM嵌入使用的示例
seq_embeds = get_esm_embedd(seq, model, batch_converter, device=device)
seq_embeds = self.project_embed(seq_embeds)
```

来源：[embeds.py#L10-L103](alphafold2_pytorch/embeds.py#L10-L103)

## 预训练过程

### 训练循环结构

预训练循环遵循标准结构：

1. 加载一批数据
2. 处理序列、坐标和掩码信息
3. 准备离散化的距离矩阵作为真实值
4. 通过模型进行前向传播
5. 计算损失（距离图预测和/或MLM）
6. 执行反向传播和优化

实现使用梯度累积，有效增加批量大小，而无需额外内存。

来源：[train_pre.py#L64-L96](train_pre.py#L64-L96)

### 模型配置

用于预训练的AlphaFold2模型可以配置多种参数：

```python
model = Alphafold2(
    dim=256,
    depth=1,
    heads=8,
    dim_head=64
).to(DEVICE)
```

关键参数包括：
- `dim`：隐藏维度大小
- `depth`：Evoformer块的数量
- `heads`：注意力头的数量
- `mlm_mask_prob`：MLM中掩码标记的概率
- `max_seq_len`：最大序列长度

来源：[train_pre.py#L51-L56](train_pre.py#L51-L56), [alphafold2.py#L469-L501](alphafold2_pytorch/alphafold2.py#L469-L501)

<CgxTip>
**提示**：在进行预训练时，从较小的模型（较少的层、较小的隐藏维度）开始，并逐步扩展规模，这样可以在验证方法之前快速迭代预训练设置。这允许你在进行更长时间的大型模型训练之前，先快速验证预训练设置。
</CgxTip>

## 运行预训练

要执行预训练，运行`train_pre.py`脚本：

```bash
python train_pre.py
```

脚本配置了合理的默认值：
- 100,000次训练迭代
- 每16批进行一次梯度累积
- 学习率为3e-4
- 蛋白质长度阈值为250个残基

来源：[train_pre.py#L14-L19](train_pre.py#L14-L19)

## 从预训练到端到端流程

预训练后，可以使用预训练权重作为端到端训练的初始化。仓库包含一个单独的脚本`train_end2end.py`，用于此目的，该脚本接受预训练模型并进行微调以进行完整结构预测。

```mermaid
flowchart LR
    A[在距离图预测上预训练] --> B[预训练权重]
    B --> C[端到端训练]
    C --> D[结构预测]
```

这种两阶段方法（预训练后进行端到端训练）有助于提高收敛速度和最终模型质量。

来源：[train_pre.py](train_pre.py), [train_end2end.py](train_end2end.py)

## 预训练的关键考虑因素

1. **数据质量**：预训练数据的质量显著影响模型性能。考虑使用SidechainNet和TrRosetta等多个数据集。
2. **计算资源**：预训练需要大量的计算资源。实现支持梯度累积以管理内存限制。
3. **嵌入选择**：不同的嵌入策略（MSA Transformer、ESM、ProtTran）在速度和质量上有所权衡。实验以找到最适合你用例的方法。
4. **评估**：定期评估预训练进度，通过监控训练损失和验证集性能来避免过拟合。

通过理解和利用这些预训练方法，你可以构建有效的蛋白质结构预测模型，在处理更具挑战性的端到端结构预测任务之前，先从学习到的表示中获益。