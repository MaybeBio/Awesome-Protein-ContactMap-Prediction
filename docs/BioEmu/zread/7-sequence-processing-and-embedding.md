---
slug:7-sequence-processing-and-embedding
blog_type:normal
---


在蛋白质结构建模领域，将氨基酸序列转化为有意义的数值表示是至关重要的第一步。BioEmu利用强大的序列处理和嵌入技术，将原始蛋白质序列转化为丰富的特征表示，从而驱动其结构生成能力。本指南将详细介绍BioEmu如何处理序列并生成嵌入，为准确的三维蛋白质结构预测奠定基础。

## 理解序列处理流程

在生成蛋白质结构之前，BioEmu需要理解蛋白质的语言。其序列处理流程负责从读取蛋白质序列到验证序列完整性，再到为嵌入生成做好准备的全过程。

整个过程始于基本的序列输入输出操作。BioEmu支持从FASTA文件读取序列，这是存储蛋白质序列的标准格式。`seq_io.py`模块为此提供了核心功能：

```python
from bioemu.seq_io import read_fasta, write_fasta, check_protein_valid

# 从FASTA文件读取序列
sequences = read_fasta("protein_sequences.fasta")

# 将序列写入FASTA文件
write_fasta(["ACDEFGHIKLMNPQRSTVWY"], "output.fasta")

# 验证序列是否仅包含标准氨基酸
check_protein_valid("ACDEFGHIKLMNPQRSTVWY")
```

这些函数确保序列格式正确且仅包含有效的氨基酸字符。BioEmu严格遵守20种常见氨基酸的IUPAC标准，拒绝任何包含无效字符的序列，以确保下游处理的一致性。

<CgxTip>处理前务必验证蛋白质序列。无效字符会导致嵌入生成失败，并可能在结构预测中产生意外结果。</CgxTip>

## 使用ColabFold生成序列嵌入

BioEmu序列处理的核心在于其能够生成同时捕获局部和全局序列特征的丰富嵌入。这些嵌入通过ColabFold创建，这是一个结合了多序列比对(MSA)信息和深度学习表示的强大工具。

BioEmu的嵌入生成过程由`get_embeds.py`中的`get_colabfold_embeds`函数处理。该函数接收蛋白质序列并返回两种类型的嵌入：

1. **单一表示**：捕获局部特征的每个残基嵌入
2. **配对表示**：捕获长程相互作用的残基对嵌入

以下是生成蛋白质序列嵌入的方法：

```python
from bioemu.get_embeds import get_colabfold_embeds

# 为蛋白质序列生成嵌入
single_embeds_file, pair_embeds_file = get_colabfold_embeds(
    seq="ACDEFGHIKLMNPQRSTVWY",
    cache_embeds_dir="./embeddings_cache"
)
```

嵌入生成过程智能且高效：

1. **缓存机制**：BioEmu首先检查给定序列的嵌入是否已存在于缓存目录中。如果找到，立即返回缓存的嵌入，节省计算时间。

2. **哈希标识**：每个序列被分配唯一的SHA256哈希值，用于标识和检索缓存的嵌入。

3. **自动ColabFold设置**：如果未安装ColabFold，BioEmu会自动在专用环境中下载并配置它。

4. **MSA生成**：对于每个序列，ColabFold搜索同源序列以构建多序列比对，为嵌入生成提供进化背景。

## 将嵌入集成到ChemGraph结构中

生成嵌入后，需要将其集成到BioEmu的数据结构中进行处理。这时就需要用到`ChemGraph`类。`ChemGraph`是一种专门的数据结构，包含蛋白质结构生成所需的所有信息：

- 序列嵌入（单一和配对表示）
- 位置信息（初始设置为NaN）
- 节点方向（初始设置为NaN）
- 表示残基连接的边索引
- 原始氨基酸序列

`sample.py`中的`get_context_chemgraph`函数展示了如何加载和集成嵌入：

```python
from bioemu.sample import get_context_chemgraph

# 创建包含嵌入的ChemGraph
chemgraph = get_context_chemgraph(
    sequence="ACDEFGHIKLMNPQRSTVWY",
    cache_embeds_dir="./embeddings_cache"
)
```

该函数执行几个关键步骤：

1. **加载嵌入**：从各自文件中加载预计算的单一和配对嵌入。

2. **重塑配对嵌入**：将配对嵌入从方阵(L×L×F)重塑为平面格式(L²×F)，以兼容图神经网络。

3. **创建边索引**：生成表示残基全连接图的边索引，其中每个残基与其他所有残基相连。

4. **初始化位置和方向**：三维坐标和方向初始化为NaN值，它们将在结构预测过程中生成。

## 嵌入如何驱动结构生成

通过此过程生成的嵌入不仅仅是静态表示——它们主动指导结构生成过程。在BioEmu的扩散模型架构中，这些嵌入作为条件信息，帮助模型生成物理上合理的蛋白质结构。

`models.py`中的`DistributionalGraphormer`模型展示了嵌入的处理方式：

```python
# 在模型的前向传播中：
single_repr = context.single_embeds  # [B, L, 384] - 单一表示
pair_repr = context.pair_embeds    # [B, L, L, 128] - 配对表示

# 将嵌入投影到模型维度
x1d = self.x1d_proj(single_repr)  # 投影单一表示
x2d = self.x2d_proj(pair_repr)    # 投影配对表示

# 添加相对位置信息
x2d = x2d + self.rp_proj(pos_sequence)[None]
```

单一表示捕获每个残基的特征，如氨基酸类型、局部二级结构倾向和进化保守性。配对表示捕获残基间的相互作用，包括接触概率、距离偏好和角度关系。

## 高级功能和自定义

BioEmu为序列处理和嵌入生成提供了多种高级功能：

### 自定义MSA文件

您可以提供自己的A3M格式MSA文件，而不是依赖自动MSA生成：

```python
single_embeds_file, pair_embeds_file = get_colabfold_embeds(
    seq="ACDEFGHIKLMNPQRSTVWY",
    msa_file="custom_msa.a3m"
)
```

当您具有领域特定知识或想使用专门的MSA生成工具时，这特别有用。

### 自定义MSA服务器

如果您可以访问专用数据库或想使用本地MSA服务器，可以指定自定义MSA服务器URL：

```python
single_embeds_file, pair_embeds_file = get_colabfold_embeds(
    seq="ACDEFGHIKLMNPQRSTVWY",
    msa_host_url="https://your-custom-msa-server.com"
)
```

### 嵌入缓存策略

BioEmu的缓存系统设计得既高效又灵活。默认情况下，嵌入存储在`~/.bioemu_embeds_cache`中，但您可以自定义此位置：

```python
single_embeds_file, pair_embeds_file = get_colabfold_embeds(
    seq="ACDEFGHIKLMNPQRSTVWY",
    cache_embeds_dir="/path/to/custom/cache"
)
```

缓存系统使用SHA256哈希确保每个唯一序列获得自己的嵌入文件，防止冲突并实现高效重用。

## 性能考虑

序列处理和嵌入生成可能计算密集，特别是对于长蛋白质。以下是一些关键考虑因素：

1. **序列长度**：由于配对表示的存在，嵌入生成的计算复杂度大致随序列长度二次方增长。

2. **MSA生成时间**：对于新序列，MSA生成可能需要几分钟，因为它涉及搜索大型数据库。

3. **缓存利用**：始终使用缓存系统避免冗余计算，特别是在多次处理相同序列时。

4. **批处理**：为多个序列生成结构时，以批处理方式进行以最大化计算效率。

## 常见问题故障排除

### 无效氨基酸字符

如果遇到关于无效氨基酸字符的错误，使用`check_protein_valid`函数识别有问题的序列：

```python
from bioemu.seq_io import check_protein_valid

try:
    check_protein_valid("ACDXFGHIKLMNPQRSTVWY")  # 包含无效的'X'
except AssertionError as e:
    print(f"无效序列: {e}")
```

### ColabFold安装问题

如果ColabFold自动安装失败，请检查`~/.bioemu_colabfold/install_log.txt`中的安装日志以获取详细错误信息。

### 嵌入缓存冲突

如果怀疑嵌入缓存有问题，可以清除缓存目录或指定新的缓存位置以强制重新生成嵌入。

## 结论

序列处理和嵌入生成构成了BioEmu蛋白质结构建模能力的基础。通过利用ColabFold强大的基于MSA的嵌入和BioEmu高效的处理流程，用户可以将原始氨基酸序列转化为同时捕获局部和全局蛋白质特征的丰富数值表示。

理解这一过程对于任何希望有效使用BioEmu的人都至关重要，无论是用于基础结构预测还是更高级的应用，如蛋白质设计和工程。您旅程的下一步将是探索这些嵌入如何在化学图操作和结构采样中使用，这将在后续指南中介绍。