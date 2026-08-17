---
slug:9-esm-2-architecture-and-design
blog_type:normal
---



ESM-2 在蛋白质语言建模领域取得了显著进步，它实现了一个专为理解蛋白质序列而优化的精密 transformer 架构。本文档将探讨 ESM-2 的架构设计、关键组件和创新点，这些使其成为蛋白质分析和表示学习的强大工具。

## 核心架构概述

ESM-2 基于 transformer 架构构建，将蛋白质序列作为离散 token 处理，类似于语言模型处理文本的方式。该模型遵循标准的仅编码器 transformer 模式，并包含多项蛋白质特定优化，增强了其从氨基酸序列中捕获生物模式和结构信息的能力。

主模型类 `ESM2` 继承自 `nn.Module`，实现了一个完整的 transformer 编码器，包含专门用于蛋白质序列处理的组件 [esm/model/esm2.py#L14-L23]。该架构设计用于处理蛋白质数据的独特特征，包括可变序列长度、生物操作的特殊 token 以及接触预测能力的需求。

### 模型配置和参数

ESM-2 提供了多个不同规模的模型变体，参数量从 800 万到 150 亿不等。每个变体遵循相同的架构模式，但具有不同的维度：

- **8M**: 6 层，320 embed_dim，8 个注意力头
- **35M**: 12 层，480 embed_dim，8 个注意力头  
- **150M**: 30 层，640 embed_dim，16 个注意力头
- **650M**: 33 层，1280 embed_dim，20 个注意力头
- **3B**: 36 层，2560 embed_dim，40 个注意力头
- **15B**: 48 层，5120 embed_dim，64 个注意力头

这些配置在预训练模块中定义，每个模型大小都有特定的加载函数 [esm/pretrained.py#L350-L551]。

## 关键架构组件

### 1. Token 嵌入和位置编码

模型首先使用 token 嵌入将氨基酸序列转换为密集向量表示。ESM-2 使用一个嵌入层，将每个氨基酸 token 映射到高维空间 [esm/model/esm2.py#L47-L52]：

```python
self.embed_tokens = nn.Embedding(
    self.alphabet_size,
    self.embed_dim,
    padding_idx=self.padding_idx,
)
```

ESM-2 的一个关键创新是使用**旋转位置嵌入**而非传统的绝对位置嵌入。这种方法能更好地处理可变长度序列，并更有效地捕获相对位置关系 [esm/rotary_embedding.py#L15-L70]。

### 2. Transformer 层

ESM-2 的核心由多个 transformer 层组成，每层实现了标准的 transformer 块，并包含蛋白质特定优化。每个 `TransformerLayer` 包含 [esm/modules.py#L84-L144]：

- **多头自注意力**机制（带旋转嵌入）
- **前馈网络**（使用 GELU 激活）
- **层归一化**（使用 ESM1bLayerNorm）
- **残差连接**（用于梯度流动）

注意力机制特别精密，实现了旋转位置嵌入，使模型能够理解氨基酸之间的相对位置 [esm/multihead_attention.py#L68-L509]。

### 3. 专用注意力机制

ESM-2 集成了旋转位置嵌入，这代表了相对于传统注意力机制的显著进步。`RotaryEmbedding` 类通过基于位置对查询和键向量应用旋转矩阵来实现这一点 [esm/rotary_embedding.py#L35-L70]：

```python
def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
    return (
        apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
        apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
    )
```

这种方法使模型能够更有效地捕获蛋白质序列中的长程依赖关系，这对理解蛋白质结构和功能至关重要。

### 4. 输出头和专用任务

ESM-2 包含多个用于不同下游任务的输出头：

- **语言建模头**：用于掩码语言建模和序列预测
- **接触预测头**：用于预测氨基酸接触，与蛋白质结构相关 [esm/modules.py#L317-L359]

接触预测头特别重要，因为它使模型能够仅从序列数据推断结构信息，展示了蛋白质中序列与结构之间的深层联系。

## 前向传播架构

ESM-2 的前向传播遵循精心设计的流程 [esm/model/esm2.py#L77-L146]：

1. **Token 处理**：输入 token 被嵌入并缩放
2. **掩码处理**：训练期间对掩码 token 的特殊处理
3. **层处理**：通过 transformer 层顺序处理
4. **归一化**：应用最终层归一化
5. **输出生成**：生成多个输出，包括 logits、表示，以及可选的注意力权重

该模型还支持从中间层提取表示，这对于可能受益于不同抽象级别的各种下游任务很有价值。

## 训练和优化特性

### Token 丢弃策略

ESM-2 实现了精密的 token 丢弃策略，有助于防止过拟合并提高泛化能力 [esm/model/esm2.py#L88-L100]。该方法根据每个批次中观察到的掩码比率动态调整掩码比例，确保一致的训练动态。

### 层归一化策略

该模型使用 ESM1bLayerNorm，这是专为蛋白质序列优化的层归一化变体 [esm/modules.py#L44-L83]。这种归一化策略有助于稳定训练并提高收敛性。

## 模型变体和扩展

ESM-2 展现了出色的扩展特性，性能随模型尺寸提升而稳定改善。该架构设计为能从小型 8M 参数模型高效扩展到大型 15B 参数模型，同时保持相同的基本设计原则。

扩展策略包括：
- **嵌入维度的线性扩展**（与模型尺寸成正比）
- **注意力头的二次扩展**（用于更大的模型）
- **增加深度**（更多层）（用于更大的模型）
- **一致的架构模式**（适用于所有尺寸）

这种扩展方法确保从小型模型学到的见解和行为能够迁移到大型模型，使架构具有鲁棒性和可预测性。

## 与下游应用的集成

ESM-2 的架构专为支持广泛的蛋白质相关任务而设计：

- **序列分析**：理解蛋白质功能和进化
- **结构预测**：用于 3D 结构推断的接触预测
- **变异效应预测**：评估突变的影响
- **蛋白质设计**：生成新型蛋白质序列

该架构的模块化设计使其能够轻松适应这些不同的任务，同时保持一致的核心表示学习方法。

## 后续步骤

要了解 ESM-2 架构如何实现特定应用，请探索以下相关主题：

- [ESMFold 结构预测系统](10-esmfold-structure-prediction-system) - 了解 ESM-2 如何驱动蛋白质结构预测
- [语言模型表示](11-language-model-representations) - 深入探讨学习到的表示
- [蛋白质的 Transformer 架构](12-transformer-architecture-for-proteins) - transformer 应用的更广泛背景
- [零样本变异预测](16-zero-shot-variant-prediction) - 架构的实际应用

ESM-2 架构代表了既定 transformer 原理与蛋白质特定创新的精心平衡，使其成为蛋白质理解和工程应用的强大基础。