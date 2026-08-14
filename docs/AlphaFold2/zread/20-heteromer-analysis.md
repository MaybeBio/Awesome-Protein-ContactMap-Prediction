---
slug:20-heteromer-analysis
blog_type:normal
---


AlphaFold-Multimer 中的异聚体分析能够预测由不同多肽链组成的蛋白质复合物，这是理解蛋白质-蛋白质相互作用和细胞机制的基础能力。该分析将 AlphaFold 的单体预测能力扩展至多链组装，同时保持结构准确性和生物学相关性。

## 异聚体检测与分类

异聚体分析流程始于复杂的序列分类机制，用于区分同聚体和异聚体复合物。系统采用基于实体的分组方式，将相同序列的链归类为同聚体，而不同序列的链则被分类为异聚体 [pipeline_multimer.py#L130-L170]。

```python
def add_assembly_features(all_chain_features):
    # 按序列同一性对链进行分组
    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_features in all_chain_features.values():
        seq = str(chain_features['sequence'])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_features)
```

分类过程为每种不同的序列类型分配唯一的实体标识符，使模型能够区分异聚体复合物内的链类型。这种基于实体的方法确保了对包含多种不同蛋白质序列的复杂组装体的正确处理。

## 异聚体复合物的 MSA 配对

多序列比对（MSA）配对是异聚体分析的基石，使模型能够识别不同蛋白质链之间的进化关系。配对算法通过基于物种的匹配运行，其中来自同一生物体不同链的序列根据序列相似性进行配对 [msa_pairing.py#L256-L320]。

<CgxTip>配对过程会过滤仅存在于一条链中的物种，并将每个物种的 MSA 大小限制为 600 条序列，以在保持计算效率的同时确保生物学相关性。</CgxTip>

配对算法遵循以下关键步骤：

1. **物种识别**：从每条链的 MSA 元数据中提取物种标识符
2. **共同物种检测**：识别在多条链中均有序列的生物体
3. **基于相似性的配对**：根据与目标序列的相似性在链间匹配序列
4. **质量过滤**：排除低质量比对和代表性不足的物种

## 链感知相对位置编码

AlphaFold-Multimer 采用了专门为异聚体复合物设计的相对位置编码机制。系统通过链感知编码区分链内和链间关系 [modules_multimer.py#L550-L650]。

当启用 `use_chain_relative` 时，模型会生成三种类型的相对特征：

1. **基于位置的编码**：标准残基位置差异，并增加"不同链"分箱
2. **实体身份编码**：指示残基是否属于相同链类型的二元特征
3. **对称性索引编码**：裁剪并独热编码的相对链索引

```python
if c.use_chain_relative:
    # 为异聚体添加"不同链"分箱
    final_offset = jnp.where(
        asym_id_same,
        clipped_offset,
        (2 * c.max_relative_idx + 1) * jnp.ones_like(clipped_offset),
    )
    
    # 用于链类型区分的实体身份
    entity_id_same = jnp.equal(entity_id[:, None], entity_id[None, :])
    
    # 对称性感知的相对链索引
    rel_sym_id = sym_id[:, None] - sym_id[None, :]
```

这种编码策略使模型能够区分同聚体界面（相同链类型）和异聚体界面（不同链类型），为准确的复合物预测提供关键的空间上下文信息。

## 特征整合与组装

异聚体分析流程整合多个特征流以创建蛋白质复合物的综合表示。系统处理配对和未配对的 MSA 特征，通过复杂的连接策略将它们组合 [msa_pairing.py#L498-L520]。

<CgxTip>特征合并过程对同聚体和异聚体组件保持独立处理，确保对链内和链间关系的最佳表示。</CgxTip>

整合工作流程包括：

1. **配对特征生成**：为具有跨链进化耦合的序列创建配对的 MSA 特征
2. **块对角构造**：以块对角格式组装未配对特征，实现独立的链表示
3. **特征连接**：沿序列维度合并配对和未配对特征
4. **模板整合**：使用链感知填充和比对处理模板特征

## 性能优化策略

异聚体分析采用多种优化策略来处理多链预测的计算复杂性：

1. **MSA 大小限制**：将每个物种的 MSA 深度限制为 600 条序列以防止内存溢出
2. **稀疏配对**：仅对在多条链中均有代表的物种序列进行配对
3. **高效填充**：对模板和 MSA 特征使用最小填充策略
4. **块对角操作**：利用高效线性代数处理稀疏矩阵运算

## 异聚体特定配置

多聚体模型为异聚体分析提供了特定的配置选项：

| 配置参数 | 描述 | 默认值 |
|------------------------|-------------|---------------|
| `use_chain_relative` | 启用链感知相对位置编码 | True |
| `max_relative_idx` | 最大相对位置索引 | 32 |
| `max_relative_chain` | 最大相对链索引 | 2 |
| `pair_msa_sequences` | 启用跨链 MSA 序列配对 | True |

这些参数控制异聚体复合物的计算效率与预测准确性之间的平衡。

## 与流水线架构的集成

异聚体分析通过 DataPipeline 类与更广泛的 AlphaFold-Multimer 流水线无缝集成 [pipeline_multimer.py#L184-L270]。处理工作流程包括：

1. **单链处理**：对每个序列独立应用单体流水线
2. **MSA 生成**：为配对生成个体和全序列 MSA
3. **特征组装**：将所有链的特征组合为统一表示
4. **模型推理**：通过多聚体特定架构处理组装后的特征

该流水线在保持与现有单体工作流程兼容性的同时，添加了异聚体特定处理步骤，确保在各种复合物类型上的稳健性能。

## 后续步骤

要全面了解多聚体预测能力，请探索 [AlphaFold-Multimer 架构](18-alphafold-multimer-architecture) 文档。要理解蛋白质复合物预测的更广泛背景，请查阅 [蛋白质复合物预测](19-protein-complex-prediction)。有关底层进化耦合机制的详细信息，请参见 [MSA 生成与处理](12-msa-generation-and-processing)。