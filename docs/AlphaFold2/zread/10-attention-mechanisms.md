---
slug:10-attention-mechanisms
blog_type:normal
---


注意力机制构成了 AlphaFold 的 Evoformer 架构的计算骨干，使模型能够捕获多序列比对（MSA）和残基对之间的复杂关系。这些机制实现了复杂的信息流模式，对于从进化信息中理解蛋白质结构至关重要。

## 核心多头注意力

AlphaFold 中的基本注意力实现遵循标准的多头注意力模式，并包含若干蛋白质特异性增强。基础的 `Attention` 类 [modules.py:579-680] 提供了核心计算框架：

```python
def __call__(self, q_data, m_data, mask, nonbatched_bias=None):
    # 使用多头设置进行查询、键、值投影
    q = jnp.einsum('bqa,ahc->bqhc', q_data, q_weights) * key_dim ** (-0.5)
    k = jnp.einsum('bka,ahc->bkhc', m_data, k_weights)
    v = jnp.einsum('bka,ahc->bkhc', m_data, v_weights)
    
    # 注意力计算及可选偏置项
    logits = jnp.einsum('bqhc,bkhc->bhqk', q, k)
    if nonbatched_bias is not None:
        logits += jnp.expand_dims(nonbatched_bias, axis=0)
    
    # 带掩码的 Softmax 及最终投影
    weights = utils.stable_softmax(logits)
    weighted_avg = jnp.einsum('bhqk,bkhc->bqhc', weights, v)
```

<CgxTip>注意力机制通过 sigmoid 激活的门控实现可选门控，使模型能够根据输入特征动态控制信息流。</CgxTip>

## MSA 级注意力机制

### 带对偏置的 MSA 行注意力

`MSARowAttentionWithPairBias` [modules.py:795-840] 代表了一项关键创新，其中 MSA 序列间的注意力由成对表示偏置引导：

```python
# 将成对表示投影为偏置项
weights = hk.get_parameter('feat_2d_weights', shape=(pair_act.shape[-1], c.num_head))
nonbatched_bias = jnp.einsum('qkc,ch->hqk', pair_act, weights)

# 应用带成对衍生偏置的注意力
attn_mod = Attention(c, self.global_config, msa_act.shape[-1])
msa_act = mapping.inference_subbatch(attn_mod, self.global_config.subbatch_size,
                                   batched_args=[msa_act, msa_act, mask],
                                   nonbatched_args=[nonbatched_bias])
```

该机制使模型能够将残基间关系信息直接整合到序列间的注意力计算中。

### MSA 列注意力

`MSAColumnAttention` [modules.py:858-890] 在每个残基位置跨序列操作，捕获进化保守模式：

```python
# 交换轴以跨序列计算注意力
msa_act = jnp.swapaxes(msa_act, -2, -3)
msa_mask = jnp.swapaxes(msa_mask, -1, -2)

# 在列方向应用注意力
mask = msa_mask[:, None, None, :]
```

### 全局 MSA 列注意力

`MSAColumnGlobalAttention` [modules.py:910-950] 实现了一种特殊注意力，其中所有序列都关注一个全局查询（该查询计算为所有序列的均值）：

```python
# 计算全局查询（序列均值）
q_avg = utils.mask_mean(q_mask, q_data, axis=1)
q = jnp.einsum('ba,ahc->bhc', q_avg, q_weights) * key_dim ** (-0.5)

# 全局注意力计算
logits = jnp.einsum('bhc,bkc->bhk', q, k)
weights = utils.stable_softmax(logits)
weighted_avg = jnp.einsum('bhk,bkc->bhc', weights, v)
```

<CgxTip>全局注意力通过每个头使用单个查询来降低计算复杂度，特别适合处理包含大量序列的深度 MSA。</CgxTip>

## 成对表示的三角形注意力

`TriangleAttention` [modules.py:963-1020] 机制作用于成对表示，捕获残基间的三元关系：

```python
# 方向特定处理
if c.orientation == 'per_column':
    pair_act = jnp.swapaxes(pair_act, -2, -3)
    pair_mask = jnp.swapaxes(pair_mask, -1, -2)

# 成对表示的自注意力
nonbatched_bias = jnp.einsum('qkc,ch->hqk', pair_act, weights)
attn_mod = Attention(c, self.global_config, pair_act.shape[-1])
pair_act = mapping.inference_subbatch(attn_mod, self.global_config.subbatch_size,
                                   batched_args=[pair_act, pair_act, mask],
                                   nonbatched_args=[nonbatched_bias])
```

三角形注意力有两种方向：
- **起始节点**：残基 i 关注所有残基 j，以残基 k 为条件
- **终止节点**：残基 j 关注所有残基 k，以残基 i 为条件

## 在 Evoformer 中的集成

注意力机制在 `EvoformerIteration` [modules.py:1751-1900] 中被策略性集成，形成全面的信息处理管道：

```python
# 带注意力的 MSA 处理
msa_act = dropout_wrapper_fn(
    MSARowAttentionWithPairBias(c.msa_row_attention_with_pair_bias, gc),
    msa_act, msa_mask, safe_key=next(sub_keys), pair_act=pair_act)

msa_act = dropout_wrapper_fn(
    MSAColumnAttention(c.msa_column_attention, gc),
    msa_act, msa_mask, safe_key=next(sub_keys))

# 带三角形注意力的成对处理
pair_act = dropout_wrapper_fn(
    TriangleAttention(c.triangle_attention_starting_node, gc),
    pair_act, pair_mask, safe_key=next(sub_keys))

pair_act = dropout_wrapper_fn(
    TriangleAttention(c.triangle_attention_ending_node, gc),
    pair_act, pair_mask, safe_key=next(sub_keys))
```

## 关键架构模式

| 注意力类型 | 输入形状 | 输出形状 | 主要功能 |
|----------------|-------------|--------------|------------------|
| 带对偏置的 MSA 行 | [N_seq, N_res, c_m] | [N_seq, N_res, c_m] | 将成对信息整合到 MSA 处理中 |
| MSA 列 | [N_seq, N_res, c_m] | [N_seq, N_res, c_m] | 捕获进化保守性 |
| MSA 全局列 | [N_seq, N_res, c_m] | [N_seq, N_res, c_m] | 高效处理深度 MSA |
| 三角形（起始/终止） | [N_res, N_res, c_z] | [N_res, N_res, c_z] | 建模三元残基关系 |

## 计算考量

AlphaFold 中的注意力实现包含多种优化策略：

1. **子批次处理**：使用 `mapping.inference_subbatch` 将大型注意力计算分解为子批次，以管理内存限制
2. **掩码机制**：复杂的掩码确保仅在有效序列位置间计算注意力
3. **偏置整合**：非批次偏置允许将结构和进化先验直接整合到注意力分数中

这些注意力机制协同工作，处理来自 MSA 的进化信息和来自成对表示的结构信息，使 AlphaFold 能够在蛋白质结构预测中实现前所未有的准确性。

要深入了解这些注意力机制如何融入更广泛的架构，请参阅 [Evoformer 模块设计](9-evoformer-module-design) 和 [模型架构概述](11-model-architecture-overview)。