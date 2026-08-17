---
slug:12-transformer-architecture-for-proteins
blog_type:normal
---



ESM（Evolutionary Scale Modeling）框架实现了专为蛋白质序列分析和理解设计的复杂 transformer 架构。这些架构利用注意力机制的强大功能来捕获蛋白质序列中的复杂模式和关系，支持从结构预测到功能注释的各种应用。

## 核心 Transformer 架构

### ESM-2 模型架构

ESM-2 模型代表了最先进的蛋白质语言模型，它基于深度 transformer 架构构建，并包含多项关键创新：

```python
class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
```

该架构由多个 transformer 层组成，包含旋转位置嵌入、专门的层归一化和接触预测能力 [esm/model/esm2.py#L14-L23]。每层遵循标准的 transformer 模式，包含自注意力机制和前馈网络，但融合了蛋白质特定的优化。

### Transformer 层实现

核心 `TransformerLayer` 实现了基础构建块：

```python
class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
    ):
```

每层包含带有旋转嵌入的自注意力、专门的层归一化（ESM1bLayerNorm）以及使用 GELU 激活的两层前馈网络 [esm/modules.py#L84-L145]。残差连接和层归一化采用预归一化模式以提高训练稳定性。

## 多头注意力机制

### 旋转位置嵌入

ESM 集成了旋转位置嵌入（RoPE），无需绝对位置嵌入即可提供位置信息：

```python
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, *_, **__):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
```

旋转嵌入根据查询和键向量的位置应用旋转矩阵，使模型能够捕获相对位置关系 [esm/rotary_embedding.py#L37-L47]。这种方法对蛋白质序列特别有效，因为局部和远程相互作用都至关重要。

### 多头注意力实现

`MultiheadAttention` 类实现了带有多项蛋白质特定优化的缩放点积注意力：

```python
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        use_rotary_embeddings: bool = False,
    ):
```

关键特性包括支持旋转嵌入、高效批处理以及蛋白质序列的专门掩码 [esm/multihead_attention.py#L74-L88]。注意力机制可以选择性返回头权重以支持可解释性和接触预测。

## 专门的 Transformer 变体

### MSA Transformer

MSA（Multiple Sequence Alignment）Transformer 扩展了架构以处理进化信息：

```python
class MSATransformer(nn.Module):
    def __init__(self, args, alphabet):
        # Implementation for MSA processing
```

该变体同时处理多序列比对，捕获同源序列中的协同进化模式 [esm/model/msa_transformer.py#L21-L23]。架构使用轴向注意力来高效处理 MSA 数据的二维特性。

### GVP-Transformer

对于反向折叠应用，ESM 实现了将几何向量感知器与 transformer 架构相结合的 GVP-Transformer：

```python
class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.
    
    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """
```

这种混合架构通过 GVP 层处理 3D 结构信息，然后输入到标准 transformer 层 [esm/inverse_folding/gvp_transformer.py#L11-L19]。

## 层归一化策略

ESM 实现了两种层归一化方法：

### ESM1LayerNorm
```python
class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
```

TensorFlow 风格的层归一化，将 epsilon 置于平方根内以确保数值稳定性 [esm/modules.py#L44-L47]。

### ESM1bLayerNorm
```python
try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
    class ESM1bLayerNorm(_FusedLayerNorm):
```

使用 Apex 的优化融合层归一化，提高 GPU 性能 [esm/modules.py#L60-L67]。

## 接触预测集成

transformer 架构包含专门用于接触预测的组件：

```python
self.contact_head = ContactPredictionHead(
    self.num_layers * self.attention_heads,
    self.prepend_bos,
    self.append_eos,
    eos_idx=self.eos_idx,
)
```

接触头处理所有层的注意力权重以预测残基间接触，利用 transformer 捕获长程依赖关系的能力 [esm/model/esm2.py#L58-L64]。

<CgxTip>旋转位置嵌入的使用消除了对绝对位置嵌入的需求，使模型对序列长度变化更加鲁棒，并能更好地捕获蛋白质中的相对结构关系。</CgxTip>

<CgxTip>ESM 的 transformer 架构通过专门的层类型支持单序列和 MSA 处理，使同一核心框架能够高效处理不同的蛋白质分析任务。</CgxTip>

## 架构比较

| 特性 | ESM-2 | MSA Transformer | GVP-Transformer |
|---------|-------|------------------|-----------------|
| 输入类型 | 单序列 | 多序列比对 | 3D 结构 + 序列 |
| 位置编码 | 旋转嵌入 | 轴向注意力 | 几何特征 |
| 主要用途 | 通用蛋白质理解 | 进化分析 | 反向折叠 |
| 注意力模式 | 标准自注意力 | 轴向注意力 | 标准 + GVP 处理 |

这种 transformer 架构构成了 ESM 蛋白质分析能力的基础，从序列理解到结构预测和设计。模块化设计允许轻松适应特定的蛋白质相关任务，同时保持基于注意力的核心学习范式。

要深入了解特定实现，请探索 [ESM-2 架构与设计](9-esm-2-architecture-and-design)或深入研究 [MSA Transformer 用于多序列比对](13-msa-transformer-for-multiple-sequence-alignment)。