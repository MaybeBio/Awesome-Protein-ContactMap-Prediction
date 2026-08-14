---
slug:17-api-reference
blog_type:normal
---
本API参考文档提供了lucidrains/alphafold2 PyTorch实现的全面文档。该实现遵循AlphaFold2论文中描述的架构，并针对PyTorch进行了特定调整。

## 核心类

### Alphafold2

实现AlphaFold2架构的主要模型类。

```python
class Alphafold2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_seq_len=2048,
        depth=6,
        heads=8,
        dim_head=64,
        max_rel_dist=32,
        num_tokens=constants.NUM_AMINO_ACIDS,
        num_embedds=constants.NUM_EMBEDDS_TR,
        max_num_msas=constants.MAX_NUM_MSA,
        max_num_templates=constants.MAX_NUM_TEMPLATES,
        extra_msa_evoformer_layers=4,
        attn_dropout=0.,
        ff_dropout=0.,
        templates_dim=32,
        templates_embed_layers=4,
        templates_angles_feats_dim=55,
        predict_angles=False,
        symmetrize_omega=False,
        predict_coords=False,
        structure_module_depth=4,
        structure_module_heads=1,
        structure_module_dim_head=4,
        disable_token_embed=False,
        mlm_mask_prob=0.15,
        mlm_random_replace_token_prob=0.1,
        mlm_keep_token_same_prob=0.1,
        mlm_exclude_token_ids=(0,),
        recycling_distance_buckets=32
    )
```

**参数:**
- `dim` (int): 整个网络的隐藏维度大小
- `max_seq_len` (int, 可选): 最大序列长度。默认为2048。
- `depth` (int, 可选): Evoformer块的数量。默认为6。
- `heads` (int, 可选): 注意力头的数量。默认为8。
- `dim_head` (int, 可选): 每个注意力头的维度。默认为64。
- `max_rel_dist` (int, 可选): 位置嵌入的最大相对距离。默认为32。
- `predict_angles` (bool, 可选): 是否预测二面角。默认为False。
- `predict_coords` (bool, 可选): 是否预测坐标。默认为False。
- `structure_module_depth` (int, 可选): 结构模块的深度。默认为4。

**前向传播方法:**

```python
def forward(
    self,
    seq,
    msa=None,
    mask=None,
    msa_mask=None,
    extra_msa=None,
    extra_msa_mask=None,
    seq_index=None,
    seq_embed=None,
    msa_embed=None,
    templates_feats=None,
    templates_mask=None,
    templates_angles=None,
    embedds=None,
    recyclables=None,
    return_trunk=False,
    return_confidence=False,
    return_recyclables=False,
    return_aux_logits=False
)
```

**参数:**
- `seq` (Tensor): 主要氨基酸序列 [batch, seq_len]
- `msa` (Tensor, 可选): 多序列比对 [batch, num_alignments, seq_len]
- `mask` (Tensor, 可选): 序列掩码 [batch, seq_len]
- `msa_mask` (Tensor, 可选): MSA掩码 [batch, num_alignments, seq_len]
- `seq_embed` (Tensor, 可选): 预计算的序列嵌入
- `msa_embed` (Tensor, 可选): 预计算的MSA嵌入
- `recyclables` (Recyclables, 可选): 来自前次迭代的回收信息

**返回值:**
- 如果 `predict_coords=False` 或 `return_trunk=True`: 返回包含预测距离、角度等的 `ReturnValues` 对象
- 如果 `predict_coords=True` 且 `return_aux_logits=True`: 返回 (坐标, `ReturnValues`) 元组
- 如果 `predict_coords=True` 且 `return_confidence=True`: 返回 (坐标, 置信度分数) 元组
- 如果 `predict_coords=True` 且无其他标志: 返回坐标 [batch, seq_len, 3]

来源: [alphafold2.py#L469-L905](alphafold2_pytorch/alphafold2.py#L469-L905)

### Evoformer

基于Transformer的核心模块，用于迭代处理MSA和成对表示。

```python
class Evoformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    )
```

**参数:**
- `depth` (int): EvoformerBlock层的数量
- `dim` (int): 隐藏维度大小
- `seq_len` (int): 最大序列长度
- `heads` (int): 注意力头的数量
- `dim_head` (int): 每个注意力头的维度
- `attn_dropout` (float): 注意力的dropout率
- `ff_dropout` (float): 前馈网络的dropout率

**前向传播方法:**

```python
def forward(
    self,
    x,
    m,
    mask=None,
    msa_mask=None
)
```

**参数:**
- `x` (Tensor): 成对表示 [batch, seq_len, seq_len, dim]
- `m` (Tensor): MSA表示 [batch, num_alignments, seq_len, dim]
- `mask` (Tensor, 可选): 成对掩码 [batch, seq_len, seq_len]
- `msa_mask` (Tensor, 可选): MSA掩码 [batch, num_alignments, seq_len]

**返回值:**
- (更新后的成对表示, 更新后的MSA表示) 元组

来源: [alphafold2.py#L448-L467](alphafold2_pytorch/alphafold2.py#L448-L467)

### EvoformerBlock

Evoformer中的单个块，用于处理MSA和成对表示。

```python
class EvoformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        global_column_attn=False
    )
```

**参数:**
- 与Evoformer相同，额外增加 `global_column_attn` 标志

**前向传播方法:**

```python
def forward(self, inputs)
```

**参数:**
- `inputs`: (成对表示, MSA表示, 成对掩码, MSA掩码) 元组

**返回值:**
- 包含更新后表示的相同元组

来源: [alphafold2.py#L412-L446](alphafold2_pytorch/alphafold2.py#L412-L446)

## 嵌入包装器

### MSAEmbedWrapper

使用预训练MSA Transformer进行嵌入的包装器。

```python
class MSAEmbedWrapper(nn.Module):
    def __init__(self, *, alphafold2)
```

**参数:**
- `alphafold2` (Alphafold2): Alphafold2模型的实例

**前向传播方法:**

```python
def forward(self, seq, msa, msa_mask=None, **kwargs)
```

**参数:**
- `seq` (Tensor): 主要序列
- `msa` (Tensor): 多序列比对
- `msa_mask` (Tensor, 可选): MSA掩码

**返回值:**
- 应用了嵌入的Alphafold2模型输出

来源: [embeds.py#L33-L75](alphafold2_pytorch/embeds.py#L33-L75)

### ESMEmbedWrapper

使用预训练ESM模型进行嵌入的包装器。

```python
class ESMEmbedWrapper(nn.Module):
    def __init__(self, *, alphafold2)
```

**参数:**
- `alphafold2` (Alphafold2): Alphafold2模型的实例

**前向传播方法:**

```python
def forward(self, seq, msa=None, **kwargs)
```

**参数:**
- `seq` (Tensor): 主要序列
- `msa` (Tensor, 可选): 多序列比对

**返回值:**
- 应用了嵌入的Alphafold2模型输出

来源: [embeds.py#L77-L103](alphafold2_pytorch/embeds.py#L77-L103)

### ProtTranEmbedWrapper

使用ProtTran (BERT)进行嵌入的包装器。

```python
class ProtTranEmbedWrapper(nn.Module):
    def __init__(self, *, alphafold2)
```

**参数:**
- `alphafold2` (Alphafold2): Alphafold2模型的实例

**前向传播方法:**

```python
def forward(self, seq, msa, msa_mask=None, **kwargs)
```

**参数:**
- 与MSAEmbedWrapper相同

**返回值:**
- 应用了嵌入的Alphafold2模型输出

来源: [embeds.py#L10-L31](alphafold2_pytorch/embeds.py#L10-L31)

## 注意力机制

### Attention

模型中使用的基础注意力机制。

```python
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len=None,
        heads=8,
        dim_head=64,
        dropout=0.,
        gating=True
    )
```

**参数:**
- `dim` (int): 输入维度
- `seq_len` (int, 可选): 序列长度
- `heads` (int, 可选): 注意力头的数量。默认为8。
- `dim_head` (int, 可选): 每个头的维度。默认为64。
- `dropout` (float, 可选): Dropout率。默认为0。
- `gating` (bool, 可选): 是否使用门控。默认为True。

**前向传播方法:**

```python
def forward(self, x, mask=None, attn_bias=None, context=None, context_mask=None, tie_dim=None)
```

**参数:**
- `x` (Tensor): 输入张量
- `mask` (Tensor, 可选): 注意力掩码
- `attn_bias` (Tensor, 可选): 注意力偏置
- `context` (Tensor, 可选): 交叉注意力的上下文
- `tie_dim` (int, 可选): 要绑定的查询维度

**返回值:**
- 注意力输出张量

来源: [alphafold2.py#L98-L190](alphafold2_pytorch/alphafold2.py#L98-L190)

### AxialAttention

用于高效处理2D数据的轴向注意力。

```python
class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        row_attn=True,
        col_attn=True,
        accept_edges=False,
        global_query_attn=False,
        **kwargs
    )
```

**参数:**
- `dim` (int): 输入维度
- `heads` (int): 注意力头的数量
- `row_attn` (bool, 可选): 启用行注意力。默认为True。
- `col_attn` (bool, 可选): 启用列注意力。默认为True。
- `accept_edges` (bool, 可选): 是否接受边输入。默认为False。
- `global_query_attn` (bool, 可选): 是否使用全局查询。默认为False。

**前向传播方法:**

```python
def forward(self, x, edges=None, mask=None)
```

**参数:**
- `x` (Tensor): 输入张量 [batch, height, width, dim]
- `edges` (Tensor, 可选): 边特征
- `mask` (Tensor, 可选): 注意力掩码

**返回值:**
- 应用了轴向注意力的更新张量

来源: [alphafold2.py#L192-L255](alphafold2_pytorch/alphafold2.py#L192-L255)

### TriangleMultiplicativeModule

用于成对表示更新的三角形乘法模块。

```python
class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim=None,
        mix='ingoing'
    )
```

**参数:**
- `dim` (int): 输入维度
- `hidden_dim` (int, 可选): 隐藏维度。默认为dim。
- `mix` (str, 可选): 混合模式，'ingoing'或'outgoing'。默认为'ingoing'。

**前向传播方法:**

```python
def forward(self, x, mask=None)
```

**参数:**
- `x` (Tensor): 输入张量 [batch, seq_len, seq_len, dim]
- `mask` (Tensor, 可选): 注意力掩码

**返回值:**
- 三角形乘法后的更新张量

来源: [alphafold2.py#L257-L317](alphafold2_pytorch/alphafold2.py#L257-L317)

## 位置编码

### FixedPositionalEmbedding

标准正弦位置嵌入。

```python
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim)
```

**参数:**
- `dim` (int): 嵌入维度

**前向传播方法:**

```python
def forward(self, n, device)
```

**参数:**
- `n` (int): 序列长度
- `device` (torch.device): 创建嵌入的设备

**返回值:**
- [sin, cos] 嵌入列表

来源: [rotary.py#L35-L45](alphafold2_pytorch/rotary.py#L35-L45)

### AxialRotaryEmbedding

用于成对数据的2D旋转嵌入。

```python
class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10)
```

**参数:**
- `dim` (int): 嵌入维度
- `max_freq` (int, 可选): 最大频率。默认为10。

**前向传播方法:**

```python
def forward(self, n, device)
```

**参数:**
- `n` (int): 序列长度
- `device` (torch.device): 创建嵌入的设备

**返回值:**
- 2D数据的[sin, cos]嵌入列表

来源: [rotary.py#L47-L67](alphafold2_pytorch/rotary.py#L47-L67)

## 输出容器

### ReturnValues

模型输出的容器。

```python
@dataclass
class ReturnValues:
    distance: torch.Tensor = None
    theta: torch.Tensor = None
    phi: torch.Tensor = None
    omega: torch.Tensor = None
    msa_mlm_loss: torch.Tensor = None
    recyclables: Recyclables = None
```

**属性:**
- `distance` (Tensor): 预测的距离图
- `theta` (Tensor): 预测的theta角
- `phi` (Tensor): 预测的phi角
- `omega` (Tensor): 预测的omega角
- `msa_mlm_loss` (Tensor): MSA掩码语言模型损失
- `recyclables` (Recyclables): 回收信息

来源: [alphafold2.py#L30-L37](alphafold2_pytorch/alphafold2.py#L30-L37)

### Recyclables

可回收表示的容器。

```python
@dataclass
class Recyclables:
    coords: torch.Tensor
    single_msa_repr_row: torch.Tensor
    pairwise_repr: torch.Tensor
```

**属性:**
- `coords` (Tensor): 预测的坐标
- `single_msa_repr_row` (Tensor): 单个MSA表示行
- `pairwise_repr` (Tensor): 成对表示

来源: [alphafold2.py#L24-L28](alphafold2_pytorch/alphafold2.py#L24-L28)

## 使用示例

### 基本模型初始化

```python
import torch
from alphafold2_pytorch import Alphafold2

model = Alphafold2(
    dim = 256,               # 嵌入维度
    depth = 4,               # 注意力网络深度
    heads = 8,               # 注意力头数
    dim_head = 64,           # 每个头维度
    predict_angles = True,   # 预测角度
    predict_coords = True    # 预测坐标
)

# 准备输入
seq = torch.randint(0, 20, (2, 128))  # 2个序列的批次，长度128
msa = torch.randint(0, 20, (2, 16, 128))  # 2个MSA的批次，16个比对，长度128
mask = torch.ones_like(seq).bool()
msa_mask = torch.ones_like(msa).bool()

# 前向传播
coords = model(seq, msa, mask=mask, msa_mask=msa_mask)
print(coords.shape)  # [2, 128, 3] - 批次，序列长度，3D坐标
```

### 使用ESM嵌入

```python
from alphafold2_pytorch import Alphafold2, ESMEmbedWrapper

# 初始化基础模型
base_model = Alphafold2(
    dim = 256,
    predict_coords = True
)

# 用ESM嵌入包装
model = ESMEmbedWrapper(alphafold2=base_model)

# 使用ESM嵌入的前向传播
seq = torch.randint(0, 20, (1, 256))
coords = model(seq)
```

### 回收预测

```python
model = Alphafold2(
    dim = 256,
    predict_coords = True,
    return_recyclables = True
)

# 第一次迭代
seq = torch.randint(0, 20, (1, 128))
coords, recyclables = model(
    seq, 
    return_recyclables=True
)

# 带回收的第二次迭代
refined_coords = model(
    seq,
    recyclables=recyclables
)
```

## 常量和配置

该实现使用了`constants.py`中定义的几个常量，包括：

- `NUM_AMINO_ACIDS`: 氨基酸类型数量
- `DISTOGRAM_BUCKETS`: 距离图预测的距离区间数量
- `THETA_BUCKETS`, `PHI_BUCKETS`, `OMEGA_BUCKETS`: 角度区间数量
- `MAX_NUM_MSA`: MSA序列的最大数量
- `MAX_NUM_TEMPLATES`: 模板的最大数量

<CgxTip>
**重要提示**: 该实现通过PyTorch的`checkpoint_sequential`支持梯度检查点，以减少内存使用。这对于在较长序列上进行训练或微调尤为重要。
</CgxTip>