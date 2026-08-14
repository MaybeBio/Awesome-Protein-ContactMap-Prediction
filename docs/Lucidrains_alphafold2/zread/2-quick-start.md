---
slug:2-quick-start
blog_type:normal
---


本指南将帮助您快速上手`alphafold2-pytorch`，这是DeepMind革命性的AlphaFold2蛋白质结构预测系统的非官方PyTorch实现。只需几分钟，您就能使用这个强大的深度学习框架进行蛋白质结构预测。

## 安装

使用pip安装包：

```bash
pip install alphafold2-pytorch
```

就这么多！核心功能现在已为您可用。

来源：[README.md#L13-L17](README.md#L13-L17)

<CgxTip>
如果您计划使用预训练嵌入（如ESM或ProtTrans），您还需要安装Nvidia的apex库。可以通过以下命令安装：

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
</CgxTip>

## 基本用法

让我们从最简单的用例开始——预测蛋白质序列的距离图（distogram）：

```python
import torch
from alphafold2_pytorch import Alphafold2

# 初始化模型
model = Alphafold2(
    dim = 256,           # 隐藏维度大小
    depth = 2,           # Transformer层数
    heads = 8,           # 注意力头数
    dim_head = 64        # 每个注意力头的维度
).cuda()

# 创建样本输入
seq = torch.randint(0, 21, (1, 128)).cuda()      # 主序列（batch_size, seq_length）
msa = torch.randint(0, 21, (1, 5, 120)).cuda()   # 多序列比对（batch_size, num_alignments, msa_length）
mask = torch.ones_like(seq).bool().cuda()        # 序列掩码
msa_mask = torch.ones_like(msa).bool().cuda()    # MSA掩码

# 获取预测
distogram = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask
) # 输出形状：(1, 128, 128, 37) - 表示成对距离概率
```

来源：[README.md#L29-L54](README.md#L29-L54), [alphafold2.py#L32-L38](alphafold2_pytorch/alphafold2.py#L32-L38)

## 预测角度

对于需要更多信息而不仅仅是距离的应用，您可以通过设置`predict_angles = True`来预测角度（theta, phi, omega）：

```python
model = Alphafold2(
    dim = 256,
    depth = 2,
    heads = 8,
    dim_head = 64,
    predict_angles = True   # 启用角度预测
).cuda()

# 输入准备与之前相同
seq = torch.randint(0, 21, (1, 128)).cuda()
msa = torch.randint(0, 21, (1, 5, 120)).cuda()
mask = torch.ones_like(seq).bool().cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

# 获取包括角度的预测
distogram, theta, phi, omega = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask
)

# 输出形状：
# distogram - (1, 128, 128, 37) - 成对距离概率
# theta     - (1, 128, 128, 25) - theta角概率  
# phi       - (1, 128, 128, 13) - phi角概率
# omega     - (1, 128, 128, 25) - omega角概率
```

来源：[README.md#L56-L86](README.md#L56-L86)

## 预测3D坐标

AlphaFold2最强大的功能是预测蛋白质结构的实际3D坐标。通过设置`predict_coords = True`来启用此功能：

```python
model = Alphafold2(
    dim = 256,
    depth = 2,
    heads = 8,
    dim_head = 64,
    predict_coords = True,                  # 启用坐标预测
    structure_module_type = 'se3',          # 使用SE3 Transformer进行结构细化
    structure_module_dim = 4,               # 结构模块维度
    structure_module_depth = 1,             # 结构细化层数
    structure_module_heads = 1,             # 结构模块注意力头数
    structure_module_dim_head = 16,         # 结构模块头维度
    structure_module_refinement_iters = 2,  # 细化迭代次数
    structure_num_global_nodes = 1          # 结构模块中的全局节点数
).cuda()

# 输入准备
seq = torch.randint(0, 21, (2, 64)).cuda()
msa = torch.randint(0, 21, (2, 5, 60)).cuda()
mask = torch.ones_like(seq).bool().cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

# 获取预测坐标
coords = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask
) # 输出形状：(2, 64 * 3, 3) - 每个残基的3个原子的3D坐标
```

默认情况下，模型预测3个主链原子（C, Cα, N）的坐标。您可以使用`atoms`参数自定义要包括的原子：

```python
model = Alphafold2(
    dim = 256,
    depth = 2,
    heads = 8,
    dim_head = 64,
    predict_coords = True,
    atoms = 'backbone-with-cbeta'  # 在预测中包括C-beta原子
).cuda()

# 现在坐标将包括C-beta原子
# 输出形状将为(batch_size, seq_length * 4, 3)
```

来源：[README.md#L88-L157](README.md#L88-L157), [README.md#L129-L166](README.md#L129-L166)

## 使用预训练嵌入

为了获得更好的性能，您可以使用来自ESM或MSA Transformer等模型的预训练蛋白质嵌入：

```python
import torch
from alphafold2_pytorch import Alphafold2
from alphafold2_pytorch.embeds import MSAEmbedWrapper

# 创建AlphaFold2模型
alphafold2 = Alphafold2(
    dim = 256,
    depth = 2,
    heads = 8,
    dim_head = 64
)

# 用MSA嵌入包装
model = MSAEmbedWrapper(
    alphafold2 = alphafold2
).cuda()

# 输入准备
seq = torch.randint(0, 21, (2, 16)).cuda()
mask = torch.ones_like(seq).bool().cuda()
msa = torch.randint(0, 21, (2, 5, 16)).cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

# 使用预训练嵌入获取预测
distogram = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask
)
```

来源：[README.md#L177-L219](README.md#L177-L219)

## 高级功能

AlphaFold2-PyTorch提供了许多高级功能，您可以在熟悉库后进行探索：

| 功能 | 描述 | 关键参数 |
|------|------|----------|
| 卷积块 | 为序列和MSA添加卷积层 | `use_conv = True` |
| 线性注意力 | 使用线性注意力以提高交叉注意力的效率 | `cross_attn_linear = True` |
| 实值距离预测 | 直接预测均值和标准差 | `predict_real_value_distances = True` |
| 模板处理 | 在预测中包含模板信息 | 传入`templates_seq`、`templates_coors`、`templates_mask` |
| 结构模块类型 | 选择不同的等变网络 | `structure_module_type = 'se3'`、`'egnn'`或`'en'` |

来源：[README.md#L235-L269](README.md#L235-L269), [README.md#L420-L449](README.md#L420-L449), [README.md#L271-L354](README.md#L271-L354), [README.md#L507-L545](README.md#L507-L545), [README.md#L588-L600](README.md#L588-L600)

## 下一步

现在您已经基本了解了如何使用alphafold2-pytorch，您可以：

1. 探索`notebooks/`目录中的Jupyter笔记本，获取实用示例
2. 尝试不同的模型配置
3. 使用您自己的蛋白质数据训练模型
4. 查看`scripts/`目录中的结构细化脚本

请记住，这是AlphaFold2的非官方实现，随着原架构更多细节的揭示，它将继续演进。