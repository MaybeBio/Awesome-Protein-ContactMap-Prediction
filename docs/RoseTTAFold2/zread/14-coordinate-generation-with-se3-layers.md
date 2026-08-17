---
slug:14-coordinate-generation-with-se3-layers
blog_type:normal
---


RoseTTAFold2 中的坐标生成利用 **SE(3)-等变 Transformer 层** 来迭代优化 3D 蛋白质结构，同时保持几何一致性。该机制构成了 3D 结构轨道的核心，通过等变神经网络操作，从序列和成对表示中精确预测原子坐标。

## SE(3)-等变架构概览

坐标生成系统通过一个复杂的流水线运行，该流水线将序列嵌入转换为精确的 3D 原子坐标，同时保持旋转和平移不变性。这是通过 **SE(3)-Transformer 架构** 实现的，该架构将蛋白质结构处理为具有等变消息传递的残基图。

```mermaid
flowchart TB
    A[输入特征] --> B[特征准备]
    B --> C[图构建]
    C --> D[SE3 Transformer 处理]
    D --> E[坐标更新]
    E --> F[侧链预测]
    F --> G[刚体变换]
    G --> H[更新后的坐标]
    
    A --> A1[MSA 嵌入<br/>(B, N, L, d_msa)]
    A --> A2[成对表示<br/>(B, L, L, d_pair)]
    A --> A3[初始坐标<br/>(B, L, 3, 3)]
    A --> A4[状态特征<br/>(B, L, d_state)]
    
    D --> D1[基于 Fiber 的处理<br/>Type-0: 不变量<br/>Type-1: 等变向量]
    D --> D2[基函数<br/>球谐函数<br/>Clebsch-Gordan 系数]
    D --> D3[径向分布函数<br/>距离加权核]
    
    E --> E1[Type-0 特征<br/>→ 状态更新]
    E --> E2[Type-1 特征<br/>→ 旋转和平移]
    
    G --> G1[四元数归一化]
    G --> G2[旋转矩阵组合]
    G --> G3[平移累积]
```

该架构通过仔细设计特征类型和基函数来保持 **SE(3)-等变性**——确保旋转或平移输入结构会导致产生同等旋转/平移的输出。系统使用 **Fiber 结构**来表示特征，其中 Type-0 特征是旋转不变标量，Type-1 特征是在旋转下协变变换的 3D 向量 [se3_transformer/model/fiber.py](/SE3Transformer/se3_transformer/model/fiber.py#L34-L44)。

来源：[network/Track_module.py](/network/Track_module.py#L490-L618), [network/SE3_network.py](/network/SE3_network.py#L12-L87)

## 特征准备与图构建

在进行 SE3 处理之前，系统准备多尺度特征并构建一个稀疏图表示，以捕获局部和长程残基相互作用。

### 节点特征嵌入

节点特征将来自多个轨道的信息组合为每个残基的统一表示：

```python
# MSA 和状态特征融合
seq = self.norm_msa(msa[:,0])  # 提取查询序列
state = self.norm_state(state)
node = self.embed_node1(seq) + self.embed_node2(state)
node = node + self.ff_node(node)
node = self.norm_node(node)
```

节点特征的形状为 `(B, L, l0_in_features)`，其中默认 `l0_in_features=32`，表示每个残基位置上的旋转不变信息 [network/Track_module.py](/network/Track_module.py#L543-L549)。

### 边特征生成

边特征使用成对表示和几何信息的组合来编码残基对关系：

**距离编码**：径向基函数（RBF）将成对距离转换为平滑、可微的表示：

```python
rbf_feat = rbf(torch.cdist(xyz[:,:,1], xyz[:,:,1])).reshape(B,L,L,-1)
```

RBF 使用 64 个分箱，覆盖 0.0 到 20.0 Å 的距离，标准差为 0.5 Å，从而创建详细的距离分布 [network/util_module.py](/network/util_module.py#L81-L85) [network/Track_module.py](/network/Track_module.py#L557-L558)。

**序列间隔**：有符号序列间隔特征帮助模型区分局部和非局部相互作用，Sergey 发现包含符号可以提供轻微的改进 [network/util_module.py](/network/util_module.py#L87-L99)。

**特征融合**：最终的边特征将成对表示与几何编码结合起来：

```python
edge = self.embed_edge1(pair) + self.embed_edge2(rbf_feat)
edge = edge + self.ff_edge(edge)
edge = self.norm_edge(edge)
```

这将产生形状为 `(B, L, L, num_edge_features)` 的边特征，其中 `num_edge_features=32` [network/Track_module.py](/network/Track_module.py#L556-L560)。

### 稀疏图构建

系统基于几何邻近性和序列邻接性创建一个连接残基的稀疏图：

```mermaid
graph LR
    A[距离计算] --> B[Top-K 选择]
    B --> C[序列邻接<br/>kmin=32]
    C --> D[边合并]
    D --> E[相对位置存储]
    
    B --> B1[为每个残基选择 K 个<br/>最近邻]
    C --> C1[连接残基<br/>|i-j| ≤ 32]
    D --> D1[Top-k 边与<br/>序列边的并集]
    E --> E1[从基计算中<br/>分离梯度]
```

图构建遵循以下步骤：

1. **距离计算**：使用 `torch.cdist` 计算 CA 原子之间的成对距离
2. **Top-K 选择**：对于每个残基，选择 `top_k=64` 个最近邻
3. **序列邻接**：始终连接 `kmin=32` 位置范围内的残基
4. **边合并**：将 top-k 边与序列边合并
5. **相对位置**：将相对位置向量作为分离的梯度存储，用于基函数计算 [network/util_module.py](/network/util_module.py#L218-L242)。

生成的图使用 DGL (Deep Graph Library) 进行高效的消息传递，边索引和预计算的相对位置作为边数据存储。

来源：[network/Track_module.py](/network/Track_module.py#L552-L569), [network/util_module.py](/network/util_module.py#L218-L242)

## SE(3)-Transformer 处理

坐标生成的核心发生在 **SE3Transformer** 中，它通过保持几何一致性的等变消息传递操作来处理图。

### 基于 Fiber 的特征表示

特征被组织成 **Fibers**——描述每个度（不可约表示类型）特征多样性的结构：

| 特征类型 | 度 | 维度 | 变换属性 |
|-------------|--------|-----------|------------------------|
| Type-0 | 0 | 1 | 旋转不变标量 |
| Type-1 | 1 | 3 | 等变 3D 向量 |
| Type-2 | 2 | 5 | 对称无迹矩阵 |

Fiber 类管理这些表示，实现不同特征类型的灵活组合 [se3_transformer/model/fiber.py](/SE3Transformer/se3_transformer/model/fiber.py#L34-L44)。

### 基函数计算

等变操作依赖于 **球谐函数** 和 **Clebsch-Gordan 系数** 来构造在旋转下正确变换的基矩阵：

```python
def get_basis_script(max_degree, use_pad_trick, spherical_harmonics, 
                     clebsch_gordon, amp):
    """计算高达 max_degree 度的成对基矩阵"""
    basis = {}
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f'{d_in},{d_out}'
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                Q_J = clebsch_gordon[idx][freq_idx]
                # 爱因斯坦求和以组合谐波和 CG 系数
                K_Js.append(torch.einsum('n f, k l f -> n l k', 
                                        spherical_harmonics[J], Q_J))
            basis[key] = torch.stack(K_Js, 2)
```

球谐函数从相对位置捕获方向信息，而 Clebsch-Gordan 系数实现不同度的耦合，所有这些都进行了缓存以提高计算效率 [se3_transformer/model/basis.py](/SE3Transformer/se3_transformer/model/basis.py#L73-L93)。

### 径向分布函数

**径向分布函数**充当距离相关的调制器，根据残基之间的几何关系对等变核进行加权：

```mermaid
flowchart LR
    A[相对距离<br/>||x-y||] --> B[额外不变量<br/>边特征]
    B --> C[MLP 网络<br/>在边之间共享]
    C --> D[径向权重<br/>R^{l,k}]
    D --> E[加权基<br/>矩阵]
    
    C --> C1[线性层]
    C1 --> C2[层归一化]
    C2 --> C3[ReLU 激活]
    C3 --> C4[线性层]
    C4 --> C5[层归一化]
    C5 --> C6[ReLU 激活]
    C6 --> C7[线性层]
```

径向函数是一个多层感知机，它接收不变的边特征（距离加上额外的不变信息）并产生等变核的权重。这种距离依赖性允许网络对几何约束进行建模，而不会破坏等变性 [se3_transformer/model/layers/convolution.py](/SE3Transformer/se3_transformer/model/layers/convolution.py#L67-L102)。

### 卷积操作

**ConvSE3** 模块通过组合以下内容执行等变消息传递：
- 源节点的输入特征
- 编码几何关系的基函数
- 调制贡献的径向权重

为了进行推理，系统实现了 **内存优化处理**，当边数超过阈值时，将边分块为 65,536 个一批，以减少内存占用，代价是额外的计算 [se3_transformer/model/layers/convolution.py](/SE3Transformer/se3_transformer/model/layers/convolution.py#L104-L165)。

来源：[SE3Transformer/se3_transformer/model/basis.py](/SE3Transformer/se3_transformer/model/basis.py#L1-L100), [SE3Transformer/se3_transformer/model/layers/convolution.py](/SE3Transformer/se3_transformer/model/layers/convolution.py#L1-L200)

## 坐标更新机制

SE3 transformer 产生两种类型的输出来驱动坐标更新：用于状态更新的 **Type-0 特征**（不变量）和用于刚体变换的 **Type-1 特征**（等变量）。

### 状态特征更新

Type-0 特征更新每个残基的不变状态表示：

```python
shift = self.se3(G, node, l1_feats, edge_feats)
state = state + shift['0'].reshape(B, L, -1)  # (B, L, C)
```

这种残差连接在结合新的几何上下文的同时保留了先前学习的信息 [network/Track_module.py](/network/Track_module.py#L571-L573)。

### 刚体变换

Type-1 特征编码被解释为增量刚体变换的 3D 向量：

```python
offset = shift['1'].reshape(B, L, 2, 3)
Ts = offset[:,:,0,:] * 10.0  # 平移
Qs = offset[:,:,1,:]         # 旋转（四元数分量）
```

系统对平移分量应用 **10.0 的缩放因子** 以提高数值稳定性，因为预测的平移通常很小 [network/Track_module.py](/network/Track_module.py#L574-L576)。

### 四元数归一化与转换

旋转向量被转换为四元数并归一化以确保有效的旋转：

```python
Qs = torch.cat((torch.ones((B,L,1),device=Qs.device), Qs), dim=-1)
Qs = normQ(Qs)  # 归一化为单位四元数
Rs = Qs2Rs(Qs)  # 转换为旋转矩阵
```

`normQ` 函数除以范数以确保单位四元数属性，`Qs2Rs` 执行从四元数到 3×3 旋转矩阵的代数转换 [network/util_module.py](/network/util_module.py#L54-L72) [network/kinematics.py](/network/kinematics.py#L14-L21) [network/kinematics.py](/network/kinematics.py#L54-L72)。

### 刚体组合

预测的变换与当前的刚体状态组合：

```python
Rs = einsum('bnij,bnjk->bnik', Rs, R_in)  # 旋转组合
Ts = Ts + T_in                             # 平移累积
```

这种组合允许模型增量地细化坐标，每次回收迭代应用微小的修正，这些修正累积起来提高了结构精度 [network/Track_module.py](/network/Track_module.py#L582-L583)。

来源：[network/Track_module.py](/network/Track_module.py#L571-L583), [network/kinematics.py](/network/kinematics.py#L14-L72)

## 侧链预测

坐标生成包括通过 **SCPred 模块** 进行侧链方向预测，该模块基于序列和更新的状态特征预测扭转角。

### 扭转角预测

侧链预测器输出全面的扭转角信息：

```python
# 输出：(B, L, 10, 2) - 10 个角度，带有 sin/cos 分量
# - phi, psi, omega: 骨架扭转角
# - chi1~4: 侧链 chi 角  
# - CB bend, CB twist, CG: 附加侧链描述符
alpha = self.sc_predictor(seqfull, state)
```

每个角度都用正弦和余弦分量表示，以确保优化期间适当的角连续性 [network/Track_module.py](/network/Track_module.py#L584-L585) [network/Track_module.py](/network/Track_module.py#L351-L399)。

### 特征整合

侧链预测器使用序列嵌入和来自 SE3 处理的更新状态特征的组合：

```python
def forward(self, seq, state):
    # seq: 隐藏嵌入 (B, L, d_msa)
    # state: SE3 type-0 特征 (B, L, d_state)
    # 输出: 预测的扭转角 (B, L, 10, 2)
```

这种整合允许模型利用序列信息和细化的几何上下文进行准确的侧链放置 [network/Track_module.py](/network/Track_module.py#L391-L399)。

来源：[network/Track_module.py](/network/Track_module.py#L351-L399), [network/Track_module.py](/network/Track_module.py#L584-L585)

## 通过回收进行迭代细化

坐标生成在 **迭代回收机制** 内运行，其中多次 SE3 层应用逐步细化结构。

### 回收循环结构

**IterativeSimulator** 协调多个回收迭代：

```python
class IterativeSimulator(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=12, n_ref_block=4, ...):
        # Extra blocks: 具有有限上下文的初始细化
        # Main blocks: 具有完整信息的完整 SE3 处理
        # Ref blocks: 用于高精度细节的最终细化
```

这种三阶段设计平衡了计算效率和预测准确性，在早期阶段使用较少的参数，在后期阶段使用更复杂的处理 [network/Track_module.py](/network/Track_module.py#L701-L715)。

### 梯度检查点

为了提高内存效率，系统支持 **梯度检查点**：

```python
def forward(self, seq, msa, msa_full, pair, xyz_in, state, idx, ...,
            use_checkpoint=False, ...):
    if use_checkpoint:
        # 在反向传播期间重新计算中间激活
        # 以额外的计算为代价减少内存
```

这使得在有限的 GPU 内存上处理更大的蛋白质成为可能 [network/Track_module.py](/network/Track_module.py#L753-L759)。

来源：[network/Track_module.py](/network/Track_module.py#L701-L759)

## 实现细节与优化

### 内存优化策略

坐标生成系统采用了几种内存优化技术：

| 技术 | 目的 | 实现 |
|-----------|---------|----------------|
| 跨步处理 | 减少长序列的内存 | 以 `STRIDE` 个残基的块进行处理 |
| 梯度检查点 | 用计算换取内存 | 在反向传播中重新计算激活 |
| 边分块 | 限制密集图的内存 | 每次处理 65,536 条边 |
| 关键操作使用 Float32 | 确保数值稳定性 | 强制 SE3 前向传播使用 float32 |

跨步机制分块处理蛋白质序列，对每个块分别应用嵌入和前馈层，同时通过图操作维护全局上下文 [network/Track_module.py](/network/Track_module.py#L531-L551) [network/SE3_network.py](/network/SE3_network.py#L78-L87)。

### 初始化策略

适当的权重初始化对于稳定训练至关重要：

```python
def reset_parameter(self):
    # 线性层的 LeCun 正态初始化
    for n, p in self.se3.named_parameters():
        if "bias" in n:
            nn.init.zeros_(p)
        elif len(p.shape) == 1:
            continue
        else:
            if "radial_func" not in n:
                p = init_lecun_normal_param(p)
            else:
                # 零初始化最终的径向函数层
                if "net.6" in n:
                    nn.init.zeros_(p)
                else:
                    nn.init.kaiming_normal_(p, nonlinearity='relu')
    
    # 零初始化最终的 SE3 层权重
    nn.init.zeros_(self.se3.graph_modules[-1].weights['0'])
    if self.l1_out > 0:
        nn.init.zeros_(self.se3.graph_modules[-1].weights['1'])
```

最终 SE3 层的零初始化确保网络最初预测恒等变换（零位移），为迭代细化提供了稳定的起点 [network/SE3_network.py](/network/SE3_network.py#L57-L74)。

来源：[network/SE3_network.py](/network/SE3_network.py#L57-L74), [network/Track_module.py](/network/Track_module.py#L531-L551)

## 配置与超参数

### SE3 层配置

SE3 transformer 参数平衡模型容量与计算效率：

| 参数 | 默认值 | 描述 |
|-----------|---------------|-------------|
| `num_layers` | 2 | SE3 transformer 层数 |
| `num_channels` | 32 | 每个度的隐藏通道 |
| `num_degrees` | 3 | 最大度 (0, 1, 2) |
| `n_heads` | 4 | 注意力头数 |
| `l0_in_features` | 32 | 输入 Type-0 通道 |
| `l0_out_features` | 16 | 输出 Type-0 通道 |
| `num_edge_features` | 32 | 边特征维度 |

Type-1 特征被配置为将 3D 向量表示（N-CA 和 CA-C 向量）作为输入处理，并输出旋转/平移信息 [network/SE3_network.py](/network/SE3_network.py#L14-L40)。

### 图构建参数

图拓扑参数控制连接性和内存使用：

| 参数 | 默认值 | 影响 |
|-----------|---------------|--------|
| `top_k` | 64 | 每个残基的最近邻 |
| `kmin` | 32 | 序列邻接范围 |
| `STRIDE` | 可变 | 用于内存优化的块大小 |

较大的 `top_k` 值捕获更多的长程相互作用，但会增加内存和计算 [network/util_module.py](/network/util_module.py#L218-L242)。

来源：[network/SE3_network.py](/network/SE3_network.py#L14-L40), [network/util_module.py](/network/util_module.py#L218-L242)

## 数学基础

坐标生成机制依赖于 **等变神经网络理论**，其中操作经过仔细设计以保持 SE(3) 对称性。

### 等变性属性

函数 **f** 是 SE(3)-等变的，如果：

```
f(g · x) = g · f(x)
```

其中 **g** 是 SE(3) 群中的任何旋转或平移，**x** 表示输入特征。SE3 transformer 通过以下方式保持此属性：
- **Type-0 操作**：不受旋转影响的完全不变标量
- **Type-1 操作**：协变旋转的向量
- **基函数**：在旋转下正确变换的球谐函数
- **径向函数**：不受旋转影响的距离相关标量

### 刚体变换

更新机制通过仔细的组合保持适当的刚体变换：

1. **预测增量变换**：来自 SE3 输出的 ΔR, Δt
2. **归一化旋转**：转换为单位四元数以确保数值稳定性
3. **与当前状态组合**：R_new = ΔR · R_old, t_new = Δt + t_old

这确保每次回收迭代都产生有效的刚体变换 [network/Track_module.py](/network/Track_module.py#L571-L583)。

## 后续步骤

了解使用 SE3 层生成坐标可深入了解 RoseTTAFold2 的几何核心。要更深入地了解相关概念：

- 了解 **[三轨设计：MSA、成对和 3D 结构轨道](6-three-track-design-msa-pair-and-3d-structure-tracks)** 以查看 3D 轨道如何与其他表示轨道集成
- 探索 **[SE(3)-等变 Transformer 网络](7-se-3-equivariant-transformer-network)** 以获取详细的理论基础
- 了解 **[用于迭代细化的回收机制](8-recycling-mechanism-for-iterative-refinement)** 以查看坐标更新如何在迭代中传播
- 查看 **[FAPE (帧对齐点误差) 损失](16-fape-frame-aligned-point-error-loss)** 以了解如何在训练期间测量坐标精度