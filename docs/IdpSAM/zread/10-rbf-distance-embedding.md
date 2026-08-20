---
slug:10-rbf-distance-embedding
blog_type:normal
---


RBF（径向基函数）距离嵌入模块将原始的两两距离值转换为高维、富有表现力的特征向量，作为编码器 Transformer 架构的 **2D 几何骨架**。如果直接将标量距离输入神经网络——这几乎无法为模型提供学习距离依赖模式的表征能力——RBF 展开会将每个距离在一组高斯核上“展宽”，生成多通道信号，下游的线性投影层随后可将其塑造成信息丰富的配对表示。这是欧几里得几何与注意力机制之间的关键桥梁。

## 在流水线中的架构角色

RBF 嵌入位于编码器前向传播的最开端，在任何 Transformer 处理发生之前，对完整的 Cα–Cα 距离图进行操作。数据流如下：

```mermaid
flowchart LR
    A["Cα 坐标<br/>(B, L, 3)"] -->|calc_dmap| B["距离图<br/>(B, 1, L, L)"]
    B -->|RBF 展开| C["高斯特征<br/>(B, 1, L, L, K)"]
    C -->|转置| D["(B, K, L, L, 1)"]
    D -->|project_dmap| E["2D 配对嵌入<br/>(B, L, L, E₂)"]
    E --> F["Transformer<br/>注意力偏置"]
```

3D 坐标首先通过 `calc_dmap` 转换为两两距离图，然后通过选定的 RBF 核进行展开，最后由一个可学习的线性层投影到 `embed_2d_dim` 维的配对表示中，该表示作为注意力偏置被注入到每一个 Transformer 块中。这确保了**几何邻近信息在编码的每一层都可用**，而不仅仅是在输入层。

来源: [coords.py](sam/coords.py#L5-L37), [encoder.py](sam/nn/autoencoder/encoder.py#L190-L199)

## GaussianSmearing — 固定中心 RBF 展开

`GaussianSmearing` 是 idpSAM 中默认且主要的 RBF 变体。它在距离范围内放置**等距分布的固定高斯中心**，并根据每个中心评估输入距离，生成类似软直方图的表示。

**数学定义：**

$$\phi_k(d) = \exp\!\left(-\frac{(d - \mu_k)^2}{2\delta^2}\right)$$

其中 $\mu_k$ 是从 `start` 到 `stop` 线性间隔的中心，$\delta$ 是相邻中心之间的均匀间距，$d$ 是输入距离。系数 $-\frac{1}{2\delta^2}$ 在构建时预计算以提高效率。

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `start` | `0.0` | 第一个高斯中心的最小距离 (Å) |
| `stop` | `5.0` | 最后一个高斯中心的最大距离 (Å) |
| `num_gaussians` | `50` | 高斯核数量（控制输出维度） |

**实现细节：** 偏移张量（中心）被注册为缓冲区——而非可学习参数——并被重塑为 `(1, K, 1, 1)`，以便与形状为 `(B, 1, L, L)` 的距离图进行广播时，生成形状为 `(B, K, L, L)` 的输出。系数以普通浮点数存储，因为它在初始化后永远不会改变。

来源: [geometric.py](sam/nn/geometric.py#L6-L19)

## ExpNormalSmearing — 可训练的指数-正态 RBF

`ExpNormalSmearing` 实现了 **PhysNet 风格**的径向基函数展开，它在评估高斯核之前对距离应用指数变换。该设计的动机源于一个观察：蛋白质中的原子间相互作用在距离上高度非线性——短距离范围内的微小距离变化远比长距离范围内的同等变化重要。指数映射自然地将表征容量集中在最关键的地方。

**数学定义：**

$$\phi_k(d) = \text{cutoff}(d) \cdot \exp\!\left(-\beta_k \left(\exp\!\left(\alpha(-d + d_{\min})\right) - \mu_k\right)^2\right)$$

其中 $\alpha = 5 / (d_{\max} - d_{\min})$ 是缩放因子，均值 $\mu_k$ 从 $\exp(-d_{\max} + d_{\min})$ 到 1 线性初始化，宽度 $\beta_k$ 基于核间距均匀初始化。关键是，`means` 和 `betas` 都可以是**可训练参数**，允许网络在训练期间调整核的位置和锐度。

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `cutoff_lower` | `0.0` | 距离下界 (Å) |
| `cutoff_upper` | `5.0` | 距离上界 (Å) — 也是截断半径 |
| `num_rbf` | `50` | RBF 核数量 |
| `trainable` | `True` | 若为 `True`，means 和 betas 为 `nn.Parameter`；否则为缓冲区 |

当 `trainable=True` 时，`means` 和 `betas` 张量被注册为 `nn.Parameter` 对象，将通过梯度下降进行更新。当 `trainable=False` 时，它们作为缓冲区被冻结——在可更新性方面与 `GaussianSmearing` 的行为完全相同，尽管函数形式仍然不同。

来源: [geometric.py](sam/nn/geometric.py#L22-L64)

## CosineCutoff — 平滑距离衰减

`CosineCutoff` 模块提供了**平滑、可微的衰减**，在截断边界处将 RBF 输出逐渐减至零。若无此模块，嵌入在 `cutoff_upper` 处将存在硬间断，这会导致梯度异常并可能破坏训练稳定性。

对于 `cutoff_lower = 0` 的常见情况：

$$\text{cutoff}(d) = \frac{1}{2}\left(\cos\!\left(\frac{d \cdot \pi}{d_{\max}}\right) + 1\right) \cdot \mathbb{1}[d < d_{\max}]$$

当 `cutoff_lower > 0` 时，在上下界之间应用对称余弦包络，并使用指示函数将区间外的贡献置零。指示掩码使用浮点转换的布尔乘法，该方法几乎处处可微，并在边界处产生精确的零值。

来源: [geometric.py](sam/nn/geometric.py#L67-L95)

## 编码器如何选择和使用 RBF 变体

`CA_TransformerEncoder` 基于 `dmap_embed_type` 配置键分派至相应的 RBF 类：

```python
if dmap_embed_type == "rbf":
    self.dmap_ca_expansion = GaussianSmearing(
        start=dmap_ca_min, stop=dmap_ca_cutoff,
        num_gaussians=dmap_ca_num_gaussians)
elif dmap_embed_type == "expnorm":
    self.dmap_ca_expansion = ExpNormalSmearing(
        cutoff_lower=dmap_ca_min, cutoff_upper=dmap_ca_cutoff,
        num_rbf=dmap_ca_num_gaussians, trainable=dmap_embed_trainable)
```

展开后，原始的 K 维 RBF 输出被投影到 2D 嵌入空间。有两种可用的投影模式，由 `use_dmap_mlp` 控制：

| `use_dmap_mlp` | 投影 | 参数 |
|----------------|-----------|------------|
| `False`（默认） | 单层线性：`K → E₂` | 最少；一个权重矩阵 |
| `True` | 带激活函数的两层 MLP：`K → E₂ → E₂` | 更具表达力；在已发表的模型中使用 |

在**已发表的 v1.0 配置**中，编码器使用 `GaussianSmearing`，包含 320 个跨越 0.0–3.0 Å 的高斯核，以及两层 MLP 投影。3.0 Å 这一较窄的截断值值得注意——它将所有表征容量集中在**局部和中程距离**上，这些距离对骨架几何最具信息量，同时忽略了长距离，后者通过 Transformer 的多跳注意力被间接捕获。

来源: [encoder.py](sam/nn/autoencoder/encoder.py#L91-L112), [models.yaml](config/models.yaml#L30-L34)

## 集成到 Transformer 注意力机制

投影后的 2D 嵌入作为 `z` 参数流入每一个编码器 Transformer 块。在 `AE_IdpGAN_TransformerBlock.forward` 内部，2D RBF 特征在与位置嵌入组合后，作为注意力偏置传入：

| `embed_2d_inject_mode` | 组合方式 | 传入注意力的形状 |
|------------------------|------------|--------------------------|
| `"concat"` | `z_hat = cat([z, p], dim=-1)` | `(B, L, L, E₂ + P)` |
| `"add"` | `z_hat = z + project(p)` | `(B, L, L, E₂)` |
| `None` | `z_hat = p`（忽略 RBF） | `(B, L, L, P)` |

v1.0 模型使用 `"concat"`，因此源自 RBF 的配对特征与相对位置嵌入作为**独立通道堆叠**，而非加性混合——在注意力机制学习融合它们之前，保留了每种模态的独特信息内容。

来源: [ca_models.py](sam/nn/autoencoder/ca_models.py#L128-L136)

## 配置参考

编码器配置中完整的 RBF 相关参数集，如 `models.yaml` 所示：

| 键 | v1.0 值 | 类型 | 描述 |
|-----|-----------|------|-------------|
| `dmap_ca_min` | `0.0` | float | 距离范围下界 (Å) |
| `dmap_ca_cutoff` | `3.0` | float | 上界 / 截断距离 (Å) |
| `dmap_ca_num_gaussians` | `320` | int | RBF 核数量 |
| `dmap_embed_type` | `"rbf"` | str | `"rbf"` 对应 GaussianSmearing，`"expnorm"` 对应 ExpNormalSmearing |
| `dmap_embed_trainable` | — | bool | 仅适用于 `"expnorm"`：means/betas 是否可学习 |
| `use_dmap_mlp` | `true` | bool | 使用两层 MLP 投影代替单层线性 |

<CgxTip>在 0–3 Å 范围内的 320 个高斯核产生了约 0.0094 Å 的核间距——远细于骨架原子典型的热涨落尺度（约 0.1 Å）。这种超分辨率是刻意的：它使得线性投影层能够充当一个可学习的滤波器组，从 RBF 表示中提取关于距离的任意平滑函数，类似于傅里叶基在给定足够项数时能够表示任何周期函数。</CgxTip>

来源: [models.yaml](config/models.yaml#L30-L34)

## 与其他模块的关系

RBF 距离嵌入是编码器中两个几何特征提取器之一，另一个是通过 `torch_chain_dihedrals` 计算的**扭转角特征**。它们共同提供了 Transformer 架构学习从结构到潜在编码映射所需的 1D（扭转）和 2D（配对距离）几何信号。RBF 模块具体回答了以下问题：*“应该如何表示两两距离，以便 Transformer 能够高效地从中学习？”*——答案是平滑、高维的固定基展开，随后是可学习的投影。

关于配套的 1D 几何特征，请参阅 [距离图与扭转特征](9-distance-map-and-torsion-features)。关于投影后的 2D 嵌入如何与注意力计算交互，请参阅 [自定义 Transformer 注意力](11-custom-transformer-attention)。