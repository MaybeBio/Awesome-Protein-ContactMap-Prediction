---
slug:9-dihedral-angle-computation
blog_type:normal
---


`torch_chain_dihedrals` 函数是 idpGAN 中连接笛卡尔坐标与扭角空间表示的可微桥梁。该函数完全在 PyTorch 中实现，可将一批 3D 骨架构象转换为相应的二面角序列——这一转换对于下游的立体化学分析以及构建[镜像选择器网络](7-mirror-image-selector-network)所需的旋转不变输入特征都至关重要。

来源：[coords.py](/idpgan/coords.py#L1-L19), [nn_models.py](/idpgan/nn_models.py#L7-L7)

## 数学基础

**二面角**（也称扭角）由四个连续键合的原子 A–B–C–D 定义。它量化了包含原子 B–C–D 的平面相对于包含原子 A–B–C 的平面的旋转量，该旋转围绕共用键向量 **b₁** = C − B 进行测量。idpGAN 的实现遵循**带符号的 atan2 约定**，生成的角度范围为 (−π, π]：

$$\phi = \text{atan2}(y, x)$$

其中：

| 分量 | 表达式 | 几何含义 |
|-----------|-----------|-------------------|
| **b₀** | −(r_{i+1} − r_i) | 原子 i 与 i+1 之间键向量的负值 |
| **b₁** | r_{i+2} − r_{i+1} | 原子 i+1 与 i+2 之间的键向量 |
| **b₂** | r_{i+3} − r_{i+2} | 原子 i+2 与 i+3 之间的键向量 |
| **x** | (b₀ × b₁) · (b₁ × b₂) | 余弦缩放投影（归一化后） |
| **y** | ((b₀ × b₁) × (b₁ × b₂)) · b₁ / ‖b₁‖ | 带符号的正弦缩放投影 |

叉积 **b₀ × b₁** 和 **b₁ × b₂** 定义了这两个平面的法向量。相较于 `acos`，更推荐使用 `atan2(y, x)` 公式，因为它避免了在 0°/180° 处的分支切割歧义，并且无需额外逻辑即可生成正确的符号。

来源：[coords.py](/idpgan/coords.py#L5-L19)

## 实现演练

函数 `torch_chain_dihedrals` 作用于分批的骨架坐标张量，并在单次向量化传递中计算所有连续的二面角：

```python
def torch_chain_dihedrals(xyz, norm=False):
    r_sel = xyz
    b0 = -(r_sel[:,1:-2,:] - r_sel[:,0:-3,:])
    b1 = r_sel[:,2:-1,:] - r_sel[:,1:-2,:]
    b2 = r_sel[:,3:,:]   - r_sel[:,2:-1,:]
    b0xb1 = torch.cross(b0, b1)
    b1xb2 = torch.cross(b2, b1)
    b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2)
    y = torch.sum(b0xb1_x_b1xb2*b1, axis=2)*(1.0/torch.linalg.norm(b1, dim=2))
    x = torch.sum(b0xb1*b1xb2, axis=2)
    dh_vals = torch.atan2(y, x)
    if not norm:
        return dh_vals
    else:
        return dh_vals/np.pi
```

切片模式 `[:, k:-m, :]` 沿残基轴移动，以形成每组四个连续的原子。对于包含 **L** 个残基的链，这会产生 **L − 3** 个二面角——每个内部可旋转键对应一个。该计算是完全批处理的：第一维度 (`N`) 索引系综中的构象，整个操作执行过程中无需 Python 循环。

来源：[coords.py](/idpgan/coords.py#L5-L19)

### 逐步数据流" flowchart LR
    A["输入 xyz<br/>(N, L, 3)"] --> B["键向量<br/>b₀, b₁, b₂"]
    B --> C["叉积<br/>b₀×b₁, b₁×b₂"]
    C --> D["双重叉积<br/>(b₀×b₁)×(b₁×b₂)"]
    D --> E["y = 点积 · ‖b₁‖⁻¹<br/>x = 点积"]
    E --> F["atan2(y, x)"]
    F --> G{"norm=True?"}
    G -->|否| H["φ ∈ (−π, π]<br/>(N, L−3)"]
    G -->|是| I["φ/π ∈ (−1, 1]<br/>(N, L−3)"]
```

来源：[coords.py](/idpgan/coords.py#L5-L19)

## API 参考

| 参数 | 类型 | 形状 | 描述 |
|-----------|------|-------|-------------|
| `xyz` | `torch.Tensor` | `(N, L, 3)` | 批量的笛卡尔坐标集。`N` = 系综大小，`L` = 链长 |
| `norm` | `bool` | — | 若为 `True`，则除以 π 将输出归一化至 (−1, 1] |
| **返回值** | `torch.Tensor` | `(N, L−3)` | 以弧度表示的二面角（若 `norm=True` 则为归一化值） |

<CgxTip>当 `norm=True` 时，输出范围变为 (−1, 1]，作为期望有界激活的神经网络层的输入时，数值稳定性更佳。然而，在选择器流水线中默认使用 `norm=False`，以保留 cos/sin 编码的自然弧度尺度。</CgxTip>

来源：[coords.py](/idpgan/coords.py#L5-L19)

## 在镜像选择器流水线中的作用

`torch_chain_dihedrals` 的主要消费者是 idpGAN 的 **ABSINTH 变体**（`ABSIdpGANGenerator`）。在基础生成器生成一个 XYZ 坐标系综后，二面角被计算并转换为旋转等变特征张量，馈送给 `StereoSelNN` 镜像选择器：

```python
# Inside ABSIdpGANGenerator.predict_idp()
dh_gen = torch_chain_dihedrals(xyz_gen)                        # (N, L-3)
dh_gen = torch.cat([zeros(N,1), dh_gen], axis=1)               # 为残基 0 前置填充
dh_gen = torch.cat([dh_gen, zeros(N,2)], axis=1)               # 为最后 2 个残基后置填充
x = torch.cat([torch.cos(dh_gen).unsqueeze(2),                 # (N, L, 1)
               torch.sin(dh_gen).unsqueeze(2),                 # (N, L, 1)
               mask.unsqueeze(2)], axis=2)                      # → (N, L, 3)
```

此流水线执行三个关键转换：

1. **填充** — 长度为 L − 3 的二面角在第一个和最后两个残基位置用零进行填充，以恢复长度 L，使角度张量与逐残基的坐标张量对齐。附加一个二进制 **掩码**（有效位置为 1，填充位置为 0），使选择器网络能够区分真实的扭角数据与填充值。

2. **圆形编码** — 原始角度被转换为 `(cos φ, sin φ)` 对。此映射在**拓扑上是正确的**：接近 −π 和 +π 的角度映射到单位圆上相近的点，避免了将原始弧度值直接用作特征时会产生的人为不连续性。

3. **立体分类** — 生成的 `(N, L, 3)` 特征张量 (cos, sin, mask) 被传递给 `StereoSelNN`，由其预测每个构象是否具有正确的手性。被标记为镜像的构象通过取反其 z 坐标进行反转。

来源：[nn_models.py](/idpgan/nn_models.py#L576-L612)

### ABSINTH 流水线中的端到端二面角流" flowchart TD
    A["生成器产生<br/>xyz_gen (N, L, 3)"] --> B["torch_chain_dihedrals<br/>→ (N, L−3)"]
    B --> C["用零填充<br/>→ (N, L)"]
    C --> D["cos/sin 编码<br/>+ 掩码 → (N, L, 3)"]
    D --> E["StereoSelNN<br/>→ σ ∈ (0, 1)"]
    E --> F{"σ < 0.5?"}
    F -->|是| G["反转: z ← −z"]
    F -->|否| H["保持不变"]
    G --> I["校正后的 xyz_gen"]
    H --> I
```

来源：[nn_models.py](/idpgan/nn_models.py#L583-L612)

## 设计原理与可微性

该实现相对于输入坐标是**完全可微的**——每一个操作（减法、叉积、点积、`linalg.norm`、`atan2`）在 PyTorch 的 autograd 框架中都具有定义良好的梯度。这一特性至关重要，因为二面角计算位于生成器网络输出的下游；在选择器网络的训练期间，梯度必须穿过二面角计算回传至生成坐标的层。

选择 `atan2` 而非 `acos` 并非仅仅为了美观。`acos` 函数在其分支点（0 和 π）处具有**无界梯度**，当二面角接近这些值时，会在反向传播期间引起数值不稳定——这在蛋白质骨架扭角中很常见。`atan2` 公式在各处都保持有界梯度，使其成为可微分流水线中数值鲁棒的选择。

<CgxTip>叉积 `b1xb2` 是通过 `cross(b2, b1)` 而非 `cross(b1, b2)` 计算的——这种刻意的符号反转被吸收到了 `atan2(y, x)` 的计算中，并不影响最终角度，但在与教科书中的参考二面角公式进行对比时，这是一个需要注意的细微实现细节。</CgxTip>

来源：[coords.py](/idpgan/coords.py#L10-L15)

## 使用模式

该函数可直接在任何批量坐标张量上调用。以下是代码库中的两种主要使用模式：

| 上下文 | 调用 | 输出形状 | 目的 |
|---------|------|-------------|---------|
| ABSINTH 选择器流水线 | `torch_chain_dihedrals(xyz_gen)` | `(N, L−3)` | 为 `StereoSelNN` 构建特征 |
| 笔记本分析 | `torch_chain_dihedrals(torch_tensor)` | `(N, L−3)` | 直接检查生成系综的扭角 |

在自定义分析中导入时：

```python
from idpgan.coords import torch_chain_dihedrals

# xyz: 形状为 (N, L, 3) 的 torch.Tensor
dihedrals = torch_chain_dihedrals(xyz)         # 弧度，形状 (N, L-3)
dihedrals_norm = torch_chain_dihedrals(xyz, norm=True)  # 归一化至 (-1, 1]
```

来源：[coords.py](/idpgan/coords.py#L5-L19), [nn_models.py](/idpgan/nn_models.py#L588-L588)

## 与其他模块的关系

二面角计算占据了一个特定的架构定位：它是 idpGAN 代码库中从笛卡尔坐标空间转换到扭角空间的**唯一模块**。其唯一的消费者是生成器流水线的 ABSINTH 变体，该变体将其与[镜像选择器网络](7-mirror-image-selector-network)配对使用。基础的 CG 模型生成器（`IdpGANGenerator`）完全在笛卡尔空间中运行，从不调用二面角计算——它直接生成并评估 XYZ 坐标和距离图。

| 模块 | 与 `torch_chain_dihedrals` 的关系 |
|--------|----------------------------------------|
| [Transformer 生成器网络](5-transformer-generator-network) | 生成作为输入的 XYZ 坐标 |
| [镜像选择器网络](7-mirror-image-selector-network) | 消费 cos/sin 编码的二面角特征 |
| [FASTA 解析与 PDB 输出](8-fasta-parsing-and-pdb-output) | 独立——处理序列 I/O，而非坐标 |
| [氨基酸特征编码](10-amino-acid-feature-encoding) | 独立——提供逐残基特征，而非几何特征 |

**下一步**：要了解二面角衍生特征如何被选择器网络消费，请参阅[镜像选择器网络](7-mirror-image-selector-network)。对于在此计算之前的坐标生成，请参阅[Transformer 生成器网络](5-transformer-generator-network)。