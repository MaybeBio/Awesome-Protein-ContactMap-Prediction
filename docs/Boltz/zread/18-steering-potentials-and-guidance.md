---
slug:18-steering-potentials-and-guidance
blog_type:normal
---


Boltz 采用双机制引导框架，将基于物理的约束注入扩散采样循环，使模型无需重新训练即可生成结构合理的预测。该框架结合了 **Fleming-Katilman (FK) 粒子重采样** —— 一种促进粒子集合中低能量构象的概率选择机制 —— 与 **物理引导更新** —— 一种基于梯度的校正，直接引导去噪坐标满足几何与化学约束。这两种机制共同将原始的神经去噪器从无约束生成器转变为受引导的采样器，使其遵循键几何、范德华极限、立体化学、链对称性、模板参考以及你指定的接触约束。

来源：[potentials.py](src/boltz/model/potentials/potentials.py#L1-L30), [diffusion.py](src/boltz/model/modules/diffusion.py#L449-L465), [main.py](src/boltz/main.py#L147-L158)

## 架构概述

引导系统作为插入扩散去噪循环的推理中介层运行。在每个去噪步产生 x₀ 预测后，引导模块会评估一组能量势，计算其梯度，并在下一个扩散步继续之前应用一种或两种校正路径。整个流程由 `AtomDiffusion.sample()` 编排，它根据 `BoltzSteeringParams` 配置有条件地激活 FK 重采样、物理引导或两者同时激活。

```mermaid
flowchart TD
    A[扩散步: 去噪 x₀] --> B{引导激活?}
    B -- 否 --> C[标准 Euler 步]
    B -- 是 --> D[评估所有势]
    D --> E[计算能量 E]
    D --> F[计算梯度 ∇E]
    
    E --> G{FK 引导?}
    G -- 是 --> H[跨粒子计算<br/>log-G 权重]
    H --> I[通过 softmax 选择<br/>重采样粒子]
    
    F --> J{物理引导?}
    J -- 是 --> K[对去噪坐标进行<br/>迭代梯度下降步]
    K --> L[缩放引导更新<br/>Δx = -∇E · step_scale · (σ_t - t̂)/t̂]
    
    I --> M[更新 atom_coords]
    L --> M
    M --> C
    
    subgraph 势库
        P1[PoseBustersPotential]
        P2[ConnectionsPotential]
        P3[VDWOverlapPotential]
        P4[SymmetricChainCOMPotential]
        P5[ChiralAtomPotential]
        P6[StereoBondPotential]
        P7[PlanarBondPotential]
        P8[ContactPotentital]
        P9[TemplateReferencePotential]
    end
    
    势库 --> D
```

`get_potentials` 工厂函数基于两个布尔标志（`fk_steering` 和 `physical_guidance_update`，在 Boltz-2 中额外包含 `contact_guidance_update`）构建激活的势集合。当启用 `fk_steering` 时，会实例化所有七个物理势；当在 Boltz-2 模式下启用 `contact_guidance_update` 时，会将 `ContactPotentital` 和 `TemplateReferencePotential` 添加到集合中。

来源：[potentials.py](src/boltz/model/potentials/potentials.py#L670-L787), [diffusion.py](src/boltz/model/modules/diffusion.py#L449-L530)

## 势抽象

每个引导势都继承自 `Potential` 抽象基类，该类定义了三阶段的计算协议：**参数准备**（`compute_args`）、**变量计算**（`compute_variable`）和**能量函数评估**（`compute_function`）。这种分离实现了清晰的混入架构，其中几何变量提取器（距离、二面角、参考对齐）与能量势面形状（平底阱）相互独立组合。

### 核心计算流水线

`compute` 方法编排了完整的能量评估。它首先调用 `compute_args` 从输入特征中提取原子索引和参数元组。如果请求了质心（COM）聚合 —— 如在 `SymmetricChainCOMPotential` 中 —— 原子会通过带有 `"mean"` 归约的 `scatter_reduce` 分散为每条链的质心代表。如果提供了参考坐标 —— 如在 `TemplateReferencePotential` 中 —— 坐标会被索引到参考原子位置。生成的坐标张量随后流经 `compute_variable`（为每个约束对生成标量几何观测量）和 `compute_function`（将该观测量映射为能量值）。

`compute_gradient` 方法与此流水线类似，但额外通过 `compute_variable` 和 `compute_function` 进行反向传播，以生成解析的逐原子梯度向量。这些梯度使用带有 `"sum"` 归约的 `scatter_reduce` 散布回原始原子索引，并且在应用了 COM 聚合时，梯度会从质心代表映射回组成原子。

来源：[potentials.py](src/boltz/model/potentials/potentials.py#L15-L200), [potentials.py](src/boltz/model/potentials/potentials.py#L91-L200)

### 平底能量函数

`FlatBottomPotential` 实现了核心能量势面：在 `lower_bounds` 和 `upper_bounds` 之间是零能量阱，在外部则是线性递增的惩罚。这是库中所有具体势共享的能量函数。对于具有边界 `[lo, hi]` 和刚度 `k` 的值 `v`：

- 若 `v < lo`：能量 = `k · (lo - v)`，导数 = `-k`
- 若 `v > hi`：能量 = `k · (v - hi)`，导数 = `+k`  
- 若 `lo ≤ v ≤ hi`：能量 = `0`，导数 = `0`

取反掩码为“保持远离”约束反转了阱的语义：边界被交换，使得惩罚应用于区间内部而非外部。`ContactPotentital` 使用此机制来同时强制残基对之间的吸引和排斥。

来源：[potentials.py](src/boltz/model/potentials/potentials.py#L231-L279)

### 几何变量提取器

三个变量提取器构成了几何主干，每个都提供一个值和一个解析梯度：

| 提取器 | 几何观测量 | 索引形状 | 梯度计算 |
|---|---|---|---|
| **DistancePotential** | ‖r_i - r_j‖（成对距离） | `[2, N_pairs]` | 沿键方向的单位向量 |
| **DihedralPotential** | 有符号二面角 φ(i,j,k,l) | `[4, N_dihedrals]` | 通过法向量的叉积链式法则 |
| **AbsDihedralPotential** | |φ|（绝对二面角） | `[4, N_dihedrals]` | 经过符号校正的 DihedralPotential 梯度 |
| **ReferencePotential** | ‖x_i - aligned_ref_i‖（类 RMSD） | `[N_atoms]` | 加权刚体对齐 + 单位位移 |

`DistancePotential` 计算成对欧几里得距离，并返回单位方向向量作为梯度。`DihedralPotential` 通过键向量的叉积（`n_ijk × n_jkl`）计算有符号扭转角，其四原子梯度源自标准的解析扭转梯度公式。`ReferencePotential` 在计算逐原子位移范数之前，通过 `weighted_rigid_align` 将参考坐标对齐到当前坐标。

来源：[potentials.py](src/boltz/model/potentials/potentials.py#L282-L363), [potentials.py](src/boltz/model/potentials/potentials.py#L366-L383)

## 具体势类型

每个具体势将一个变量提取器与 `FlatBottomPotential` 组合，并实现 `compute_args` 以从特征化输入中提取特定于约束的索引和边界。下表总结了所有九种势：

| 势 | 变量类型 | 约束域 | 关键参数 | 引导间隔 |
|---|---|---|---|---|
| **PoseBustersPotential** | 距离 | RDKit 键/角度/冲突边界 | `bond_buffer`, `angle_buffer`, `clash_buffer` | 1 |
| **ConnectionsPotential** | 距离 | 链间连接键 | `buffer` | 1 |
| **VDWOverlapPotential** | 距离 | 范德华重叠防止 | `buffer` (0.225) | 5 |
| **SymmetricChainCOMPotential** | 距离 (COM) | 链质心分离 | `buffer` (指数调度) | 4 |
| **ChiralAtomPotential** | 二面角 | 手性中心手征性 | `buffer` (0.5236 ≈ π/6) | 1 |
| **StereoBondPotential** | 绝对二面角 | 双键立体化学 | `buffer` (0.5236 ≈ π/6) | 1 |
| **PlanarBondPotential** | 绝对二面角 | 双键平面性 | `buffer` (0.2618 ≈ π/12) | 1 |
| **ContactPotentital** | 距离 | 你指定的接触约束 | `union_lambda` (指数调度) | 4 |
| **TemplateReferencePotential** | 参考 | 模板结构对齐 | 来自 `template_force_threshold` | 2 |

### 物理化学势（Boltz-1 和 Boltz-2）

**PoseBustersPotential** 强制执行由 RDKit 计算的键、角度和非键合对的距离边界。它从特征中读取 `rdkit_bounds_index`、`rdkit_lower_bounds` 和 `rdkit_upper_bounds`，然后应用缓冲区缩放：键获得 ±12.5% 的容差，角度获得 ±12.5%，冲突获得 10% 的向内缓冲区且无上限。非键合对强制执行 `0.35 + mean(vdw_radii)` 的范德华下限，以防止原子重叠。

**VDWOverlapPotential** 计算所有唯一的链间原子对（排除已连接的链和单原子离子），并惩罚低于按 `(1 - buffer)` 缩放的范德华半径之和的距离。这防止了非共价连接的链之间发生空间冲突。

**SymmetricChainCOMPotential** 的独特之处在于使用质心聚合。它从 `symmetric_chain_index` 识别对称链对，计算每条链的质心，并强制执行最小质心距离。缓冲区参数遵循从 1.0 Å 到 5.0 Å（α = -2.0）的 `ExponentialInterpolation` 调度，这意味着约束随着去噪的进行而收紧。

**ConnectionsPotential** 强制连接的原子（来自 `connected_atom_index`）保持在彼此的 `buffer` Å 范围内，以维持跨共价键和肽键的链连接性。

来源：[potentials.py](src/boltz/model/potentials/potentials.py#L386-L530), [potentials.py](src/boltz/model/potentials/potentials.py#L532-L597)

### 立体化学势

**ChiralAtomPotential** 约束手性中心的不当二面角以维持正确的 R/S 构型。对于正手性原子，二面角必须超过 `buffer`（0.5236 弧度 ≈ 30°）；对于负手性原子，它必须低于 `-buffer`。这确保了手性碳周围的四面体几何得以保留。

**StereoBondPotential** 使用绝对二面角强制双键处的 E/Z 立体化学。对于顺式 (Z) 取向，二面角必须低于 `buffer`；对于反式 (E)，它必须超过 `π - buffer`。使用 `AbsDihedralPotential`（而非有符号的）反映了平面约束的对称性。

**PlanarBondPotential** 约束双键周围的不当二面角以维持平面性。它为每个双键构造两个不当二面角索引（使用 `[1,2,3,0]` 和 `[4,5,0,3]` 排列模式），并强制每个二面角保持在 `buffer`（0.2618 弧度 ≈ 15°）以下。

来源：[potentials.py](src/boltz/model/potentials/potentials.py#L532-L597)

### 接触与模板势（仅限 Boltz-2）

**ContactPotentital** 强制执行你指定的接触约束，这些约束从 `contact_pair_index`、`contact_thresholds`、`contact_negation_mask` 和 `contact_union_index` 中读取。取反掩码反转了排斥接触的边界。联合索引实现了一种 **软联合** 机制：与其同时要求所有接触，不如对 `-union_lambda * energy` 的指数进行 softmax，在每个联合组内选择最满足的约束。`union_lambda` 参数遵循从 8.0 到 0.0（α = -2.0）的 `ExponentialInterpolation` 调度，随着去噪的进行，从硬选择过渡到软平均。

**TemplateReferencePotential** 通过 `weighted_rigid_align` 将预测的 Cβ 坐标对齐到模板结构，并惩罚超过 `template_force_threshold` 的偏差。只有具有有效模板条目的残基（由 `template_mask_cb` 指示）受到约束，并且仅当存在 `template_force` 特征时该势才激活。

来源：[potentials.py](src/boltz/model/potentials/potentials.py#L599-L667)

## 调度系统

参数可以通过 `ParameterSchedule` 抽象在去噪过程中变化。每个调度将归一化时间 `t ∈ [0, 1]`（其中 1 = 去噪开始，0 = 去噪结束）映射到参数值。

| 调度类型 | 公式 | 用例 |
|---|---|---|
| **ExponentialInterpolation** | `start + (end - start) · (e^(αt) - 1) / (e^α - 1)` | 缓冲区和权重的平滑上升或衰减 |
| **PiecewiseStepFunction** | 跨阈值的阶跃函数 | 特定去噪阶段的离散权重转换 |

具有负 α 的 `ExponentialInterpolation` 产生一条凸曲线，在开始时变化缓慢，而在接近结束时变化迅速 —— 这用于 `SymmetricChainCOMPotential` 的缓冲区（1.0→5.0, α=-2.0）和 `ContactPotentital` 的 `union_lambda`（8.0→0.0, α=-2.0），两者都在结构接近收敛时收紧约束。`PiecewiseStepFunction` 用于 `VDWOverlapPotential` 的引导权重（t=0.4 之前为 0.125，之后为 0.0）和 `ContactPotentital` 的引导权重（在阈值 0.25 和 0.75 处从 0.0→0.5→1.0 递增）。

来源：[schedules.py](src/boltz/model/potentials/schedules.py#L1-L38), [potentials.py](src/boltz/model/potentials/potentials.py#L202-L211)

## FK 粒子重采样

Fleming-Katilman 引导通过为每个输入维护一组 `num_particles` 个独立去噪轨迹来运作。在重采样间隔（每 `fk_resampling_interval` 步）处，使用其 `resampling_weight` 参数将每个粒子去噪预测的能量评估为所有激活势的加权和。然后以与 `softmax(ll_difference + fk_lambda · log_G)` 成正比的概率重采样粒子，其中：

- **log_G** 捕获累积能量改善：在第 0 步，`log_G = -E`；在后续步骤中，`log_G = E_{prev} - E_{curr}`，衡量自上次重采样以来能量下降了多少。
- **ll_difference** 考虑了由物理引导更新引入的似然偏移（当两种机制均激活时）：`ll_diff = (ε² - (ε + Δguided)²) / (2σ²)`，其中 ε 是此步添加的噪声，Δguided 是累积的引导位移。
- **fk_lambda**（默认 4.0）控制选择压力 —— 较高的值更积极地偏好低能量粒子。

重采样后，所有逐粒子状态（原子坐标、去噪预测、能量轨迹、引导更新、token 表示）都根据重采样索引进行复制。在最后一步，每个粒子组仅保留一个样本。

<CgxTip>`num_particles` 参数会乘以基础 `multiplicity`，因此带有 `num_particles=3` 和 `multiplicity=2` 的 FK 引导将运行 6 条并行轨迹。内存消耗与该乘积呈线性关系。</CgxTip>

来源：[diffusion.py](src/boltz/model/modules/diffusion.py#L460-L678), [main.py](src/boltz/main.py#L148-L158)

## 物理引导更新

物理引导直接对去噪坐标预测 x₀ 应用梯度下降。对于 `num_gd_steps` 次迭代，它累积所有 `guidance_weight > 0` 且步索引可被 `guidance_interval` 整除的势的能量梯度：

```
guidance_update -= Σ (guidance_weight_i · ∇E_i(x₀ + guidance_update))
```

`guidance_interval` 参数充当计算节流阀 —— 具有较高间隔的势在 GD 循环中评估频率较低，从而降低了计算成本高但变化缓慢的约束的开销。在迭代 GD 循环之后，去噪预测被校正：`x₀_corrected = x₀ + guidance_update`。然后该校正被转换为后续 Euler 步的噪声空间位移：

```
scaled_guidance_update = -step_scale · (σ_t - t̂) / t̂ · guidance_update
```

该缩放更新跨步跟踪（并与随机增强一起旋转），以实现 FK 重采样中的 `ll_difference` 校正。`ll_difference` 项确保 FK 重采样不会重复计算引导的效果：接收到较大引导校正的粒子已经被“推”向低能量区域，因此它们表面的能量优势被似然惩罚部分抵消。

<CgxTip>`physical_guidance_update` 标志控制梯度是否实际修改坐标。当仅启用 `fk_steering`（没有物理引导）时，所有 `guidance_weight` 值都设置为 0.0，这意味着计算能量仅用于重采样，而不应用梯度校正。这产生了一种没有直接坐标操作的纯选择机制。</CgxTip>

来源：[diffusion.py](src/boltz/model/modules/diffusion.py#L606-L637), [potentials.py](src/boltz/model/potentials/potentials.py#L670-L754)

## 默认势配置

`get_potentials` 工厂使用经过精心调整的默认参数构建势集合。下表显示了当 FK 引导和物理引导均激活时的完整默认配置：

| 势 | 引导权重 | 引导间隔 | 重采样权重 | 缓冲区 / 关键参数 |
|---|---|---|---|---|
| SymmetricChainCOMPotential | 0.5 | 4 | 0.5 | ExpInterp(1.0→5.0, α=-2.0) |
| VDWOverlapPotential | Piecewise([0.4]→[0.125, 0.0]) | 5 | Piecewise([0.6]→[0.01, 0.0]) | 0.225 |
| ConnectionsPotential | 0.15 | 1 | 1.0 | 2.0 |
| PoseBustersPotential | 0.01 | 1 | 0.1 | bond=0.125, angle=0.125, clash=0.10 |
| ChiralAtomPotential | 0.1 | 1 | 1.0 | 0.5236 (π/6) |
| StereoBondPotential | 0.05 | 1 | 1.0 | 0.5236 (π/6) |
| PlanarBondPotential | 0.05 | 1 | 1.0 | 0.2618 (π/12) |
| **仅限 Boltz-2：** | | | | |
| ContactPotentital | Piecewise([0.25,0.75]→[0.0,0.5,1.0]) | 4 | 1.0 | union_lambda: ExpInterp(8.0→0.0, α=-2.0) |
| TemplateReferencePotential | 0.1 | 2 | 1.0 | 来自 template_force_threshold |

权重层级揭示了设计哲学：**连接性和手性**被最强力地执行（高引导权重、高重采样权重、频繁评估），而 **RDKit 边界**被温和地应用（低权重 0.01），因为神经网络已经在很大程度上遵循了它们。**范德华重叠**主要通过重采样而非梯度校正来处理，因为其全成对计算成本高昂，且在早期去噪阶段其梯度势面充满噪声。

来源：[potentials.py](src/boltz/model/potentials/potentials.py#L670-L787)

## 引导参数参考

`BoltzSteeringParams` 数据类暴露了所有你可配置的引导选项：

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `fk_steering` | bool | `False` | 启用 FK 粒子重采样 |
| `num_particles` | int | `3` | 每个样本的并行粒子数 |
| `fk_lambda` | float | `4.0` | FK 重采样 softmax 中的选择压力 |
| `fk_resampling_interval` | int | `3` | 每 N 个扩散步重采样一次 |
| `physical_guidance_update` | bool | `False` | 启用基于梯度的坐标校正 |
| `contact_guidance_update` | bool | `True` | 启用接触/模板势 (Boltz-2) |
| `num_gd_steps` | int | `20` | 每步的梯度下降迭代次数 |

当 `fk_steering=False` 且 `physical_guidance_update=False` 时，不会实例化任何势，采样作为标准的无条件去噪进行。当 `fk_steering=True` 但 `physical_guidance_update=False` 时，所有 `guidance_weight` 值都被归零 —— 势仍通过其 `resampling_weight` 贡献于重采样能量，但不发生梯度校正。`contact_guidance_update` 标志特定于 Boltz-2，它激活 `ContactPotentital` 和 `TemplateReferencePotential`，它们分别需要接触对和模板特征。

来源：[main.py](src/boltz/main.py#L147-L158), [potentials.py](src/boltz/model/potentials/potentials.py#L670-L787)

## 与扩散采样器的交互

引导系统在 `AtomDiffusion.sample()` 内的三个关键点挂载到去噪循环：

**1. 去噪前增强** —— 对所有粒子坐标应用随机旋转/平移。当物理引导激活时，累积的 `scaled_guidance_update` 会共同旋转，以保持全局坐标系的一致性。

**2. 去噪后评估** —— 神经网络生成 `atom_coords_denoised`（即 x₀ 预测）后，同时执行能量计算（用于 FK）和梯度计算（用于物理引导）。FK 重采样和 GD 校正在 Euler 步之前应用于去噪预测。

**3. 带有校正预测的 Euler 步** —— 标准更新 `x_{t+1} = x_noisy + step_scale · (σ_t - t̂) · (x_noisy - x₀)/t̂` 使用了经引导校正的 x₀，因此校正自然地通过扩散 ODE 传播。

`steering_t` 变量（计算为 `1.0 - step_idx / num_sampling_steps`）驱动所有调度评估，确保随时间变化的参数（如缓冲区和 union_lambda）从宽松（早期、高噪声）平滑过渡到严格（晚期、低噪声）。

来源：[diffusion.py](src/boltz/model/modules/diffusion.py#L449-L712), [diffusionv2.py](src/boltz/model/modules/diffusionv2.py#L1-L35)

## 设计权衡与实践考量

双机制架构反映了 **探索** 与 **利用** 之间的基本权衡。FK 重采样是纯选择性的 —— 它从不创造新构象，只放大现有构象 —— 这使其安全但受限于粒子集合的多样性。物理引导是建设性的 —— 它可以将坐标推入去噪器从未探索过的区域 —— 但如果梯度势面条件不良，则存在引入伪影的风险。FK 重采样中的 `ll_difference` 项通过抵消从引导中过度受益的粒子来部分缓解此问题。

`guidance_interval` 节流机制利用了不同约束在去噪期间以不同速率变化的观察结果。快速变化的约束（连接性、手性）在每次 GD 迭代中评估，而缓慢变化的约束（范德华重叠、链质心）评估频率较低，在不牺牲准确性的情况下降低了范德华计算的 O(N²) 成对开销。

对于决定启用哪种机制的你：**仅 FK 引导** 以 `num_particles × base_memory` 的代价提供了低风险的改进；**仅物理引导** 提供了更强的校正，但在低步数时可能过度约束；**两者结合** 并带有 `ll_difference` 校正提供了最佳平衡，正如 Boltz-2 默认配置所证明的那样（`contact_guidance_update=True`）。

来源：[diffusion.py](src/boltz/model/modules/diffusion.py#L560-L637), [main.py](src/boltz/main.py#L147-L158)

如需更深入地探索引导所修改的扩散采样循环，请参阅[基于扩散的结构模块](9-diffusion-based-structure-module)。如需了解输入到 Boltz-2 引导势的接触和模板特征，请参阅[模板与接触条件化](19-template-and-contact-conditioning)。如需了解训练期间使用的损失函数（不涉及引导），请参阅[损失函数与验证](20-loss-functions-and-validation)。