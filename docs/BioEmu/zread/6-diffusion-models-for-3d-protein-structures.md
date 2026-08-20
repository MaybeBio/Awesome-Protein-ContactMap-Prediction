---
slug:6-diffusion-models-for-3d-protein-structures
blog_type:normal
---


扩散模型已在多个领域彻底改变了生成建模技术，而其在3D蛋白质结构中的应用代表了计算生物学领域的重大进步。在bioemu中，扩散模型通过学习精心设计的噪声添加过程的逆过程来生成逼真的蛋白质结构。本文将探讨bioemu如何实现专门针对蛋白质结构生成独特挑战的扩散模型。

## 理解蛋白质扩散模型

扩散模型的核心原理是通过逐渐向数据添加噪声（前向过程），然后学习逆转这一过程（逆向过程）来生成新样本。对于蛋白质结构而言，这带来了独特挑战，因为蛋白质存在于**SO(3)旋转空间**而非简单的欧几里得空间。

Bioemu通过精密的双组件方法处理这种复杂性：

1. **位置扩散**：在欧几里得空间中对原子三维坐标建模
2. **方向扩散**：在SO(3)空间中对氨基酸残基的旋转框架建模

这种双重方法使模型能够同时捕获原子的空间排列以及对蛋白质稳定性和功能至关重要的局部几何约束。

来源：[sde_lib.py#L51-L73](src/bioemu/sde_lib.py#L51-L73), [so3_sde.py#L20-L52](src/bioemu/so3_sde.py#L20-L52)

## 蛋白质扩散的随机微分方程（SDE）

Bioemu使用随机微分方程（SDE）实现扩散过程，SDE为扩散过程提供了连续时间公式化表达。该框架包含多个专门的SDE类：

### 基础SDE框架

`SDE`基类定义了所有扩散过程的接口：

```python
class SDE:
    @abc.abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """返回漂移项f和扩散系数g，满足dx = f * dt + g * sqrt(dt) * 标准高斯分布"""
        pass
```

这个抽象方法定义了扩散过程的核心，其中`漂移项`代表确定性分量，`扩散项`代表SDE的随机分量。

来源：[sde_lib.py#L51-L59](src/bioemu/sde_lib.py#L51-L59)

### 用于旋转的SO(3)扩散

蛋白质结构需要对旋转分量进行特殊处理。Bioemu实现了`SO3SDE`，一个专门用于3D空间旋转的SDE：

```python
class SO3SDE(SDE, torch.nn.Module):
    def __init__(self, eps_t: float = 1e-4, num_sigma: int = 1000, 
                 num_omega: int = 1000, omega_exponent: int = 3, 
                 l_max: int = 1000, tol: float = 1e-6, ...):
```

这里的关键创新是使用**IGSO(3)分布**（SO(3)上的不变高斯分布），它为3D旋转添加噪声提供了数学上合理的方法。IGSO(3)分布基于级数展开：

```math
f(ω) = ∑_{l=0}^∞ (2l+1) exp(-l(l+1)σ²/2) * sin((l+½)ω) / sin(ω/2)
```

这使模型能够采样符合3D旋转群几何特性的旋转。

来源：[so3_sde.py#L20-L49](src/bioemu/so3_sde.py#L20-L49), [so3_sde.py#L44-L46](src/bioemu/so3_sde.py#L44-L46)

### 用于位置的方差保持SDE

对于原子位置，bioemu使用方差保持SDE（VPSDE），在整个扩散过程中保持稳定的噪声水平：

```python
class CosineVPSDE(BaseVPSDE):
    """具有余弦噪声调度表的方差保持SDE"""
    
    def __init__(self, s: float = 0.008):
        self.s = s
        self.c = np.cos(s / (1 + s) * np.pi / 2)
```

余弦调度表提供了平滑的噪声过渡，非常适合蛋白质结构生成。

来源：[sde_lib.py#L156-L162](src/bioemu/sde_lib.py#L156-L162)

## 逆向过程：蛋白质结构去噪

扩散模型的真正威力在于其逆转噪声添加过程的能力。Bioemu实现了精密的去噪算法，可以从随机噪声生成蛋白质结构。

### 欧拉-丸山预测器

逆向过程的核心在`EulerMaruyamaPredictor`类中实现：

```python
class EulerMaruyamaPredictor:
    def __init__(self, *, corruption: SDE, noise_weight: float = 1.0, 
                 marginal_concentration_factor: float = 1.0):
```

该预测器处理标准欧几里得更新和特殊的SO(3)更新：

```python
def update_given_drift_and_diffusion(self, *, x: torch.Tensor, dt: torch.Tensor, 
                                   drift: torch.Tensor, diffusion: torch.Tensor):
    # 使用SO(3)上SDE的特殊更新或标准更新到下一步
    if isinstance(self.corruption, SO3SDE):
        mean = apply_rotvec_to_rotmat(x, drift * dt, tol=self.corruption.tol)
        sample = apply_rotvec_to_rotmat(mean, self.noise_weight * diffusion * 
                                       torch.sqrt(dt.abs()) * z, tol=self.corruption.tol)
    else:
        mean = x + drift * dt
        sample = mean + self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z
    return sample, mean
```

关键区别在于SO(3)更新使用旋转组合而非简单加法，尊重旋转群的几何特性。

来源：[denoiser.py#L16-L31](src/bioemu/denoiser.py#L16-L31), [denoiser.py#L59-70](src/bioemu/denoiser.py#L59-70)

### 高级去噪算法

Bioemu为不同用例提供多种去噪算法：

1. **Heun方法**：二阶求解器，提供更精确的采样
2. **DPM求解器**：专门用于VPSDE的求解器，具有改进的稳定性

`heun_denoiser`函数实现了具有自适应噪声水平和二阶修正的精密采样方案：

```python
def heun_denoiser(*, sdes: dict[str, SDE], N: int, eps_t: float, max_t: float, 
                 device: torch.device, batch: Batch, score_model: torch.nn.Module, 
                 noise: float) -> ChemGraph:
```

该函数协调整个去噪过程，管理位置扩散和方向扩散之间的相互作用。

来源：[denoiser.py#L143-154](src/bioemu/denoiser.py#L143-154)

## 评分模型：学习逆转扩散

扩散模型中的"评分"指的是数据分布对数概率的梯度。Bioemu的评分模型学习预测该评分，该评分指导逆向扩散过程。

### 架构概述

主要评分模型架构围绕`DistributionalGraphormer`构建：

```python
class DistributionalGraphormer(nn.Module):
    def __init__(self, dim_model: int = 512, dim_pair: int = 256, 
                 num_layers: int = 8, num_heads: int = 32, ...):
```

该架构设计为对旋转和平移**等变**，意味着旋转或平移输入蛋白质结构会导致相应旋转或变换的输出。这一特性对于学习蛋白质结构的有意义表示至关重要。

来源：[models.py#L148-185](src/bioemu/models.py#L148-185)

### 时间嵌入

评分模型的一个关键组件是时间嵌入，它对扩散时间步进行编码：

```python
class SinusoidalPositionEmbedder(nn.Module):
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # 重新缩放输入标量，使其在[0, 1000.]范围内
        time = (time - self.min_input) * 1000.0 / (self.max_input - self.min_input)
        embeddings = torch.exp(torch.arange(self.half_dim, device=device) * self.embedding_factor)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

这种正弦嵌入使模型能够根据扩散过程中的当前噪声水平调整其行为。

来源：[models.py#L19-69](src/bioemu/models.py#L19-69)

## 整合所有组件：采样流程

Bioemu中完整的蛋白质结构生成过程遵循以下步骤：

### 1. 序列到嵌入

首先，使用ColabFold将氨基酸序列转换为数值嵌入：

```python
def get_context_chemgraph(sequence: str, cache_embeds_dir: str | Path | None = None, 
                         msa_file: str | Path | None = None, 
                         msa_host_url: str | None = None) -> ChemGraph:
    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence, cache_embeds_dir=cache_embeds_dir, 
        msa_file=msa_file, msa_host_url=msa_host_url)
```

这些嵌入捕获蛋白质序列的进化信息和结构约束。

来源：[sample.py#L176-189](src/bioemu/sample.py#L176-189)

### 2. 扩散采样

核心采样过程从随机噪声生成蛋白质结构：

```python
def generate_batch(score_model: torch.nn.Module, sequence: str, 
                  sdes: dict[str, SDE], batch_size: int, seed: int, 
                  denoiser: Callable, ...) -> dict[str, torch.Tensor]:
    context_chemgraph = get_context_chemgraph(...)
    context_batch = Batch.from_data_list([context_chemgraph] * batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sampled_chemgraph_batch = denoiser(
        sdes=sdes, device=device, batch=context_batch, score_model=score_model)
```

该函数协调整个生成过程，从序列嵌入到最终3D坐标。

来源：[sample.py#L219-259](src/bioemu/sample.py#L219-259)

### 3. 后处理和输出

最后，生成的结构转换为标准格式（PDB和XTC）进行分析和可视化：

```python
save_pdb_and_xtc(pos_nm=positions, node_orientations=node_orientations,
                 topology_path=output_dir / "topology.pdb",
                 xtc_path=output_dir / "samples.xtc",
                 sequence=sequence, filter_samples=filter_samples)
```

<CgxTip>
生成的蛋白质结构可以被过滤，以去除具有长键距离或空间冲突的非物理样本，确保只保留生物学上合理的结构。
</CgxTip>

来源：[sample.py#L165-172](src/bioemu/sample.py#L165-172)

## 关键创新和优势

Bioemu的扩散模型实现包含几项关键创新：

### 1. 几何感知

通过将蛋白质结构视为SE(3)（3D空间中刚性变换群）的元素，bioemu尊重分子结构的基本几何特性。这种几何感知导致生成更逼真且物理上合理的结构。

### 2. 双重扩散过程

位置和方向的分别处理使模型能够在适当的粒度级别捕获蛋白质结构的不同方面。位置捕获整体折叠，而方向捕获局部几何约束。

### 3. 高效采样

实现包含多种采样算法（Heun、DPM），在准确性和计算效率之间取得平衡，使生成多样化蛋白质结构变得实用。

### 4. 与进化信息整合

通过利用ColabFold嵌入，模型整合了自然蛋白质中观察到的进化约束和结构模式，导致生成生物学上更相关的结构。

## 实际应用

Bioemu中的扩散模型实现了几项重要应用：

1. **蛋白质设计**：生成具有所需特性的新颖蛋白质结构
2. **结构预测**：从给定序列的可能结构分布中采样
3. **构象采样**：探索柔性蛋白质的构象景观
4. **药物发现**：生成用于虚拟筛选和结合位点分析的蛋白质结构

## 结论

扩散模型代表了3D蛋白质结构生成的强大方法，而bioemu的实现提供了一个尊重分子结构几何约束的精密框架。通过结合微分几何的高级数学概念与深度学习技术，bioemu能够生成多样化、物理上合理的蛋白质结构，加速结构生物学和药物发现研究。

成功的关键在于数学严谨性与实际实现之间的谨慎平衡，确保生成的结构既在理论上合理又在生物学上相关。随着该领域的不断发展，我们可以期待更复杂的扩散模型，进一步缩小计算预测与实验现实之间的差距。