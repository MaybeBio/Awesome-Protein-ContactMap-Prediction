---
slug:10-training-and-customizing-models
blog_type:normal
---


在BioEmu中训练和定制蛋白质结构模型需要理解该框架的架构、损失函数和配置选项。本指南将带您了解核心组件，并提供训练自定义模型的实用示例。

## 理解训练框架

BioEmu实现了基于3D蛋白质结构扩散模型的复杂训练流程。训练过程的核心是最小化一个专门评估蛋白质折叠度的损失函数——该指标衡量蛋白质结构与其天然构象的匹配程度。

核心训练基础设施围绕两个主要组件构建：

1. **损失计算**（`src/bioemu/training/loss.py`）：实现指导模型训练的PPFT（蛋白质折叠分数目标）损失函数
2. **折叠度评估**（`src/bioemu/training/foldedness.py`）：计算预测结构与参考接触的匹配程度

来源：[loss.py#L18-L64](src/bioemu/training/loss.py#L18-L64), [foldedness.py#L24-L40](src/bioemu/training/foldedness.py#L24-L40)

## PPFT损失函数

PPFT损失是BioEmu训练的基石。它通过比较蛋白质结构的折叠度与目标值来优化结构。其工作原理如下：

```python
def calc_ppft_loss(
    *,
    score_model: torch.nn.Module,
    sdes: dict[str, SDE],
    batch: list[ChemGraph],
    n_replications: int,
    mid_t: float,
    N_rollout: int,
    record_grad_steps: set[int],
    reference_info_lookup: dict[str, ReferenceInfo],
    target_info_lookup: dict[str, TargetInfo],
) -> torch.Tensor:
```

该损失函数执行以下关键步骤：
1. 通过快速展开过程生成多个结构样本
2. 通过与参考接触对比计算每个样本的折叠度
3. 估算平均折叠度与目标折叠度之间的平方误差

<CgxTip>
PPFT损失采用巧妙的数学技巧计算平方误差的无偏估计，无需显式计算均值，从而提高训练效率。
</CgxTip>

来源：[loss.py#L18-L64](src/bioemu/training/loss.py#L18-L64), [loss.py#L130-L167](src/bioemu/training/loss.py#L130-L167)

## 模型架构

BioEmu采用名为`DistributionalGraphormer`的复杂神经网络架构预测蛋白质结构。该模型设计为对旋转和平移具有等变性——这是3D结构预测的关键特性。

主要模型组件包括：

1. **SinusoidalPositionEmbedder**：将扩散时间步编码为嵌入向量
2. **RelativePositionBias**：捕获残基间的序列距离信息
3. **StructureModule**：使用注意力机制处理3D结构信息

```python
class DistributionalGraphormer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        dim_pair: int = 256,
        num_layers: int = 8,
        num_heads: int = 32,
        dim_single_rep: int = 64,
        dim_hidden: int = 1024,
        num_buckets: int = 64,
        max_distance_relative: int = 128,
        dropout: float = 0.1,
    ):
```

该模型接收含噪蛋白质结构并预测"分数"——本质上是调整每个原子位置以趋向更接近天然构象的方向和幅度。

来源：[models.py#L148-L185](src/bioemu/models.py#L148-L185), [models.py#L326-L346](src/bioemu/models.py#L326-L346)

## 训练配置

训练通过YAML配置文件控制，这些文件指定去噪过程的参数。框架支持不同的采样策略：

```yaml
# DPM求解器配置
_target_: bioemu.shortcuts.dpm_solver
_partial_: true
eps_t: 0.001
max_t: 0.99
N: 50
noise: 0.0
```

关键参数包括：
- `eps_t`：最小噪声水平（去噪起点）
- `max_t`：最大噪声水平（去噪终点）
- `N`：去噪步数
- `noise`：采样过程中添加的额外噪声

来源：[dpm.yaml](src/bioemu/config/denoiser/dpm.yaml)

## 自定义模型训练

要训练自定义模型，您需要：

1. **准备数据**：将蛋白质结构转换为带序列嵌入的ChemGraph格式
2. **设置参考信息**：定义接触图和目标折叠度值
3. **配置训练参数**：调整损失函数参数和模型超参数

以下是设置训练的简化示例：

```python
from bioemu.training.loss import calc_ppft_loss
from bioemu.models import DiGConditionalScoreModel
from bioemu.training.foldedness import ReferenceInfo, TargetInfo

# 初始化模型
model = DiGConditionalScoreModel(
    dim_model=512,
    num_layers=8,
    num_heads=32
)

# 设置参考和目标信息
reference_info = ReferenceInfo(
    contact_indices=contact_indices,
    contact_distances_angstrom=contact_distances,
    sequence=protein_sequence
)

target_info = TargetInfo(
    p_fold_thr=0.7,  # 折叠度=0.5时的FNC阈值
    steepness=10.0,   # sigmoid函数的陡峭度
    p_fold_target=0.8 # 目标折叠度
)

# 计算损失
loss = calc_ppft_loss(
    score_model=model,
    sdes=sde_dict,
    batch=protein_batch,
    n_replications=4,
    mid_t=0.5,
    N_rollout=20,
    record_grad_steps={10, 15},
    reference_info_lookup={"protein1": reference_info},
    target_info_lookup={"protein1": target_info}
)
```

来源：[loss.py#L18-L29](src/bioemu/training/loss.py#L18-L29), [foldedness.py#L24-L40](src/bioemu/training/foldedness.py#L24-L40)

## 高级定制

进行更高级的定制时，您可以修改：

1. **损失函数组件**：调整折叠度计算方式或接触评分方式
2. **模型架构**：更改层数、注意力头数或嵌入维度
3. **训练计划**：修改噪声计划或展开过程

折叠度计算使用sigmoid函数将天然接触比例（FNC）转换为折叠度分数：

```python
def foldedness_from_fnc(fnc: torch.Tensor, p_fold_thr: float, steepness: float) -> torch.Tensor:
    return torch.sigmoid(2 * steepness * (fnc - p_fold_thr))
```

您可以调整`p_fold_thr`和`steepness`参数来控制模型评估蛋白质结构的严格程度。

来源：[foldedness.py#L58-L70](src/bioemu/training/foldedness.py#L58-L70), [foldedness.py#L143-L167](src/bioemu/training/foldedness.py#L143-L167)

## 训练最佳实践

训练自定义模型时，请牢记以下最佳实践：

1. **从预训练模型开始**：对现有模型进行微调而非从头训练
2. **监控折叠度指标**：训练期间同时跟踪FNC和折叠度
3. **使用适当的批次大小**：平衡内存使用与训练稳定性
4. **调整学习率**：蛋白质结构训练通常需要仔细的学习率调度
5. **在多样化结构上验证**：确保模型在不同蛋白质折叠类型上具有泛化能力

该训练框架在保持蛋白质结构预测所需数学严谨性的同时，兼顾了灵活性。通过理解核心组件并遵循这些指南，您可以有效地训练和定制模型，以满足特定的蛋白质结构预测需求。