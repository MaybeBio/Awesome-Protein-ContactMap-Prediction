---
slug:16-confidence-metrics-plddt-pae
blog_type:normal
---


AlphaFold 提供两种主要置信度指标来量化预测蛋白质结构的可靠性：预测局部距离差异检验（pLDDT）和预测比对误差（PAE）。这些指标对于解释预测质量并就下游应用做出明智决策至关重要。

## pLDDT：残基级置信度

**预测 LDDT（pLDDT）**提供 0-100 范围的残基级置信度分数，用于衡量局部原子位置的预期准确性。与传统 LDDT 需要参考结构不同，pLDDT 直接从模型的内部表示中预测得出。

### 架构与计算

pLDDT 预测使用 `PredictedLDDTHead` 模块 [modules.py#L1079-L1131]，该模块通过两层神经网络处理结构模块的单表示：

```python
# 输入：structure_module 表示 [N_res, c_s]
act = representations['structure_module']
# 层归一化和两个 ReLU 层
act = LayerNorm()(act)
act = Linear(num_channels)(act)
act = relu(act)
act = Linear(num_channels)(act) 
act = relu(act)
# 输出：logits [N_res, num_bins]
logits = Linear(num_bins)(act)
```

最终的 pLDDT 分数通过对 logits 应用 softmax 并计算跨区间的期望值计算得出 [confidence.py#L24-L38]：

```python
def compute_plddt(logits: np.ndarray) -> np.ndarray:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
    probs = scipy.special.softmax(logits, axis=-1)
    predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100
```

### 置信度分类

pLDDT 分数分为四个置信度等级 [confidence.py#L41-L52]：

| 分数范围 | 类别 | 描述 |
|-------------|----------|-------------|
| 90-100 | 高 (H) | 非常高置信度，主链 RMSD 通常 < 1Å |
| 70-90 | 中 (M) | 良好置信度，主链 RMSD 通常 1-2.5Å |
| 50-70 | 低 (L) | 低置信度，主链 RMSD 通常 2.5-4Å |
| 0-50 | 无序 (D) | 非常低置信度，通常为内在无序区域 |

<CgxTip>
置信度分类遵循 AlphaFold 蛋白质结构数据库使用的相同方案，可与公开可用的预测结果直接比较。
</CgxTip>

## PAE：残基间比对置信度

**预测比对误差（PAE）**矩阵提供所有残基对之间的成对置信度分数，衡量最优比对后的预期位置误差。该指标对于评估结构域排列和多聚体界面特别有价值。

### PAE 计算流程

PAE 预测使用 `PredictedAlignedErrorHead` 模块 [modules.py#L1181-L1222]，该模块作用于配对表示：

```python
# 输入：配对表示 [N_res, N_res, c_z]
act = representations['pair']
# 直接线性投影到误差区间
logits = Linear(num_bins)(act)
# 生成从 0 到 max_error_bin 的区间边界
breaks = jnp.linspace(0.0, self.config.max_error_bin, self.config.num_bins - 1)
```

PAE 矩阵通过将 logits 转换为概率并计算预期误差得出 [confidence.py#L121-L149]：

```python
def compute_predicted_aligned_error(logits, breaks):
    aligned_confidence_probs = scipy.special.softmax(logits, axis=-1)
    predicted_aligned_error, max_predicted_aligned_error = (
        _calculate_expected_aligned_error(breaks, aligned_confidence_probs)
    )
    return {
        'aligned_confidence_probs': aligned_confidence_probs,
        'predicted_aligned_error': predicted_aligned_error,
        'max_predicted_aligned_error': max_predicted_aligned_error,
    }
```

### TM 分数集成

PAE 数据支持预测 TM 分数的计算，提供全局质量评估。`predicted_tm_score` 函数 [confidence.py#L178-L239] 实现了标准 pTM 和界面 pTM（ipTM）计算：

- **pTM**：使用所有残基对的整体结构置信度
- **ipTM**：多聚体预测的界面置信度，仅考虑链间接触

TM 分数计算遵循标准公式 [confidence.py#L212-L221]：
```python
d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
tm_per_bin = 1.0 / (1 + np.square(bin_centers) / np.square(d0))
predicted_tm_term = np.sum(probs * tm_per_bin, axis=-1)
```

## 输出格式与集成

### JSON 序列化

两种置信度指标均支持标准化 JSON 输出以供下游应用使用：

- **pLDDT JSON**：带置信度类别的残基级分数 [confidence.py#L55-L75]
- **PAE JSON**：带最大误差值的完整误差矩阵 [confidence.py#L152-L175]

PAE JSON 格式符合 AlphaFold 数据库规范，可与现有可视化工具无缝集成。

### 模型训练集成

两个置信度头均与主模型进行端到端训练：

- **pLDDT 损失**：预测 LDDT 区间与真实 LDDT 区间的交叉熵 [modules.py#L1133-L1140]
- **PAE 损失**：预测比对误差分布与真实比对误差分布的交叉熵 [modules.py#L1224-L1240]

<CgxTip>
置信度指标不是后处理添加项，而是模型训练目标的组成部分，确保它们反映模型的真实预测不确定性。
</CgxTip>

## 实际应用

### 结构验证

- **高置信度区域**（pLDDT > 90）适用于详细的原子分析和药物设计
- **低置信度区域**（pLDDT < 50）通常表明内在无序性或建模挑战
- **PAE 分析**揭示多结构域蛋白质的结构域边界和潜在柔性

### 多聚体复合物

对于蛋白质复合物，置信度指标变得更为关键：

- **ipTM 分数**评估界面预测质量
- **PAE 热图**识别可靠的链间接触
- **链特异性 pLDDT**评估单个亚基质量

## 与预测工作流的集成

置信度指标在最终结构预测阶段计算，是完整 AlphaFold 输出的重要组成部分。它们为研究人员提供必要的质量评估，以适当解释预测结果并就实验验证策略做出明智决策。

要了解这些指标如何融入更广泛的预测流程，请参阅[结构预测工作流](15-structure-prediction-workflow)文档。生成这些置信度分数的模型架构在[模型架构概述](11-model-architecture-overview)中有详细说明。