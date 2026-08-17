---
slug:11-advanced-configuration-options
blog_type:normal
---


ColabFold 提供了广泛的配置选项，允许研究人员针对特定用例微调蛋白质结构预测过程。本指南解释了关键的高级参数及其对预测质量、速度和资源使用的影响。

## MSA 配置选项

多序列比对（MSA）生成是影响预测精度的关键组件。ColabFold 提供了多种生成和自定义 MSA 的方法。

### MSA 生成方法

```python
msa_mode = "mmseqs2_uniref_env"  # 选项: "mmseqs2_uniref_env", "mmseqs2_uniref", "single_sequence", "custom"
```

- **mmseqs2_uniref_env**：默认且推荐选项。搜索 UniRef30 和环境序列以实现最大多样性。
- **mmseqs2_uniref**：仅搜索 UniRef30，当您想排除环境序列时有用。
- **single_sequence**：完全绕过 MSA 生成，仅使用输入序列。这会导致预测速度更快，但可能准确性较低。
- **custom**：允许上传您自己的预生成 MSA 文件（支持 a3m、fasta 等格式）。

来源：[colabfold/colabfold.py](colabfold/colabfold.py)，[AlphaFold2.ipynb](AlphaFold2.ipynb)

### 序列配对选项

```python
pair_mode = "unpaired_paired"  # 选项: "unpaired_paired", "paired", "unpaired"
```

此设置控制复杂结构中不同链的序列如何配对：

- **unpaired_paired**：使用同一物种的配对序列和非配对 MSA。这是复杂预测的默认且最稳健选项。
- **paired**：仅使用跨链成功配对的序列。这可以提高具有强共进化信号的复杂结构的准确性。
- **unpaired**：为每条链生成单独的 MSA。当链不共进化或来自不同生物体时有用。

高级 MSA 过滤选项：

```python
pair_cov = 50  # 覆盖率阈值（0-100%）
pair_qid = 20  # 序列同一性阈值（0-100%）
```

这些参数在配对前对序列进行预过滤，这在处理包含旁系的复杂结构时很有帮助。

来源：[beta/AlphaFold2_advanced.ipynb#L230-L250](beta/AlphaFold2_advanced.ipynb)

### MSA 过滤

```python
cov = 0  # 覆盖率阈值（0, 25, 50, 75, 90, 95）
qid = 0  # 序列同一性阈值（0, 15, 20, 25, 30, 40, 50）
```

这些参数允许根据以下条件过滤 MSA：
- **cov**：与查询的最小覆盖率（%）
- **qid**：与查询的最小序列同一性（%）

增加这些值会过滤掉更远的序列，有助于聚焦于密切相关蛋白。

<CgxTip>
对于保守性较差的难靶点，将这些值保持在 0 以保留所有可用序列。对于具有高度保守蛋白的非常大 MSA，过滤（例如，cov=75, qid=30）可以减少计算需求而不牺牲质量。
</CgxTip>

来源：[beta/AlphaFold2_advanced.ipynb#L260-L284](beta/AlphaFold2_advanced.ipynb)

### 区域选择

```python
trim = ""  # 示例："5-9,20" 或 "A1-A3,B5-B7"
trim_inverse = False
```

- **trim**：指定要从模型中排除的区域，使用 1 索引位置。
- **trim_inverse**：启用时，保留指定区域而不是修剪。

此功能适用于聚焦特定域或去除无序区域。

来源：[beta/AlphaFold2_advanced.ipynb#L258-L265](beta/AlphaFold2_advanced.ipynb)

## 模型执行选项

这些选项控制神经网络模型的执行方式，影响预测质量和资源使用。

### 模型选择

```python
num_models = 5  # 选项：1-5
use_ptm = True  # 使用 AlphaFold-Multimer 参数
```

- **num_models**：尝试不同模型参数的数量（1-5）。使用所有 5 个模型推荐以获得最佳结果，因为不同模型可能在不同的结构类型上表现优异。
- **use_ptm**：在 AlphaFold 的原始参数和微调的“ptm”模型参数之间切换。ptm 模型优化了蛋白质复合物预测，并启用 PAE（预测对齐误差）计算。

来源：[beta/AlphaFold2_advanced.ipynb#L316-L321](beta/AlphaFold2_advanced.ipynb)

### 采样策略

```python
num_ensemble = 1  # 选项：1, 2, 4, 8
max_recycles = 3  # 选项：0, 1, 3, 6, 12, 24, 48
is_training = False
num_samples = 1  # 选项：1, 2, 4, 8, 16, 32
```

- **num_ensemble**：网络主干多次运行，每次使用不同的 MSA 聚类中心随机选择。较高值（例如 8）接近 CASP14 设置，但需要更多计算。
- **max_recycles**：控制结构反馈到网络中进行精化的次数。更多循环通常提高准确性，但增加计算时间。
- **is_training**：启用时，激活模型的随机部分（dropout），可以生成更多样化的结构。
- **num_samples**：尝试的随机种子数量。结合 `is_training=True`，可以生成多样化的预测集合。

来源：[beta/AlphaFold2_advanced.ipynb#L321-L330](beta/AlphaFold2_advanced.ipynb)

### 性能优化

```python
use_turbo = True
max_msa = "512:1024"  # 选项："512:1024", "256:512", "128:256", "64:128", "32:64"
subsample_msa = True
```

- **use_turbo**：引入优化（一次性编译，交换参数，调整 max_msa）以加速执行并减少内存使用。
- **max_msa**：定义要使用的 MSA 序列的最大数量，格式为 `max_msa_clusters:max_extra_msa`。降低这些值减少 GPU 内存需求，但可能影响模型质量。
- **subsample_msa**：自动对非常大的 MSA 进行子采样，以防止预处理期间出现内存问题。

<CgxTip>
对于 GPU 内存有限的复杂预测，尝试将 `max_msa` 降低到 "256:512" 或更低。从较高值开始，如果遇到内存错误，逐渐降低。权衡在于内存使用和预测准确性之间。
</CgxTip>

来源：[beta/AlphaFold2_advanced.ipynb#L290-L300](beta/AlphaFold2_advanced.ipynb)

## 模板选项

模板可以提供额外的结构信息以指导预测。

```python
template_mode = "none"  # 选项："none", "pdb100", "custom"
```

- **none**：不使用模板信息。预测仅基于序列和生成的 MSA。
- **pdb100**：自动在 PDB100 数据库中搜索模板。
- **custom**：允许上传您自己的 PDB 或 mmCIF 格式模板。

当可用结构与目标密切相关时，模板通常会提高预测质量。它们对于 MSA 深度有限的蛋白特别有帮助。

来源：[AlphaFold2.ipynb#L71-L73](AlphaFold2.ipynb)

## 结构精化选项

初始结构预测后，ColabFold 可以使用 Amber 力场对结构进行可选精化。

```python
num_relax = 0  # 选项：0, 1, 5（或 "None", "Top1", "Top5", "All"）
```

- **0/None**：跳过精化（最快）
- **1/Top1**：仅精化最高排名的模型
- **5/Top5**：精化五个最高排名的模型
- **All**：精化所有生成的模型

Amber 精化改善了侧链几何形状并解决了轻微的立体冲突，但很少改变整体主链结构。它大约使运行时间翻倍，但在需要精确侧链位置时推荐使用。

来源：[AlphaFold2.ipynb#L73-L75](AlphaFold2.ipynb)，[beta/AlphaFold2_advanced.ipynb#L400-L450](beta/AlphaFold2_advanced.ipynb)

## 排名和输出选项

这些选项控制预测的排名和生成的输出文件。

### 模型排名

```python
rank_by = "pLDDT"  # 选项："pLDDT", "pTMscore"
```

- **pLDDT**：根据平均预测局部距离差异测试评分对模型进行排名。最适合单个域。
- **pTMscore**：根据预测的 TM 评分对模型进行排名。推荐用于蛋白质复合物，因为它考虑整体结构质量。

注意：`pTMscore` 仅在 `use_ptm=True` 时可用。

来源：[beta/AlphaFold2_advanced.ipynb#L287-L288](beta/AlphaFold2_advanced.ipynb)

### 输出文件

```python
save_to_txt = True
save_pae_json = True
```

- **save_to_txt**：将距离图和接触信息保存到文本文件
- **save_pae_json**：将预测对齐误差数据保存为与 AlphaFold DB 兼容的 JSON 格式

这些输出对于预测置信度和域间交互的下游分析非常有价值。

来源：[beta/AlphaFold2_advanced.ipynb#L500-L550](beta/AlphaFold2_advanced.ipynb)

## 可视化选项

ColabFold 提供了多种选项以自定义结构可视化。

```python
color = "lDDT"  # 选项："chain", "lDDT", "rainbow"
show_sidechains = False
show_mainchains = False
dpi = 100  # 图表分辨率
```

- **color**：控制 3D 结构的颜色
  - **lDDT**：按预测置信度着色
  - **chain**：不同链颜色不同
  - **rainbow**：N 端到 C 端按彩虹渐变着色
- **show_sidechains**：切换氨基酸侧链的可见性
- **show_mainchains**：切换主链作为棍棒的可见性
- **dpi**：控制 2D 置信度图表的分辨率

对于动画，还有更多选项：

```python
use_pca = True
color_by_plddt = False
```

这些选项控制多个预测模型在动画中的对齐和着色方式。

来源：[beta/AlphaFold2_advanced.ipynb#L475-L480](beta/AlphaFold2_advanced.ipynb)，[beta/AlphaFold2_advanced.ipynb#L550-L570](beta/AlphaFold2_advanced.ipynb)

## 常见用例的示例配置

### 高置信度单一蛋白预测

```python
msa_mode = "mmseqs2_uniref_env"
template_mode = "pdb100"
num_models = 5
max_recycles = 6
num_ensemble = 1
use_ptm = True
num_relax = 1  # 仅精化顶级模型
rank_by = "pLDDT"
```

### 资源受限预测

```python
msa_mode = "mmseqs2_uniref"  # 排除环境序列
template_mode = "none"
num_models = 1
max_recycles = 3
use_turbo = True
max_msa = "128:256"
num_relax = 0  # 跳过精化
```

### 最大结构多样性

```python
msa_mode = "mmseqs2_uniref_env"
num_models = 5
is_training = True
num_samples = 8
num_ensemble = 2
use_ptm = True
```

### 蛋白质复合物预测

```python
pair_mode = "unpaired_paired"
rank_by = "pTMscore"
num_models = 5
max_recycles = 12
use_ptm = True
```

## 结论

ColabFold 的高级配置选项允许对预测过程进行精细控制。通过理解和调整这些参数，您可以针对特定蛋白、研究问题和计算资源优化预测。尝试不同的配置，以找到准确性和计算效率之间的最佳平衡，满足您特定的研究需求。