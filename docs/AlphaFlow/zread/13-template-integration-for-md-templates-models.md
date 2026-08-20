---
slug:13-template-integration-for-md-templates-models
blog_type:normal
---


模板集成是 AlphaFlow MD+Templates 模型系列中的关键架构组件，实现了分子动力学轨迹数据与结构模板约束的融合。本文档阐述了模板的处理方式、如何集成到模型架构中，以及在基于 MD 的蛋白质结构预测的训练和推理过程中如何利用这些模板。

## 模板处理流程

模板集成过程始于从 mmCIF 和 PDB 文件中系统地提取和处理结构模板。DataPipeline 类通过专门的模板命中解析和特征化工作流来协调模板特征的生成。

通过针对结构数据库（通常是 PDB70）的序列比对搜索来识别模板，并根据序列相似性和比对置信度对顶部命中进行排序。`_parse_template_hits` 方法解析 HHR（隐马尔可夫模型）格式的比对文件以提取模板信息 [data_pipeline.py#L480-L518]。这些原始命中随后通过模板特征化管道转换为张量化的特征表示。

### 模板特征架构

模板特征遵循一种结构化的架构设计，旨在捕捉几何结构和序列-结构关系：

| 特征类别 | 特征名称 | 形状 | 用途 |
|-----------------|--------------|-------|---------|
| 序列 | `template_aatype` | [NUM_TEMPLATES, NUM_RES] | 每个模板的氨基酸类型 |
| 几何 | `template_all_atom_positions` | [NUM_TEMPLATES, NUM_RES, None, None] | 完整原子坐标 |
| 几何 | `template_all_atom_mask` | [NUM_TEMPLATES, NUM_RES, None] | 有效原子位置指示符 |
| 几何 | `template_pseudo_beta` | [NUM_TEMPLATES, NUM_RES, None] | Cβ（或 Gly 的 Cα）位置 |
| 几何 | `template_pseudo_beta_mask` | [NUM_TEMPLATES, NUM_RES] | 伪 β 有效性掩码 |
| 主链 | `template_backbone_rigid_tensor` | [NUM_TEMPLATES, NUM_RES, None, None] | 刚性框架表示 |
| 主链 | `template_backbone_rigid_mask` | [NUM_TEMPLATES, NUM_RES] | 刚性框架有效性 |
| 扭转角 | `template_torsion_angles_sin_cos` | [NUM_TEMPLATES, NUM_RES, None, None] | Sin/cos 编码的角度 |
| 扭转角 | `template_torsion_angles_mask` | [NUM_TEMPLATES, NUM_RES, None] | 扭转角有效性 |
| 元数据 | `template_sum_probs` | [NUM_TEMPLATES, None] | 模板命中概率总和 |
| 元数据 | `template_mask` | [NUM_TEMPLATES] | 模板有效性掩码 |

来源：[config.py#L252-L273]

模板特征架构支持同时处理多个模板，其中 `NUM_TEMPLATES` 充当动态占位符，能够适应实际可用模板命中的数量。配置参数控制训练（`max_templates=4`）和推理（`max_templates=4`）期间的模板限制 [config.py#L350-L323]。

## 模板集成架构

模板特征通过专门的模板处理堆栈集成到 AlphaFlow 架构中，该堆栈与主要的 Evoformer 处理管道并行运行。集成遵循分层递进过程：原始模板特征 → 模板嵌入 → 模板对堆栈 → 基于注意力的融合 → 主干集成。

### 模板堆栈架构

模板处理堆栈由三个主要组件组成：

1. **模板角度嵌入器**：将扭转角特征转换为序列嵌入。嵌入器将 57 维输入特征（表示 sin/cos 编码角度加上掩码信息）映射到序列嵌入维度 `c_m` [config.py#L418-L422]。

2. **模板对嵌入器**：将几何和位置模板特征转换为成对表示。该模块接受 88 维输入（编码相对位置、方向和兼容性分数），并输出维度为 `c_t=64` 的模板对嵌入 [config.py#L423-L426]。

3. **模板对堆栈**：在模板对嵌入上实现三角形自注意力。该堆栈通过多个三角形注意力和转换层块处理模板成对关系，使模型能够跨模板推理结构模式 [config.py#L427-L440]。

```mermaid
flowchart TB
    subgraph Template_Features
        T1[template_aatype<br/>[T, L]]
        T2[template_all_atom_positions<br/>[T, L, A, 3]]
        T3[template_torsion_angles_sin_cos<br/>[T, L, 7, 2]]
        T4[template_mask<br/>[T]]
    end
    
    subgraph Template_Embedding
        AE[Template Angle Embedder<br/>c_in=57 → c_out=c_m=256]
        PE[Template Pair Embedder<br/>c_in=88 → c_out=c_t=64]
        TPS[Template Pair Stack<br/>2 blocks, triangular attention]
    end
    
    subgraph Template_Pointwise_Attention
        TPA[Template Pointwise Attention<br/>4 heads, c_hidden=16]
    end
    
    subgraph Trunk_Integration
        EVO[Evoformer Stack]
        SM[Structure Module]
    end
    
    T1 --> AE
    T2 --> PE
    T3 --> AE
    T4 --> AE
    T4 --> PE
    
    PE --> TPS
    TPS --> TPA
    
    TPA -->|z_template + z_msa| EVO
    EVO --> SM
    
    style TPS fill:#e1f5ff
    style TPA fill:#fff4e1
```

模板逐点注意力机制充当关键的集成枢纽，其中模板衍生的成对表示关注 MSA 衍生的成对表示。该注意力块使用 4 个头和 16 的隐藏维度运行，计算模板和查询成对特征的加权组合 [config.py#L441-L449]。

来源：[config.py#L412-L465]

## MD+Templates 模型配置

MD+Templates 模型代表了一种专门的模型变体，它结合了分子动力学轨迹数据和结构模板约束。这些模型的独特之处在于能够同时利用 MD 衍生的结构特征和基于同源性的模板信息。

### 模板集成控制

模板子系统提供了几个配置选项，用于控制模板如何集成到模型中：

| 配置参数 | 类型 | 默认值 | 用途 |
|-----------------------|------|---------|---------|
| `model.template.enabled` | bool | True | 模板处理的主开关 |
| `data.common.use_templates` | bool | True | 启用模板特征提取 |
| `data.common.use_template_torsion_angles` | bool | True | 包含扭转角特征 |
| `model.template.average_templates` | bool | False | 平均模板嵌入以提高效率 |
| `model.template.offload_templates` | bool | False | 卸载到 CPU 以节省内存 |
| `data.train.max_templates` | int | 4 | 训练期间的最大模板数 |
| `data.predict.max_templates` | int | 4 | 推理期间的最大模板数 |

来源：[config.py#L302-L303], [config.py#L452-L465]

### 模型预设配置

不同的模型预设根据其预期用例启用或禁用模板：

```python
# 启用模板的模型（适用于 MD+Templates）
"model_1":  # 带有模板的主要推理模型
    c.data.common.use_templates = True
    c.model.template.enabled = True

"model_2":  # 带有模板的替代推理模型
    c.data.common.use_templates = True
    c.model.template.enabled = True

# 禁用模板的模型（无模板变体）
"model_3", "model_4", "model_5":
    c.model.template.enabled = False
```

MD+Templates 模型专门利用启用模板的配置（model_1 和 model_2）来结合 MD 轨迹数据和模板约束 [config.py#L96-L154]。

来源：[config.py#L96-L154]

## 用于 MD 集成的额外输入处理

MD+Templates 模型引入了超出标准模板处理的额外结构输入能力。`extra_input` 标志启用了辅助输入成对嵌入通路，可以合并额外的结构信息，例如 MD 衍生的构象。

### 额外输入成对嵌入

当在模型初始化期间启用 `extra_input=True` 时，AlphaFold 模型会实例化额外的嵌入组件以处理额外的结构输入：

```python
if extra_input:
    self.extra_input_pair_embedding = Linear(
        self.config.input_pair_embedder.no_bins, 
        self.config.evoformer_stack.c_z,
        init="final",
    )   
    self.extra_input_pair_stack = InputPairStack(**self.config.input_pair_stack)
```

在前向传播期间，如果批次中存在 `extra_all_atom_positions`，模型会根据这些额外坐标计算伪 β 距离，并通过额外输入成对堆栈进行处理 [alphafold.py#L238-L250]。

来源：[alphafold.py#L103-L109], [alphafold.py#L238-L250]

这种额外输入通路允许 MD+Templates 模型同时合并：
1. **模板特征**：来自已知结构的基于同源性的结构约束
2. **MD 轨迹特征**：来自分子动力学模拟的构象样本
3. **流匹配目标**：用于扩散训练的噪声结构表示

### 集成点

模板衍生的特征和额外输入特征在模型的前向传播中汇聚于成对表示 `z`：

```python
# 模板集成（如果启用）
if self.template_config.enabled:
    z_template = self.template_pair_stack(t, template_mask, ...)
    z = z + self.template_pointwise_attention(z_template, z, ...)

# 额外输入集成（如果可用）
if self.extra_input and 'extra_all_atom_positions' in batch:
    extra_inp_z = self._get_extra_input_pair_embeddings(...)
    z = z + extra_inp_z
```

这种加法集成允许模型合并多个结构信息源，同时保持在训练期间对每个贡献进行加权的能力 [alphafold.py#L150-L235]。

来源：[alphafold.py#L150-L235]

## 训练期间的模板处理

训练 MD+Templates 模型需要仔细管理模板特征，特别是在数据增强期间裁剪和子采样序列及模板时。

### 模板子采样和裁剪

`random_crop_to_size` 变换函数实现了感知模板的裁剪，该裁剪在集成成员之间维护模板一致性：

```python
def random_crop_to_size(
    protein,
    crop_size,
    max_templates,
    shape_schema,
    subsample_templates=False,
    seed=None,
):
    # 特定于模板的处理
    if "template_mask" in protein:
        num_templates = protein["template_mask"].shape[-1]
        subsample_templates = subsample_templates and num_templates
    
    if subsample_templates:
        templates_crop_start = _randint(0, num_templates)
        templates_select_indices = torch.randperm(
            num_templates, device=protein["seq_length"].device, generator=g
        )
```

该函数应用了特定的模板裁剪逻辑，在尊重模板维度的同时维持残基对应关系。模板特征在裁剪前被随机排列，以减少训练期间的位置偏差 [input_pipeline.py#L25-L106]。

来源：[input_pipeline.py#L25-L106]

### 固定大小模板填充

`make_fixed_size` 函数将模板填充到一致的大小以便进行批次处理，类似于填充 MSA 和序列特征的方式：

```python
def make_fixed_size(
    protein,
    shape_schema,
    num_res=0,
    num_templates=0,
):
    pad_size_map = {
        NUM_RES: num_res,
        NUM_TEMPLATES: num_templates,
    }
    # 基于 shape_schema 对模板特征应用填充
```

这确保了训练期间可以高效处理具有不同模板计数的批次 [input_pipeline.py#L110-L120]。

来源：[input_pipeline.py#L110-L120]

## 内存优化策略

模板处理可能会消耗大量内存，特别是对于具有多个模板的长序列。AlphaFlow 提供了几种优化策略：

### 模板平均

当 `average_templates=True` 时，模板嵌入在模板维度上进行平均，而不是单独处理。这将内存使用量从 O(T × L²) 减少到 O(L²)，其中 T 是模板数量，L 是序列长度 [config.py#L459]。

### 模板卸载

`offload_templates` 选项在模板嵌入未被主动使用时将其移动到 CPU 内存，从而以增加 CPU-GPU 传输为代价显著减少 GPU 内存消耗。当为长序列推理设置 `offload_inference` 时，这会自动启用 [config.py#L460-L465]。

### 低内存注意力

模板处理支持与其他模型组件相同的低内存注意力 (LMA) 和 FlashAttention 优化，通过 `globals.use_lma` 和 `globals.use_flash` 设置进行全局控制 [config.py#L375-L379]。

<CgxTip>
对于超过 1000 个残基的序列推理，同时启用 `offload_templates=True` 和 `use_lma=True`，与默认配置相比，可减少约 60-80% 的内存使用量。
</CgxTip>

来源：[config.py#L459-L465], [config.py#L375-L379]

## MD 数据的模板特征提取

处理用于模板集成的 MD 轨迹需要专门的特征提取，以考虑到 MD 构象与静态晶体学模板相比的动态性质。

### 特定于 MD 的特征生成

`make_pdb_features` 函数提供了一条专门为 MD 轨迹处理设计的模板特征生成通路：

```python
def make_pdb_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.,
) -> FeatureDict:
    # 从蛋白质对象生成特征
    # 如果指定，应用置信度过滤
    # 返回与模板管道兼容的特征字典
```

当 `is_distillation=True` 时，此函数应用额外的处理，适用于蒸馏训练场景，其中模型从教师模型或高质量的 MD 轨迹中学习 [data_pipeline.py#L125-L130]。

来源：[data_pipeline.py#L125-L130]

### MD 的模板命中生成

基于 MD 的模板生成与基于同源性的模板生成不同。无需搜索结构数据库，MD 轨迹可以通过以下方式作为“合成模板”进行处理：

1. 从 MD 轨迹中采样帧
2. 计算伪 β 距离和扭转角
3. 计算逐帧质量分数（相对于参考的 RMSD、能量等）
4. 排序并选择帧作为模板命中
5. 转换为模板特征格式

此工作流允许 MD+Templates 模型同时利用外部同源模板和内部 MD 衍生的结构约束。

来源：[data_pipeline.py#L480-L518]

## 后续步骤

要全面了解模板集成在更广泛的 AlphaFlow 架构中的作用：

- **[Evoformer 和折叠主干架构](8-evoformer-and-folding-trunk-architecture)**：了解集成了模板的成对表示如何流经主干处理
- **[PDB 和 MD 数据集的训练管道](10-training-pipeline-for-pdb-and-md-datasets)**：了解针对不同数据源训练期间如何配置模板特征
- **[模板处理和特征提取](19-template-processing-and-feature-extraction)**：深入了解模板特征化算法和比对策略