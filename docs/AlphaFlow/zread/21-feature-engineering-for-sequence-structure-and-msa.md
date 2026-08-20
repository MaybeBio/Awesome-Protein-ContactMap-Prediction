---
slug:21-feature-engineering-for-sequence-structure-and-msa
blog_type:normal
---


本页面详细介绍了特征工程流水线，该流水线将原始蛋白质序列、结构和多序列比对（MSA）转换为 AlphaFlow 架构所需的丰富特征表示。理解此流水线对于有效使用模型至关重要，无论是训练新模型还是执行推理。

## 特征流水线架构

特征工程系统遵循分层设计，原始数据流经多个转换阶段。在顶层，`FeaturePipeline` 类协调从 numpy 数组到处理后的 PyTorch 张量的转换，准备模型输入。此流水线集成了基于配置的转换，以及针对训练、评估和预测场景的特定模式处理。

架构流从包含序列信息、MSA 比对和可选结构数据的原始特征字典开始。这些数据经过基于配置的转换器，应用裁剪、掩码、MSA 采样和特征选择操作。最终输出由符合模型预期特征模式的适当形状的张量组成。

来源：[feature_pipeline.py](alphaflow/data/feature_pipeline.py#L115-L132)

## 序列特征工程

### One-Hot 编码和序列级特征

序列特征构成特征表示的基础。`make_sequence_features` 函数构建了一组全面的序列级属性，包括通过使用 22 种可能残基（20 种标准氨基酸加上 X 代表未知和间隙）的 one-hot 表示进行的氨基酸类型编码 [data_pipeline.py](alphaflow/data/data_pipeline.py#L30-L49)。每个残基位置接收一个表示其身份的 22 维向量。

除了氨基酸编码外，序列特征还包括用于位置信息的残基索引、序列长度元数据，以及用于跟踪蛋白质来源的域标识符。这些特征支持模型对序列顺序的理解，并为残基特定预测提供上下文。

### 结构特征集成

当通过 PDB 或 mmCIF 文件提供结构信息时，流水线将原子坐标和掩码直接集成到特征字典中。`make_mmcif_features` 函数提取全原子位置及其相应的存在掩码，指示结构中实际存在哪些原子 [data_pipeline.py](alphaflow/data/data_pipeline.py#L52-L85)。这种集成使得对结构目标的监督训练成为可能，同时保持与仅序列输入的兼容性。

对于基于蒸馏的训练，流水线结合置信度阈值以过滤低置信度结构区域。高置信度区域通过超过可配置阈值（默认 50.0）的 B 因子识别，并选择性应用原子掩码以将训练集中在可靠的结构数据上 [data_pipeline.py](alphaflow/data/data_pipeline.py#L125-L140)。

## MSA 特征工程

### MSA 编码和删除矩阵

多序列比对提供了准确结构预测所需的进化信息。`make_msa_features` 函数处理比对序列集合及其相应的删除矩阵，将其转换为统一的特征表示 [data_pipeline.py](alphaflow/data/data_pipeline.py#L143-L176)。MSA 中的每个氨基酸使用 HHBLITS 氨基酸到 ID 的映射转换为整数标识符，实现紧凑表示。

删除矩阵捕获相对于查询序列的插入和删除事件，提供关于插入缺失模式的进化上下文。这些矩阵存储为整数计数，并在特征处理期间转换为 float32 张量。流水线移除重复序列以避免冗余信息，同时保留比对多样性。

### MSA 转换流水线

输入流水线对 MSA 特征应用一系列复杂的转换，从残基类型修正和随机替换未知氨基酸进行正则化开始 [input_pipeline.py](alphaflow/data/input_pipeline.py#L146-L156)。掩码 MSA 操作在训练期间随机遮蔽比对的一部分，通过基于轮廓、同序列和均匀掩码策略的可配置概率实现。

关键的 MSA 转换包括：
- **MSA 采样**：基于特定于模式的簇限制的动态比对序列选择（预测为 512，训练/评估为 128）[config.py](alphaflow/config.py#L320-L348)
- **额外 MSA 裁剪**：管理超出主簇的额外比对序列（默认 1024 个序列）[input_pipeline.py](alphaflow/data/input_pipeline.py#L234-L237)
- **基于簇的特征**：启用时的最近邻聚类和汇总 [input_pipeline.py](alphaflow/data/input_pipeline.py#L229-L231)
- **轮廓生成**：捕获每个位置氨基酸频率的 HHblits 轮廓构建 [input_pipeline.py](alphaflow/data/input_pipeline.py#L155)

<CgxTip>MSA 重采样策略对于推理一致性至关重要。当禁用 `resample_msa_in_recycling` 时，固定种子确保在所有循环迭代中相同的 MSA 采样，而启用重采样会以一致性为代价引入多样性。</CgxTip>

## 结构特征工程

### 原子级表示

对于监督训练和基于结构的输入，流水线生成全面的原子级特征。转换序列创建多个原子表示，包括 atom14（蛋白质建模中使用的每个残基标准 14 个原子）和 atom37（完整的 PDB 原子集）[input_pipeline.py](alphaflow/data/input_pipeline.py#L174-L176)。每个表示包括指示哪些原子存在的存在掩码。

`make_atom14_positions` 函数将 atom37 坐标转换为 atom14 框架，将 PDB 原子映射到模型中使用的简化表示 [input_pipeline.py](alphaflow/data/input_pipeline.py#L181)。此转换包括处理多个 atom37 位置映射到同一 atom14 位置的歧义情况。

### 扭转角和框架特征

结构特征不仅限于笛卡尔坐标，还包括扭转角和刚体框架。转换流水线通过 `atom37_to_frames` 计算主链框架，为每个残基建立局部坐标系 [input_pipeline.py](alphaflow/data/input_pipeline.py#L182)。这些框架支持旋转不变的表示，并促进基于框架的注意力机制。

扭转角通过 `atom37_to_torsion_angles` 提取，将主链和侧链二面角捕获为正弦和余弦对，以避免角度不连续性 [input_pipeline.py](alphaflow/data/input_pipeline.py#L183)。这些角度表示对于现实构象生成至关重要，并用于监督目标和模板特征。

### 伪 Beta 特征

流水线生成伪 beta 位置，表示所有残基的 Cβ 原子位置的近似值（对于甘氨酸使用 Cα）[input_pipeline.py](alphaflow/data/input_pipeline.py#L184)。这些特征提供适合成对距离预测的简化表示，并作为几何推理的锚点。查询和模板结构都接收带有相应存在掩码的伪 beta 特征。

## 模板特征工程

### 模板选择和处理

当启用模板时，流水线通过多阶段处理集成结构模板。模板特征从 TemplateHitFeaturizer 的命中识别和特征化开始，该转换器将比对命中转换为结构特征字典 [data_pipeline.py](alphaflow/data/data_pipeline.py#L480-L517)。根据评分指标选择信息量最大的模板。

模板特征反映了应用于查询序列的结构特征，包括：
- 模板全原子位置和掩码 [config.py](alphaflow/config.py#L253-L263)
- 模板伪 beta 表示 [config.py](alphaflow/config.py#L265-L266)
- 用于结构上下文的模板主链刚体框架 [config.py](alphaflow/config.py#L260-L263)
- 启用时的模板扭转角 [config.py](alphaflow/config.py#L267-L273)

### 模板扭转角嵌入

流水线支持可选的模板扭转角嵌入，由 `embed_template_torsion_angles` 配置控制 [config.py](alphaflow/config.py#L198)。启用时，模板二面角转换为正弦/余弦表示并集成到模板特征集中，提供来自同源结构的直接角度约束。

模板特征经过特定转换，包括氨基酸类型修正（`fix_templates_aatype`）和掩码生成（`make_template_mask`），以处理模板结构中缺失或模糊的区域 [input_pipeline.py](alphaflow/data/input_pipeline.py#L158-L162)。

## 特征模式和维度

### 综合特征字典

配置定义了一个完整的模式，指定所有特征的预期维度。关键维度占位符包括 `NUM_RES` 用于序列长度，`NUM_MSA_SEQ` 用于 MSA 序列计数，`NUM_EXTRA_SEQ` 用于额外比对序列，以及 `NUM_TEMPLATES` 用于模板计数 [config.py](alphaflow/config.py#L201-L204)。这些占位符在运行时根据实际输入维度解析。

该模式包含 60 多个不同的特征键，组织为以下类别：
- **序列特征**：aatype、residue_index、seq_length、seq_mask [config.py](alphaflow/config.py#L211-L250)
- **MSA 特征**：msa、msa_feat、msa_mask、deletion_matrix、true_msa [config.py](alphaflow/config.py#L225-L274)
- **结构特征**：all_atom_positions、atom14_positions、pseudo_beta、backbone_frames [config.py](alphaflow/config.py#L212-L248)
- **模板特征**：template_aatype、template_all_atom_positions、template_torsion_angles [config.py](alphaflow/config.py#L252-L273)
- **元数据特征**：resolution、is_distillation、no_recycling_iters、use_clamped_fape [config.py](alphaflow/config.py#L233-L275)

### 特定于模式的特征选择

不同的操作模式激活不同的特征子集。无监督特征形成在所有模式中使用的核心集，包括序列和 MSA 信息 [config.py](alphaflow/config.py#L292-L301)。监督特征在训练和评估期间添加，提供用于学习的结构目标 [config.py](alphaflow/config.py#L307-L314)。

流水线根据配置有选择地启用模板特征，特定于模板的激活标志控制是否使用模板数据以及是否包含扭转角嵌入 [config.py](alphaflow/config.py#L302-L303)。

## 数据增强和预处理

### 随机裁剪和固定大小处理

为了训练效率，流水线实现了复杂的裁剪策略。`random_crop_to_size` 函数对序列、MSA 和模板应用随机裁剪，同时保持维度一致性 [input_pipeline.py](alphaflow/data/input_pipeline.py#L24-L109)。这使得能够在可变长度蛋白质上进行训练，同时为 GPU 处理准备固定大小的批次。

裁剪后，`make_fixed_size` 将特征填充到预定维度，确保样本间一致的张量形状 [input_pipeline.py](alphaflow/data/input_pipeline.py#L111-L145)。填充策略尊重特征语义，为不同特征类型使用适当的值（例如，位置特征使用零，掩码使用特定哨兵值）。

### 循环和钳制特征

流水线生成与循环相关的特征，包括 `no_recycling_iters` 和 `use_clamped_fape` [feature_pipeline.py](alphaflow/data/feature_pipeline.py#L97-L110)。钳制特征控制 FAPE（框架对齐点误差）损失使用钳制还是未钳制的距离，在训练期间随机激活，概率由 `clamp_prob` 配置控制 [config.py](alphaflow/config.py#L306)。

<CgxTip>训练使用 90% 的钳制 FAPE 值概率（clamp_prob=0.9），通过限制大结构偏差的影响来稳定学习。推理始终使用未钳制值（概率 0.0）以允许生成期间完全的结构自由。</CgxTip>

## 模型架构中的特征集成

### 输入对堆栈处理

特征表示通过专用输入处理器馈送到模型架构中。InputPairStack 实现了 AlphaFold 论文中的算法 16，通过三角形注意力和乘法更新块处理成对特征 [input_stack.py](alphaflow/model/input_stack.py#L171-L240)。每个块结合三角形注意力（起始和结束节点变体）与三角形乘法（传出和传入方向）以及对转换层。

这种三角形处理使模型能够推理残基对之间的几何关系，注意力机制捕获距离和方向特征中的方向模式。堆栈配置了通道维度（`c_t` 用于对特征，`c_hidden_tri_att` 用于注意力，`c_hidden_tri_mul` 用于乘法更新）和多头注意力头数 [config.py](alphaflow/config.py#L189-L190, L398-L402)。

### 配置驱动的架构

模型架构维度通过可以全局修改的 FieldReferences 指定。关键通道维度包括 `c_z`（对表示，默认 128）、`c_m`（MSA 表示，默认 256）、`c_t`（模板表示，默认 64）、`c_e`（额外 MSA 表示，默认 64）和 `c_s`（单表示，默认 384）[config.py](alphaflow/config.py#L187-L191)。

输入嵌入器配置转换维度，包括目标特征的 `tf_dim`（22）和 MSA 特征的 `msa_dim`（49），指定原始特征如何投影到模型的表示空间 [config.py](alphaflow/config.py#L390-L396)。

## 流水线编排

### 数据流水线集成

DataPipeline 类编排来自不同输入源的特征组装 [data_pipeline.py](alphaflow/data/data_pipeline.py#L409-L730)。它支持多种输入格式，包括 FASTA 文件、PDB 结构、mmCIF 文件和 ProteinNet 核心文件，并为每种格式提供专门的处理方法。流水线在统一的工作流中处理 MSA 生成、模板识别和特征集成。

对于多序列 FASTA 输入，流水线实现“AlphaFold-Gap”技术，允许以受控间隙大小（默认 200 个残基）处理连接序列 [data_pipeline.py](alphaflow/data/data_pipeline.py#L721-L730)。这使得能够高效处理多链或多域预测。

### 比对运行器集成

AlignmentRunner 处理 MSA 生成的外部工具执行，支持针对 Uniref90、Mgnify、BFD 和 PDB70 等数据库的 jackhmmer、hhblits 和 hhsearch 搜索 [data_pipeline.py](alphaflow/data/data_pipeline.py#L214-L360)。可配置的命中限制控制检索的比对数量，不同数据库有单独的设置（例如，uniref_max_hits=10000，mgnify_max_hits=5000）。

比对过程将结果保存到输出目录，启用跨多次运行的缓存和重用。流水线通过专门的方法处理单链和多链蛋白质，将这些比对结果解析为 MSA 特征 [data_pipeline.py](alphaflow/data/data_pipeline.py#L418-L551)。

## 后续步骤

理解特征工程为有效使用 AlphaFlow 奠定了基础。为了实际应用，请继续：

- [使用 MMseqs2 生成 MSA](18-msa-generation-with-mmseqs2) - 了解如何生成构成 MSA 特征核心的进化比对
- [模板处理和特征提取](19-template-processing-and-feature-extraction) - 探索如何识别结构模板并将其集成到特征流水线中
- [PDB 和 ATLAS 数据集的数据预处理](20-data-preprocessing-for-pdb-and-atlas-datasets) - 了解不同数据集的端到端数据准备工作流

对于模型端处理，请参阅 [输入堆栈和特征表示](9-input-stack-and-feature-representations) 以了解这些特征如何被架构使用，以及 [Evoformer 和折叠主干架构](8-evoformer-and-folding-trunk-architecture) 了解下游特征转换层。