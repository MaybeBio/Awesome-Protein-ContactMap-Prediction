---
slug:11-data-augmentation-and-ensembling-strategies
blog_type:normal
---



Uni-Fold 实现了一个精密的数据增强和集成框架，通过多样化的训练和推理策略来增强模型的鲁棒性和预测准确性。该系统结合了多种增强技术和集成平均方法，以提高蛋白质结构预测的性能。

## 数据增强管道

数据增强管道分为三个不同的处理阶段，每个阶段在特征转换工作流程中都有特定用途。

### 非集成数据转换

非集成转换应用确定性预处理步骤，确保所有集成成员之间的一致性。这些操作包括 MSA 校正、掩码生成和模板处理：

```python
def nonensembled_fns(common_cfg, mode_cfg):
    """输入管道中不进行集成的数据转换器。"""
    operators = []
    if mode_cfg.random_delete_msa:
        operators.append(data_ops.random_delete_msa(common_cfg.random_delete_msa))
    operators.extend([
        data_ops.cast_to_64bit_ints,
        data_ops.correct_msa_restypes,
        data_ops.squeeze_features,
        data_ops.randomly_replace_msa_with_unknown(0.0),
        data_ops.make_seq_mask,
        data_ops.make_msa_mask,
    ])
```

这些转换通过选择性 MSA 删除和未知残基替换引入受控的随机性，同时确保数据一致性 [unifold/data/process.py#L9-L54]。

### 集成数据转换

集成转换引入随机变化，在不同集成成员之间创建多样性。这是核心增强策略的实现之处：

#### MSA 采样和聚类

MSA 采样过程采用 Gumbel-max 采样进行概率序列选择，使不同集成成员能够获得多样化的 MSA 表示：

```python
operators.append(
    data_ops.sample_msa(
        max_msa_clusters,
        keep_extra=True,
        gumbel_sample=gumbel_sample,
        biased_msa_by_chain=mode_cfg.biased_msa_by_chain,
    )
)
```

Gumbel-max 技巧提供了从概率分布中高效采样的方法，允许每个集成成员拥有不同的 MSA 子集，同时保持序列多样性 [unifold/data/process.py#L94-L161]。

#### 块 MSA 删除

对于单体预测，块删除随机移除连续的 MSA 区域以模拟不完整的序列信息：

```python
def block_delete_msa(protein, config):
    num_seq = protein["msa"].shape[0]
    block_num_seq = torch.floor(
        torch.tensor(num_seq, dtype=torch.float32) * config.msa_fraction_per_block
    ).to(torch.int32)
    
    del_block_starts = torch.from_numpy(np.random.randint(0, num_seq, [nb]))
    del_blocks = del_block_starts[:, None] + torch.arange(0, block_num_seq)
```

该技术通过在部分可用的 MSA 上训练来提高模型对缺失序列数据的鲁棒性 [unifold/data/data_ops.py#L298-L334]。

#### BERT 式训练的掩码 MSA

掩码 MSA 实现通过策略性地掩码残基来创建 BERT 式训练目标：

```python
def make_masked_msa(protein, config, replace_fraction, gumbel_sample=False, share_mask=False):
    categorical_probs = (
        config.uniform_prob * random_aa
        + config.profile_prob * protein["hhblits_profile"]
        + config.same_prob * one_hot(protein["msa"], 22)
    )
    
    mask_position = torch.from_numpy(np.random.rand(*sh) < replace_fraction)
    mask_position &= protein["msa_mask"].bool()
```

该方法结合了均匀、基于谱和基于同一性的采样，为自监督学习创建真实的掩码模式 [unifold/data/data_ops.py#L560-L607]。

### 裁剪和大小标准化

裁剪策略适应不同的蛋白质类型和大小，实现了单链和多聚体特异的方法：

#### 单链裁剪

对于单体蛋白质，选择随机连续区域来创建固定大小的输入：

```python
def crop_to_size_single(protein, crop_size, shape_schema, seed):
    num_res = protein["aatype"].shape[0] if "aatype" in protein else protein["msa_mask"].shape[1]
    crop_idx = get_single_crop_idx(num_res, crop_size, seed)
    protein = apply_crop_idx(protein, shape_schema, crop_idx)
    return protein
```

#### 多聚体空间裁剪

多聚体复合物使用空间感知裁剪来保留接口区域：

```python
def crop_to_size_multimer(protein, crop_size, shape_schema, seed, spatial_crop_prob, ca_ca_threshold):
    """裁剪到指定大小。"""
```

这确保在裁剪过程中保持关键的相互作用接口 [unifold/data/data_ops.py#L1239-L1242]。

## 集成策略

### 推理时集成

Uni-Fold 支持在推理期间配置集成大小，允许对多个随机预测进行平均以提高准确性：

```python
parser.add_argument("--num_ensembles", type=int, default=2)
config.data.predict.num_ensembles = args.num_ensembles
```

每个集成成员使用不同的随机种子以确保多样化的采样和增强结果 [unifold/inference.py#L249-L251]。

### 模板采样

模板子采样引入了结构模板使用的多样性：

```python
if args.sample_templates:
    config.data.predict.subsample_templates = True
```

这使每个集成成员能够获得不同的模板子集，增强结构预测的多样性 [unifold/inference.py#L85-L86]。

### 回收迭代

回收机制允许迭代优化预测，具有可配置的迭代次数：

```python
parser.add_argument("--max_recycling_iters", type=int, default=3)
config.data.common.max_recycling_iters = args.max_recycling_iters
config.globals.max_recycling_iters = args.max_recycling_iters
```

更多的回收迭代通常会提高预测准确性，但代价是增加计算量 [unifold/inference.py#L244-L246, L78-L80]。

## 配置和实现

### 增强参数

关键增强参数可通过模型配置系统进行配置：

- **MSA 采样**：控制 Gumbel 采样行为和对特定链的偏向
- **块删除**：配置要删除的 MSA 块的数量和大小
- **掩码**：设置 MSA 掩码的比例和策略
- **裁剪**：确定裁剪大小和多聚体的空间偏好

### 管道集成

增强管道通过 `process_features` 函数与训练和推理工作流程无缝集成，该函数根据配置参数组合所有转换步骤 [unifold/data/process.py#L162-L210]。

<CgxTip> 集成和非集成转换的分离对于在引入多样性的同时保持一致性至关重要。非集成操作确保所有集成成员获得相同的基础预处理，而集成操作则创建有效模型集成平均所需的变异。</CgxTip>

<CgxTip> Gumbel-max 采样提供了一种高效且可微的方式来从分类分布中采样，使其成为集成管道中 MSA 和模板采样的理想选择。这种方法在保持不同集成成员所需的随机性的同时，在计算上也很高效。</CgxTip>

## 性能考虑

增强和集成策略旨在平衡计算效率和预测准确性。关键优化包括：

- **分块处理**：大蛋白质被分块处理以管理内存使用
- **选择性集成**：并非所有转换都进行集成以减少计算开销
- **可配置多样性**：用户可根据可用计算资源调整集成大小和增强强度

这些策略使 Uni-Fold 能够扩展到不同大小的蛋白质复合物，同时通过有效的数据增强和集成保持高预测准确性。

有关模型架构以及这些增强如何与核心 AlphaFold 实现集成的更多详细信息，请参阅 [PyTorch 中的 AlphaFold 模型实现](6-alphafold-model-implementation-in-pytorch)。要了解这些特征如何提取和处理，请参阅 [特征提取和 MSA 处理](9-feature-extraction-and-msa-processing)。