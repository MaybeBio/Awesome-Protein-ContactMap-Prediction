---
slug:8-chemical-graph-operations
blog_type:normal
---


化学图是bioemu框架中表示生物分子结构的基础数据结构。它提供了一种统一的表示方式，弥合了序列信息、3D坐标和进化特征之间的差距，使深度学习模型能够高效处理蛋白质结构预测和设计任务。

## Bioemu中的化学图理解

bioemu中的化学图不仅仅是简单的图表示——它是一种综合性的数据结构，封装了蛋白质结构建模所需的所有关键信息。其核心在于，`ChemGraph`类扩展了PyTorch Geometric的`Data`类，提供了专为生物分子结构定制的专业表示。`来源：[chemgraph.py#L12-L19](src/bioemu/chemgraph.py#L12-L19)`

`ChemGraph`包含几个关键组成部分：
- **节点朝向**：表示每个残基方向的3D旋转矩阵
- **位置**：每个残基的3D坐标（纳米单位）
- **边索引**：残基之间的连接信息
- **单残基嵌入**：来自进化分析的每个残基特征
- **残基对嵌入**：残基之间的成对特征
- **序列**：作为字符串的氨基酸序列
- **系统ID**：蛋白质的可选标识符

这种丰富的表示方式使框架能够以统一的方式同时处理序列信息、结构约束和进化背景。

## 创建化学图

化学图通常通过多步骤过程从蛋白质序列创建，涉及生成嵌入和建立图结构。`get_context_chemgraph`函数展示了这一过程：`来源：[sample.py#L176-L216](src/bioemu/sample.py#L176-L216)`

```python
def get_context_chemgraph(
    sequence: str,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
) -> ChemGraph:
    n = len(sequence)
    
    # 使用ColabFold生成嵌入
    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    
    # 将嵌入加载为张量
    single_embeds = torch.from_numpy(np.load(single_embeds_file))
    pair_embeds = torch.from_numpy(np.load(pair_embeds_file))
    
    # 创建全连接边索引
    edge_index = torch.cat([
        [
            torch.arange(n).repeat_interleave(n).view(1, n**2),
            torch.arange(n).repeat(n).view(1, n**2),
        ],
        dim=0,
    ])
    
    # 初始化位置和朝向（将在采样过程中填充）
    pos = torch.full((n, 3), float("nan"))
    node_orientations = torch.full((n, 3, 3), float("nan"))
    
    return ChemGraph(
        edge_index=edge_index,
        pos=pos,
        node_orientations=node_orientations,
        single_embeds=single_embeds,
        pair_embeds=pair_embeds,
        sequence=sequence,
    )
```

此函数创建了一个包含结构生成所需所有必要组件的化学图。注意位置和朝向最初设置为NaN值——它们将在结构采样过程中被填充。

## 不同表示之间的转换

bioemu中化学图操作最强大的方面之一是能够在不同结构表示之间进行转换。`convert_chemgraph.py`模块提供了几个关键函数用于这些转换：`来源：[convert_chemgraph.py](src/bioemu/convert_chemgraph.py)`

### 从框架到全原子位置

`get_atom37_from_frames`函数将粗粒度的框架信息（位置和朝向）转换为详细的全原子位置，遵循蛋白质结构建模中使用的标准atom37表示：

```python
def get_atom37_from_frames(
    pos: torch.Tensor,           # (num_residues, 3) 单位：nm
    node_orientations: torch.Tensor,  # (num_residues, 3, 3)
    sequence: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 转换为刚体表示
    positions = pos.view(1, -1, 3)
    orientations = node_orientations.view(1, -1, 3, 3)
    rots = Rotation(rot_mats=orientations)
    rigids = Rigid(rots=rots, trans=positions)
    
    # 获取残基类型
    aatype = torch.tensor([
        residue_constants.restype_order.get(x, 0) 
        for x in sequence
    ], device=device)
    
    # 计算主链原子
    atom_37, atom_37_mask = compute_backbone(
        bb_rigids=rigids,
        psi_torsions=torch.zeros(1, positions.shape[1], 2, device=device),
        aatype=aatype,
    )
    
    # 调整氧原子位置
    atom_37 = _adjust_oxygen_pos(atom_37, pos_is_known=None)
    
    return atom_37, atom_37_mask, aatype
```

这种转换对于从扩散过程中使用的粗粒度表示生成详细蛋白质结构至关重要。

### 氧原子位置推算

转换过程中一个特别有趣的方面是氧原子位置的智能推算。`_adjust_oxygen_pos`函数使用几何推理将氧原子放置在物理上合理的位置：`来源：[convert_chemgraph.py#L214-L293](src/bioemu/convert_chemgraph.py#L214-L293)`

```python
def _adjust_oxygen_pos(atom_37: torch.Tensor) -> torch.Tensor:
    # 获取从Cα和下一个残基的氮原子到羰基的向量
    calpha_to_carbonyl = (atom_37[:-1, 2, :] - atom_37[:-1, 1, :])
    nitrogen_to_carbonyl = (atom_37[:-1, 2, :] - atom_37[1:, 0, :])
    
    # 归一化向量
    calpha_to_carbonyl = calpha_to_carbonyl / (
        torch.norm(calpha_to_carbonyl, keepdim=True, dim=1) + 1e-7
    )
    nitrogen_to_carbonyl = nitrogen_to_carbonyl / (
        torch.norm(nitrogen_to_carbonyl, keepdim=True, dim=1) + 1e-7
    )
    
    # 计算氧原子位置
    carbonyl_to_oxygen = calpha_to_carbonyl + nitrogen_to_carbonyl
    carbonyl_to_oxygen = carbonyl_to_oxygen / (
        torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-7
    )
    
    # 将氧原子放置在正确的键长位置
    atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * C_O_BOND_LENGTH
```

此函数展示了框架对化学真实性的关注，确保生成的结构保持适当的键长和键角。

## 将结构保存为标准格式

化学图操作还包括将生成的结构保存为标准分子文件格式的实用工具。`save_pdb_and_xtc`函数将采样的结构批次转换为PDB和XTC文件，用于可视化和进一步分析：`来源：[convert_chemgraph.py#L398-L461](src/bioemu/convert_chemgraph.py#L398-L461)`

```python
def save_pdb_and_xtc(
    pos_nm: torch.Tensor,           # (batch_size, N, 3) 单位：nm
    node_orientations: torch.Tensor,  # (batch_size, N, 3, 3)
    sequence: str,
    topology_path: str | Path,
    xtc_path: str | Path,
    filter_samples: bool = True,
) -> None:
    # 转换为埃单位并居中
    pos_angstrom = pos_nm * 10.0
    pos_angstrom = pos_angstrom - pos_angstrom.mean(axis=1, keepdims=True)
    
    # 将第一帧写入为PDB拓扑
    _write_pdb(
        pos=pos_angstrom[0],
        node_orientations=node_orientations[0],
        sequence=sequence,
        filename=topology_path,
    )
    
    # 将所有帧转换为atom37表示
    xyz_angstrom = []
    for i in range(batch_size):
        atom_37, atom_37_mask, _ = get_atom37_from_frames(
            pos=pos_angstrom[i], 
            node_orientations=node_orientations[i], 
            sequence=sequence
        )
        xyz_angstrom.append(atom_37.view(-1, 3)[atom_37_mask.flatten()].cpu().numpy())
    
    # 创建轨迹并保存为XTC
    topology = mdtraj.load_topology(topology_path)
    traj = mdtraj.Trajectory(xyz=np.stack(xyz_angstrom) * 0.1, topology=topology)
    
    if filter_samples:
        traj = filter_unphysical_traj(traj)
    
    traj.save_xtc(xtc_path)
```

<CgxTip>
`save_pdb_and_xtc`中的过滤步骤特别重要，因为它移除了可能具有不合理键长或空间位阻的非物理结构。这确保只有物理上合理的结构才会保存到输出文件中。
</CgxTip>

## 与采样流程的集成

化学图是bioemu中结构生成流程的核心。采样过程使用化学图作为输入和输出表示，模型迭代地预测位置和朝向的更新：`来源：[sample.py#L219-L265](src/bioemu/sample.py#L219-L265)`

```python
def generate_batch(
    score_model: torch.nn.Module,
    sequence: str,
    sdes: dict[str, SDE],
    batch_size: int,
    seed: int,
    denoiser: Callable,
    cache_embeds_dir: str | Path | None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
) -> dict[str, torch.Tensor]:
    # 创建上下文化学图
    context_chemgraph = get_context_chemgraph(
        sequence=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    
    # 创建用于处理的批次
    context_batch = Batch.from_data_list([context_chemgraph] * batch_size)
    
    # 使用去噪过程采样结构
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sampled_chemgraph_batch = denoiser(
        sdes=sdes,
        device=device,
        batch=context_batch,
        score_model=score_model,
    )
    
    # 提取位置和朝向
    sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
    pos = torch.stack([x.pos for x in sampled_chemgraphs])
    node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs])
    
    return {"pos": pos, "node_orientations": node_orientations}
```

这种集成展示了化学图如何作为框架不同组件之间的接口，实现从序列处理到结构生成和输出的信息无缝流动。

## 最佳实践和注意事项

在bioemu中使用化学图时，有几个重要考虑因素需要牢记：

1. **单位很重要**：框架内部计算使用纳米单位，但转换为埃单位用于PDB输出。处理位置数据时始终要注意单位。

2. **内存效率**：对于大型蛋白质，化学图可能占用大量内存。框架会根据序列长度自动调整批次大小以管理内存使用。

3. **验证**：框架包含验证函数来检查生成的结构是否物理合理，包括键长和空间位阻检查。

4. **灵活性**：化学图表示设计为灵活的，允许轻松集成不同类型的结构模型和采样算法。

bioemu中的化学图操作代表了蛋白质结构建模的一种复杂方法，结合了图神经网络的强大功能与蛋白质化学和几何学的领域特定知识。通过提供捕获序列和结构信息的统一表示，这些操作能够生成高质量的蛋白质结构，既尊重进化约束又符合物理现实。