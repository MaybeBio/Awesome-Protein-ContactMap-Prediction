---
slug:4-protein-structure-modeling-tutorial
blog_type:normal
---


欢迎来到 BioEmu 蛋白质结构建模教程！在本指南中，您将学习如何使用 BioEmu 从氨基酸序列生成 3D 蛋白质结构。BioEmu 是一个强大的工具，它能够根据蛋白质单体的氨基酸序列，从近似平衡结构分布中进行采样。通过本教程的学习，您将能够生成自己的蛋白质结构模型，并理解这一过程背后的关键概念。

## 理解蛋白质结构生成

蛋白质结构预测是计算生物学中的一个基本问题。传统的分子动力学模拟等方法计算成本高昂，而像 AlphaFold2 这样的机器学习方法则提供单一结构预测。BioEmu 采用了一种不同的方法，通过生成代表蛋白质天然构象多样性的结构集合。

BioEmu 使用**扩散模型**——一种通过逆转渐进加噪过程来学习创建样本的生成模型。可以将其理解为从随机噪声开始，逐步精修成生物学上合理形状的蛋白质结构雕刻过程。这种方法使 BioEmu 能够生成多样化、物理上真实的蛋白质结构，捕捉蛋白质天然的灵活性和动态特性。

BioEmu 功能的核心围绕**ChemGraph**表示展开，它将蛋白质结构编码为图，其中节点代表氨基酸残基，边代表它们之间的相互作用。每个节点包含位置信息和方向数据，边则捕获残基之间的成对关系。

来源：[README.md#L14-16](README.md#L14-16), [chemgraph.py#L12-18](src/bioemu/chemgraph.py#L12-18)

## 设置环境

在开始生成蛋白质结构之前，让我们确保您的环境已正确设置。BioEmu 设计用于 Linux 系统，需要 Python 3.10 或更高版本。

```bash
# 安装 BioEmu
pip install bioemu
```

<CgxTip>
首次使用 BioEmu 采样结构时，它会自动在单独的虚拟环境中设置 ColabFold，用于 MSA（多序列比对）和嵌入生成。此设置默认使用 `~/.bioemu_colabfold` 目录。
</CgxTip>

在本教程中，我们将使用一个名为 chignolin 的小型测试蛋白质，这是一个 10 个残基的微型蛋白质，常被用作蛋白质结构预测的基准。其小巧的尺寸使其非常适合快速实验和学习。

来源：[README.md#L27-35](README.md#L27-35)

## 首次蛋白质结构生成

让我们生成您的第一个蛋白质结构！我们将使用 BioEmu 的命令行界面为 chignolin 蛋白质序列创建 10 个样本结构。

```bash
python -m bioemu.sample --sequence GYDPETGTWG --num_samples 10 --output_dir ~/test-chignolin
```

此命令告诉 BioEmu：
- 使用氨基酸序列 "GYDPETGTWG"（chignolin）
- 生成 10 个不同的结构样本
- 将结果保存在 `~/test-chignolin` 目录中

或者，您可以使用 Python API 以获得更大的灵活性：

```python
from bioemu.sample import main as sample
sample(sequence='GYDPETGTWG', num_samples=10, output_dir='~/test_chignolin')
```

首次运行此命令时，模型参数将自动从 Hugging Face 下载。BioEmu 支持两个模型版本：
- `bioemu-v1.0`：原始预印本中使用的检查点
- `bioemu-v1.1`：改进版本，具有更好的蛋白质稳定性性能

默认情况下，BioEmu 使用 `bioemu-v1.1`，它在大多数用例中提供最佳结果。

来源：[README.md#L38-51](README.md#L38-51), [sample.py#L44-47](src/bioemu/sample.py#L44-47)

## 理解采样过程

让我们深入了解 BioEmu 如何生成蛋白质结构。采样过程涉及几个关键步骤，它们协同工作以将随机噪声分布转化为真实的蛋白质结构。

### 步骤 1：序列处理和嵌入生成

首先，BioEmu 处理您的氨基酸序列并生成捕获进化和结构信息的嵌入。这是使用 ColabFold 完成的，它会创建：

1. **单残基嵌入**：捕获每个氨基酸信息的残基级特征
2. **成对嵌入**：捕获残基对之间关系的特征

这些嵌入至关重要，因为它们为模型提供了关于蛋白质序列的生物学背景，包括进化约束和潜在的相互作用模式。

```python
# 这会在后台自动发生
single_embeds, pair_embeds = get_colabfold_embeds(
    seq=sequence,
    cache_embeds_dir=cache_embeds_dir,
    msa_file=msa_file,
    msa_host_url=msa_host_url,
)
```

来源：[sample.py#L184-189](src/bioemu/sample.py#L184-189), [sample.py#L176-216](src/bioemu/sample.py#L176-216)

### 步骤 2：图表示

BioEmu 将蛋白质表示为**ChemGraph**，这是一种特殊的图结构，包含：

- `pos`：每个残基的 3D 坐标（以纳米为单位）
- `node_orientations`：表示每个残基方向的 3×3 旋转矩阵
- `edge_index`：残基之间的连接信息
- `single_embeds` 和 `pair_embeds`：来自步骤 1 的进化嵌入
- `sequence`：原始氨基酸序列

这种图表示使 BioEmu 能够使用几何深度学习技术高效处理蛋白质结构。

来源：[chemgraph.py#L12-18](src/bioemu/chemgraph.py#L12-18)

### 步骤 3：基于扩散的结构生成

BioEmu 功能的核心是基于扩散的采样过程。这包括：

1. **从噪声开始**：过程从随机的 3D 坐标和方向开始
2. **渐进去噪**：使用训练好的神经网络逐步精修结构
3. **基于分数的指导**：在每一步，模型预测如何修改结构使其更像蛋白质

BioEmu 提供两种主要的去噪算法：

- **DPM（扩散概率模型）**：一种快速高效的采样器，适用于大多数蛋白质
- **Heun**：一种更准确但稍慢的采样器，可以产生更高质量的结果

```python
# 去噪过程
sampled_chemgraph_batch = denoiser(
    sdes=sdes,
    device=device,
    batch=context_batch,
    score_model=score_model,
)
```

去噪过程处理两种类型的数据：
- **位置**：每个残基的 3D 坐标
- **节点方向**：每个残基局部坐标系的旋转方向

来源：[denoiser.py#L143-154](src/bioemu/denoiser.py#L143-154), [denoiser.py#L259-269](src/bioemu/denoiser.py#L259-269)

### 步骤 4：结构过滤和输出

生成结构后，BioEmu 会自动过滤掉不合理的样本，这些样本可能具有：
- 空间冲突（原子过于接近）
- 链不连续（主链断裂）
- 其他几何不一致性

剩余的结构以两种格式保存：
- **PDB 文件**：用于可视化的标准蛋白质结构格式
- **XTC 文件**：包含所有生成结构的轨迹格式

```python
# 将样本转换为 PDB 和 XTC 格式
save_pdb_and_xtc(
    pos_nm=positions,
    node_orientations=node_orientations,
    topology_path=output_dir / "topology.pdb",
    xtc_path=output_dir / "samples.xtc",
    sequence=sequence,
    filter_samples=filter_samples,
)
```

来源：[sample.py#L165-172](src/bioemu/sample.py#L165-172), [README.md#L60-61](README.md#L60-61)

## 高级配置选项

虽然默认设置适用于许多蛋白质，但 BioEmu 提供了几个高级选项用于微调采样过程：

### 批量大小优化

BioEmu 根据蛋白质长度自动调整批量大小以优化内存使用。`batch_size_100` 参数控制 100 个残基蛋白质的批量大小，实际批量大小计算如下：

```python
batch_size = int(batch_size_100 * (100 / len(sequence)) ** 2)
```

这种二次方缩放考虑了较长蛋白质增加的计算复杂性。

### 去噪器配置

您可以通过提供自己的配置文件或选择不同的去噪器类型来自定义去噪过程：

```python
# 使用 Heun 去噪器代替 DPM
sample(
    sequence='GYDPETGTWG', 
    num_samples=10, 
    output_dir='~/test_chignolin',
    denoiser_type='heun'
)
```

### 使用自定义 MSA 文件

如果您有自己的多序列比对（MSA）数据，可以将其作为 A3M 文件提供，而不是依赖 ColabFold 的自动 MSA 生成：

```python
# 使用自定义 MSA 文件
sample(
    sequence='path/to/your/msa.a3m',
    num_samples=10,
    output_dir='~/test_chignolin'
)
```

来源：[sample.py#L42-52](src/bioemu/sample.py#L42-52), [sample.py#L124-128](src/bioemu/sample.py#L124-128)

## 性能考虑

生成蛋白质结构所需的时间取决于几个因素：

| 序列长度 | 时间（分钟） | 说明 |
|----------------|----------------|-------|
| 100 个残基 | ~4 分钟 | 在 A100 GPU 上快速生成 |
| 300 个残基 | ~40 分钟 | 中等计算时间 |
| 600 个残基 | ~150 分钟 | 需要大量计算资源 |

这些时间是在使用 `batch_size_100=20` 设置的 80 GB VRAM A100 GPU 上测量的。您的实际时间可能因硬件配置而异。

<CgxTip>
对于较长的蛋白质（>300 个残基），考虑减少批量大小或使用更多内存的 GPU 以避免内存不足错误。
</CgxTip>

来源：[README.md#L53-58](README.md#L53-58)

## 可视化结果

BioEmu 完成采样过程后，您会在输出目录中找到几个文件：

- `topology.pdb`：显示蛋白质序列的参考结构
- `samples.xtc`：包含所有生成结构的轨迹文件
- `batch_*.npz`：原始样本数据文件（中间格式）

要可视化这些结构，您可以使用分子可视化工具，如：
- **PyMOL**：强大的分子可视化程序
- **VMD**：可视化分子动力学
- **ChimeraX**：高级分子可视化和分析

例如，在 PyMOL 中加载您的结构：
```bash
pymol ~/test-chignolin/topology.pdb ~/test-chignolin/samples.xtc
```

这将加载参考结构和轨迹，让您能够播放不同的生成构象并分析样本的结构多样性。

## 后续步骤

恭喜！您已成功使用 BioEmu 生成了第一批蛋白质结构。以下是继续探索的一些方法：

1. **尝试不同的蛋白质**：尝试为不同长度和复杂性的蛋白质生成结构
2. **与实验结构比较**：如果您可以访问实验结构（例如来自 PDB），将它们与 BioEmu 的预测进行比较
3. **探索侧链重建**：使用 BioEmu 的侧链松弛工具为您的骨架结构添加原子细节
4. **分析结构多样性**：检查生成的结构集合以理解蛋白质的灵活性和动态特性

BioEmu 为蛋白质结构建模和分析开辟了令人兴奋的可能性。通过生成多样化的结构集合而非单一预测，它提供了更全面的蛋白质行为视图，这对于药物发现、蛋白质工程和基础生物学研究都具有重要价值。