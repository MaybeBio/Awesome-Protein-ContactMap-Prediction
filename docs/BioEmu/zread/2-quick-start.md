---
slug:2-quick-start
blog_type:normal
---


欢迎使用 BioEmu！本指南将帮助您在几分钟内生成首批蛋白质结构样本。BioEmu 是一款强大工具，它采用扩散模型技术，根据氨基酸序列从蛋白质结构的平衡分布中进行采样。

## 您将完成的任务

完成本指南后，您将能够：
- 安装 BioEmu 及其依赖项
- 为给定氨基酸序列生成蛋白质结构样本
- 了解基本工作流程和输出格式
- 掌握查找生成结构的方法

## 前置要求

开始前，请确保您已具备：
- 已安装 **Python 3.10 或更高版本**
- **Linux 操作系统**（BioEmu 仅支持 Linux）
- **GPU 访问权限**（推荐用于加速采样，CPU 也可用但速度较慢）
- **8GB+ 内存**（小型蛋白质需求，大型蛋白质需要更多）

## 安装

使用 pip 安装 BioEmu 非常简单：

```bash
pip install bioemu
```

此命令将安装 BioEmu 及其所有核心依赖项，包括 PyTorch、生物信息学库和蛋白质结构生成所需的必要工具。
来源：[pyproject.toml#L11-L25](pyproject.toml#L11-L25), [README.md#L27-L32](README.md#L27-L32)

<CgxTip>
首次使用 BioEmu 时，它会自动在独立虚拟环境中设置 ColabFold，用于 MSA（多序列比对）和嵌入生成。此一次性设置将在 `~/.bioemu_colabfold` 目录中完成。
</CgxTip>

## 您的首个蛋白质结构样本

让我们为一个简单的测试序列生成蛋白质结构。我们将使用 chignolin 微型蛋白质序列（10 个氨基酸）作为示例。

### 方法 1：命令行界面

最快捷的入门方式是使用命令行界面：

```bash
python -m bioemu.sample --sequence GYDPETGTWG --num_samples 10 --output_dir ~/test-chignolin
```

此命令指示 BioEmu：
- 使用氨基酸序列 `GYDPETGTWG`
- 生成 **10 个不同的结构样本**
- 将结果保存到 `~/test-chignolin` 目录
来源：[README.md#L39-L42](README.md#L39-L42), [sample.py#L268-L275](src/bioemu/sample.py#L268-L275)

### 方法 2：Python API

如需程序化访问，可以使用 Python API：

```python
from bioemu.sample import main as sample

sample(
    sequence='GYDPETGTWG', 
    num_samples=10, 
    output_dir='~/test_chignolin'
)
```

此方法与命令行版本效果相同，但为集成到更大工作流程中提供了更多灵活性。
来源：[README.md#L44-L49](README.md#L44-L49), [sample.py#L39-L53](src/bioemu/sample.py#L39-L53)

## 理解采样过程

运行 BioEmu 时，后台会执行以下操作：

1. **模型下载**：BioEmu 自动从 HuggingFace 下载预训练模型权重（默认：`bioemu-v1.1`）
2. **序列处理**：验证并准备您的氨基酸序列
3. **嵌入生成**：ColabFold 为您的序列生成 MSA 和嵌入
4. **结构采样**：扩散模型生成蛋白质主链结构
5. **输出转换**：将结果转换为标准 PDB 和 XTC 格式

整个过程完全自动化，无需手动干预。BioEmu 会为您处理蛋白质结构生成的所有复杂步骤。
来源：[sample.py#L82-L89](src/bioemu/sample.py#L82-L89), [sample.py#L95-L99](src/bioemu/sample.py#L95-L99)

## 预期结果：输出文件

采样完成后，您的输出目录（`~/test-chignolin`）将包含：

| 文件 | 描述 | 用途 |
|------|------|------|
| `sequence.fasta` | 输入氨基酸序列 | 参考文件 |
| `topology.pdb` | 蛋白质拓扑信息 | 结构模板 |
| `samples.xtc` | 生成的结构轨迹 | 主要结果 |
| `batch_*.npz` | 原始采样数据（临时） | 中间文件 |

最重要的文件是 `topology.pdb` 和 `samples.xtc`，它们包含您生成的蛋白质结构。您可以使用 PyMOL、VMD 或 Chimera 等分子可视化工具查看这些文件。
来源：[sample.py#L165-L172](src/bioemu/sample.py#L165-L172), [convert_chemgraph.py](src/bioemu/convert_chemgraph.py)

<CgxTip>
BioEmu 会自动过滤掉非物理结构（存在空间位阻或链断裂的结构），因此输出样本数量可能少于请求数量。这属于正常现象，可确保结果质量。
</CgxTip>

## 采样时间预期

采样时间取决于蛋白质长度和硬件。以下是在 A100 GPU 上生成 1000 个样本的大致时间：

| 序列长度 | 预计时间 |
|----------|----------|
| 100 个残基 | 4 分钟 |
| 300 个残基 | 40 分钟 |
| 600 个残基 | 150 分钟 |

对于我们的 10 个残基示例，即使在 CPU 上采样也应在一分钟内完成。
来源：[README.md#L53-L58](README.md#L53-L58), [sample.py#L124-L128](src/bioemu/sample.py#L124-L128)

## 后续步骤

恭喜！您已成功使用 BioEmu 生成了首批蛋白质结构。以下是继续探索的几种方式：

- **尝试更长序列**：实验 100-300 个残基的蛋白质
- **调整采样参数**：使用不同去噪器（`--denoiser_type heun`）或批处理大小
- **可视化结果**：在您喜爱的分子查看器中加载 XTC 文件
- **探索高级功能**：了解侧链重建和 MD 弛豫

更详细的教程和高级用法，请参阅我们的[蛋白质结构建模教程](4-protein-structure-modeling-tutorial)。
来源：[sample.py#L43-L47](src/bioemu/sample.py#L43-L47), [config/denoiser/](src/bioemu/config/denoiser/)

## 常见问题排查

如遇到问题：

1. **安装失败**：确保您在 Linux 上使用 Python 3.10+
2. **内存错误**：减小 `batch_size_100` 参数或使用更短序列
3. **采样缓慢**：强烈建议使用 GPU 加速
4. **无输出文件**：检查输出目录的写入权限

如需更多帮助，请查阅[安装与设置](3-installation-and-setup)指南或在 GitHub 仓库创建问题。