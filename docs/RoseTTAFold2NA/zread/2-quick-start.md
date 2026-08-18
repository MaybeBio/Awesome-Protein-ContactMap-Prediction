---
slug:2-quick-start
blog_type:normal
---


RoseTTAFold2NA 是一种先进的深度学习模型，用于预测蛋白质-核酸复合物的三维结构。本指南将帮助您在几分钟内完成首次预测。

## 您将完成的内容

通过本指南，您将能够：
- 安装 RoseTTAFold2NA 及其依赖项
- 为蛋白质-RNA 和蛋白质-DNA 复合物准备输入文件
- 运行您的首次结构预测
- 理解和解释输出文件

## 前置要求

开始前，请确保您具备：
- 搭载 NVIDIA GPU 的 Linux 系统（推荐）
- 已安装 Conda 包管理器
- 至少 100GB 可用磁盘空间用于存储数据库
- 可以下载数据库和模型权重的网络连接

## 安装

安装过程包括环境设置、依赖项下载以及获取必要的数据库。

### 1. 克隆仓库

首先，从 GitHub 克隆 RoseTTAFold2NA 仓库：

```bash
git clone https://github.com/uw-ipd/RoseTTAFold2NA.git
cd RoseTTAFold2NA
```

### 2. 创建 Conda 环境

RoseTTAFold2NA 需要特定的 Python 环境以及多种生物信息学工具。提供的 conda 环境文件可简化此过程：

```bash
conda env create -f RF2na-linux.yml
conda activate RF2NA
```

### 3. 安装 SE3 Transformer

SE3 Transformer 是三维结构预测的关键组件。使用包含的版本进行安装：

```bash
cd SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ..
```

### 4. 下载模型权重

下载预训练模型权重，其中包含结构预测的学习参数：

```bash
cd network
wget https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz
tar xvfz RF2NA_apr23.tgz
cd ..
```

<CgxTip>
模型权重文件约为 1.1GB。下载前请确保有足够的磁盘空间和稳定的网络连接。
</CgxTip>

### 5. 下载所需数据库（快速启动时可选）

为获得完整功能，RoseTTAFold2NA 需要多个生物数据库。虽然这些是获得最佳性能所必需的，但首次运行时可以使用提供的示例跳过此步骤：

```bash
# 蛋白质数据库
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# 结构模板
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz
```

来源：[README.md#L11-L39](README.md#L11-L39), [RF2na-linux.yml](RF2na-linux.yml)

## 首次预测

让我们使用提供的示例文件运行您的首次结构预测。我们将预测蛋白质-RNA 复合物结构。

### 理解输入格式

RoseTTAFold2NA 接受带有特定前缀的 FASTA 文件以指示分子类型：
- `P:` 表示蛋白质序列
- `R:` 表示 RNA 序列  
- `D:` 表示双链 DNA
- `S:` 表示单链 DNA

### 运行预测

导航到示例目录并运行预测脚本：

```bash
cd example
../run_RF2NA.sh rna_pred P:rna_binding_protein.fa R:RNA.fa
```

此命令指示 RoseTTAFold2NA：
1. 创建名为 `rna_pred` 的输出文件夹
2. 处理来自 `rna_binding_protein.fa` 的蛋白质序列
3. 处理来自 `RNA.fa` 的 RNA 序列
4. 预测复合物的三维结构

对于蛋白质-DNA 复合物，应使用：

```bash
../run_RF2NA.sh dna_pred P:dna_binding_protein.fa D:DNA.fa
```

来源：[run_RF2NA.sh#L22-L132](run_RF2NA.sh#L22-L132), [README.md#L84-L86](README.md#L84-L86)

### 预测过程说明

预测过程涉及多个自动化步骤：

1. **MSA 生成**：对蛋白质，使用 HHblits 创建多序列比对；对 RNA，使用专门工具生成比对。
2. **模板搜索**：对蛋白质，在 PDB 数据库中搜索结构同源物。
3. **联合处理**：预测蛋白质-RNA 复合物时，基于分类学信息合并 MSA。
4. **结构预测**：神经网络预测复合物中所有原子的三维坐标。

整个过程通常需要 10-30 分钟，具体取决于序列长度和可用计算资源。

来源：[run_RF2NA.sh#L28-L118](run_RF2NA.sh#L28-L118)

## 理解输出

成功完成后，您将在指定目录（如 `rna_pred/`）中找到多个输出文件：

### 主要输出文件

| 文件 | 描述 | 格式 |
|------|-------------|---------|
| `models/model_00.pdb` | 带置信度分数的预测三维结构 | PDB 格式 |
| `models/model_00.npz` | 详细预测数据 | NumPy 存档 |

### 解释 PDB 文件

PDB 文件包含预测的三维结构，其中：
- 原子坐标表示预测位置
- B 因子列包含每个残基的置信度分数（值越高表示越可信）
- 链标识符分隔不同分子（蛋白质、RNA、DNA）

您可以在任何分子查看器（如 PyMOL、Chimera 或 NGLView）中可视化此文件。

### 理解 NPZ 文件

NumPy 存档包含可使用 Python 加载的附加预测指标：

```python
import numpy as np
data = np.load('rna_pred/models/model_00.npz')

# 访问预测指标
distogram = data['dist']  # 预测距离矩阵 (L x L x 37)
lddt_scores = data['lddt']  # 每残基置信度分数 (L)
pae_matrix = data['pae']  # 预测比对误差 (L x L)
```

这些指标有助于评估预测质量：
- **LDDT 分数**：范围 0-100，表示局部置信度
- **PAE 矩阵**：显示残基对之间的预期误差
- **距离图**：预测残基间的距离分布

来源：[README.md#L93-L99](README.md#L93-L99), [predict.py#L24-L28](network/predict.py#L24-L28)

## 后续步骤

恭喜！您已成功使用 RoseTTAFold2NA 完成首次结构预测。以下是继续探索的几种方式：

1. **尝试您自己的序列**：使用您的蛋白质和核酸序列创建 FASTA 文件
2. **探索高级功能**：了解配对的蛋白质-RNA MSA 以提高准确性
3. **优化性能**：在运行脚本中调整 CPU 和内存参数
4. **可视化结果**：使用分子可视化工具分析您的预测

更多详细信息，请查看[安装指南](3-installation-guide)和[输入准备](5-input-preparation-for-proteins-and-rna)文档。

<CgxTip>
请记住，预测质量在很大程度上取决于输入序列质量和可用的同源序列。为获得最佳结果，请确保您的序列完整且错误最少。
</CgxTip>