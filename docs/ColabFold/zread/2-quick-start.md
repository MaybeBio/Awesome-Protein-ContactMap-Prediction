---
slug:2-quick-start
blog_type:normal
---


ColabFold 通过 Google Colab 笔记本和本地安装，使蛋白质结构预测变得人人可及。本指南将帮助你在几分钟内运行你的第一次蛋白质结构预测。

## 选择你的路径

使用 ColabFold 有两种主要方式：

| 方法 | 优点 | 适用于 |
|------|------|--------|
| **Google Colab** | 无需安装，免费使用 GPU | 快速预测，初学者 |
| **本地安装** | 完全控制，批量处理 | 大型项目，隐私保护 |

## Google Colab 快速入门（3分钟）

使用 ColabFold 最快的方式是通过 Google Colab：

1. **打开 AlphaFold2 笔记本**：点击此链接：[AlphaFold2 with MMseqs2](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb)

2. **输入你的蛋白质序列**：
   - 在第一个单元格中输入你的氨基酸序列（或粘贴 FASTA 序列）
   - 对于蛋白质复合物，使用 `:` 分隔链（例如，`SEQUENCE1:SEQUENCE2`）
   - 给你的任务命名（可选）

   ```python
   query_sequence = 'PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK'
   jobname = 'my_protein'
   ```
   来源：[AlphaFold2.ipynb#L77-L79](AlphaFold2.ipynb#L77-L79)

3. **运行预测**：点击菜单中的“运行时” → “运行所有”

笔记本将：
- 安装所需的依赖项
- 下载适当的模型
- 生成多序列比对（MSA）
- 预测你的蛋白质结构
- 显示预测结果

<CgxTip>
对于小型蛋白质（<100 氨基酸），预测可能需要 1-5 分钟，对于更大的蛋白质和复合物则可能更久。免费的 Colab GPU 可以处理大约 1000-2000 氨基酸的序列，具体取决于分配的 GPU。
</CgxTip>

## 本地安装（10分钟）

对于常规使用或批量处理，请在本地安装 ColabFold：

### 选项 1：使用 LocalColabFold（推荐给大多数用户）

[LocalColabFold](https://github.com/YoshitakaMo/localcolabfold) 为 Windows（WSL2）、macOS 和 Linux 提供了一个简单的安装脚本：

```bash
# 按照此处的安装说明操作：
# https://github.com/YoshitakaMo/localcolabfold
```
来源：[README.md#L54-L55](README.md#L54-L55)

### 选项 2：使用 Poetry（适用于开发者）

```bash
# 安装 poetry
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.in-project true

# 克隆并安装 ColabFold
git clone https://github.com/sokrypton/ColabFold
cd ColabFold
poetry install -E alphafold

# 激活环境
source .venv/bin/activate

# 安装支持 CUDA 的 JAX
pip install -q "jax[cuda]>=0.3.8,<0.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
来源：[Contributing.md#L5-L28](Contributing.md#L5-L28)

### 选项 3：使用 Docker

对于基于 Docker 的安装，请按照 [wiki](https://github.com/sokrypton/ColabFold/wiki/Running-ColabFold-in-Docker) 中的说明操作。
来源：[README.md#L63-L64](README.md#L63-L64)

## 在本地运行你的第一次预测

安装完成后，运行你的第一次预测：

```bash
# 单个蛋白质预测
colabfold_batch input.fasta output_dir

# 对于蛋白质复合物，创建一个 CSV 文件，格式如下：
# id,sequence
# complex1,SEQUENCE1:SEQUENCE2
```
来源：[README.md#L70-L72](README.md#L70-L72)

### 优化资源使用

为了更好地利用 GPU，分开生成 MSA 和结构预测：

```bash
# 步骤 1：生成 MSA（CPU 密集型）
colabfold_batch input.fasta output_dir --msa-only

# 步骤 2：运行预测（GPU 密集型）
colabfold_batch input.fasta output_dir
```
来源：[README.md#L74-L77](README.md#L74-L77)

## 下一步做什么？

- **更多模型**：尝试 [ESMFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb)（更快但准确性较低）或 [RoseTTAFold2](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/RoseTTAFold2.ipynb)
- **批量处理**：使用 [AlphaFold2_batch](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2_batch.ipynb)
- **高级选项**：探索参数如模板、MSA 方法和解弛设置
- **大规模预测**：设置本地 MSA 数据库（参见 [Large Scale Structure Predictions](https://github.com/sokrypton/ColabFold/wiki/Large-scale-structure-prediction)）

## 关键术语

- **MSA**：多序列比对，进化相关的蛋白质序列集合
- **模板**：用于指导预测的已知蛋白质结构
- **循环**：预测算法的迭代次数（更多可以提高准确性）
- **解弛**：使用物理力场（Amber）优化预测结构