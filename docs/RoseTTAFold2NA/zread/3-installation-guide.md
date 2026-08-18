---
slug:3-installation-guide
blog_type:normal
---


RoseTTAFold2NA是一款用于预测蛋白质-核酸复合物结构的强大工具。本指南将引导您完成完整的安装过程，从系统要求到验证步骤。通过本指南，您将拥有一个功能完备的RoseTTAFold2NA环境，随时可以进行结构预测。

## 系统要求

在安装RoseTTAFold2NA之前，请确保您的系统满足以下要求：

- **操作系统**：Linux（已在Ubuntu 18.04/20.04上测试）
- **GPU**：支持CUDA 11.7的NVIDIA GPU
- **内存**：至少64GB RAM（推荐用于数据库操作）
- **存储**：数据库和模型需要至少500GB可用磁盘空间
- **软件**：Conda包管理器（Miniconda或Anaconda）

<CgxTip>安装过程需要下载大型数据库（总计超过500GB）。在开始之前，请确保您有足够的磁盘空间和稳定的网络连接。</CgxTip>

## 分步安装指南

### 1. 克隆代码库

首先，从GitHub克隆RoseTTAFold2NA代码库：

```bash
git clone https://github.com/uw-ipd/RoseTTAFold2NA.git
cd RoseTTAFold2NA
```

这将创建代码库的本地副本，包含所有必需的脚本和配置文件。
来源：[README.md#L11-L15](README.md#L11-L15)

### 2. 创建Conda环境

RoseTTAFold2NA使用Conda环境管理依赖项。`RF2na-linux.yml`文件包含所有必需的包：

```bash
conda env create -f RF2na-linux.yml
```

此命令创建一个名为`RF2NA`的新Conda环境，包含Python 3.10和所有必需的包，包括：
- 支持CUDA 11.7的PyTorch
- 生物信息学工具（HHsuite、BLAST、HMMER、Infernal）
- 深度学习库（DGL、PyG）
- 序列比对工具（MAFFT、CD-HIT）

根据您的网络连接速度，此过程可能需要10-20分钟。
来源：[RF2na-linux.yml](RF2na-linux.yml)、[README.md#L17-L22](README.md#L17-L22)

### 3. 安装SE3 Transformer

RoseTTAFold2NA需要NVIDIA的SE(3)-Transformer进行3D等变变换。请单独安装：

```bash
conda activate RF2NA
cd SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ..
```

SE3 Transformer通过`e3nn`库和其他依赖项添加了专门的3D几何深度学习功能。
来源：[README.md#L23-L30](README.md#L23-L30)、[SE3Transformer/requirements.txt](SE3Transformer/requirements.txt)、[SE3Transformer/setup.py](SE3Transformer/setup.py)

### 4. 下载预训练权重

将预训练模型权重（1.1GB）下载到网络目录：

```bash
cd network
wget https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz
tar xvfz RF2NA_apr23.tgz
ls weights/  # 应包含1.1GB的权重文件
cd ..
```

这些权重包含结构预测的训练参数，是工具正常运行所必需的。
来源：[README.md#L32-L39](README.md#L32-L39)

## 数据库设置

RoseTTAFold2NA需要多个生物数据库进行序列比对和模板检测。请按以下顺序下载：

### 蛋白质数据库

```bash
# UniRef30 [46GB]
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD [272GB]
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# 结构模板
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz
```

### RNA数据库

```bash
mkdir -p RNA
cd RNA

# Rfam [300MB]
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.full_region.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz
gunzip Rfam.cm.gz
cmpress Rfam.cm

# RNAcentral [12GB]
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/rfam/rfam_annotations.tsv.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/id_mapping/id_mapping.tsv.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_species_specific_ids.fasta.gz
../input_prep/reprocess_rnac.pl id_mapping.tsv.gz rfam_annotations.tsv.gz  # 约8分钟
gunzip -c rnacentral_species_specific_ids.fasta.gz | makeblastdb -in - -dbtype nucl -parse_seqids -out rnacentral.fasta -title "RNACentral"

# NCBI nt数据库 [151GB]
update_blastdb.pl --decompress nt
cd ..
```

<CgxTip>数据库下载可能需要几小时甚至几天，具体取决于您的网络速度。建议在夜间下载或使用下载管理器处理大文件。</CgxTip>

这些数据库有不同的用途：
- **UniRef30**和**BFD**：用于生成蛋白质多序列比对（MSA）
- **PDB100**：用于结构模板检测
- **Rfam**和**RNAcentral**：用于RNA序列比对和注释
- **NCBI nt**：用于核苷酸序列搜索
来源：[README.md#L41-L77](README.md#L41-L77)

## 验证安装

使用提供的示例文件运行测试预测来验证您的安装：

```bash
conda activate RF2NA
cd example

# 测试蛋白质/RNA预测
../run_RF2NA.sh rna_pred rna_binding_protein.fa R:RNA.fa

# 测试蛋白质/DNA预测
../run_RF2NA.sh dna_pred dna_binding_protein.fa D:DNA.fa
```

### 预期输出

如果安装成功，您应该看到：

1. **日志文件**：在`log`目录中显示MSA生成和结构预测的进度
2. **模型输出**：在`models`子目录中：
   - `model_00.pdb`：预测的3D结构，B因子列中包含每个残基的置信度分数
   - `model_00.npz`：NumPy存档文件，包含：
     - `dist`（L × L × 37）：预测的距离直方图
     - `lddt`（L）：每个残基的置信度分数
     - `pae`（L × L）：预测的比对误差矩阵

预测过程通常需要10-30分钟，具体取决于序列长度和可用的计算资源。
来源：[README.md#L79-L100](README.md#L79-L100)、[run_RF2NA.sh](run_RF2NA.sh)

## 故障排除

### 常见问题

1. **CUDA错误**：确保您已安装CUDA 11.7和兼容的NVIDIA驱动程序
2. **内存问题**：如果遇到内存不足错误，请增加`run_RF2NA.sh`中的`MEM`变量
3. **数据库路径问题**：验证所有数据库都已解压到正确的目录中
4. **Conda环境激活**：运行预测前始终激活RF2NA环境

### 环境变量

脚本会自动设置必需的环境变量，但您可以自定义：
- `CPU`：使用的CPU数量（默认：8）
- `MEM`：最大内存（GB）（默认：64）
- `HHDB`：PDB数据库路径（自动设置为`pdb100_2021Mar03/pdb100_2021Mar03`）

这些参数可以在`run_RF2NA.sh`脚本中修改以匹配您的系统规格。
来源：[run_RF2NA.sh#L19-L20](run_RF2NA.sh#L19-L20)

## 后续步骤

成功安装和验证后，您可以：

1. **准备您自己的序列**：按照输入准备指南格式化您的蛋白质和核酸序列
2. **运行自定义预测**：使用已学习的命令结构预测您的序列结构
3. **探索高级功能**：了解配对的蛋白质/RNA预测和自定义参数设置

安装过程已完成，您现在拥有一个功能完备的RoseTTAFold2NA环境，可用于蛋白质-核酸结构预测！