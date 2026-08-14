---
slug:4-database-setup
blog_type:normal
---


设置所需的数据库是成功运行AlphaFold 3的关键步骤。本指南将指导您了解数据库需求、下载过程、优化技术以及这些数据库在预测工作流程中的使用方式。

## AlphaFold 3需要哪些数据库？

AlphaFold 3需要多个遗传和结构数据库来有效进行预测。这些数据库对于以下方面至关重要：

1. **生成蛋白质和RNA序列的多序列比对（MSA）**
2. **寻找蛋白质链的结构模板**
3. **支持配体构象生成**

以下是所需数据库的详细列表：

| 数据库 | 用途 | 压缩大小 | 未压缩大小 |
|--------|------|-----------|-------------|
| PDB mmCIF文件 | 结构信息 | ~50 GB | ~150 GB |
| BFD（小型） | 蛋白质序列数据库 | ~65 GB | ~185 GB |
| MGnify | 元基因组序列 | ~35 GB | ~95 GB |
| UniRef90 | 非冗余蛋白质簇 | ~25 GB | ~65 GB |
| UniProt | 综合蛋白质数据库 | ~50 GB | ~85 GB |
| PDB seqres | PDB序列 | ~0.1 GB | ~0.3 GB |
| RNACentral | 非编码RNA数据库 | ~10 GB | ~20 GB |
| NT | 核苷酸数据库 | ~15 GB | ~25 GB |
| RFam | RNA家族数据库 | ~2 GB | ~5 GB |

**总大小：** ~252 GB（压缩），~630 GB（未压缩）

<CgxTip>
💡 强烈建议使用SSD存储数据库，因为这可以显著提高遗传搜索阶段的搜索性能。
</CgxTip>

来源：[fetch_databases.sh](fetch_databases.sh), [docs/installation.md#obtaining-genetic-databases](docs/installation.md)

## 下载数据库

AlphaFold 3提供了一个便捷的脚本，自动化下载和设置所有所需数据库。以下是使用方法：

1. 确保已安装必要的工具：
```bash
# 对于基于Debian的系统
sudo apt install wget zstd
```

2. 运行数据库下载脚本：
```bash
cd alphafold3  # 导航到AlphaFold 3存储库目录
./fetch_databases.sh [<DB_DIR>]
```

如果未指定`<DB_DIR>`，默认位置为`$HOME/public_databases`。

<CgxTip>
⚠️ 下载目录不应是AlphaFold 3存储库的子目录。这会导致Docker构建变慢，因为它会在镜像创建过程中复制大型数据库。
</CgxTip>

该脚本从托管在Google Cloud Storage上的镜像下载数据库，所有版本与AlphaFold 3论文中使用的版本相同。由于总下载量较大（~252 GB），建议在`screen`或`tmux`会话中运行。

来源：[fetch_databases.sh](fetch_databases.sh), [docs/installation.md#obtaining-genetic-databases](docs/installation.md)

## 数据库目录结构

成功运行fetch脚本后，您的数据库目录应具有以下结构：

```
<DB_DIR>/
├── mmcif_files/          # 包含~200k PDB mmCIF文件的目录
├── bfd-first_non_consensus_sequences.fasta
├── mgy_clusters_2022_05.fa
├── nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
├── pdb_seqres_2022_09_28.fasta
├── rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
├── rnacentral_active_seq_id_90_cov_80_linclust.fasta
├── uniprot_all_2021_04.fa
└── uniref90_2022_05.fa
```

验证所有文件是否存在并具有适当的读取权限。如果遇到权限问题，您可能需要运行：

```bash
sudo chmod 755 --recursive <DB_DIR>
```

来源：[docs/installation.md#obtaining-genetic-databases](docs/installation.md)

## 使用SSD优化数据库性能

为了获得最佳性能，强烈建议使用SSD存储数据库，特别是在遗传搜索阶段。AlphaFold 3提供了两个辅助脚本用于此目的：

### 1. 挂载SSD（适用于云环境）

如果您使用的是带有附加SSD的云实例，需要挂载：

```bash
src/scripts/gcp_mount_ssd.sh [<SSD_MOUNT_PATH>]
```

该脚本将未挂载的驱动器挂载并格式化为指定路径。默认挂载路径为`/mnt/disks/ssd`。

### 2. 将数据库复制到SSD

要将数据库从原始位置复制到SSD：

```bash
src/scripts/copy_to_ssd.sh [<DB_DIR>] [<SSD_DB_DIR>]
```

该脚本尽可能多地将数据库文件复制到SSD。默认源目录为`$HOME/public_databases`，默认目标目录为`/mnt/disks/ssd/public_databases`。

如果您的SSD不足以容纳所有数据库，您仍可以通过将最频繁访问的数据库复制到SSD，并将其余数据库保留在较慢的存储上受益。AlphaFold 3支持指定多个数据库目录。

来源：[docs/installation.md#obtaining-genetic-databases](docs/installation.md)

## 在Docker或Singularity中使用数据库

在容器中运行AlphaFold 3时，您需要通过卷挂载使数据库对容器可访问。

### Docker示例：

```bash
docker run -it \
    --volume $HOME/af_input:/root/af_input \
    --volume $HOME/af_output:/root/af_output \
    --volume <MODEL_PARAMETERS_DIR>:/root/models \
    --volume <DB_DIR>:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output
```

### 使用多个数据库目录：

如果您有一些数据库在SSD上，而其他数据库在较慢的磁盘上，可以同时挂载并指定多个数据库目录：

```bash
docker run -it \
    # ... 其他卷 ...
    --volume <SSD_DB_DIR>:/root/public_databases \
    --volume <DB_DIR>:/root/public_databases_fallback \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    # ... 其他参数 ...
    --db_dir=/root/public_databases \
    --db_dir=/root/public_databases_fallback \
    --output_dir=/root/af_output
```

对于Singularity，使用`--bind`标志代替`--volume`即可。

来源：[docs/installation.md#building-the-docker-container-that-will-run-alphafold-3](docs/installation.md)

## AlphaFold 3如何使用数据库

了解AlphaFold 3如何利用这些数据库可以帮助您优化设置：

### 1. 多序列比对（MSA）构建

对于蛋白质和RNA链，AlphaFold 3通过搜索遗传数据库构建MSA：

- **蛋白质MSA**：使用Jackhmmer在UniRef90、BFD、MGnify、UniProt和PDB序列上进行搜索生成
- **RNA MSA**：使用Nhmmer在RNACentral、NT和RFam数据库上进行搜索生成

这些搜索识别出进化上相关的序列，为模型提供重要的共进化信息。

### 2. 模板搜索

对于蛋白质链，AlphaFold 3搜索PDB mmCIF文件以找到可能指导预测的结构模板。

### 3. 自定义输入与数据库搜索

用户可以选择：

- **自动生成MSA**：AlphaFold 3执行数据库搜索以构建MSA（默认）
- **自定义MSA输入**：用户提供预计算的MSA，跳过数据库搜索
- **无MSA模式**：不使用任何MSA信息运行

同样对于模板，用户可以：
- **使用自动模板搜索**：AlphaFold 3搜索PDB以找到模板
- **提供自定义模板**：跳过数据库搜索模板
- **无模板运行**：完全禁用模板使用

### 4. 数据库搜索过程

在自动生成MSA的情况下，过程如下：

1. 对于每个蛋白质链，Jackhmmer搜索每个蛋白质数据库
2. 对于每个RNA链，Nhmmer搜索每个RNA数据库
3. 结果被过滤和处理以构建MSA
4. 对于蛋白质链，HHsearch在PDB中查找匹配模板
5. 所有这些信息被输入模型进行预测

来源：[docs/input.md#multiple-sequence-alignment](docs/input.md), [docs/input.md#structural-templates](docs/input.md)

## 数据库问题排查

以下是一些常见的数据库相关问题和解决方案：

### 1. 磁盘空间不足

**问题**：由于磁盘空间不足，下载失败。
**解决方案**：确保在开始下载前至少有700 GB的空闲空间。

### 2. 遗传搜索缓慢

**问题**：MSA生成步骤耗时过长。
**解决方案**：使用提供的脚本将数据库移动到SSD。

### 3. 数据库文件缺失或损坏

**问题**：AlphaFold 3报告文件缺失或在数据库访问时失败。
**解决方案**：验证数据库结构并重新下载任何缺失或损坏的文件。

### 4. 权限错误

**问题**：出现关于访问或权限被拒绝的错误消息。
**解决方案**：确保所有数据库文件具有适当的读取权限：
```bash
sudo chmod 755 --recursive <DB_DIR>
```

### 5. 数据库路径配置

**问题**：AlphaFold 3找不到数据库。
**解决方案**：仔细检查Docker/Singularity命令中的路径，并确保卷挂载正确。

## 结论

正确设置数据库对AlphaFold 3的性能至关重要。尽管下载和设置过程需要大量的磁盘空间和时间，但提供的脚本使其相对简单。使用SSD存储数据库以获得最佳性能，特别是在遗传搜索阶段，是强烈推荐的。

通过了解AlphaFold 3如何使用这些数据库，您可以根据具体需求和可用资源做出明智的设置决策。