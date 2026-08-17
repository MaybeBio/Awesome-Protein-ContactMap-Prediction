---
slug:3-installation-options
blog_type:normal
---


ColabFold 提供多种安装选项，以满足不同用户的需求，从无需设置的云解决方案到高级本地安装。本指南涵盖了所有可用选项、其要求以及逐步安装说明。

## 快速参考表

| 安装方法 | 难度 | 要求 | 适用于 |
|----------|------|------|--------|
| Google Colab | ⭐ 最简单 | 互联网连接，Google 账户 | 初学者，快速预测，无需设置 |
| LocalColabFold | ⭐⭐ 简单 | 16GB+ RAM，20GB 磁盘空间 | 桌面用户，支持 Windows/macOS/Linux |
| Pip 安装 | ⭐⭐⭐ 中等 | Python 3.9+，依赖包 | 命令行用户，批量处理 |
| Docker 容器 | ⭐⭐⭐ 中等 | Docker，NVIDIA 驱动 | 隔离环境，可重复性 |
| MSA 服务器设置 | ⭐⭐⭐⭐⭐ 高级 | Linux 服务器，128GB+ RAM | 研究组，高通量分析 |

## 1. Google Colab（无需安装）

使用 ColabFold 最简单的方法是通过 Google Colab 笔记本，无需安装，在云端运行。

### 可用笔记本

| 笔记本 | 功能 | 链接 |
|--------|------|------|
| AlphaFold2_mmseqs2 | 单体、复合体、模板 | [在 Colab 中打开](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb) |
| AlphaFold2_batch | 批量处理 | [在 Colab 中打开](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2_batch.ipynb) |
| ESMFold | 快速预测 | [在 Colab 中打开](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb) |
| RoseTTAFold2 | 替代模型 | [在 Colab 中打开](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/RoseTTAFold2.ipynb) |

### 优点
- 无需设置
- 可访问免费的 GPU 资源
- 笔记本定期更新
- 简单的用户界面

### 限制
- 受限于 Colab 的资源和超时限制
- 最大序列长度取决于 GPU 分配（T4/P100 约 2000aa，K80 约 1000aa）
- 需要互联网连接

来源：[README.md](README.md)

## 2. LocalColabFold

LocalColabFold 是一个专用安装程序，将 ColabFold 功能带到本地机器，支持多种操作系统。

### 支持的平台
- Windows 10 或更高版本（使用 Windows 子系统 for Linux 2）
- macOS
- Linux

### 安装
按照 [LocalColabFold 仓库](https://github.com/YoshitakaMo/localcolabfold) 中的说明进行操作。

### 优点
- 用户友好的安装
- 本地运行（无超时）
- 跨平台支持
- 参数配置的 GUI

### 要求
- 推荐 16GB+ RAM
- 支持 CUDA 的 GPU（可选但推荐）
- 20GB+ 磁盘空间

来源：[README.md](README.md)

## 3. Python 包安装

对于命令行导向的用户，ColabFold 可以直接通过 pip 或 poetry 安装。

### 使用 pip

```bash
pip install "colabfold[alphafold]"
```

### 使用 poetry

```bash
# 如果尚未安装 poetry，请先安装
curl -sSL https://install.python-poetry.org | python3 -

# 克隆并安装
git clone https://github.com/sokrypton/ColabFold
cd ColabFold
poetry install -E alphafold

# 激活环境
source .venv/bin/activate

# 安装支持 CUDA 的 jax
pip install -q "jax[cuda]>=0.3.8,<0.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 可用命令

安装后，您将可以使用以下命令行工具：
- `colabfold_batch` - 运行结构预测
- `colabfold_search` - 使用 MMseqs2 生成 MSA
- `colabfold_split_msas` - 拆分 MSA 以进行分析
- `colabfold_relax` - 放松预测的结构

### 要求
- Python 3.9+
- pyproject.toml 中列出的依赖包
- 支持 CUDA 的 GPU（可选但推荐）

来源：[Contributing.md](Contributing.md), [pyproject.toml](pyproject.toml)

## 4. Docker 容器

Docker 提供一个隔离环境，所有依赖项均已预配置。

### 使用预构建镜像

```bash
docker pull ghcr.io/sokrypton/colabfold:1.5.5
docker run --gpus all -it ghcr.io/sokrypton/colabfold:1.5.5 colabfold_batch --help
```

### 从源代码构建

```bash
git clone https://github.com/sokrypton/ColabFold
cd ColabFold
docker build -t colabfold .
docker run --gpus all -it colabfold colabfold_batch --help
```

### 优点
- 跨机器环境一致
- 无依赖冲突
- 易于部署到不同系统
- 可重复性

### 要求
- 已安装 Docker
- NVIDIA 容器工具包（用于 GPU 支持）
- 支持 CUDA 的 GPU（可选但推荐）

来源：[Dockerfile](Dockerfile), [README.md](README.md)

## 5. MSA 服务器设置（高级）

对于研究组或高通量应用，设置专用 MSA 服务器可以提供更快、更高效的预测。

### 数据库设置

首先，设置必要的数据库（需要约 940GB 存储）：

```bash
MMSEQS_NO_INDEX=1 ./setup_databases.sh /path/to/db_folder
```

### 服务器安装选项

#### 选项 1：快速设置脚本

```bash
# 导航到 MsaServer 目录
cd MsaServer
# 运行设置脚本
./setup-and-start-local.sh
```

#### 选项 2：Systemd 服务（推荐用于生产）

1. 先运行设置脚本以下载必要组件
2. 修改 `systemd-example-mmseqs-server.service` 文件
3. 安装服务：

```bash
sudo cp systemd-example-mmseqs-server.service /etc/systemd/system/mmseqs-server.service
sudo systemctl daemon-reload
sudo systemctl start mmseqs-server.service
```

### 内存优化

为了最佳性能，数据库应常驻系统内存：

```bash
cd databases
sudo vmtouch -f -w -t -l -d -m 1000G *.idx
```

### 要求
- Linux 服务器
- 128GB+ RAM（推荐 768GB-1TB 用于预计算索引）
- 940GB+ 存储空间用于数据库
- 已安装 MMseqs2

### 与 ColabFold 配合使用

```bash
# 使用 colabfold_batch
colabfold_batch --host-url https://your-server.example.org input_sequences.fasta out_dir
```

来源：[MsaServer/README.md](MsaServer/README.md), [README.md](README.md)

## 6. GPU 加速 MSA 搜索

ColabFold 支持 GPU 加速的 MSA 搜索，通过 [MMseqs2-GPU](https://www.biorxiv.org/content/10.1101/2024.11.13.623350v1)。

### GPU 数据库设置

```bash
GPU=1 ./setup_databases.sh /path/to/db_folder
```

### 运行 GPU 搜索

```bash
colabfold_search --mmseqs /path/to/bin/mmseqs input_sequences.fasta /path/to/db_folder msas --gpu 1
```

### 可选 GPU 服务器

对于频繁搜索或最小延迟：

```bash
# 启动 GPU 服务器
mmseqs gpuserver /path/to/db_folder/colabfold_envdb_202108_db --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 &
PID1=$!
mmseqs gpuserver /path/to/db_folder/uniref30_2302_db --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 &
PID2=$!

# 使用 GPU 服务器运行搜索
colabfold_search /path/to/bin/mmseqs input_sequences.fasta /path/to/db_folder msas --gpu 1 --gpu-server 1
```

来源：[README.md](README.md)

## 选择合适的安装选项

<CgxTip>
**快速决策指南：**
- **只是试试？** 使用 Google Colab（无需设置）
- **常规桌面使用？** 使用 LocalColabFold（用户友好）
- **命令行/脚本？** 使用 pip 安装
- **需要可重复性？** 使用 Docker
- **大型研究组？** 设置 MSA 服务器
</CgxTip>

在选择安装选项时，请考虑：

1. **可用硬件** - GPU 加速显著提高性能
2. **技术专长** - 一些选项需要更高级的技能
3. **序列复杂性** - 更大或更复杂的蛋白质可能需要本地安装
4. **吞吐量需求** - 对于大量预测，考虑 MSA 服务器设置

## 常见安装问题排查

- **内存问题**：结构预测需要大量 RAM（推荐 16GB+）
- **CUDA 错误**：确保 NVIDIA 驱动与 CUDA 版本要求匹配
- **依赖冲突**：Docker 或虚拟环境可以帮助隔离依赖
- **性能缓慢**：推荐使用 GPU 加速以提高速度
- **MSA 生成失败**：考虑设置本地 MSA 服务器以保证可靠访问

如需进一步支持，请加入 [ColabFold Discord 频道](https://discord.gg/gna8maru7d) 与其他用户和开发者交流。