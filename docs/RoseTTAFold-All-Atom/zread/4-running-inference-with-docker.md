---
slug:4-running-inference-with-docker
blog_type:normal
---


Docker 为 RoseTTAFold-All-Atom (RFAA) 提供了一种简化的手动环境设置替代方案，消除了创建复杂的 conda 环境和进行依赖管理的需求。本指南将引导初学者使用 Docker 容器来构建、配置和运行 RFAA 推理。

![RoseTTAFold-All-Atom Architecture](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/raw/main/img/RFAA.png)

## Docker 架构概览

基于 Docker 的方法将整个 RFAA 计算环境封装到一个可复现的容器中。该容器包含 CUDA 支持、Python 运行时、所有必需的科学库以及预安装的 RFAA 代码库。

下图说明了 Docker 推理工作流程：

```mermaid
flowchart TB
    A[主机系统] --> B[Docker 构建<br/>docker build . -t rfaa:latest]
    B --> C[Docker 镜像<br/>+ CUDA 11.8<br/>+ Python 3.11<br/>+ RFAA 依赖]
    
    C --> D[容器启动<br/>docker run --gpus all]
    
    D --> E[卷挂载]
    E --> E1[/workdir<br/>输入文件与配置]
    E --> E2[/mnt/databases/rfaa<br/>UniRef30 & BFD]
    E --> E3[/pdb100_2021Mar03<br/>模板数据库]
    E --> E4[/weights<br/>模型权重]
    
    E1 --> F[Hydra 配置<br/>docker.yaml]
    F --> G[run_inference.py]
    
    E2 --> G
    E3 --> G
    E4 --> G
    
    G --> H[ModelRunner 执行]
    H --> I[特征构建]
    I --> J[带循环的前向传播]
    J --> K[输出生成]
    
    K --> L[/workdir/outputs<br/>预测 PDB<br/>+ 辅助文件]
    
    style A fill:#e1f5ff
    style C fill:#fff4e1
    style G fill:#e8f5e9
    style L fill:#f3e5f5
```

<CgxTip>Docker 容器采用多阶段构建策略：首先拉取 micromamba 基础镜像，然后在其上叠加 NVIDIA 的 CUDA 11.8 支持，最后安装 RFAA 依赖。这既保持了最终镜像的优化，又确保了所有依赖的版本正确。</CgxTip>

## 系统要求

在开始基于 Docker 的推理之前，请确保你的系统满足以下先决条件：

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|-----------|-------------------|-------------|
| GPU | 支持 CUDA 11.8 的 NVIDIA GPU | RTX 3090, A100 或同等配置 |
| 显存 | 16 GB VRAM | 24 GB+ VRAM（用于大复合物） |
| 系统内存 | 32 GB | 64 GB+ |
| 磁盘空间 | 20 GB（用于容器） | 400 GB+（用于数据库） |

### 软件要求

- **Docker Engine**：版本 20.10 或更高，并已安装 NVIDIA Container Toolkit
- **NVIDIA 驱动**：版本 450.80.02 或更高（与 CUDA 11.8 兼容）
- **操作系统**：Linux（已测试 Ubuntu 20.04/22.04）

## 构建 Docker 镜像

Docker 镜像构建过程会创建一个包含所有预装依赖的独立环境。导航至仓库根目录并执行：

```bash
docker build . -t rosetta-fold-all-atom:latest
```

此命令会触发 [Dockerfile](Dockerfile#L1-L72) 中定义的多阶段构建过程，该过程包括：

1. **阶段 1**：设置来自 `mambaorg/micromamba:1.5.0` 的 micromamba 包管理器基础镜像
2. **阶段 2**：基于 `nvidia/cuda:11.8.0-base-ubuntu22.04` 构建，以提供 GPU 支持
3. **系统依赖**：安装基本工具，包括 aria2, wget, curl 和构建工具 [Dockerfile#L28-L37]
4. **Python 环境**：通过 micromamba 安装 Python 3.11 [Dockerfile#L38-L39]
5. **RFAA 安装**：复制仓库代码并安装 SE3Transformer 作为依赖 [Dockerfile#L44-L46]
6. **依赖安装**：执行 [install_dependencies.sh](install_dependencies.sh#L1-L23) 下载 cs-blast 用于序列处理
7. **BLAST 设置**：下载并配置旧版 BLAST 2.2.26 [Dockerfile#L53-L58]
8. **环境配置**：安装 [environment.yaml](environment.yaml#L1-L330) 中定义的所有 Python 包，包括 PyTorch 2.0.1, TensorFlow 2.11.0 和特定领域工具

<CgxTip>Docker 构建过程不在镜像中包含数据库或模型权重，以保持容器大小可控。这些必须在运行时作为外部卷挂载，正如在第 [68-70](Dockerfile#L68-L70) 行设置的环境变量所反映的那样。</CgxTip>

## 下载所需资源

运行推理之前，你必须在 Docker 之外下载并准备数据库和模型权重：

### 模型权重

下载预训练模型权重：

```bash
wget http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFAA_paper_weights.pt
```

### 序列数据库

下载 MSA 生成数据库（进行准确预测所必需）：

```bash
# UniRef30 数据库 (~46 GB)
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD 数据库 (~272 GB)
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# PDB 模板数据库 (~81 GB)
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz
```

来源：[README.md](README.md#L57-L76)

## 准备输入配置

RFAA 使用 Hydra 进行配置管理。Docker 示例配置提供在 [examples/docker/docker.yaml](examples/docker/docker.yaml#L1-L22) 中：

```yaml
defaults:
  - base
checkpoint_path: /weights/RFAA_paper_weights.pt
database_params:
  sequencedb: "/pdb100_2021Mar03/pdb100_2021Mar03"
  hhdb: "/pdb100_2021Mar03/pdb100_2021Mar03"
  command: make_msa.sh
  num_cpus: 4
  mem: 64
  job_name: "/workdir/3fap"

protein_inputs:
  A:
    fasta_file: /workdir/3fap_A.fasta
  B: 
    fasta_file: /workdir/3fap_B.fasta

sm_inputs:
  C:
    input: /workdir/ARD_ideal.sdf
    input_type: "sdf"
```

此配置演示了蛋白质-蛋白质-小分子复合物预测。要点如下：

1. **路径是相对于容器的**：所有路径都使用 `/workdir/` 前缀，该前缀在运行时映射到你的当前目录
2. **数据库路径**：引用 docker run 命令中定义的挂载卷位置
3. **链命名**：每个分子组件（蛋白质链、小分子）都需要唯一的单字符链标识符
4. **检查点位置**：模型权重路径指向挂载的权重卷

[base.yaml](rf2aa/config/inference/base.yaml#L1-L71) 配置提供了模型架构的默认参数，包括循环迭代次数 (MAXCYCLE: 4)、模板搜索设置 (n_templ: 4) 和序列处理约束。

## 使用 Docker 运行推理

核心推理执行遵循以下工作流程：

```mermaid
flowchart LR
    A[准备输入文件] --> B[创建/编辑 docker.yaml]
    B --> C[启动容器]
    C --> D[运行 run_inference.py]
    D --> E[ModelRunner 初始化]
    E --> F[加载模型与权重]
    F --> G[解析配置与构建特征]
    G --> H[执行前向传播]
    H --> I[生成输出]
    I --> J[访问 /workdir 中的结果]
    
    style A fill:#e3f2fd
    style C fill:#fff3e0
    style F fill:#f1f8e9
    style I fill:#f3e5f5
```

### 完整的 Docker 运行命令

从包含输入文件的目录执行以下命令：

```bash
docker run --gpus all \
    -v `pwd`:/workdir/ \
    -v $path_to_uniref30:/mnt/databases/rfaa/latest/UniRef30_2020_06/ \
    -v $path_to_bfd:/mnt/databases/rfaa/latest/bfd/ \
    -v $path_to_pdb100_2021Mar03:/pdb100_2021Mar03/ \
    -v $path_to_RFAA_paper_weights.pt:/weights/RFAA_paper_weights.pt \
    rosetta-fold-all-atom:latest \
    python -m rf2aa.run_inference -cd /workdir/ --config-name docker
```

来源：[README.md](README.md#L123-L131)

### 卷挂载详解

| 卷挂载 | 用途 | 容器路径 |
|-------------|---------|----------------|
| `pwd`:/workdir/ | 输入文件（FASTA, SDF）和配置 | `/workdir/` |
| UniRef30 路径 | MSA 生成数据库 | `/mnt/databases/rfaa/latest/UniRef30_2020_06/` |
| BFD 路径 | 大型序列数据库 | `/mnt/databases/rfaa/latest/bfd/` |
| PDB100 路径 | 模板数据库 | `/pdb100_2021Mar03/` |
| 权重路径 | 模型检查点文件 | `/weights/RFAA_paper_weights.pt` |

请将每个 `$path_to_*` 变量替换为主机系统上相应资源的绝对路径。

## 使用示例进行测试

仓库在 `examples/docker/` 目录中包含一个完整的示例。要测试你的 Docker 设置：

1. **导航至示例目录**：
   ```bash
   cd examples/docker/
   ```

2. **验证输入文件是否存在**：
   - `3fap_A.fasta` 和 `3fap_B.fasta`：蛋白质链序列
   - `docker.yaml`：预配置的推理设置
   - `3fap_aux.pt`：辅助数据文件

3. **运行示例**（请替换为实际的数据库路径）：
   ```bash
   docker run --gpus all \
       -v `pwd`:/workdir/ \
       -v /your/path/to/UniRef30_2020_06:/mnt/databases/rfaa/latest/UniRef30_2020_06/ \
       -v /your/path/to/bfd:/mnt/databases/rfaa/latest/bfd/ \
       -v /your/path/to/pdb100_2021Mar03:/pdb100_2021Mar03/ \
       -v /your/path/to/RFAA_paper_weights.pt:/weights/RFAA_paper_weights.pt \
       rosetta-fold-all-atom:latest \
       python -m rf2aa.run_inference -cd /workdir/ --config-name docker
   ```

来源：[README.md](README.md#L133-L134)

## 理解推理过程

Docker 容器执行 [run_inference.py](rf2aa/run_inference.py#L1-L210) 脚本，该脚本协调整个预测工作流程：

### ModelRunner 初始化

当推理开始时，[ModelRunner](rf2aa/run_inference.py#L21-L32) 类会初始化几个关键组件：

1. **化学数据**：从 [chemical.py](rf2aa/chemical.py) 加载原子类型定义和化学参数
2. **FFindex 数据库**：使用配置的 HHsearch 路径打开模板数据库
3. **设备检测**：自动检测并选择 CUDA GPU（如果可用）
4. **分子数据库**：加载用于小分子处理的理想 SDF 字符串

### 配置解析

[parse_inference_config](rf2aa/run_inference.py#L34-L94) 方法处理你的 YAML 配置：

```python
# 对于蛋白质输入（第 38-51 行）
protein_input = generate_msa_and_load_protein(
    self.config.protein_inputs[chain]["fasta_file"],
    chain,
    self
)

# 对于小分子（第 73-86 行）
sm_input = load_small_molecule(
    self.config.sm_inputs[chain]["input"],
    self.config.sm_inputs[chain]["input_type"],
    self
)
```

此阶段验证输入，生成 MSA（多序列比对），并加载分子结构。

### 模型执行

推理管道遵循以下步骤：

1. **特征构建** ([construct_features](rf2aa/run_inference.py#L112-L113))：将原始输入转换为模型兼容的张量
2. **前向传播** ([run_model_forward](rf2aa/run_inference.py#L115-L127))：执行带有循环迭代的神经网络
3. **输出生成** ([write_outputs](rf2aa/run_inference.py#L130-L149))：写入预测结构和置信度指标

模型默认使用 4 个循环周期（可通过 `MAXCYCLE` 参数配置），迭代地优化预测结构。

## 输出文件

成功完成后，容器会将输出写入 `/workdir/` 目录，该目录对应于主机系统上的当前工作目录：

### 主要输出

- **`{job_name}.pdb`**：预测的 3D 结构，每个残基的置信度分数存储在 B-factor 字段中

### 辅助输出

- **`{job_name}_aux.pt`**：包含详细预测指标的 PyTorch 文件，包括：
  - `plddts`：预测的局部距离差异测试分数（每个残基的置信度）
  - `pae`：预测的比对误差矩阵（残基间距离的置信度）
  - `pde`：小分子相互作用的预测距离误差
  - `mean_plddt`：全局置信度分数

来源：[run_inference.py](rf2aa/run_inference.py#L130-L149)

有关解读这些输出和置信度指标的详细指导，请参阅 [理解模型输出](5-understanding-model-outputs)。

## 常见问题排查

### 未检测到 GPU

**问题**：容器报告 "CUDA not available" 或回退到 CPU

**解决方案**：
- 验证 NVIDIA 驱动已安装：`nvidia-smi`
- 安装 NVIDIA Container Toolkit 并重启 Docker 守护进程
- 确保 docker run 命令中包含 `--gpus all` 标志
- 检查 CUDA 版本与你的 GPU 的兼容性

### 数据库路径错误

**问题**：数据库组件出现 FileNotFoundError

**解决方案**：
- 验证卷挂载中的绝对路径是否正确
- 确保数据库目录包含解压后的文件（而不是 tar.gz 归档文件）
- 检查 docker.yaml 中的环境变量是否与挂载的卷路径匹配

### 内存不足错误

**问题**：CUDA 内存不足或系统 RAM 耗尽

**解决方案**：
- 减小批次大小或序列长度
- 减少 `n_templ` 参数以使用更少的模板
- 使用显存更大的 GPU
- 减少 `num_cpus` 参数以限制并发进程数

### SignalP6 不可用

**注意**：由于许可限制，SignalP6 未包含在 Docker 容器中，在基于 Docker 的推理期间将不可用。这只影响依赖信号肽检测的预测，不会影响标准结构预测任务。

来源：[README.md](README.md#L135)

## 高级配置

### 自定义资源分配

在你的 docker.yaml 配置中修改资源参数：

```yaml
database_params:
  num_cpus: 8        # 增加以加快 MSA 生成
  mem: 128           # 为大型数据库分配更多 RAM
  
loader_params:
  MAXCYCLE: 6        # 更多循环迭代（精度更高，速度更慢）
  n_templ: 8         # 使用更多结构模板
```

来源：[base.yaml](rf2aa/config/inference/base.yaml#L20-L27)

### 替代输入类型

Docker 工作流程支持除示例之外的各种输入类型：

- **核酸预测**：在你的 YAML 中配置 `na_inputs`
- **共价修饰**：使用 `covale_inputs` 进行翻译后修饰
- **高阶复合物**：组合多条蛋白质链、核酸和小分子

请参阅 [rf2aa/config/inference/](rf2aa/config/inference) 中的配置示例，了解不同预测类型的模板。

## 后续步骤

成功运行基于 Docker 的推理后，探索以下主题以加深理解：

- **[理解模型输出](5-understanding-model-outputs)**：学习解读置信度指标并验证预测
- **[Hydra 配置管理](6-hydra-configuration-management)**：掌握针对复杂场景的高级配置组合
- **[蛋白质结构预测](9-protein-structure-prediction)**：仅蛋白质预测的详细指南
- **[蛋白质-小分子复合物预测](11-protein-small-molecule-complex-prediction)**：配体结合结构的专用工作流程

对于有兴趣扩展 Docker 工作流程的开发者，[安装和设置](3-installation-and-setup) 页面提供了手动环境设置过程的深入解析，这有助于针对特定需求自定义 Dockerfile。