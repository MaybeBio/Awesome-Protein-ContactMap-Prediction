---
slug:2-quick-start
blog_type:normal
---


几分钟内让 EquiFold 跑起来——从环境配置到预测你的第一个蛋白质结构。本指南将引导你完成安装、输入数据准备，以及本仓库内置的**抗体**和**微蛋白质**预测模式的推理流程。

## 前提条件

EquiFold 依赖 Python 3.9，以及 PyTorch 和 PyTorch Geometric 生态系统。本项目在配备 CUDA 11.3 的 **NVIDIA A100 GPU** 上开发并验证，同时也提供了仅 CPU 的回退配置。

**启用 GPU 的环境**（推荐）：

```bash
conda create -n ef python=3.9 -y
conda activate ef
conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch -y
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install e3nn pytorch-lightning biopython pandas tqdm einops
```

**仅 CPU 的环境**（用于无 GPU 环境下的测试）：

```bash
conda create -n ef python=3.9 -y
conda activate ef
conda install pytorch=1.12 -c pytorch -y
conda install pyg -c pyg
pip install e3nn pytorch-lightning biopython pandas tqdm einops
```

<CgxTip>安装 PyTorch Geometric 扩展库（`torch-scatter`、`torch-sparse` 等）时，`-f` 标志必须指向与你实际的 PyTorch + CUDA 版本相匹配的 URL。版本不匹配的构建会导致推理阶段的 scatter/gather 操作发生隐蔽的运行时故障。</CgxTip>

来源: [README.md](/README.md#L10-L24)

## 逐步工作流

整个预测流水线——从原始序列到压缩的 PDB——遵循以下流程：

```mermaid
flowchart LR
    A["📋 准备输入 CSV"] --> B["⚙️ 加载模型权重 + 配置"]
    B --> C["🧬 将序列编码为 CG 特征"]
    C --> D["🔄 运行 EquiFold 推理"]
    D --> E["📦 写入 .pdb.gz 输出"]
    
    style A fill:#e8f4fd,stroke:#1a73e8
    style B fill:#e8f4fd,stroke:#1a73e8
    style C fill:#fef7e0,stroke:#f9ab00
    style D fill:#fef7e0,stroke:#f9ab00
    style E fill:#e6f4ea,stroke:#1e8e3e
```

每个步骤直接映射到 [run_inference.py](/run_inference.py) 中的代码：通过 pandas 读取输入 CSV，从 `models/` 目录加载模型权重，经由 `sequence_to_feats` 将序列编码为粗粒度特征，`NN` 模型执行迭代精修，最后将预测的原子坐标写为 gzip 压缩的 PDB 文件。

来源: [run_inference.py](/run_inference.py#L1-L103), [utils_data.py](/utils_data.py#L424-L471)

## 预打包模型权重

本仓库在 `models/` 目录中提供了两种预训练模型变体，无需任何训练即可直接使用：

| 模型 | 配置文件 | 权重文件 | 领域 | 迭代模块 |
|-------|-------------|--------------|--------|-----------------|
| `ab` | `ab_config.json` | `ab_weights.pt` | 抗体（重链 + 轻链） | 6 |
| `science` | `science_config.json` | `science_weights.pt` | 微蛋白质（单链） | 4 |

每个配置 JSON 都包含超参数（学习率、通道维度、交互类型等），在加载时会作为 `**kwargs` 直接传递给 `NN` 构造函数。关键的架构差异在于：抗体模型使用 **6 个迭代精修模块**，截断半径为 64Å；而 science 模型使用 **4 个模块**，截断半径为 32Å——这反映了两个领域不同的结构复杂度。

来源: [models/ab_config.json](/models/ab_config.json#L1-L1), [models/science_config.json](/models/science_config.json#L1-L1), [run_inference.py](/run_inference.py#L60-L66)

## 准备输入数据

EquiFold 接受 **CSV 文件**作为输入序列。具体格式因模型而异：

### 抗体模型（`--model ab`）

| 列名 | 是否必需 | 描述 |
|--------|----------|-------------|
| `uid` | ✅ | 每个抗体的唯一标识符（用作输出文件名） |
| `heavy` | ✅ | 重链的氨基酸序列 |
| `light` | ✅ | 轻链的氨基酸序列 |

**示例**（`tests/data/inference_ab_input.csv`）：

```csv
uid,heavy,light
6mh2,EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS,DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK
```

对于抗体，重链和轻链序列会独立处理为粗粒度特征，然后以 `len(seq1) + MAX_DIST` 个残基的**空间偏移量**进行拼接，以防止两条链之间发生边缘重叠。

### 微蛋白质模型（`--model science`）

| 列名 | 是否必需 | 描述 |
|--------|----------|-------------|
| `uid` | ✅ | 每个蛋白质的唯一标识符 |
| `seq` | ✅ | 氨基酸序列（单链） |

**示例**（`tests/data/inference_science_input.csv`）：

```csv
uid,seq
EHEE_rd3_0145,GSSEQTYTFDNSKQAKKFAKELKKKGIPFQLHQKNGKWQVTKQ
```

<CgxTip>在单个 CSV 中，所有 `uid` 值必须**唯一**——推理脚本会强制断言此约束。重复的 UID 将导致运行时崩溃并抛出 `AssertionError`。</CgxTip>

来源: [tests/data/inference_ab_input.csv](/tests/data/inference_ab_input.csv#L1-L2), [tests/data/inference_science_input.csv](/tests/data/inference_science_input.csv#L1-L3), [run_inference.py](/run_inference.py#L68-L76)

## 运行推理

激活环境并准备好输入 CSV 后，使用 `run_inference.py` 入口脚本运行预测：

**抗体预测：**

```bash
python run_inference.py \
  --model ab \
  --model_dir models \
  --seqs tests/data/inference_ab_input.csv \
  --ncpu 1 \
  --out_dir out_tests
```

**微蛋白质预测：**

```bash
python run_inference.py \
  --model science \
  --model_dir models \
  --seqs tests/data/inference_science_input.csv \
  --ncpu 1 \
  --out_dir out_tests
```

### CLI 参数

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--model` | `ab` | 模型变体：`ab`（抗体）或 `science`（微蛋白质） |
| `--model_dir` | `models` | 包含 `{model}_weights.pt` 和 `{model}_config.json` 的目录 |
| `--seqs` | *(必需)* | 包含序列的输入 CSV 文件路径 |
| `--ncpu` | `1` | 用于并行计算输入特征的 CPU 进程数 |
| `--out_dir` | `out` | 预测 PDB 文件的输出目录 |

### 底层运行机制

推理脚本按顺序执行四个操作：

1. **模型加载** — 将配置 JSON 反序列化，并作为关键字参数传递给 `NN(**config)`，然后通过 `model.load_state_dict()` 加载预训练权重，并将模型设置为可用设备（CUDA 或 CPU）上的评估模式。
2. **特征计算** — 使用 `--ncpu` 个 worker 的多进程，将每条序列转换为粗粒度图特征（`cg_cgidx`、`cg_resnum`、scatter 索引、目标原子映射）。
3. **批量推理** — 带有自定义 `collate_fn` 的 `DataLoader` 每次向模型输入一个结构，设置 `compute_loss=False` 且 `return_struct=True`，从而提取最终迭代预测的坐标。
4. **PDB 输出** — 预测的原子坐标经由 `x_to_pdb()` 转换为 PDB 格式，并以 gzip 压缩文件（`{uid}.pred.pdb.gz`）的形式写入输出目录。

来源: [run_inference.py](/run_inference.py#L57-L103)

## 理解输出

成功运行后，输出目录中每个输入序列都会对应生成一个 **gzip 压缩的 PDB 文件**：

```
out_tests/
├── 6mh2.pred.pdb.gz          # 抗体预测
└── EHEE_rd3_0145.pred.pdb.gz # 微蛋白质预测
```

使用 `gunzip` 或 `zcat` 解压：

```bash
zcat out_tests/6mh2.pred.pdb.gz > 6mh2.pred.pdb
```

每个 PDB 文件包含标准的 `ATOM` 记录，带有所有重原子的预测 3D 坐标，并组织为单个模型（`MODEL 1`）。残基编号、原子名称和残基类型均通过粗粒度 scatter 映射重建，从 EquiFold 的粗粒度预测中生成完整的原子级结构。

> **已知局限**：该研究版本偶尔可能会产生具有非物理键几何结构或空间位阻冲突的结构。这些问题正着手在未来的版本中予以解决。

来源: [run_inference.py](/run_inference.py#L94-L103), [utils_data.py](/utils_data.py#L155-L198), [README.md](/README.md#L6-L7)

## 后续方向

既然你已能够运行预测，接下来可以探索其架构与内部机制：

- **[架构概述](3-architecture-overview)** — 了解完整的系统设计及各组件的连接方式
- **[粗粒度表示](4-coarse-grained-representation)** — 深入探讨将每个残基定义为一组刚体的新型 CG 方案
- **[输入数据流水线](10-input-data-pipeline)** — 了解原始序列如何转化为供神经网络使用的图结构特征
- **[推理与 PDB 输出](11-inference-and-pdb-output)** — 推理循环及坐标重建的详细演练