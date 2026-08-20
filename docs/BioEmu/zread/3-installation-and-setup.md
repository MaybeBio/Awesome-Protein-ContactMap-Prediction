---
slug:3-installation-and-setup
blog_type:normal
---


BioEmu 是一款强大的生物分子模拟器，可根据蛋白质单体的氨基酸序列，从其近似平衡结构分布中进行采样。本指南将指导您完成安装过程和初始设置，帮助您开始蛋白质结构建模工作。

## 前置要求

在安装 BioEmu 之前，请确保满足以下前置条件：

- **操作系统**：Linux（BioEmu 仅提供 Linux 版本）
- **Python**：3.10 或更高版本
- **硬件**：建议使用 GPU 以获得最佳性能（需兼容 CUDA 的驱动程序）
- **包管理器**：用于安装 Python 包的 pip

若要使用侧链重建和分子动力学松弛等高级功能，您还需要：
- **Conda**：用于处理复杂依赖关系的 conda 包管理器
- **CUDA 12 兼容驱动**：OpenMM 和 HPacker 功能的必需组件

## 基础安装

安装 BioEmu 的最简单方法是通过 pip：

```bash
pip install bioemu
```

此命令将安装 BioEmu 及其所有核心依赖项，包括：
- `mdtraj>=1.9.9`：用于分子轨迹分析
- `torch_geometric>=2.6.1` 和 `torch>=2.6.0`：用于深度学习功能
- `bio>=1.5.9`：用于生物序列处理
- `huggingface-hub`：用于模型权重管理
- 以及其他核心功能所需的必要包

来源：[pyproject.toml#L12-L25](pyproject.toml#L12-L25)、[README.md#L27-L32](README.md#L27-L32)

## 首次设置与自动依赖项

当您首次使用 BioEmu 进行结构采样时，它会在独立的虚拟环境中自动设置其他依赖项：

### ColabFold 设置

首次运行 BioEmu 时，它会自动设置 **ColabFold** 用于多序列比对（MSA）和嵌入生成。默认情况下，此设置使用 `~/.bioemu_colabfold` 目录。

```bash
# 首次使用时会自动执行
python -m bioemu.sample --sequence GYDPETGTWG --num_samples 10 --output_dir ~/test-chignolin
```

<CgxTip>
若要更改默认的 ColabFold 安装目录，请在首次使用前设置环境变量。设置过程是自动处理的，无需手动干预。
</CgxTip>

来源：[README.md#L34-L35](README.md#L34-L35)

## 可选功能安装

### 分子动力学和侧链重建

若要使用侧链重建和分子动力学松弛等高级功能，请安装可选依赖项：

```bash
pip install bioemu[md]
```

这将安装 `openmm[cuda12]==8.2.0` 及其他 MD 模拟所需的包。

首次使用侧链重建功能时，BioEmu 会在独立的 conda 环境中自动设置 **HPacker**：

```bash
# 侧链重建使用示例
python -m bioemu.sidechain_relax --pdb-path path/to/topology.pdb --xtc-path path/to/samples.xtc
```

<CgxTip>
HPacker 安装要求 conda 已添加到 PATH 环境变量中，并需要 CUDA 12 兼容驱动。首次调用此模块时，它会默认创建一个名为 "hpacker" 的独立 conda 环境。您可以通过在首次使用前设置 `HPACKER_ENV_NAME` 环境变量来自定义此名称。
</CgxTip>

来源：[pyproject.toml#L39-L41](pyproject.toml#L39-L41)、[README.md#L82-L98](README.md#L82-L98)、[sidechain_relax.py#L35-L36](src/bioemu/sidechain_relax.py#L35-L36)

### 开发环境安装

如果您计划为 BioEmu 做出贡献或修改代码，请安装开发依赖项：

```bash
pip install bioemu[dev]
```

这将安装用于测试、代码质量和开发工作流程的额外包：
- `pytest` 和 `pytest-cov`：用于测试
- `pre-commit`：用于代码质量检查

来源：[pyproject.toml#L34-L38](pyproject.toml#L34-L38)

## 安装验证

安装完成后，通过运行简单测试来验证 BioEmu 是否正常工作：

### 命令行界面测试

```bash
# 使用小型蛋白质序列进行测试
python -m bioemu.sample --sequence GYDPETGTWG --num_samples 10 --output_dir ~/test-chignolin
```

这应在指定的输出目录中生成蛋白质结构样本。您应该看到如下文件：
- `samples.xtc`：包含采样结构的轨迹文件
- `topology.pdb`：拓扑信息
- `sequence.fasta`：输入序列文件

### Python API 测试

```python
from bioemu.sample import main as sample
sample(sequence='GYDPETGTWG', num_samples=10, output_dir='~/test_chignolin')
```

如果安装成功，这些命令将无错误完成并生成预期的输出文件。

来源：[README.md#L38-L49](README.md#L38-L49)

## 替代安装方法

### Azure AI Foundry

对于偏好无需本地安装的云端解决方案的用户，BioEmu 已在 Azure AI Foundry 上提供：

1. 访问 [Azure AI Foundry 中的 BioEmu](https://ai.azure.com/explore/models/BioEmu/version/1/registry/azureml)
2. 使用您的 Azure 账户登录（如需要可免费创建）
3. 点击"部署"并选择项目
4. 选择"Standard_NC24ads_A100_v4"虚拟机实例
5. 点击"部署"并等待端点就绪（约 30 分钟）
6. 从"使用"选项卡复制端点 URL 和 API 密钥

然后使用 Python API 与部署的模型交互：

```python
import base64
import os
import requests

ENDPOINT_URL = "https://<ONLINE_ENDPOINT_NAME>.<REGION>.inference.ml.azure.com/score"
ENDPOINT_API_KEY = "<ONLINE_ENDPOINT_API_KEY>"
OUTPUT_DIR = os.path.expanduser("~/test-chignolin")

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": ("Bearer " + ENDPOINT_API_KEY),
}
data = {
    "input_data": {
        "sequence": "GYDPETGTWG",
        "num_samples": 10,
    }
}

response = requests.post(url=ENDPOINT_URL, headers=headers, json=data)
result = response.json()
# 处理并保存结果...
```

来源：[AZURE_AI_FOUNDRY.md#L5-L66](AZURE_AI_FOUNDRY.md#L5-L66)

## 常见问题排查

### 安装失败

如果在安装过程中遇到问题：

1. **Python 版本**：确保使用 Python 3.10 或更高版本
   ```bash
   python --version
   ```

2. **权限问题**：尝试使用用户权限安装
   ```bash
   pip install --user bioemu
   ```

3. **依赖冲突**：先创建虚拟环境
   ```bash
   python -m venv bioemu_env
   source bioemu_env/bin/activate
   pip install bioemu
   ```

### 运行时问题

1. **CUDA 兼容性**：对于 MD 功能，确保拥有 CUDA 12 兼容驱动
   ```bash
   nvidia-smi  # 检查 CUDA 版本
   ```

2. **内存问题**：BioEmu 可能占用大量内存。对于大型蛋白质，请考虑：
   - 减少采样数量（`--num_samples`）
   - 使用更小的批次大小
   - 确保足够的 GPU 内存（建议：80GB VRAM 以获得最佳性能）

3. **ColabFold/HPacker 设置**：如果自动设置失败：
   - 检查网络连接
   - 验证 conda 是否正确安装并添加到 PATH
   - 通过检查仓库中的设置脚本尝试手动设置

来源：[README.md#L53-L61](README.md#L53-L61)、[setup_sidechain_relax.sh#L1-L17](src/bioemu/hpacker_setup/setup_sidechain_relax.sh#L1-L17)

## 后续步骤

恭喜！您现已成功安装 BioEmu 并可以开始使用。以下是一些建议的后续步骤：

1. **尝试快速入门示例**：运行验证部分的示例来生成您的第一批蛋白质结构
2. **探索教程**：查看[蛋白质结构建模教程](4-protein-structure-modeling-tutorial)获取全面指导
3. **尝试不同序列**：为您感兴趣的蛋白质序列尝试结构采样
4. **探索高级功能**：如果安装了可选依赖项，请尝试侧链重建和 MD 松弛功能

请记住，BioEmu 专为蛋白质单体设计。对于多聚体结构，您可能需要探索替代方法或文档中提到的"连接子技巧"。