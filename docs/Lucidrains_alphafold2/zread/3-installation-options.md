---
slug:3-installation-options
blog_type:normal
---


<text>

本指南涵盖了AlphaFold2 PyTorch实现的全部安装方法，从基本设置到包含可选组件的高级配置。

## 基本安装

安装AlphaFold2 PyTorch实现的最简单方式是通过pip：

```bash
pip install alphafold2-pytorch
```

此命令会自动安装包及其核心依赖项。对于大多数只想探索基本功能的用户来说，这已经足够了。

来源：[README.md#L15-L17](README.md#L15-L17)

## 核心依赖项

通过pip安装包时，以下依赖项会自动安装：

| 类别 | 依赖项 |
|------|--------|
| **核心PyTorch** | torch>=1.6 |
| **AI/ML库** | einops>=0.3, En-transformer>=0.2.3, invariant-point-attention, transformers |
| **蛋白质科学** | mdtraj>=1.8, proDy, sidechainnet, biopython, mp-nerf>=0.1.5 |
| **3D处理** | pytorch3d |
| **实用工具** | numpy, requests, tqdm |

来源：[setup.py#L18-L33](setup.py#L18-L33)

## 高级安装选项

### 从源代码安装

对于想要贡献或修改代码的开发者：

```bash
git clone https://github.com/lucidrains/alphafold2.git
cd alphafold2
pip install -e .
```

运行测试并验证安装：

```bash
python setup.py test
```

来源：[setup.py#L34-L39](setup.py#L34-L39)

### 安装稀疏注意力支持

为了使用Microsoft Deepspeed的稀疏注意力（可以显著加速长序列的自注意力操作），你需要：

1. 使用提供的脚本安装Deepspeed：

```bash
sh install_deepspeed.sh
```

2. 安装Triton包：

```bash
pip install triton
```

<CgxTip>
**注意：** 稀疏注意力仅支持自注意力，不支持交叉注意力。在初始化模型时需要指定`max_seq_len`参数。
</CgxTip>

来源：[README.md#L390-L416](README.md#L390-L416)

### 预训练嵌入支持

要使用预训练嵌入（ESM, MSA Transformer或ProtTrans），你需要安装NVIDIA的apex库以优化操作：

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

这使你能使用来自预训练模型（如Facebook AI的ESM）的嵌入，从而提高性能。

来源：[README.md#L177-L187](README.md#L177-L187)

## 可选外部包

### 结构精修工具

对于最终的结构精修步骤，推荐使用以下工具：

1. **PyRosetta** - 用于快速放松协议：
   - 从[http://www.pyrosetta.org](http://www.pyrosetta.org)下载PyRosetta轮文件（需要学术邮箱）
   - 安装轮文件：
     ```bash
     cd downloads_folder
     pip install pyrosetta_wheel_filename.whl
     ```

2. **OpenMM Amber** - 用于分子动力学模拟：
   - 按照[ParmEd的OpenMM Amber文档](https://parmed.github.io/ParmEd/html/omm_amber.html)中的说明操作

来源：[README.md#L640-L646](README.md#L640-L646)

## 系统要求

- **Python版本**：3.7（如分类器中指定）
- **硬件**：虽然没有明确说明，但建议使用GPU加速，特别是对于较大的蛋白质结构

来源：[setup.py#L45](setup.py#L45)

## 安装故障排除

### pytorch3d依赖项的常见问题

pytorch3d依赖项有时难以安装。如果遇到问题：

1. 首先尝试直接从PyTorch生态系统安装：
   ```bash
   pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/$(python -c "import torch; print(torch.__version__)")/$(python -c "import torch; print(torch.version.cuda.replace('.', ''))")/index.html
   ```

2. 如果失败，你可能需要按照[pytorch3d的仓库](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)中的说明从源代码安装

### 其他依赖项

某些依赖项可能需要额外的系统包。在Ubuntu/Debian上：

```bash
sudo apt-get install -y build-essential cmake
```

## 开发环境

对于开发工作，建议设置虚拟环境：

```bash
python -m venv alphafold2-env
source alphafold2-env/bin/activate  # 在Windows上：alphafold2-env\Scripts\activate
pip install -e .
```

这为项目创建了一个隔离环境，并以可编辑模式安装。

## Docker安装

虽然仓库中没有正式提供，但你可以创建一个Docker容器以确保一致的环境：

```dockerfile
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# 安装系统依赖项
RUN apt-get update && apt-get install -y git build-essential cmake

# 克隆仓库
RUN git clone https://github.com/lucidrains/alphafold2.git
WORKDIR /alphafold2

# 安装包及其依赖项
RUN pip install -e .

# 安装可选依赖项
RUN pip install triton
RUN sh install_deepspeed.sh

# 添加任何其他自定义设置

# 设置工作目录
WORKDIR /workspace
```

保存为`Dockerfile`并构建：
```bash
docker build -t alphafold2-pytorch .
```

使用GPU支持运行：
```bash
docker run --gpus all -it alphafold2-pytorch
```
</text>