---
slug:5-working-with-jupyter-notebooks
blog_type:normal
---


Jupyter笔记本提供了一种交互式的方式来实验AlphaFold2 PyTorch实现。本指南将逐步介绍包含的笔记本，解释如何使用它们进行蛋白质结构分析和预测。

## 仓库笔记本简介

lucidrains/alphafold2仓库在`notebooks/`目录中包含两个Jupyter笔记本：

1. **egnn_esm_end2end.ipynb**：一个端到端的实现，结合了AlphaFold2、EGNN（E(n)等变图神经网络）和ESM（进化尺度建模）嵌入，用于蛋白质结构预测
2. **structure_utils_tests.ipynb**：用于处理蛋白质结构分析、比较和排列的实用工具测试

这些笔记本展示了AlphaFold2实现的实际应用，同时为您自己的实验提供了基础。

来源：[notebooks/egnn_esm_end2end.ipynb](notebooks/egnn_esm_end2end.ipynb), [notebooks/structure_utils_tests.ipynb](notebooks/structure_utils_tests.ipynb)

## 开始使用笔记本

### 前置条件

在运行笔记本之前，请确保您已：

1. 安装了alphafold2-pytorch包及其依赖项
2. 能够访问GPU资源（推荐以提高性能）
3. 对蛋白质结构概念有基本的了解

### 环境设置

这两个笔记本都需要特定的Python包。`egnn_esm_end2end.ipynb`笔记本包含环境设置单元格，用于安装所有必需的依赖项：

```python
# 安装基本依赖项
!pip install sidechainnet proDy einops

# 安装PyTorch Geometric依赖项
!pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
!pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
!pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
!pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
!pip install torch-geometric

# 安装alphafold2-pytorch
!pip install alphafold2-pytorch
```

此设置在可用的情况下启用GPU加速，并配置所有必需的库。

来源：[notebooks/egnn_esm_end2end.ipynb#L56-L64](notebooks/egnn_esm_end2end.ipynb#L56-L64), [notebooks/egnn_esm_end2end.ipynb#L286-L290](notebooks/egnn_esm_end2end.ipynb#L286-L290)

## 使用结构工具测试笔记本

`structure_utils_tests.ipynb`笔记本展示了`alphafold2_pytorch/utils.py`中的实用函数，用于分析和比较蛋白质结构。

### 加载蛋白质结构

笔记本首先从PDB文件加载蛋白质结构：

```python
import mdtraj
import numpy as np
import torch

# 从PDB文件加载蛋白质结构
prot = mdtraj.load_pdb("data/1h22_protein_chain_1.pdb").xyz[0].transpose()
```

仓库在`notebooks/data/`目录中包含示例PDB文件，您可以使用它们进行测试。

来源：[notebooks/structure_utils_tests.ipynb#L44-L55](notebooks/structure_utils_tests.ipynb#L44-L55), [notebooks/structure_utils_tests.ipynb#L64-L65](notebooks/structure_utils_tests.ipynb#L64-L65)

### 计算结构比较指标

笔记本展示了用于比较蛋白质结构的各种指标，这些指标在NumPy和PyTorch版本中都有实现：

```python
# 创建一个轻微扰动的结构用于比较
pred = prot + (2*np.random.rand(*prot.shape) - 1) * 1

# 使用NumPy计算指标
rmsd     = RMSD(prot, pred)
gdt_ha   = GDT(prot, pred, mode="HA")
gdt_ts   = GDT(prot, pred, mode="TS")
tm_score = TMscore(prot, pred)

# 转换为PyTorch张量并再次计算
prot, pred = torch.tensor(prot), torch.tensor(pred)
rmsd     = RMSD(prot, pred)
gdt_ha   = GDT(prot, pred, mode="HA")
gdt_ts   = GDT(prot, pred, mode="TS")
tm_score = TMscore(prot, pred)
```

这些指标对于评估预测蛋白质结构的质量至关重要：
- **RMSD（均方根偏差）**：较低值表示结构相似度更高
- **GDT-TS（全局距离测试-总分）**：较高值（0-1）表示对齐更好
- **GDT-HA（全局距离测试-高精度）**：GDT-TS的更严格版本
- **TM-Score（模板建模评分）**：较高值（0-1）表示全局拓扑匹配更好

来源：[notebooks/structure_utils_tests.ipynb#L81-L82](notebooks/structure_utils_tests.ipynb#L81-L82), [notebooks/structure_utils_tests.ipynb#L106-L127](notebooks/structure_utils_tests.ipynb#L106-L127)

### 结构对齐和叠加

笔记本展示了Kabsch算法用于蛋白质结构的最佳叠加：

```python
# 创建一个旋转和扰动的结构
R = np.array([[0.25581, -0.77351, 0.57986],
              [-0.85333, -0.46255, -0.24057],
              [0.45429, -0.43327, -0.77839]])
pred = prot + (2*np.random.rand(*prot.shape) - 1) * 1 
pred = np.dot(R, pred)

# 使用PyTorch对齐结构
pred_mod_, prot_mod_ = kabsch_torch(torch.tensor(pred).double(), torch.tensor(prot).double())
rmsd_torch(prot_mod_, pred_mod_), tmscore_torch(prot_mod_, pred_mod_)

# 使用NumPy对齐结构
pred_mod, prot_mod = kabsch_numpy(pred, prot)
rmsd_numpy(prot_mod, pred_mod), tmscore_numpy(prot_mod, pred_mod)
```

Kabsch算法找到最优旋转矩阵，将一个结构叠加到另一个结构上，最小化RMSD。

来源：[notebooks/structure_utils_tests.ipynb#L153-L211](notebooks/structure_utils_tests.ipynb#L153-L211)

### 从距离矩阵重建3D结构

笔记本还展示了使用多维尺度（MDS）将距离矩阵转换回3D坐标：

```python
# 从3D坐标创建距离矩阵
dist_mat = torch.cdist(prot.t(), prot.t())

# 选择骨干原子进行手性校正
N_mask  = torch.tensor(prot_traj.topology.select("name == N and backbone")).unsqueeze(0)
CA_mask = torch.tensor(prot_traj.topology.select("name == CA and backbone")).unsqueeze(0) 
C_mask  = torch.tensor(prot_traj.topology.select("name == C and backbone")).unsqueeze(0)

# 从距离矩阵重建3D坐标
preds, stresses = MDScaling(dist_mat.cpu(), 
                           iters=5, tol=1e-5, fix_mirror=1, eigen=True,
                           N_mask=N_mask, CA_mask=CA_mask, C_mask=C_mask, verbose=2)
pred, stress = preds[0], stresses[0]
```

此过程对于输出距离矩阵而不是直接3D坐标的结构预测模型非常重要。

来源：[notebooks/structure_utils_tests.ipynb#L245-L247](notebooks/structure_utils_tests.ipynb#L245-L247), [notebooks/structure_utils_tests.ipynb#L276-L344](notebooks/structure_utils_tests.ipynb#L276-L344)

## 使用端到端模型笔记本

`egnn_esm_end2end.ipynb`笔记本展示了使用AlphaFold2结合EGNN和ESM嵌入进行蛋白质结构预测的完整工作流程。

### 加载蛋白质数据

笔记本使用SidechainNet库从CASP7数据集加载蛋白质数据：

```python
import sidechainnet
from sidechainnet.utils.sequence import ProteinVocabulary as VOCAB
VOCAB = VOCAB()

# 从SidechainNet加载数据
dataloaders_ = sidechainnet.load(casp_version=7, 
                                with_pytorch="dataloaders", 
                                batch_size=1, 
                                dynamic_batching=False)

# 根据长度过滤蛋白质
MIN_LEN = 100
MAX_LEN = 100000
train_examples_storer = [get_prot(dataloader_=dataloaders_, vocab_=VOCAB,
                                  min_len=MIN_LEN, max_len=MAX_LEN, verbose=0)
                         for i in range(3)]
```

这从CASP7数据集加载蛋白质结构和序列，并根据长度标准进行过滤。

来源：[notebooks/egnn_esm_end2end.ipynb#L511-L514](notebooks/egnn_esm_end2end.ipynb#L511-L514), [notebooks/egnn_esm_end2end.ipynb#L554-L560](notebooks/egnn_esm_end2end.ipynb#L554-L560)

### 初始化AlphaFold2模型

笔记本展示了如何初始化一个简化的AlphaFold2模型：

```python
from alphafold2_pytorch import Alphafold2
import alphafold2_pytorch.constants as constants
import alphafold2_pytorch.utils as af2utils

# 设置设备和参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
constants.DEVICE = device
DEVICE = constants.DEVICE

# 使用自定义参数初始化模型
model = Alphafold2(
    dim = 128,
    depth = 1,
    heads = 1,
    dim_head = 16,
    predict_coords = False,
    num_backbone_atoms = 4,
).to(DEVICE)
```

这创建了一个适合实验的简化AlphaFold2模型。参数可以根据您的计算资源和需求进行调整。

来源：[notebooks/egnn_esm_end2end.ipynb#L497-L500](notebooks/egnn_esm_end2end.ipynb#L497-L500), [notebooks/egnn_esm_end2end.ipynb#L678-L686](notebooks/egnn_esm_end2end.ipynb#L678-L686)

### 自定义蛋白质编码

笔记本实现了一个自定义蛋白质编码函数，将蛋白质序列和坐标转换为适合模型的特征：

```python
def encode_whole_protein(seq, true_coords, padding_seq, needed_info, free_mem=False):
    """对整个蛋白质进行编码为点和向量。"""
    device, precise = true_coords.device, true_coords.type()
    
    # 创建原子位置的掩码
    cloud_mask = torch.tensor(scn_cloud_mask(seq[:-padding_seq or None])).bool().to(device)
    flat_mask = rearrange(cloud_mask, 'l c -> (l c)')
    coords_wrap = rearrange(true_coords, '(l c) d -> l c d', c=14)[:-padding_seq or None]
    
    # 编码位置和原子身份
    # [编码原子位置和身份的代码]
    
    # 编码键
    # [编码键的代码]
    
    # 合并特征
    # [合并特征的代码]
    
    return whole_point_enc, whole_bond_idxs, whole_bond_enc, embedd_info
```

此函数将原始蛋白质数据转换为适合神经网络模型的结构化表示。

来源：[notebooks/egnn_esm_end2end.ipynb#L706-L759](notebooks/egnn_esm_end2end.ipynb#L706-L759)

## 使用笔记本的有效技巧

<CgxTip>
**性能提示**：对于大型蛋白质或复杂模型，考虑在配备强大GPU的机器上运行笔记本。AlphaFold2实现计算密集，GPU加速显著提高性能。笔记本会自动检测并使用可用的GPU。
</CgxTip>

### 管理内存

在处理蛋白质结构数据时，内存管理至关重要：

1. 在编码函数中使用`free_mem=True`选项以在可能的情况下释放内存
2. 在处理大型蛋白质之间调用`gc.collect()`以强制垃圾回收
3. 在内存有限时处理较小的批次或较小的蛋白质

### 定制模型

笔记本中的AlphaFold2模型可以通过调整各种参数进行定制：

```python
model = Alphafold2(
    dim = 256,           # 增加以获得更复杂的表示
    depth = 4,           # 增加以获得更深的网络
    heads = 8,           # 增加以获得更多的注意力头
    dim_head = 64,       # 调整注意力头的维度
    predict_coords = True, # 设置为True以直接预测坐标
    num_backbone_atoms = 4,
).to(DEVICE)
```

通过调整这些参数，可以在模型性能和计算需求之间取得平衡。

## 结论

lucidrains/alphafold2仓库中的Jupyter笔记本提供了强大的蛋白质结构分析和预测工具。`structure_utils_tests.ipynb`笔记本提供了用于比较和排列蛋白质结构的实用工具，而`egnn_esm_end2end.ipynb`笔记本展示了蛋白质结构预测的完整工作流程。

通过理解和扩展这些笔记本，您可以将AlphaFold2 PyTorch实现应用于自己的蛋白质结构预测任务，从学术研究到药物发现应用。

要进一步扩展这些示例，可以考虑：
- 结合来自多序列比对（MSAs）的进化信息等额外特征
- 对特定蛋白质家族进行微调
- 实现针对特定结构属性的定制损失函数
- 使用领域特定知识扩展模型，以适应您感兴趣的蛋白质