---
slug:25-structure-output-generation
blog_type:normal
---


结构输出生成是 RoseTTAFold-All-Atom 推理流程的最终阶段，在此阶段，模型预测的原子表示被转换为标准分子格式。该过程涉及从模型的内部表示中提取 3D 坐标、计算置信度指标，以及编写适合下游分析和可视化的结构化输出文件。

## 输出处理流程

输出生成在模型完成循环迭代的前向传播后开始。模型返回一个包含结构预测和置信度指标的综合输出元组。`ModelRunner.write_outputs()` 方法通过处理模型输出并将其转换为人类可读的格式来协调这最后阶段。该方法接收输入特征和模型输出，然后执行三个主要操作：置信度指标计算、链组织和文件写入。

**来源：** [run_inference.py](rf2aa/run_inference.py#L130-L152)

## 置信度指标计算

RoseTTAFold-All-Atom 产生三个不同的置信度指标，用于量化不同层级的预测可靠性。这些指标通过一个去分箱过程从分箱 logits 计算得出，该过程将离散概率分布转换为连续值。

### 预测 lDDT (Local Distance Difference Test)

lDDT 指标表示每个残基的置信度分数，范围为 0 到 1。模型在 `nbin` 个类别上输出分箱的 lDDT 预测值，这些预测值通过 softmax 进行非归一化处理，并使用间隔为 `1.0/nbin` 的分箱边缘进行加权平均，从而转换为连续值。较高的 pLDDT 值表示对每个残基的局部结构准确性有更大的信心。

**来源：** [run_inference.py](rf2aa/run_inference.py#L158-L163)

### 预测对齐误差 (PAE)

PAE 捕捉残基对之间的预期对齐误差，对于评估结构域级别的准确性至关重要。去分箱过程应用 0.5 Å 的分箱步长，分箱中心从 0.25 Å 开始向上延伸。PAE 预测结果是一个 2D 矩阵，其中 `PAE[i,j]` 表示将残基 i 叠加到残基 j 时的预期误差。

**来源：** [run_inference.py](rf2aa/run_inference.py#L165-L172)

### 预测距离误差 (PDE)

PDE 提供互补的距离准确性信息，具有更精细的 0.3 Å 分箱分辨率。该指标根据对称化的对特征计算得出，有助于评估整个结构中残基间距离预测的精度。

**来源：** [run_inference.py](rf2aa/run_inference.py#L174-L181)

### 汇总统计

`calc_pred_err()` 方法将这些原始指标聚合成汇总统计数据，包括平均 pLDDT、平均 PAE、蛋白质特异性 PAE（屏蔽小分子相互作用）和组分间 PAE（捕获蛋白质-配体界面准确性）。这些统计数据能够快速评估整体预测质量。

**来源：** [run_inference.py](rf2aa/run_inference.py#L181-L202)

## 链组织

在写入坐标之前，系统必须使用 `same_chain` 二进制矩阵将原子组织到适当的链中。该矩阵通过块对角结构指示哪些残基属于同一条链。`Ls_from_same_chain_2d()` 函数通过识别 `same_chain[i,j] = 1` 的连续块来提取链长度，返回一个用于 PDB 文件链字母分配的链长度列表。

**来源：** [run_inference.py](rf2aa/run_inference.py#L135-L138), [util.py](rf2aa/util.py#L891-L905)

## PDB 文件生成

坐标输出使用标准 PDB 格式，通过 `util.py` 中的 `writepdb()` 和 `writepdb_file()` 函数写入。系统在写入过程中区分标准蛋白质残基和小分子配体。

### 原子类型映射

系统维护了一个从内部数字索引到 PDB 标准原子名称的原子类型映射。此映射修正了早期模型版本中可能存在的原子编号分配问题。当 `remap_atomtype=True` 时，系统使用 `ChemData` 单例的 `atomnum2atomtype` 字典在模型的内部原子编号和标准 PDB 原子类型约定之间进行转换。

**来源：** [util.py](rf2aa/util.py#L346-L363)

### 蛋白质原子写入

对于标准蛋白质残基（索引 < len(ChemData.aa2long)），函数遍历 `ChemData.aa2long[seq]` 中定义的所有原子。每个原子都作为一个 ATOM 记录写入，格式包括："ATOM" 关键字、序列号、原子名称、残基名称、链标识符、残基编号、坐标、占用率 (1.0) 和 B-因子（截断至 [0, 1] 范围）。缺失的原子（被屏蔽或坐标为 NaN）将被跳过。

**来源：** [util.py](rf2aa/util.py#L396-L410)

### 配体原子写入

小分子配体（索引 >= len(ChemData.aa2long)）作为 HETATM 记录写入。由于 RFAA 字母表中不包含氢原子，系统会特殊处理氢原子。原子名称可以显式提供或从原子类型映射中提取。配体原子使用特殊的残基名称（默认为 'LG1'），并从 `idx_pdb.max() + 10` 开始编号，以避免与蛋白质残基编号冲突。

**来源：** [util.py](rf2aa/util.py#L375-L393)

### 键连接

当提供 `bond_feats` 时，系统会分析键特征矩阵以识别共价键（特征值为 1-4 的特征）。键记录作为 CONECT 条目写入，引用在写入期间存储的原子序列号。这保留了化学连接信息，对于小分子和共价修饰尤为重要。

**来源：** [util.py](rf2aa/util.py#L412-L420)

## 坐标数据结构

模型在 `xyz_allatom` 中输出原子坐标，其中包含所有原子的预测位置。该张量的形状为 `(1, L, NTOTAL, 3)`，其中 L 是序列长度，NTOTAL 是 36，代表 RFAA 字母表中每个残基的最大原子数。这些坐标是循环过程的最终产物，经过多次结构预测和细化迭代的精炼。

**来源：** [model/RoseTTAFoldModel.py](rf2aa/model/RoseTTAFoldModel.py#L375-L376)

## 输出文件

推理会在指定的输出目录中生成两个主要输出文件：

### PDB 文件 (`{job_name}.pdb`)

主要结构输出，包含：
- 所有蛋白质原子的 ATOM 记录，带有坐标和置信度 B-因子
- 小分子配体的 HETATM 记录
- 共价键的 CONECT 记录
- 基于输入序列组织的链分配

### 辅助文件 (`{job_name}_aux.pt`)

一个 PyTorch 序列化字典，包含：
- `plddts`：每残基置信度分数
- `pae`：预测对齐误差矩阵
- `pde`：预测距离误差矩阵
- `mean_plddt`：整体结构置信度
- `mean_pae`、`pae_prot`、`pae_inter`：分类的 PAE 统计数据

<CgxTip>
PDB 文件中的 B-因子直接编码了预测的 lDDT 置信度分数（截断至 [0,1]）。这些可以在分子查看器中可视化为颜色梯度，以识别高置信度和低置信度区域，通常较暖的颜色表示较高的置信度。
</CgxTip>

**来源：** [run_inference.py](rf2aa/run_inference.py#L140-L150)

## 与推理流程的集成

输出生成与基于循环的推理过程无缝集成。`run_model_forward()` 方法通过 `recycle_step_legacy()` 执行多个循环周期，迭代地细化结构预测。最后一次迭代的输出直接传递给 `write_outputs()`，确保生成的坐标代表模型最精炼的预测。

**来源：** [run_inference.py](rf2aa/run_inference.py#L115-L129), [training/recycling.py](rf2aa/training/recycling.py#L10-L48)

## 后续步骤

生成结构输出后，你可能需要探索：

- **[置信度指标计算](24-confidence-metrics-calculation)** - 详细了解如何解释 pLDDT、PAE 和 PDE 指标
- **[前向传播和循环](23-forward-pass-and-recycling)** - 结构如何通过循环迭代细化
- **[理解模型输出](5-understanding-model-outputs)** - 除坐标外的所有输出类型概述
- **[蛋白质-小分子复合物预测](11-protein-small-molecule-complex-prediction)** - 包含配体结构的特殊考虑因素