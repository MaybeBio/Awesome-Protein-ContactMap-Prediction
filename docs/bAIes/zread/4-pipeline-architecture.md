---
slug:4-pipeline-architecture
blog_type:normal
---


bAIes-IDP 流水线通过四个阶段将 AlphaFold-2（或 ColabFold）的结构预测转化为本质无序蛋白质的原子系综：**准备 → 预处理 → 转换 → 模拟**。每个阶段都在各自独立的工作目录中运行，接收明确的输入并产生可验证的输出——这使得该流水线既可审计又可恢复。该架构支持两种并行模式：**bAIes**（使用 AF2 距离直方图约束的贝叶斯信息系综）和 **Coil**（无实验约束的随机线团基线），二者共享通用的准备和转换主干，但在预处理和模拟阶段产生分歧。

来源: [README.md](/README.md#L1-L54), [tutorial/bAIes/README.md](/tutorial/bAIes/README.md#L1-L151)

## 端到端流程

下图展示了两种流水线变体在所有四个阶段中的完整数据流。实线箭头表示文件依赖关系；虚线箭头表示 bAIes 流水线独有的阶段。

```mermaid
flowchart TD
    subgraph Inputs["0. External Inputs"]
        AF2["AlphaFold-2 / ColabFold<br/>prediction outputs"]
        AF2 --> PDB_AF["relaxed_model.pdb<br/>(AF2 structure)"]
        AF2 --> DISTO["result_model.pkl<br/>or .npy distograms"]
    end

    subgraph Step1["1. Preparation"]
        PDB_AF --> S1["step1-prepare_gmx.bash<br/>gmx pdb2gmx + gmx trjconv"]
        S1 --> GRO["idp.gro"]
        S1 --> TOP["idp.top"]
        S1 --> ITP["idp.itp"]
        S1 --> PDB_GMX["idp.pdb"]
    end

    subgraph Step2["2. Preprocessing (bAIes only2)"]
        PDB_GMX --> S2["step2-preprocess.bash<br/>→ preprocess_bAIes.py"]
        PDB_AF --> S2
        DISTO --> S2
        S2 --> DAT[""baies_params.dat<br/>(f-itted restraint params)""]
        S2 --> NDX["atom_list.ndx<br/>(restrained atom list)"]
        S2 --> PLUMED["plumed.dat<br/>(PLUMED bias config)"]
    end

    subgraph Step3["3. Conversion"]
        GRO --> S3["step3-conversion.bash<br/>intermol + make_ff.py"]
        PDB_GMX --> S3
        TOP --> S3
        CMAP["cmap_20240524.cmap<br/>(dihedral correction maps)"] --> S3
        S3 --> LMP_IN["idp_nvt.in<br/>(LAMMPS input script)"]
        S3 --> LMP_DATA["idp_nvt.data<br/>(LAMMPS topology)"]
    end

    subgraph Step4["4. Simulation"]
        LMP_IN --> S4["lmp -in idp_nvt.in"]
        LMP_DATA --> S4
        CMAP --> S4
        PLUMED -.-> S4
        NDX -.-> S4
        DAT -.-> S4
        S4 --> TRAJ["traj_idp.xtc<br/>(ensemble trajectory)"]
    end

    style Step2 fill:#e8f4e8,stroke:#2d7d2d
    style Step4 fill:#e8e8f4,stroke:#2d2d7d
```

**Coil 流水线**完全省略了预处理阶段，直接从准备阶段进入转换阶段再到模拟阶段，生成一个无约束的随机线团系综，作为用于比较的具有物理意义的基线。

来源: [scripts/step1-prepare_gmx.bash](/scripts/step1-prepare_gmx.bash#L1-L7), [scripts/step2-preprocess.bash](/scripts/step2-preprocess.bash#L1-L39), [scripts/step3-conversion.bash](/scripts/step3-conversion.bash#L1-L33)

## 阶段 1：准备 — GROMACS 拓扑生成

准备阶段将 AlphaFold-2 的输出（具有非标准原子命名的 PDB 模型）桥接到 GROMACS 生态系统中，使用 **amber99SB-ILDN** 力场生成自洽的拓扑。该拓扑作为所有下游力场转换的唯一事实来源。

脚本 `step1-prepare_gmx.bash` 依次执行两条 GROMACS 命令。首先，`gmx pdb2gmx` 将 AF2 弛豫后的 PDB 转换为 GROMACS 坐标（`.gro`）、拓扑（`.top`）和包含拓扑（`.itp`）文件，选择水模型为 `none` 并忽略氢原子（`-ignh`）。其次，`gmx trjconv` 对结构重新居中，输出一个保留了 GROMACS 原子编号的干净 `.pdb` 文件——这是一个关键细节，因为 AF2 输出与经 GROMACS 处理的结构之间的原子索引可能不同，而此重新编号的 PDB 正是 PLUMED 为其 `MOLINFO` 指令所引用的文件。

| 输入 | 命令 | 输出文件 | 用途 |
|---|---|---|---|
| `relaxed_model.pdb` | `gmx pdb2gmx` | `idp.gro`, `idp.top`, `idp.itp` | 力场拓扑 + 坐标 |
| `idp.gro` | `gmx trjconv` | `idp.pdb` | 具有正确原子编号的重新居中 PDB |

来源: [scripts/step1-prepare_gmx.bash](/scripts/step1-prepare_gmx.bash#L1-L7), [tutorial/bAIes/README.md](/tutorial/bAIes/README.md#L37-L47)

## 阶段 2：预处理 — 距离直方图拟合与 PLUMED 生成

这是 bAIes 流水线的计算核心，也是其区别于随机线团模拟的阶段。预处理阶段接收 AF2 距离直方图（残基间距离的概率分布），将每个选定的残基对拟合至解析模型，并输出定义模拟期间施加的贝叶斯约束势的三个 PLUMED 配置文件。

### 距离直方图读取与残基对选择

`preprocess_bAIes.py` 脚本首先读取距离直方图文件——可以是 AlphaFold-2 的 `.pkl` pickle 文件（通过 `read_pkl`）或 ColabFold 的 `.npy` 文件（通过 `read_npy`）。AF2 距离直方图包含横跨 2.0–22.0 Å 的 **64 个分箱**；最后一个分箱（覆盖 >22 Å）因不可靠而被丢弃。原始 logit 通过 softmax 转换为概率分布。同时，解析 PDB 结构以提取每个残基的 CB 原子位置（甘氨酸则为 CA），建立距离直方图残基索引与 MD 原子编号之间的映射。

随后，残基对将通过三个条件进行**过滤**：(1) **序列间隔**截断值（`-seqsep`，默认为 3）移除相邻残基，以保留来自基础力场的螺旋信息；(2) **距离截断值**（可以是固定的 Å 值或残基对特定的 `cutoff_matrix`）移除最可能距离超过阈值的残基对；(3) **链过滤器**（`-chains`）可选地将多链系统限制为链内或链间残基对。

### 距离直方图拟合

每个选定残基对的概率分布均被拟合至解析模型。支持两种模型族：

| 模型 | 公式 | 适用场景 |
|---|---|---|
| **高斯** (`-model gauss`) | $A \cdot \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ | 默认模型；适用于单峰距离分布 |
| **对数正态** (`-model lognorm`) | $A \cdot \frac{1}{x\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)$ | 更适用于短距离处的偏态分布 |

对于**单峰**拟合（`-nmodes 1`），直接使用 `scipy.optimize.curve_fit`，并在随机初始条件下进行迭代回退。对于**多峰**拟合，`lmfit` 库在 1–N 个峰之间执行模型选择，选择简化 χ² 最接近 1 的拟合。距离直方图中的最可能距离（`dmax`）作为第一个峰的初始条件种子，以改善收敛性。

### PLUMED 文件组装

拟合完成后，`step2-preprocess.bash` 将组装带有四条 PLUMED 指令的最终 `plumed.dat` 文件：

```
#MOLINFO STRUCTURE=idp.pdb
batoms: GROUP NDX_FILE=atom_list.ndx NDX_GROUP=batoms
baies: BAIES ATOMS=batoms DATA_FILE=baies_params.dat PRIOR=JEFFREYS TEMP=2.478541306
PRINT ARG=baies.ene FILE=COLVAR STRIDE=500
bbias: BIASVALUE ARG=baies.ene STRIDE=2
```

`PRIOR` 关键字在架构上具有重要意义：**JEFFREYS** 选择 Jeffreys 先验用于完整的 bAIes 采样，而 **NONE** 则生成 bAIes-N 变体（无贝叶斯先验的距离直方图约束）。温度 `2.478541306` 是 LAMMPS 单位下的约化温度（以 kcal/mol 为单位的 k_BT）。

| 输出文件 | 内容 | 消费者 |
|---|---|---|
| `baies_params.dat` | 每个残基对的拟合 (μ, σ², 缩放系数) | PLUMED BAIES 操作 |
| `atom_list.ndx` | `[batoms]` 组下的原子索引 | PLUMED GROUP 操作 |
| `plumed.dat` | PLUMED 偏置配置 | LAMMPS (通过 PLUMED fix) |

<CgxTip>使用 ColabFold 距离直方图时，请确保使用 `*_distmat/` 子目录下的 `_prob_distributions.npy` 文件——这是概率分布，而非原始 logit。`read_npy` 函数假定输入为已转换的概率，且具有特定的 64 分箱网格（单位为 nm）。</CgxTip>

来源: [GROMACS](/scripts/preprocess_bAIes.py#L#L1-50), [0 [scripts/step2-preprocess.bash](E0-39), [tutorial/bAIes/README.md](/tutorial/bAIes/README.md#L49-L92)

## 阶段 3：转换 — GROMACS 至 LAMMPS 及力场简化

转换阶段执行两阶段转换：首先将 GROMACS 拓扑转化为 LAMMPS 格式，随后应用**力场简化**，即用最小的 LJ 截断加 CMAP 二面角校正图替换完整的 amber99SB-ILDN 非键相互作用。

### 阶段 A：Intermol 转换

`intermol` Python 库执行从 GROMACS（`.gro` + `.top`）到 LAMMPS（`.lmp` + `.input`）的机械式文件格式转换。这将产生中间文件——`idp_converted.input` 和 `idp_converted.lmp`——它们保留了完整的 amber99SB-ILDN 力场，包括所有对系数、特殊键校正以及 k 空间静电相互作用。

### 阶段 B：通过 `make_ff.py` 进致力场简化

`make_ff.py` 脚本（在脚本目录中称为 `remove_nonbonded_cmap_plumed.py`）读取中间 LAMMPS 文件并应用两项关键修改：

1. **非键简化**：带有长程静电的完整成对 LJ 相互作用被替换为截断半径为 2 Å 的 `lj/cut 2.0` 对势——这实际上移除了键合邻域之外的所有非键相互作用。所有对系数均被显式枚举（包括通过算术混合计算的交叉项），并且 `pair_modify shift yes` 确保了截断处的能量连续性。

2. **CMAP 注入**：从 `cmap_20240524.cmap` 中读取骨架 φ/ψ 二面角校正图，并将其注入到 LAMMPS 输入脚本（通过 `fix drycmap all cmap`）和数据文件（追加具有残基类型特定映射索引的 CMAP 交叉项）中。该校正图用捕获 Ramachandran 取向偏好的残基特异性二维网格替换了全力场的扭转分量。

3. **模拟样板**：该脚本追加默认的模拟设置——能量最小化、NVT 恒温器（298.1 K 下的 `temp/csvr`）、PLUMED fix、XTC 轨迹输出以及 2 μs 的生产运行。

模拟盒被设置为一个立方体，边长由 `-cube` 参数指定（教程中默认为 200 Å），并以原点为中心。

| 输入文件 | 操作 | 输出文件 |
|---|---|---|
| `idp.gro`, `idp.top` | `intermol.convert` | `idp_converted.input`, `idp_converted.lmp` |
| `idp_converted.*`, `idp.pdb`, `cmap_*.cmap` | `make_ff.py` | `idp_nvt.in`, `idp_nvt.data` |

<CgxTip>在编译前，必须将 LAMMPS 中的 CMAP 补丁（`installation/` 中的 `patch_cmap.txt`）应用于 `fix_cmap.cpp`——未打补丁的 LAMMPS CMAP 实现无法正确处理 bAIes 偏置交换所需的能量核算。详情参见 [LAMMPS CMAP 补丁](15-lammps-cmap-patch)。</CgxTip>

来源: [scripts/step3-conversion.bash](/scripts/step3-conversion.bash#L1-L33), [scripts/make_ff.py](/scripts/make_ff.py#L1-L50), [tutorial/bAIes/README.md](/tutorial/bAIes/README.md#L94-L122)

## 阶段 4：模拟 — 系综生产

模拟阶段将最终的 LAMMPS+PLUMED 系统收敛为一条采样自 bAIes 后验系综的轨迹。来自阶段 2 和 3 的所有文件被收集至同一目录下：

| 文件 | 来源 | 角色 |
|---|---|---|
| `idp_nvt.in` | 阶段 3 | LAMMPS 输入脚本（力场 + 积分器设置） |
| `idp_nvt.data` | 阶段 3 | LAMMPS 拓扑（键、角度、二面角、CMAP 交叉项） |
| `cmap_20240524.cmap` | 阶段 3 | 二面角校正图网格数据 |
| `plumed.dat` | 阶段 2 (仅 bAIes) | PLUMED 偏置配置 |
| `baies_params.dat` | 阶段 2 (仅 bAIes) | 拟合的距离直方图约束参数 |
| `atom_list.ndx` | 阶段 2 (仅 bAIes) | bAIes 约束的原子索引组 |
| `idp.pdb` | 阶段 1 | PLUMED MOLINFO 的结构参考 |

模拟通过 `lmp -in idp_nvt.in` 启动。输入脚本中可调的关键参数包括轨迹转储频率（`dump ... xtc <stride>`）、恒温器耦合以及总时间步数（`run <N>`）。模拟后分析可直接在 XTC 输出上利用 GROMACS 工具进行——例如，`gmx trjconv -f traj_idp.xtc -s idp.pdb -o traj_idp_dt100ps.xtc -dt 100` 用于时间跨步。

来源: [tutorial/bAIes/README.md](/tutorial/bAIes/README.md#L124-L151), [scripts/make_ff.py](/scripts/make_ff.py#L130-L165)

## 流水线变体：bAIes vs bAIes-N vs Coil

流水线架构支持三种不同的模拟模式，它们共享准备和转换基础设施，但在约束势和力场上有所不同：

| 特性 | bAIes | bAIes-N | Coil |
|---|---|---|---|
| **距离直方图约束** | ✅ 从 AF2 拟合 | ✅ 从 AF2 拟合 | ❌ 无 |
| **贝叶斯先验** | Jeffreys (`PRIOR=JEFFREYS`) | 无 (`PRIOR=NONE`) | 不适用 |
| **PLUMED 偏置** | BAIES + BIASVALUE | BAIES + BIASVALUE | 无 |
| **预处理阶段** | 必需 | 必需 (使用 `prior=NONE`) | 跳过 |
| **力场** | 简化 + CMAP | 简化 + CMAP | 简化 + CMAP |
| **物理解释** | 贝叶斯后验系综 | 受约束系综 | 无约束线团基线 |
| **教程路径** | `tutorial/bAIes/` | `tutorial/bAIes/` (编辑 prior) | `tutorial/coil/` |

**bAIes** 变体应用 Jeffreys 先验，该先验自然地惩罚过度精确的约束，产生一个兼顾 AF2 预测置信度与物理力场的后验分布。**bAIes-N** 移除此先验，按原值施加距离直方图约束——有助于对比，但对预测不确定性的鲁棒性较差。**Coil** 作为零模型：相同的简化力场但不包含任何 AF2 导出的偏置，代表随机聚合物在相同模拟条件下的形态。

来源: [scripts/step2-preprocess.bash](/scripts/step2-preprocess.bash#L13-L15), [tutorial/coil/README.md](/tutorial/coil/README.md#L1-L81)

## 文件流总结

下表追踪了流水线中的每一个文件，展示了其生产阶段、消费阶段和格式：

| 文件 | 格式 | 生产者 | 消费者 | 内容 |
|---|---|---|---|---|
| `result_model_*.pkl` | Pickle | AlphaFold-2 (外部) | 阶段 2 | 原始距离直方图 (logit + 分箱) |
| `*_prob_distributions.npy` | NumPy | ColabFold (外部) | 阶段 2 | 预转换的概率分布 |
| `relaxed_model_*.pdb` | PDB | AlphaFold-2 (外部) | 阶段 1 | AF2 弛豫结构 |
| `idp.gro` | GROMACS GRO | 阶段 1 | 阶段 3 | 坐标 + 盒子 |
| `idp.top` | GROMACS TOP | 阶段 1 | 阶段 3 | 系统拓扑 |
| `idp.itp` | GROMACS ITP | 阶段 1 | (被 .top 引用) | 分子拓扑 |
| `idp.pdb` | PDB | 阶段 1 | 阶段 2, 3, 4 | 重新编号的结构 |
| `baies_params.dat` | PLUMED DAT | 阶段 2 | 阶段 4 | 拟合的约束参数 |
| `atom_list.ndx` | NDX | 阶段 2 | 阶段 4 | 原子索引组 |
| `plumed.dat` | PLUMED | 阶段 2 | 阶段 4 | 偏置配置 |
| `idp_nvt.in` | LAMMPS input | 阶段 3 | 阶段 4 | 模拟脚本 |
| `idp_nvt.data` | LAMMPS data | 阶段 3 | 阶段 4 | 拓扑 + CMAP 交叉项 |
| `cmap_20240524.cmap` | CMAP | 阶段 3 (提供) | 阶段 4 | 二面角校正网格 |
| `traj_idp.xtc` | XTC | 阶段 4 | 后分析 | 系综轨迹 |

来源: [tutorial/bAIes/README.md](/tutorial/bAIes/README.md#L1-L151), [tutorial/coil/README.md](/tutorial/coil/README.md#L1-L81)

## 后续阅读

流水线架构建立了骨架框架——每个阶段都包含值得单独探索的巨大实现深度：

- **距离直方图内部机制**：[距离直方图读取与拟合](5-distogram-reading-and-fitting)——详述模型选择、拟合收敛及质量诊断
- **PLUMED 配置**：[PLUMED 文件生成](6-plumed-file-generation)——BAIES 操作语法、先验选择与温度缩放
- **力场机制**：[GROMACS 至 LAMMPS 转换](7-gromacs-to-lammps-conversion)与 [CMAP 校正图](8-cmap-correction-maps)——简化原理与 CMAP 补丁要求
- **模拟执行**：[bAIes 系综模拟](9-baies-ensemble-simulations)与 [随机线团模拟](10-random-coil-simulations)——生产运行配置与轨迹分析
- **输入规范**：[AlphaFold-2 与 ColabFold 输入](11-alphafold-2-and-colabfold-inputs)与 [残基对截断矩阵](12-residue-pair-cutoff-matrix)——距离直方图格式与残基对距离阈值逻辑