---
slug:13-benchmark-protein-systems
blog_type:normal
---


bAIes-IDP 基准测试套件为 20 个内在无序蛋白 (IDP) 系统提供了**可直接运行的 LAMMPS 模拟数据**，涵盖四种不同的无序类别。这些系统重现了 Schnapka, Morozova, Sen & Bonomi (2025) 所报告的结果，可作为评估 bAIes 系综质量相对于实验可观测量的验证参考点。每个系统均在三种模拟方案下预打包——**bAIes**、**bAIes-N** 和 **coil**——从而在一致的力场框架内，实现偏置系综与无偏随机线圈基线之间的直接比较。

来源: [README.md](/benchmark/README.md#L1-L38)

## 系统目录与分类

20 个基准测试蛋白被组织为**四个功能类别**，反映了 IDP 系综建模中不同的挑战。此分类不仅是分类学上的划分——每个类别都探测了 AlphaFold-2 预测的不同失效模式，并需要不同的 bAIes 校正策略。

| 类别 | 系统 | 数量 | 残基范围 | 挑战 |
| :--- | :--- | :---: | :---: | :--- |
| **Disordered** | Ab40, p61_Hck, emerin_67-170, His-PknG_1-75, Colicin_N_T_domain, Hug1_ | 6 | 40–105 | 完全无序；AF2 预测完全缺乏结构基础 |
| **Structure motifs** | PaaA2, Nt-SOCS5, Alb3-A3CT, idr_SSRP1, UBact, FCP1 | 6 | 64–101 | 在无序区域内存在局部结构 |
| **AF2 prediction errors** | asyn, ACTR, NHE1, Nsp2_CtlIDR, spm_FrpC, drkN | 6 | 45–179 | AF2 预测出置信度高但错误的局部结构 |
| **Multidomain** | GS8, Ubq2, Ubq3 | 3 | 162–486 | 由柔性连接子连接的多个折叠结构域 |

来源: [README.md](/benchmark/README.md#L15-L36)

### 详细系统规格

| 蛋白标签 | 类别 | 残基 | 原子 | 键 | 二面角 | 交叉项 | 盒子尺寸 (Å) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Ab40 | Disordered | 40 | 598 | 604 | 2,128 | 38 | 200 |
| p61_Hck | Disordered | 78 | — | — | — | — | — |
| emerin_67-170 | Disordered | 105 | — | — | — | — | — |
| His-PknG_1-75 | Disordered | 95 | — | — | — | — | — |
| Colicin_N_T_domain | Disordered | 90 | — | — | — | — | — |
| Hug1_ | Disordered | 73 | — | — | — | — | — |
| PaaA2 | Structure motifs | 71 | 1,165 | 1,175 | 4,208 | 69 | 200 |
| Nt-SOCS5 | Structure motifs | 70 | — | — | — | — | — |
| Alb3-A3CT | Structure motifs | 101 | — | — | — | — | — |
| idr_SSRP1 | Structure motifs | 89 | — | — | — | — | — |
| UBact | Structure motifs | 64 | — | — | — | — | — |
| FCP1 | Structure motifs | 85 | — | — | — | — | — |
| asyn | AF2 errors | 140 | 2,016 | 2,027 | 7,209 | 138 | 200 |
| ACTR | AF2 errors | 71 | — | — | — | — | — |
| NHE1 | AF2 errors | 120 | — | — | — | — | — |
| Nsp2_CtlIDR | AF2 errors | 45 | — | — | — | — | — |
| spm_FrpC | AF2 errors | 179 | — | — | — | — | — |
| drkN | AF2 errors | 59 | — | — | — | — | — |
| GS8 | Multidomain | 486 | 7,480 | 7,577 | 27,369 | 484 | 2,000 |
| Ubq2 | Multidomain | 162 | 2,599 | 2,618 | 9,429 | 160 | 2,000 |
| Ubq3 | Multidomain | 228 | — | — | — | — | — |

<CgxTip>多结构域系统 (GS8, Ubq2, Ubq3) 使用 **2,000 Å** 的模拟盒子，而单结构域系统使用 **200 Å** 的盒子。这反映了多结构域蛋白可及的构象空间显著增大，并防止在扩展构型中出现周期性边界伪影。</CgxTip>

来源: [Ab40_nvt.data](/benchmark/bAIes/Ab40/Ab40_nvt.data#L1-L20), [GS8_nvt.data](/benchmark/bAIes/GS8/GS8_nvt.data#L1-L20), [Ubq2_nvt.data](/benchmark/bAIes/Ubq2/Ubq2_nvt.data#L1-L20), [PaaA2_nvt.data](/benchmark/bAIes/PaaA2/PaaA2_nvt.data#L1-L20), [asyn_nvt.data](/benchmark/bAIes/asyn/asyn_nvt.data#L1-L20)

## 基准测试目录架构

基准测试数据组织在 `benchmark/` 目录下，包含三个平行的子目录，每个模拟方案对应一个。每个蛋白系统目录包含 LAMMPS 就绪的模拟输入，其中 bAIes 和 bAIes-N 方案在公共力场基础设施之上添加了 PLUMED 偏置文件。

```
benchmark/
├── bAIes/          ← 完整 bAIes 系综 (Jeffreys 先验)
│   └── <protein>/
│       ├── <protein>_nvt.data        # LAMMPS 拓扑与坐标
│       ├── <protein>_nvt.in          # LAMMPS 输入脚本
│       ├── atom_list_matrix.ndx      # PLUMED 的原子索引组
│       ├── baies_gauss_matrix.dat    # 拟合的距离图高斯分布
│       ├── dry_ff_20240524_correct.cmap  # CMAP 校正图
│       └── plumed_<protein>.dat      # PLUMED 偏置配置
├── bAIes-N/        ← 无 Jeffreys 先验的 bAIes (平坦先验)
│   └── <protein>/
│       └── (与 bAIes 相同的 6 文件结构)
└── coil/           ← 随机线圈参考 (无 bAIes 偏置)
    └── <protein>/
        ├── <protein>_nvt.data
        ├── <protein>_nvt.in
        └── dry_ff_20240524_correct.cmap
```

来源: [benchmark/](/benchmark/README.md#L1-L38)

## 模拟方案变体

三个基准测试子目录编码了用于验证 bAIes 的三种模拟方案。理解它们的结构差异对于解释基准测试比较至关重要。

### bAIes — 带 Jeffreys 先验的偏置系综

**bAIes** 方案通过 PLUMED 应用完整的贝叶推断结构系综 (bAIes) 偏置，使用 **Jeffreys 先验** (`PRIOR=JEFFREYS`)。此先验施加了无信息、尺度不变的正则化，防止系综过拟合于从距离图导出的高斯约束。每个 bAIes 系统的 PLUMED 配置都遵循相同的模式：

```plumed
#MOLINFO STRUCTURE=<protein>.pdb
batoms: GROUP NDX_FILE=atom_list_matrix.ndx NDX_GROUP=batoms
baies: BAIES ATOMS=batoms DATA_FILE=baies_gauss_matrix.dat PRIOR=JEFFREYS TEMP=2.478541306
PRINT ARG=baies.ene FILE=COLVAR STRIDE=500
bbias: BIASVALUE ARG=baies.ene STRIDE=2
```

关键参数为：`TEMP=2.478541306` (LAMMPS real 单位制下的约化温度，对应于 300 K)，`STRIDE=500` 用于能量输出，`STRIDE=2` 用于偏置评估。`baies_gauss_matrix.dat` 文件将拟合的 AlphaFold-2 距离图编码为一组 pairwise 距离约束 (每对原子的高斯均值 μ 和宽度 σ)。

来源: [plumed_Ab40.dat](/benchmark/bAIes/Ab40/plumed_Ab40.dat#L1-L6), [plumed_GS8.dat](/benchmark/bAIes/GS8/plumed_GS8.dat#L1-L6)

### bAIes-N — 无先验的偏置系综

**bAIes-N** 方案使用相同的从距离图导出的偏置，但**无先验** (`PRIOR=NONE`)。此变体通过“原生”运行偏置来孤立 Jeffreys 先验的效应——距离图高斯分布直接应用，无需贝叶正则化。比较 bAIes 与 bAIes-N 可确切揭示先验在多大程度上校正了过度自信的 AlphaFold-2 距离预测。

```plumed
# bAIes-N 变体 — 仅 PRIOR 行不同
baies: BAIES ATOMS=batoms DATA_FILE=baies_gauss_matrix.dat PRIOR=NONE TEMP=2.478541306
```

来源: [plumed_Ab40.dat](/benchmark/bAIes-N/Ab40/plumed_Ab40.dat#L1-L6)

### Coil — 无偏随机线圈参考

**coil** 方案完全移除了 bAIes 偏置。这些目录仅包含 LAMMPS 数据文件、输入脚本和 CMAP 校正图——没有 PLUMED 文件，没有高斯矩阵，没有原子索引组。coil 模拟对纯力场 (带 CMAP 校正) 进行采样，建立了用于衡量 bAIes 改进程度的零模型基线。

| 文件 | bAIes | bAIes-N | coil |
| :--- | :---: | :---: | :---: |
| `*_nvt.data` | ✅ | ✅ | ✅ |
| `*_nvt.in` | ✅ | ✅ | ✅ |
| `dry_ff_*.cmap` | ✅ | ✅ | ✅ |
| `atom_list_matrix.ndx` | ✅ | ✅ | — |
| `baies_gauss_matrix.dat` | ✅ | ✅ | — |
| `plumed_*.dat` | ✅ | ✅ | — |

来源: [benchmark/bAIes/Ab40/](/benchmark/bAIes/Ab40/plumed_Ab40.dat#L1-L6), [benchmark/bAIes-N/Ab40/](/benchmark/bAIes-N/Ab40/plumed_Ab40.dat#L1-L6), [benchmark/coil/Ab40/](/benchmark/coil/Ab40/Ab40_nvt.in#L1-L15)

## 距离图导出的偏置数据

`baies_gauss_matrix.dat` 文件是将 AlphaFold-2 预测桥接到分子动力学的核心输入。每行指定一个 pairwise 距离约束，作为原子 `atom_i` 和 `atom_j` 之间原子间距的高斯分布，均值为 `mu` (单位 Å)，标准差为 `sigma` (单位 Å)。例如，在 Ab40 系统中：

```
#! FIELDS Id atom_i atom_j mu sigma
#! SET model gaussian
1  7  64  1.044356  0.306678
2  44 135  1.229774  0.437833
3  64 135  1.140156  0.364508
...
```

相应的 `atom_list_matrix.ndx` 文件定义了 **PLUMED 原子组** `[ batoms ]`，它收集了高斯矩阵中引用的所有原子索引。这些是 Cβ (对于甘氨酸为 Cα) 原子，其 pairwise 距离受 bAIes 偏置约束。高斯约束的数量随 AlphaFold-2 距离图中置信预测的接触数而变化，并在四个无序类别间差异显著——无序系统往往具有更少且更弱的约束，而多结构域系统则携带许多高置信度的结构域间接触。

来源: [baies_gauss_matrix.dat](/benchmark/bAIes/Ab40/baies_gauss_matrix.dat#L1-L16), [atom_list_matrix.ndx](/benchmark/bAIes/Ab40/atom_list_matrix.ndx#L1-L4)

## LAMMPS 输入配置

所有基准测试系统共享一致的 LAMMPS 输入配置，使用从 GROMACS 转换而来的 **AWM** (全原子) 干力场。输入脚本建立如下：

- **单位**: `real` (Å, kcal/mol, amu, fs)
- **原子风格**: `full` (带电荷的分子拓扑)
- **成键项**: `harmonic` 键，`harmonic` 角，`charmm` + `multi/harmonic` 二面角
- **非成键项**: 截断值为 2.0 Å 的 `lj/cut 2.0` (隐式溶剂干力场)
- **CMAP 校正**: 通过 `fix drycmap all cmap dry_ff_20240524_correct.cmap` 应用

CMAP 校正图 (`dry_ff_20240524_correct.cmap`) 与 `20240524` 力场参数化版本锁定，并在给定方案内的所有系统中保持一致。`pair_coeff` 块编码了由干力场转换导出的 24–29 种原子类型特定的 Lennard-Jones 参数 (ε, σ, 截断值)。

<CgxTip>所有 20 个系统间一致的力场参数确保了基准测试差异**仅源于 bAIes 偏置** (或其缺失)，而非力场变异。这种受控设置对于 [bAIes vs bAIes-N vs Coil](14-baies-vs-baies-n-vs-coil) 中的公平方案比较至关重要。</CgxTip>

来源: [Ab40_nvt.in](/benchmark/bAIes/Ab40/Ab40_nvt.in#L1-L20), [Ab40_nvt.in (coil)](/benchmark/coil/Ab40/Ab40_nvt.in#L1-L20)

## 运行基准测试

要执行基准测试模拟，请导航至相应方案下所需的蛋白目录，并调用带 PLUMED 补丁的 LAMMPS (针对 bAIes/bAIes-N)：

```bash
# bAIes 或 bAIes-N (需要 LAMMPS+PLUMED)
cd benchmark/bAIes/<protein>/
lmp -in <protein>_nvt.in -plumed plumed_<protein>.dat

# Coil (纯 LAMMPS，无需 PLUMED)
cd benchmark/coil/<protein>/
lmp -in <protein>_nvt.in
```

注意，`*_nvt.in` 脚本仅包含力场和拓扑声明——必须根据你的模拟要求追加积分时间步长、恒温器和运行步长。有关为每个系统尺寸类别设置适当 NVT 系综参数的指导，请参见 [模拟参数调优](16-simulation-parameter-tuning)。

来源: [README.md](/benchmark/README.md#L1-L38)

## 后续步骤

- **定量比较方案**: [bAIes vs bAIes-N vs Coil](14-baies-vs-baies-n-vs-coil) — 了解 Jeffreys 先验和 bAIes 偏置各自如何对四个无序类别的系综质量做出贡献。
- **学习预处理流水线**: [距离图读取与拟合](5-distogram-reading-and-fitting) — 追踪 AlphaFold-2 预测如何成为此处使用的 `baies_gauss_matrix.dat` 文件。
- **设置你自己的系统**: [快速开始](2-quick-start) — 将 bAIes 流水线应用于感兴趣的新 IDP。