---
slug:18-alphaflex-workflow-overview
blog_type:normal
---


**AlphaFlex (AFX-IDPForge) 流水线**是一个四阶段的计算工作流，它将 AlphaFold2 预测的结构转化为包含本征无序区域（IDR）的蛋白质的物理真实构象系综。与基础 IDPForge 系统（孤立地处理单个无序片段）不同，AlphaFlex 编排了端到端的处理过程：从 IDR 分类和数据库标记，经由逐区域模板构建和基于扩散的构象异构体生成，到全长多结构域模型的蒙特卡洛拼接、AMBER 弛豫以及结构验证。该流水线专为高通量蛋白质组规模的处理而设计，同时在每个阶段都保持可恢复性和 HPC 可并行性。

来源: [README.md](/AlphaFlex/README.md#L1-L146), [config.py](/AlphaFlex/config.py#L1-L123)

## 流水线架构

AlphaFlex 流水线通过四个顺序阶段处理蛋白质，每个阶段消费前一阶段的输出，并将结果写入 `Pipeline_Outputs/` 下的专用子目录中。核心 `config.py` 模块统管所有可调参数和路径默认值，确保跨脚本行为的一致性。

```mermaid
flowchart TD
    subgraph Inputs
        DB[("AlphaFlex_database_Nov2025.json<br/>IDR 边界 + 相互作用")]
        LEN[("AF2_9606_HUMAN_v4_num_residues.json<br/>残基数")]
        PDB[("AlphaFold2 PDB 库")]
        KNOT[("knot_screening.json<br/>预计算拓扑")]
    end

    subgraph Step1["步骤 1: 案例标记"]
        S1[Step_1_case_label.py] -->|IDR 类型<br/>+ 类别| S1B[Step_1B_subset_label.py]
    end

    subgraph Step2["步骤 2: 模板创建"]
        S2[Step_2_mk_ldr_template.py]
        S2 -->|尾段/环段| MK_LDR[mk_ldr_template.py<br/>静态支架]
        S2 -->|连接段| MK_FLEX[mk_flex_template.py<br/>动态支架]
        S2 -->|类别 0| IDP_JSON[idp_cases_to_run.json]
    end

    subgraph Step3["步骤 3: 构象异构体生成"]
        S3[Step_3_sample_conformer.py]
        S3 -->|IDR 区域| SAMPLE_LDR[sample_ldr.py<br/>SE(3) 扩散]
        S3 -->|完整 IDP| SAMPLE_IDP[sample_idp.py<br/>无条件扩散]
        S3 --> VALIDATE[弛豫 → 修复 → 验证循环]
    end

    subgraph Step4["步骤 4: 拼接与最小化"]
        S4[Step_4_ldr_stitch.py]
        S4 -->|单一 IDR| FASTPASS[快速通道:<br/>直接系综组装]
        S4 -->|多重 IDR| MC[蒙特卡洛拼接:<br/>运动学链组装]
        MC --> RELAX[AMBER 弛豫<br/>+ 修复 + 验证]
    end

    Inputs --> Step1
    Step1 -->|"已标记 DB +<br/>子集 .txt"| Step2
    PDB --> Step2
    Step2 -->|".npz 模板"| Step3
    Step3 -->|"已验证 .pdb<br/>构象异构体池"| Step4
    PDB --> Step4
    KNOT --> Step3
    KNOT --> Step4

    Step4 --> RESULT[("最终系综 PDB<br/>类别/长度/蛋白质/")]
```

来源: [README.md](/AlphaFlex/README.md#L49-L141), [config.py](/AlphaFlex/config.py#L26-L123)

## IDR 分类系统

AlphaFlex 流水线的基础抽象是其**四层 IDR 拓扑层级**，它决定了每个无序区域的下游处理方式。该分类是层级化的——更高的类别意味着更大的结构复杂性，并需要更复杂的拼接策略。

| 类别 | 名称 | 条件 | 模板策略 | 拼接复杂度 |
|:--------:|:----:|:----------|:------------------|:-------------------|
| **0** | IDP | 整个蛋白质是无序的（无折叠结构域） | 通过 `sample_idp.py` 进行无条件扩散 | 快速通道（无需拼接） |
| **1** | 尾段 | 仅有 N/C 端 IDR；所有内部区域均已折叠 | 通过 `mk_ldr_template.py` 构建静态支架（冻结折叠区域） | 低 |
| **2** | 连接段 | **非相互作用**的折叠结构域之间的内部 IDR | 通过 `mk_flex_template.py` 构建动态支架（结构域被视为具有随机位移的独立刚体） | 中（多体对齐） |
| **3** | 环段 | **相互作用的**折叠结构域之间的内部 IDR（平均 PAE < 15 Å） | 通过 `mk_ldr_template.py` 构建静态支架（冻结折叠区域） | 高（受限拓扑） |

杂合蛋白质（类别 1–3）中的每个 IDR 都会接收一个**无序结构域标签**（`D1`, `D2`, …）和一个**侧翼结构域注释**，用于标识相邻的折叠结构域（`F1`, `F2`, …）。侧翼结构域之间的相互作用状态——源自 AlphaFold2 预测对齐误差（PAE）矩阵——区分了连接段与环段，并直接控制调用哪种模板生成策略。

来源: [Step_1_case_label.py](/AlphaFlex/Step_1_case_label.py#L1-L10), [Step_1_case_label.py](/AlphaFlex/Step_1_case_label.py#L111-L197)

## 数据输入与资源

流水线依赖于存储在 `AlphaFlex/Data_Inputs/` 中的四种预计算数据资源：

| 文件 | 格式 | 内容 | 使用者 |
|------|--------|---------|---------|
| `AlphaFlex_database_Nov2025.json` | JSON 字典 | 每个蛋白质的 IDR 边界（`idrs`）、区域间平均 PAE（`mean_pae`）及折叠结构域相互作用 | 步骤 1, 2, 4 |
| `AF2_9606_HUMAN_v4_num_residues.json` | JSON 字典 | 来自 AlphaFold2 9606 Human v4 的总残基数 | 步骤 1, 1B |
| `knot_screening.json` | JSON 字典 | 每个结构域的纽结存在与否（`label`）及纽结类型（`closure_polys`） | 步骤 3, 4 |
| `Test_Structures/O14653.pdb` | PDB | 用于流水线测试的类别 3 蛋白质样本 | 步骤 2, 3, 4 |

主数据库是核心信息资产：每个键是一个 UniProt 编号，值编码了折叠区域与无序区域之间的空间关系。步骤 1 使用计算出的标签增强此数据库，生成驱动所有下游处理的**已标记数据库**。

来源: [README.md](/AlphaFlex/README.md#L7-L15), [config.py](/AlphaFlex/config.py#L18-L22)

## 逐步流水线演练

### 步骤 1: 案例标记 → `Step_1_Labeling/`

此步骤读取主数据库，并将每个蛋白质分类到四层类别系统中。对于每个 IDR，它计算：残基**范围**、**类型**（尾段/连接段/环段/IDP）、序数**标签**（D1, D2, …）以及**侧翼结构域**。分类逻辑应用严格的层级：如果存在任何环段 IDR，则不论其他 IDR 类型如何，该蛋白质均为类别 3；连接段优先于尾段；跨越整个序列的单一 IDR 产生类别 0。

输出是一个增强的 JSON 数据库和一个显示各类别分布的 `idr_type_summary.txt` 报告。

来源: [Step_1_case_label.py](/AlphaFlex/Step_1_case_label.py#L30-L199), [README.md](/AlphaFlex/README.md#L49-L62)

### 步骤 1B: 子集过滤 → `Step_1_Labeling/custom_subsets/`

步骤 1B 从已标记数据库中创建过滤后的 UniProt ID 列表，用于定向处理。它在两种模式下运行：

- **基本模式**（仅限长度）：根据 `[min_len, max_len]` 内的总残基数过滤蛋白质。无论 IDR 组成如何，所有匹配的蛋白质均被纳入。
- **高级模式**：额外根据各类型 IDR 数量（尾段/连接段/环段）及统一应用于蛋白质中每个 IDR 的全局 IDR 长度范围进行过滤。数量匹配可以是**精确**（默认）或**最小**（`--min_mode`）。

输出包括过滤后的 ID 列表（`.txt`）、每个蛋白质的报告表，以及——当激活高级过滤器时——带有分箱计数的 IDR 长度分布直方图。

来源: [Step_1B_subset_label.py](/AlphaFlex/Step_1B_subset_label.py#L1-L64), [README.md](/AlphaFlex/README.md#L64-L80)

### 步骤 2: 模板创建 → `Step_2_Templates/`

步骤 2 为扩散模型生成每个 IDR 的 `.npz` 结构模板。对于子集 ID 列表中的每个蛋白质，它遍历每个已标记的 IDR，并根据 IDR 类型分派相应的模板脚本：

- **尾段/环段 IDR** → `mk_ldr_template.py`：创建一个**静态支架**，其中指定 IDR 之外的所有区域均保持冻结。无序区域沿连接其侧翼折叠结构域的向量分布构象进行初始化。
- **连接段 IDR** → `mk_flex_template.py`：创建一个**动态支架**，其中两个相邻的折叠结构域被指定为独立的刚体，并彼此随机位移，模拟非相互作用结构域的灵活性。
- **类别 0（完整 IDP）** → 不创建模板；相反，蛋白质的序列被记录到 `idp_cases_to_run.json` 中，以便在步骤 3 中通过 `sample_idp.py` 单独处理。

可选的**截断模式**（`--truncate`）仅基于 IDR 及其相邻的折叠结构域构建每个模板，从而减小大型蛋白质的模板尺寸。截断附档（`.trunc.json`）记录了偏移量和窗口边界，用于步骤 4 重建期间的回溯编号。一种**嫁接模式**替代方案（`--max_residues`）限制模板大小，同时记录供 `graft_back` 工具使用的嫁接规格。

来源: [Step_2_mk_ldr_template.py](/AlphaFlex/Step_2_mk_ldr_template.py#L1-L66), [Step_2_mk_ldr_template.py](/AlphaFlex/Step_2_mk_ldr_template.py#L160-L286), [README.md](/AlphaFlex/README.md#L82-L95)

### 步骤 3: 构象异构体生成 → `Step_3_Raw_Conformers/`

步骤 3 通过迭代的**生成–弛豫–修复–验证**循环，为每个 IDR 模板生成已验证的构象异构体池。每个 IDR 独立处理，因此具有多个 IDR 的蛋白质会生成独立的逐 IDR 构象异构体池。循环过程如下：

1. **生成 + 弛豫**：调用 `sample_ldr.py`（用于 IDR 区域）或 `sample_idp.py`（用于完整 IDP）以生成扩散构象异构体，随即使用 `configs/sample.yml` 的配置通过 AMBER 最小化进行弛豫。
2. **修复**：检查每个弛豫结构是否存在 D-型氨基酸（手性反转）和断裂的组氨酸环键。如果进行了任何修复，则应用修复并重新弛豫。
3. **验证**：运行统一验证，检查手性、键完整性、冲突分数（使用**自适应智能阈值**）和骨架拓扑（针对预计算的 `knot_screening.json` 进行纽结检测）。

循环继续进行，直到达到目标构象异构体数量（`SAMPLE_N_CONFS`，默认 10）或耗尽最大总尝试次数（`SAMPLE_MAX_TOTAL_ATTEMPTS`，默认 500）。通过 `.step3_state.json` 实现**状态持久化**，支持从集群抢占中恢复——在重启时，被终止运行遗留的孤立弛豫文件将被自动恢复。通过 `--total_splits` 和 `--split_index` 支持跨作业确定性分片的 HPC 并行执行。

来源: [Step_3_sample_conformer.py](/AlphaFlex/Step_3_sample_conformer.py#L1-L143), [Step_3_sample_conformer.py](/AlphaFlex/Step_3_sample_conformer.py#L178-L241), [README.md](/AlphaFlex/README.md#L97-L113)

### 步骤 4: 拼接与最小化 → `Step_4_Final_Models/`

步骤 4 从逐 IDR 构象异构体池组装全长蛋白质模型。其行为根据 IDR 数量产生分岔：

**单一 IDR / 完整 IDP（快速通道）**：收集步骤 3 的所有已验证构象异构体，并将它们直接组合成多模型系综 PDB。不执行拼接或额外的最小化。

**多重 IDR（蒙特卡洛拼接）**：运行随机组装循环，重复直至达到目标构象异构体数量或耗尽最大尝试次数。运动学链组装算法的过程如下：

1. 构建一个**片段映射**，将蛋白质划分为交替的折叠（F）和无序（D）片段。
2. 使用 AlphaFold2 预测中的第一个折叠结构域为结构提供种子。
3. 对于后续的每个无序片段，从逐 IDR 池中**随机抽取一个构象异构体**（带随机洗牌的轮询）。
4. 使用 Kabsch 对齐，在参考结构与构象异构体结构之间，对前一个折叠结构域的中间 11 个残基（“对齐桩”）进行**叠合**。
5. 使用从对齐桩开始的已对齐构象异构体坐标**覆盖**参考结构。重复此过程直至所有片段拼接完成。
6. **弛豫前修复**：翻转拼接结构中的任何 D-型氨基酸。
7. **AMBER 最小化**（ff14SB），对折叠结构域施加谐和约束，允许 IDR 和连接区域自由弛豫。
8. **弛豫后修复**：检查弛豫期间引入的 D-型氨基酸和断裂的 HIS 环键；如有需要，应用修复并重新弛豫。
9. **验证**：手性、键完整性、冲突分数（自适应智能阈值）、骨架拓扑（纽结检测）以及**折叠结构域曲率门控**（确保弛豫未使结构良好区域的变形超过可配置比率）。
10. 将所有有效构象异构体组合成单个多模型系综 PDB。

最终输出组织为 `Step_4_Final_Models/<Category>/<Length_Label>/<protein_id>/<protein_id>_ensemble_n<N>.pdb`，其中长度标签将蛋白质分层至各个分箱（0–250, 251–500, 501–1000 等）。

来源: [Step_4_ldr_stitch.py](/AlphaFlex/Step_4_ldr_stitch.py#L1-L127), [Step_4_ldr_stitch.py](/AlphaFlex/Step_4_ldr_stitch.py#L218-L340), [README.md](/AlphaFlex/README.md#L115-L141)

## 配置参考

所有流水线参数集中存放在 `config.py` 中，并可通过每个步骤脚本的 CLI 参数进行覆盖。下表按流水线阶段汇总了关键可调参数：

| 参数 | 默认值 | 阶段 | 描述 |
|-----------|---------|:-----:|-------------|
| `VERBOSE` | `True` | 全部 | 启用详细的逐蛋白质控制台日志 |
| `SUBSET_MIN_LENGTH` / `SUBSET_MAX_LENGTH` | 0 / 250 | 1B | 蛋白质长度范围过滤器 |
| `SUBSET_TAIL_COUNT` / `SUBSET_LINKER_COUNT` / `SUBSET_LOOP_COUNT` | 2 / 1 / 1 | 1B | 各类型 IDR 数量约束 |
| `SUBSET_EXACT_COUNT` | `True` | 1B | 精确数量匹配 vs. 最小数量匹配 |
| `SUBSET_IDR_MIN_LENGTH` / `SUBSET_IDR_MAX_LENGTH` | None | 1B | 全局 IDR 长度范围（应用于所有 IDR） |
| `TEMPLATE_N_CONFS` | 200 | 2 | 生成的模板构象数量 |
| `TEMPLATE_SEED_SKEW` | 0.5 | 2 | [d/2, d] 上尾段/环段模板的种子距离偏度 |
| `TEMPLATE_FOLD_PER_SIDE` | 10 | 2 | 嫁接模式下每个连接处保留的折叠残基数 |
| `SAMPLE_N_CONFS` | 10 | 3 | 每个 IDR 的目标已验证构象异构体数 |
| `SAMPLE_BATCH_SIZE` | 6 | 3 | 扩散批次大小 |
| `SAMPLE_MAX_TOTAL_ATTEMPTS` | 500 | 3 | 每个 IDR 的最大生成尝试次数 |
| `SAMPLE_FOLD_CURV_RATIO` | 0.5 | 3 | 折叠结构域曲率门控比率 |
| `SAMPLE_JUNCTION_KAPPA` | 0.12 | 3 | 连接处曲率门控 (Å⁻¹) |
| `STITCH_N_CONFORMERS` | 10 | 4 | 每个蛋白质的目标拼接构象异构体数 |
| `STITCH_MAX_ATTEMPTS` | 500 | 4 | 最大拼接尝试次数 |
| `RELAX_STIFFNESS` | 10.0 | 4 | 折叠结构域上的 AMBER 约束强度 |
| `ALIGNMENT_STUB_HALF_SIZE` | 5 | 4 | 对齐桩的半窗口（中间 11 个残基） |
| `STITCH_BASE_CLASH_THRESHOLD` | 10.0 | 4 | 基础冲突分数阈值（自适应） |
| `STITCH_CLASH_INCREMENT` | 5.0 | 4 | 每个尝试梯度的冲突阈值升级步长 |

来源: [config.py](/AlphaFlex/config.py#L1-L123)

## 工具模块

`AlphaFlex/utils/` 包提供了主要由步骤 4 使用的共享功能：

| 模块 | 用途 |
|--------|---------|
| `stitch.py` | 运动学链组装、片段映射、BioPython 叠合、轮询构象异构体抽取、系综目录发现、PDB 加载、长度分箱及截断池重编号 |
| `smart_scoring.py` | 自适应冲突阈值计算，随尝试梯度升级，平衡验证严格度与吞吐量 |
| `graft_back.py` | 嫁接模式重建：将截断子系综映射回全长编号，并将构象异构体坐标嫁接到原始 AlphaFold2 支架上 |
| `truncate.py` | 为大型蛋白质模板尺寸缩减计算 IDR 截断窗口（IDR + 相邻折叠结构域） |
| `file_ops.py` | 流水线各步骤共享的通用文件 I/O 辅助工具 |

`stitch.py` 中的 `assemble_kinematic_chain` 函数是步骤 4 的算法核心：它将蛋白质划分为片段映射，然后使用对齐桩叠合策略，迭代地将无序构象异构体对齐并覆盖到不断增长的模型上。

来源: [stitch.py](/AlphaFlex/utils/stitch.py#L1-L23), [stitch.py](/AlphaFlex/utils/stitch.py#L291-L398)

## HPC 与可恢复性

AlphaFlex 流水线专为 HPC 集群上的蛋白质组级吞吐量而设计。关键操作特性包括：

- **确定性分片**：步骤 3 和 4 接受 `--total_splits` 和 `--split_index` 参数，通过 `numpy.array_split` 将蛋白质 ID 列表划分为非重叠块，实现高度并行的作业阵列。
- **状态持久化**：步骤 3 在每轮生成后写入 `.step3_state.json`，记录累积尝试次数。在重启时，脚本检测现有已验证文件，恢复孤立的中间文件，并从持久化的状态恢复执行。
- **进度跟踪**：步骤 2 写入 `Step_2_progress.txt` 记录最后处理的蛋白质 ID，允许在中断后批量恢复。
- **逐蛋白质日志**：步骤 3 和 4 将逐蛋白质日志流式传输至 `logs/StepN/perprotein/split_K/<ID>.log`，支持通过 `tail -f` 对长时间运行的集群作业进行实时监控。
- **本地并行性**：步骤 4 额外支持 `--workers` 用于拼接任务的多进程本地并行处理。

来源: [Step_3_sample_conformer.py](/AlphaFlex/Step_3_sample_conformer.py#L30-L56), [Step_2_mk_ldr_template.py](/AlphaFlex/Step_2_mk_ldr_template.py#L111-L129), [README.md](/AlphaFlex/README.md#L109-L141)

<CgxTip>流水线在步骤 3 中迭代的生成-弛豫-修复-验证循环是主要的吞吐量瓶颈。在 `config.py` 中调整 `SAMPLE_BATCH_SIZE` 和 `SAMPLE_MAX_TOTAL_ATTEMPTS` 可权衡 GPU 利用率与构象异构体产率——更大的批次可摊销模型加载开销，但会增加每轮的内存压力。步骤 3–4 中的自适应冲突阈值会随着尝试次数增加自动放宽验证严格度，防止在结构复杂的蛋白质上出现无限拒绝循环。</CgxTip>

<CgxTip>对于具有多个 IDR 的蛋白质，步骤 4 中的蒙特卡洛拼接在设计上具有随机性——每次运行都会产生不同的系综。`ALIGNMENT_STUB_HALF_SIZE` 参数（默认为 5，产生 11 残基桩）控制连接对齐的刚性：较短的桩容许结构域边界处更多的局部变形，而较长的桩以降低构象多样性为代价强制更严格的连续性。</CgxTip>

## 接下来去哪里

AlphaFlex 工作流概述建立了端到端处理的主线。要深入了解特定阶段：

- **[案例标记与 IDR 分类](19-case-labeling-and-idr-classification)** — 四层类别系统、基于相互作用的连接段/环段区分以及子集过滤策略的详细机制
- **[蒙特卡洛拼接与组装](20-monte-carlo-stitching-and-assembly)** — 运动学链算法、对齐桩叠合、AMBER 弛豫协议和自适应验证门控的深入探讨
- **[使用折叠模板进行 IDR 采样](13-idr-sampling-with-folded-templates)** — `sample_ldr.py` 如何利用步骤 2 的 `.npz` 模板，在折叠结构域上下文条件下进行 SE(3) 骨架扩散
- **[AMBER 弛豫与修复](15-amber-relaxation-and-repair)** — 步骤 3 和 4 中使用的 ff14SB 最小化协议、手性修复及组氨酸环键校正
- **[结构验证流水线](16-structure-validation-pipeline)** — 统一验证检查，包括自适应冲突评分、纽结检测和折叠结构域曲率门控