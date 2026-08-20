---
slug:3-input-formats-and-examples
blog_type:normal
---


Disobind 接受两种格式的蛋白质对输入——**CSV** 和 **FASTA**——每种格式针对不同的使用场景而设计。CSV 格式是首选且更灵活的选项，支持通过引用 UniProt 编号批量预测多个蛋白质对。FASTA 格式允许预测缺少 UniProt 条目的自定义序列，但每个文件仅限一对蛋白质。两种格式还支持可选的 **Disobind+AF2** 模式，在该模式下，你可以提供 AlphaFold2 的结构预测以及序列信息，以获得更精细的结果。

## 输入要求

在准备输入文件之前，请牢记以下约束条件：

| 要求 | 详情 |
|:---|:---|
| **仅限二元复合物** | Disobind 在蛋白质对 (AB) 上运行。对于非二元复合物 (ABC)，需将其分解为二元对 (AB, BC, AC) 并分别运行。 |
| **蛋白质 1 必须是 IDR** | 假定对中的第一个蛋白质是内在无序区 (IDR)。蛋白质 2 可以是有序的，也可以是无序的。 |
| **假定存在相互作用对** | Disobind 预测蛋白质在*何处*相互作用（接触图和界面残基），而非*是否*相互作用。 |
| **残基范围验证** | 指定的 `start`–`end` 范围不得超过相应蛋白质序列的长度。 |

来源: [run_disobind.py](/run_disobind.py#L4-L14), [run_disobind.py](/run_disobind.py#L435-L453), [README.md](/README.md#L43-L46)

## CSV 格式（首选）

CSV 格式是提供输入的推荐方式。每一行代表一个需要 Disobind 预测的蛋白质片段对。根据你想要 Disobind-only 还是 Disobind+AF2 预测，共有**两种变体**。

### 仅 Disobind（6 个字段）

```
UniProt_ID1,start1,end1,UniProt_ID2,start2,end2
```

| 字段 | 描述 | 示例 |
|:---|:---|:---|
| `UniProt_ID1` | IDR（蛋白质 1）的 UniProt 编号 | `P04273` |
| `start1` | 蛋白质 1 片段的起始残基位置（UniProt 编号） | `95` |
| `end1` | 蛋白质 1 片段的终止残基位置（UniProt 编号） | `193` |
| `UniProt_ID2` | 结合伴侣（蛋白质 2）的 UniProt 编号 | `P04273` |
| `start2` | 蛋白质 2 片段的起始残基位置 | `95` |
| `end2` | 蛋白质 2 片段的终止残基位置 | `192` |

使用此格式时，Disobind 会通过 UniProt REST API 自动下载两个编号的完整 UniProt 序列，然后提取指定的残基片段进行预测。

### Disobind+AF2（12 个字段）

```
UniProt_ID1,start1,end1,UniProt_ID2,start2,end2,AF2_struct_file,AF2_pae_file,chain1,chain2,offset1,offset2
```

| 字段 | 描述 | 示例 |
|:---|:---|:---|
| `AF2_struct_file` | AF2 预测结构文件（`.pdb` 或 `.cif`）的路径 | `./example/unrelaxed_model_4_multimer_v3_pred_4.pdb` |
| `AF2_pae_file` | AF2 PAE 数据文件（`.json`）的路径 | `./example/pae_model_4_multimer_v3_pred_4.json` |
| `chain1` | AF2 结构中对应蛋白质 1 的链 ID | `B` |
| `chain2` | AF2 结构中对应蛋白质 2 的链 ID | `C` |
| `offset1` | 蛋白质 1 的残基位置偏移量（若 AF2 结构与 UniProt 范围匹配则为 0） | `0` |
| `offset2` | 蛋白质 2 的残基位置偏移量（若 AF2 结构与 UniProt 范围匹配则为 0） | `0` |

<CgxTip>**偏移量** 字段用于解释 AF2 残基编号与 UniProt 编号之间的差异。如果 AF2 结构对应完整的 UniProt 序列或完全对应指定的片段范围，则将两者均设置为 `0`。</CgxTip>

### CSV 示例

代码仓库在 `example/test.csv` 中包含一个可直接运行的示例：

```csv
P04273,95,193,P04273,95,192
P04273,95,193,P04273,95,193,./example/unrelaxed_model_4_multimer_v3_pred_4.pdb,./example/pae_model_4_multimer_v3_pred_4.json,B,C,0,0
```

- **第 1 行**：对朊病毒蛋白 (P04273) 片段 95–193 与其自身片段 95–192 相互作用的仅 Disobind 预测。
- **第 2 行**：对相同朊病毒蛋白片段 95–193 的 Disobind+AF2 预测，结合了 AF2 多聚体模型（链 B 和 C，无残基偏移）进行增强。

来源: [example/test.csv](/example/test.csv#L1-L2), [run_disobind.py](/run_disobind.py#L312-L359), [README.md](/README.md#L52-L65)

## FASTA 格式

FASTA 格式适用于**没有 UniProt 编号**的蛋白质。你需要直接提供氨基酸序列。每个文件必须包含**恰好两个条目**——对中的每个蛋白质各一个。

### 仅 Disobind（3 个头字段）

```
>Protein_ID1,start1,end1
SEQUENCE1
>Protein_ID2,start2,end2
SEQUENCE2
```

### Disobind+AF2（7 个头字段）

```
>Protein_ID1,start1,end1,AF2_struct_file,AF2_pae_file,chain1,offset1
SEQUENCE1
>Protein_ID2,start2,end2,AF2_struct_file,AF2_pae_file,chain2,offset2
SEQUENCE2
```

| 头字段 | 仅 Disobind | Disobind+AF2 | 描述 |
|:---|:---:|:---:|:---|
| `Protein_ID` | ✓ | ✓ | 蛋白质的标识符（用于输出标记） |
| `start` | ✓ | ✓ | 片段的起始残基位置 |
| `end` | ✓ | ✓ | 片段的终止残基位置 |
| `AF2_struct_file` | — | ✓ | AF2 结构文件（`.pdb`/`.cif`）的路径 |
| `AF2_pae_file` | — | ✓ | AF2 PAE 文件（`.json`）的路径 |
| `chain` | — | ✓ | 该蛋白质在 AF2 结构中的链 ID |
| `offset` | — | ✓ | 该蛋白质的残基位置偏移量 |

<CgxTip>在 FASTA 模式下，提供的序列被假定为**完整蛋白质序列**。Disobind 将直接使用此序列，而不是从 UniProt 获取。`start`/`end` 值定义了要从该完整序列中提取用于预测的片段。</CgxTip>

### FASTA 示例

代码仓库在 `example/test.fasta` 中包含一个示例：

```
>prion,95,193,./example/unrelaxed_model_4_multimer_v3_pred_4.pdb,./example/pae_model_4_multimer_v3_pred_4.json,B,0
MANLSYWLLALFVAMWTDVGLCKKRPKPGGWNTGGSRYPGQGSPGGNRYPPQGGGTWGQPHGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHNQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPMMHFGNDWEDRYYRENMNRYPNQVYYRPVDQYNNQNNFVHDCVNITIKQHTVTTTTKGENFTETDIKIMERVVEQMCTTQYQKESQAYYDGRRSSAVLFSSPPVILLISFLIFLMVG

>prion,95,193,./example/unrelaxed_model_4_multimer_v3_pred_4.pdb,./example/pae_model_4_multimer_v3_pred_4.json,C,0
MANLSYWLLALFVAMWTDVGLCKKRPKPGGWNTGGSRYPGQGSPGGNRYPPQGGGTWGQPHGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHNQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPMMHFGNDWEDRYYRENMNRYPNQVYYRPVDQYNNQNNFVHDCVNITIKQHTVTTTTKGENFTETDIKIMERVVEQMCTTQYQKESQAYYDGRRSSAVLFSSPPVILLISFLIFLMVG
```

这定义了对朊病毒蛋白同源二聚体的 Disobind+AF2 预测，其中两条链映射到相同的 AF2 多聚体结构（分别为链 B 和 C）。

来源: [example/test.fasta](/example/test.fasta#L1-L5), [run_disobind.py](/run_disobind.py#L240-L308), [README.md](/README.md#L67-L72)

## AF2 补充文件

使用 Disobind+AF2 模式时，你必须从 AlphaFold2 多聚体预测中提供两个附加文件：

| 文件 | 格式 | 内容 | 关键字段 |
|:---|:---|:---|:---|
| **结构文件** | `.pdb` 或 `.cif` | 3D 坐标，B 因子列中包含每个残基的 pLDDT 分数 | 原子坐标，B 因子 (pLDDT) |
| **PAE 文件** | `.json` | 预测对齐误差矩阵 | `predicted_aligned_error` |

PAE JSON 文件必须是一个 JSON 数组，其中包含一个键为 `"predicted_aligned_error"` 的对象，该对象将非对称 PAE 矩阵存储为二维数组。Disobind 通过计算 `(PAE + PAE^T) / 2` 在内部对其进行对称化。结构文件使用 Biopython 的 `PDBParser`（对于 `.pdb`）或 `MMCIFParser`（对于 `.cif`）进行解析，并从 Cα B 因子列读取每个残基的 pLDDT 分数。

### 置信度过滤

Disobind 在结合 AF2 预测时应用了三个内部置信度阈值：

| 参数 | 默认值 | 含义 |
|:---|:---:|:---|
| `dist_threshold` | 8 Å | 定义残基接触的最大 Cα–Cα 距离 |
| `plddt_threshold` | 70 | 认为残基被可靠预测的最小 pLDDT 分数 |
| `pae_threshold` | 5 | 认为残基对被可靠对齐的最大 PAE 值 |

来源: [run_disobind.py](/run_disobind.py#L966-L1090), [run_disobind.py](/run_disobind.py#L70-L76)

## 在 CSV 和 FASTA 之间选择

| 特性 | CSV | FASTA |
|:---|:---:|:---:|
| 每个文件包含多对蛋白质 | ✓ | ✗ |
| 需要 UniProt 编号 | ✓ | ✗ |
| 支持自定义序列 | ✗ | ✓ |
| 仅 Disobind 模式 | ✓ | ✓ |
| Disobind+AF2 模式 | ✓ | ✓ |
| 自动下载序列 | ✓ | ✗ |

当你的蛋白质有 UniProt 条目时，请使用 **CSV**——你可以在单个文件中批量处理数十对蛋白质，并让 Disobind 自动获取序列。当处理未存入 UniProt 的自定义或合成序列时，请使用 **FASTA**。

来源: [run_disobind.py](/run_disobind.py#L212-L237), [README.md](/README.md#L48-L51)

## 运行示例

使用以下命令结合提供的示例输入执行 Disobind：

```bash
python run_disobind.py -i csv -f ./example/test.csv
```

对于 FASTA 示例：

```bash
python run_disobind.py -i fasta -f ./example/test.fasta
```

两条命令均使用默认设置：在 CPU 上以粗粒度分辨率 1 预测界面残基。有关命令行标志的完整列表，请参见[快速入门](2-quick-start)；有关理解结果的详细信息，请参见[输出解释](10-output-interpretation)。

来源: [run_disobind.py](/run_disobind.py#L1312-L1357), [README.md](/README.md#L76-L80)

## 格式验证

Disobind 会对你的输入执行严格验证。如果格式不正确，将引发以下错误：

| 条件 | 错误类型 | 消息 |
|:---|:---|:---|
| CSV 行既不是 6 个也不是 12 个逗号分隔字段 | `ValueError` | `Incorrect input format...` |
| FASTA 文件包含 ≠ 2 个条目 | `ValueError` | `Expected exactly 2 FASTA entries, found N...` |
| FASTA 头既不是 3 个也不是 7 个逗号分隔字段 | `ValueError` | `Incorrect input header format...` |
| 指定的残基范围超出序列长度 | `ValueError` | `Sequence length (L residues) is shorter than the specified residue range S-E...` |
| AF2 结构文件不是 `.pdb` 或 `.cif` | `ValueError` | `Incorrect file format.. Supported .pdb/.cif only.` |
| PAE 文件不是 `.json` | `ValueError` | `Incorrect file format.. Supported .json only.` |
| PAE JSON 缺少 `predicted_aligned_error` 键 | `ValueError` | `PAE matrix not found...` |