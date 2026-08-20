---
slug:18-msa-generation-with-mmseqs2
blog_type:normal
---


多重序列比对 (MSA) 生成是 AlphaFlow 模型的一个关键预处理步骤，通过同源序列提供进化上下文。AlphaFlow 支持两种互补的 MSA 生成方法：一种基于远程 API 的服务，另一种是使用 MMseqs2 的本地数据库搜索。本文档涵盖了这两种方法、它们的配置选项以及与推理流程的集成。

## MSA 生成方法

AlphaFlow 通过两种不同的工作流提供了 MSA 生成的灵活性，每种方法都针对不同的用例和计算资源进行了优化。

![AlphaFlow MSA Pipeline](https://github.com/bjing2016/alphaflow/blob/master/assets/6uof_A_animation.gif?raw=true)

### 通过 ColabFold 服务器的远程 API

远程 API 方法提供了获取 MSA 最简单的途径，利用 ColabFold 基础设施进行可扩展的、基于云端的序列搜索。这种方法非常适合没有本地数据库资源的用户，或者处理中等数量序列的用户。

`scripts/mmseqs_query.py` 中基于 API 的实现处理了从提交到结果检索的完整工作流：

来源：[scripts/mmseqs_query.py](/scripts/mmseqs_query.py#L20-L283)

```python
def run_mmseqs2(x, prefix, use_env=True, use_filter=True,
                use_templates=False, filter=None, use_pairing=False, pairing_strategy="greedy",
                host_url="https://api.colabfold.com",
                user_agent: str = "") -> Tuple[List[str], List[str]]:
```

该函数协调三个关键操作：向 ColabFold API 端点提交作业、使用指数退避轮询状态以确保可靠性，以及检索压缩结果。对于从 CSV 文件批量处理序列：

来源：[scripts/mmseqs_query.py](/scripts/mmseqs_query.py#L284-L295)

```python
df = pd.read_csv(args.split, index_col='name')
os.makedirs(args.outdir, exist_ok=True)

msas = run_mmseqs2(list(df.seqres), prefix='/tmp/', user_agent='bjing2016/alphaflow')
os.system('rm -r /tmp/_env')

for name, msa in zip(df.index, msas):
    os.makedirs(f'{args.outdir}/{name}/a3m/', exist_ok=True)
    with open(f'{args.outdir}/{name}/a3m/{name}.a3m', 'w') as f:
        f.write(msa)
```

### 本地 MMseqs2 数据库搜索

对于生产工作流或需要控制数据库版本和搜索参数的场景，本地搜索方法提供了完全的自定义能力。此方法需要在本地下载和配置 MMseqs2 数据库，但可以为大批量处理提供更高的吞吐量以及完整的参数控制。

`scripts/mmseqs_search.py` 中的本地搜索实现提供了全面的单体和配对搜索功能：

来源：[scripts/mmseqs_search.py](/scripts/mmseqs_search.py#L25-L147)

```python
def mmseqs_search_monomer(
    dbbase: Path,
    base: Path,
    uniref_db: Path = Path("uniref30_2202_db"),
    template_db: Path = Path(""),  # 默认不使用
    metagenomic_db: Path = Path("colabfold_envdb_202108_db"),
    mmseqs: Path = Path("mmseqs"),
    use_env: bool = True,
    use_templates: bool = False,
    filter: bool = True,
    expand_eval: float = math.inf,
    align_eval: int = 10,
    diff: int = 3000,
    qsc: float = -20.0,
    max_accept: int = 1000000,
    s: float = 8,
    db_load_mode: int = 2,
    threads: int = 32,
):
```

对于需要多个链之间进行配对比对的复杂结构，`mmseqs_search_pair` 函数实现了专门的比对逻辑：

来源：[scripts/mmseqs_search.py](/scripts/mmseqs_search.py#L148-L327)

```python
def mmseqs_search_pair(
    dbbase: Path,
    base: Path,
    uniref_db: Path = Path("uniref30_2202_db"),
    mmseqs: Path = Path("mmseqs"),
    s: float = 8,
    threads: int = 64,
    db_load_mode: int = 2,
):
```

## MSA 生成流程

完整的 MSA 生成工作流遵循从输入序列到模型就绪特征的结构化过程。

```mermaid
flowchart TD
    A[Input CSV<br/>name, seqres] --> B{Generation Method}
    B --> C[Remote API]
    B --> D[Local MMseqs2]
    C --> E[Submit to ColabFold API]
    E --> F[Poll Status<br/>Retry on RATELIMIT]
    F --> G[Download tar.gz]
    G --> H[Extract A3M Files]
    D --> I[Create Query Database]
    I --> J[Search UniRef30<br/>3 iterations, s=8]
    J --> K[Expand Alignments<br/>mode=0, max_seq_id=0.95]
    K --> L[Optional: Template Search]
    L --> M[Optional: Metagenomic DB]
    M --> N[Merge Results<br/>uniref.a3m + mgnify.a3m]
    N --> H
    H --> O[Directory Structure<br/>{name}/a3m/{name}.a3m]
    O --> P[Data Pipeline Parsing]
    P --> Q[Feature Generation<br/>deletion_matrix_int, msa]
    Q --> R[AlphaFlow Model Input]
```

## 数据库配置和参数

MSA 的质量和深度直接影响模型性能。AlphaFlow 支持多个数据库源和全面的参数调整。

### 数据库来源

MMseqs2 搜索利用分层数据库方法，每个数据库提供独特的进化信息：

| Database | Purpose | Default Path | Notes |
|----------|---------|--------------|-------|
| UniRef30 | Primary evolutionary signal | `uniref30_2202_db` | Clustered UniProtKB at 30% identity |
| Metagenomic (ColabFold envdb) | Environmental diversity | `colabfold_envdb_202108_db` | BFD + Mgnify + MetaEuk + SMAG databases |
| Template (PDB70) | Structural templates | Optional | Not used by default for AlphaFlow |

来源：[scripts/mmseqs_search.py](/scripts/mmseqs_search.py#L44-L61)

### 搜索参数

关键参数控制同源检测的灵敏度和选择性：

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `-s` (sensitivity) | 8.0 | 1.0-8.0 | Higher values find more distant homologs but increase runtime |
| `--num-iterations` | 3 | 1-4 | Iterative search refinement |
| `-e` (e-value) | 0.1 (search), 10 (align) | Variable | Statistical significance threshold |
| `--max-seqs` | 10000 | 1000-50000 | Maximum sequences per search round |
| `--diff` | 3000 | 0-10000 | Coverage score difference threshold |
| `--qsc` | 0.8 (filtered) / -20.0 (unfiltered) | -20.0-1.0 | Sequence identity weighting |

来源：[scripts/mmseqs_search.py](/scripts/mmseqs_search.py#L80-L94)

当启用过滤时，过滤逻辑会自动调整参数：

来源：[scripts/mmseqs_search.py](/scripts/mmseqs_search.py#L50-L56)

```python
if filter:
    align_eval = 10
    qsc = 0.8
    max_accept = 100000
```

## 与数据流程的集成

一旦生成 MSA，AlphaFlow 的数据流程会解析并将其转换为模型兼容的特征。`alphaflow/data/data_pipeline.py` 中的 `DataPipeline` 类处理来自 A3M 和 Stockholm 格式的 MSA 解析：

来源：[alphaflow/data/data_pipeline.py](/alphaflow/data/data_pipeline.py#L418-L478)

```python
def _parse_msa_data(
    self,
    alignment_dir: str,
    alignment_index: Optional[Any] = None,
) -> Mapping[str, Any]:
```

解析函数支持直接文件系统访问和大规模数据集的索引二进制数据库。对于每个 MSA 文件，它提取序列和缺失矩阵，然后生成特征：

来源：[alphaflow/data/data_pipeline.py](/alphaflow/data/data_pipeline.py#L143-L176)

```python
def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix],
) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    # Deduplicate sequences across MSAs
    # Convert to integer encoding using HHBLITS_AA_TO_ID
    # Generate deletion_matrix_int, msa, num_alignments features
```

<CgxTip>
数据流程会自动对多个 MSA 来源（UniRef30 和宏基因组数据库）的序列进行去重，以防止冗余，同时保留独特的进化信号。当处理包含数千个序列的深度 MSA 时，这种优化对于内存效率至关重要。
</CgxTip>

## 使用示例

### 远程 API 生成

使用 ColabFold API 为 CSV 文件中的序列生成 MSA：

```bash
python -m scripts.mmseqs_query --split splits/atlas_test.csv --outdir ./alignment_dir
```

此命令从 `atlas_test.csv` 读取序列，将其提交给 ColabFold API，并将生成的 A3M 文件保存为 `{alignment_dir}/{name}/a3m/{name}.a3m` 格式。

来源：[README.md](/README.md#L91-L93)

### 本地数据库搜索

对于本地 MSA 生成，首先使用 ColabFold 设置脚本下载数据库，然后运行搜索：

```bash
# Download databases (one-time setup)
bash scripts/download_atlas.sh

# Run local search
python -m scripts.mmseqs_search_helper \
  --split splits/atlas_test.csv \
  --db_dir /path/to/dbbase \
  --outdir ./alignment_dir
```

辅助脚本会创建一个临时的 FASTA 文件，并调用完整的 MMseqs2 搜索流程：

来源：[scripts/mmseqs_search_helper.py](/scripts/mmseqs_search_helper.py#L17-L18)

```python
cmd = f'python -m scripts.mmseqs_search /tmp/tmp.fasta {args.db_dir} {args.outdir}'
os.system(cmd)
```

来源：[README.md](/README.md#L91-L94)

### 高级本地配置

对于需要自定义搜索参数的生产工作流，请直接调用本地搜索模块并使用微调参数：

```bash
python -m scripts.mmseqs_search \
  query.fasta \
  /path/to/dbbase \
  ./output_dir \
  -s 7.5 \
  --use-env 1 \
  --filter 1 \
  --threads 32 \
  --db1 uniref30_2202_db \
  --db3 colabfold_envdb_202108_db
```

降低灵敏度 (`-s 7.5`) 可以将运行时间减少约 40%，同时为保守性较好的蛋白质家族保留大部分同源物。

来源：[scripts/mmseqs_search.py](/scripts/mmseqs_search.py#L329-L446)

## 性能考量

### 计算要求

| Method | Throughput | Hardware Requirements | Setup Complexity |
|--------|------------|----------------------|------------------|
| Remote API | ~2-5 分钟/序列 | Internet connection | Low |
| Local (UniRef only) | ~30-60 秒/序列 | 32+ CPU cores, 64GB+ RAM | Medium |
| Local (Full) | ~2-5 分钟/序列 | 64+ CPU cores, 128GB+ RAM | High |

### 权衡

方法的选择涉及在吞吐量、控制和资源可用性之间取得平衡：

- **远程 API**：最适合探索性分析、中等批次大小（<100 个序列）以及没有数据库基础设施的用户。该 API 实现了自动重试逻辑和速率限制，以优雅地处理服务器负载。

- **本地搜索**：对于大规模生产工作流（>100 个序列）、需要固定数据库版本的可重现研究或处理机密数据时必不可少。数据库下载和设置的前期投资会在更快的迭代时间上获得回报。

来源：[scripts/mmseqs_query.py](/scripts/mmseqs_query.py#L145-L193)

### MSA 深度与模型性能

AlphaFlow 模型是使用默认参数生成的 MSA 进行训练的。修改灵敏度或过滤阈值可能会影响预测质量：

- **更深的 MSA**（更高的 `-s`，更低的 e-value 阈值）：提供更强的进化约束，特别是对远缘同源物有益。然而，过度的深度会带来收益递减并增加计算成本。

- **更浅的 MSA**（更低的 `-s`，更高的 e-value）：生成速度更快，但可能会遗漏关键的共进化信号。可以考虑用于具有丰富实验数据的已研究蛋白质家族。

<CgxTip>
对于 AlphaFlow-MD+Templates 模型，即使提供了模板结构，仍然需要 MSA。来自 MSA 的进化信息通过识别保守区域和指导参考结构周围的构象采样，补充了结构模板。
</CgxTip>

## 故障排除

### 常见问题

| Issue | Symptoms | Solution |
|-------|----------|----------|
| API rate limiting | "RATELIMIT" status, repeated retries | Add random delays between submissions, process in smaller batches |
| Database not found | FileNotFoundError for database files | Verify database paths, download complete databases using setup script |
| Empty MSA output | Zero sequences in A3M files | Check input sequence format (FASTA), verify database contains relevant homologs |
| Memory errors | Process killed during local search | Reduce `--max-seqs`, increase swap space, or use remote API |

### 验证

通过检查以下内容来验证 MSA 生成质量：

1. **序列覆盖率**：查询序列应作为第一条序列出现，且覆盖率为 100%
2. **同源物数量**：典型范围：小家族 10-100，大家族 1000-10000
3. **多样性**：确保并非所有序列都相同（表明过滤过于激进）
4. **文件格式**：A3M 文件应使用小写字母表示插入，大写字母表示比对区域

## 后续步骤

生成 MSA 后，AlphaFlow 工作流程中的后续步骤包括：

- [Template Processing and Feature Extraction](19-template-processing-and-feature-extraction) - 了解如何为 MD+Templates 模型准备模板结构
- [Feature Engineering for Sequence, Structure, and MSA](21-feature-engineering-for-sequence-structure-and-msa) - 了解 MSA 如何转换为模型特征
- [Inference Pipeline and Sampling Process](14-inference-pipeline-and-sampling-process) - 使用准备好的 MSA 运行 AlphaFlow 模型

有关完整的推理命令和配置选项，请参阅主 README 中的 [Running Inference](#running-inference) 部分。