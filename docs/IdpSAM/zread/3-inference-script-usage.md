---
slug:3-inference-script-usage
blog_type:normal
---


`generate_ensemble.py` 脚本是使用 idpSAM 生成内禀无序肽构象系综的主要命令行接口。它将完整的两阶段推理流程——**潜变量扩散采样**及随后的**解码器重建**——封装为单次调用，并可通过 cg2all 选择性地追加全原子重建。本页将向你逐一介绍每个参数、执行流程以及预期获得的输出产物。

来源：[generate_ensemble.py](scripts/generate_ensemble.py#L1-L113), [model.py](sam/model.py#L1-L434)

## 快速命令参考

最简单的用法，在 CPU 上为一条肽生成 1,000 个 Cα 构象，如下所示：

```bash
python scripts/generate_ensemble.py \
  -c config/models.yaml \
  -s MFDNASTRNNKRERGKRQGKQTRTQRHADRSQT \
  -o peptide \
  -n 1000
```

若要启用 GPU 加速和全原子重建，请添加 `-d cuda` 和 `-a` 标志：

```bash
python scripts/generate_ensemble.py \
  -c config/models.yaml \
  -s MFDNASTRNNKRERGKRQGKQTRTQRHADRSQT \
  -o peptide \
  -n 1000 \
  -a \
  -d cuda
```

<CgxTip>当生成大型系综（n > 500）时，强烈建议通过 `-d cuda` 使用 GPU 执行——扩散采样阶段是性能瓶颈，能从 CUDA 加速中显著受益。</CgxTip>

来源：[generate_ensemble.py](scripts/generate_ensemble.py#L14-L49), [README.md](README.md#L47-L59)

## 参数参考

该脚本暴露了三个**必需**参数和六个可选参数。下表列出了每个标志、其类型、默认值及用途。

| 标志 | 类型 | 默认值 | 描述 |
|------|------|---------|-------------|
| `-c` / `--config_fp` | `str` | *(必需)* | idpSAM 模型的 YAML 或 JSON 配置文件路径。默认使用 `config/models.yaml`。 |
| `-s` / `--seq` | `str` | *(必需)* | 使用单字母代码表示的氨基酸序列（如 `MFDNASTRN…`）。仅接受 20 种标准残基。 |
| `-o` / `--out_path` | `str` | *(必需)* | 基础输出路径。根据所选输出格式自动追加文件扩展名。 |
| `-u` / `--out_fmt` | `str` | `dcd` | 坐标存储格式。可选：`numpy` (`.npy`)、`dcd` (DCD 轨迹 + PDB 拓扑)。 |
| `-n` / `--n_samples` | `int` | `1000` | 要生成的构象数量。 |
| `-t` / `--n_steps` | `int` | `100` | 扩散逆向步数。范围：1–1000。步数越多 → 质量越高，但运行时间更长。 |
| `-b` / `--batch_size` | `int` | `250` | 扩散采样与%解码的批次大小。如遇 GPU 内存错误，请减小此值。 |
| `--cg2all_batch_size` | `int` | `None` | cg2all 全原子转换的批次大小。未设置时默认为 `--batch_size`。 |
| `-a` / `--all_atom` | 标志 | `False` | 若设置此标志，则通过 cg2all 将 Cα 构象转换为全原子结构（需安装 cg2all）。 |
| `-d` / `--device` | `str` | `cpu` | idpSAM 模型使用的 PyTorch 设备。可选：`cpu`、`cuda`。 |
| `--cg2all_device` | `str` | `cpu` | cg2all 使用的 PyTorch 设备。与 `-d` 分离，因为即使 PyTorch 检测到 CUDA，cg2all 也可能无法检测到。 |
| `-q` / `--quiet` | 标志 | `False` | 隐藏所有打印输出。 |

来源：[generate_ensemble.py](scripts/generate_ensemble.py#L16-L49)

## 执行流程

该脚本按顺序经历四个阶段。下图可视化了从参数解析到最终输出的控制流：

```mermaid
flowchart TD
    A["解析并验证 CLI 参数"] --> B["初始化 SAM 模型<br/>(加载配置、权重、网络)"]
    B --> C["扩散采样<br/>生成潜变量编码"]
    C --> D["解码器重建<br/>潜变量 → Cα xyz 坐标"]
    D --> E["保存输出文件<br/>(FASTA, 轨迹, 拓扑)"]
   :    E -->%F{--all_atom 标志是否设置?}
#    F --2-- 是 --> G["cg2all 重建<br?/>Cα → 全原子"]
     F -- 否 --> H["4"完成"]
     G -->&>H
```

*（注：为保持原文格式，*%Mermaid 语法中*的英文标签已按原结构翻译$转换为中文字)文，如遇&影响渲染，请以原文英文标签/结构为准）*

每个阶段直接对应 `SAM` 包装类上的一个方法调用：

1. **模型初始化** — `SAM(config_fp=…, device=…, verbose=…)` 从配置文件和权重路径加载 epsilon 网络、解码器、标准缩放器和扩散调度器。[model.py](sam/model.py#L46-L122)
2. **采样** — `model.sample()` 内部调用 `model.generate()`（扩散逆向过程 → 潜变量编码），然后调用 `model.decode()`（解码器网络 → Cα 坐标）。[model.py](sam/model.py#L134-L190)
3. **保存** — `model.save()` 按请求的格式将坐标数据写入磁盘。[model.py](sam/model.py#L341-L393)
4. **全原子重建**（可选） — `model.cg2all()` 借助 `convert_cg2all` 命令行工具执行。[model.py](sam/model.py#L396-L434)

来源：[generate_ensemble.py](scripts/generate_ensemble.py#L88-L113), [model.py](sam/model.py#L134-L190)

## 输入验证

在任何计算开始之前，脚本会验证你的输入：

- **序列检查**：氨基酸序列必须匹配正则表达式 `^[QWERTYIPASDFGHKLCVNM]*$` —— 仅允许 20 种标准单字母残基代码。非标准或小写字符将引发 `ValueError`。[generate_ensemble.py](scripts/generate_ensemble.py#L56-L58)
- **全原子兼容性**：设置 `-a` 时，脚本会验证：(1) cg2all 可导入，(2) `--out_fmt` 为 `dcd`（唯一兼容 cg2all 的格式），以及 (3) `--cg2all_batch_size` 能被 `--n_samples` 整除。[generate_ensemble.py](scripts/generate_ensemble.py#L61-L81)

<CgxTip>若出现 `ImportError: The cg2all library is not installed`，请在重新使用 `-a` 标志运行之前，通过 `pip install git+http://github.com/huhlim/cg2all`（仅 CPU 版）进行安装。</CgxTip>

来源：[generate_ensemble.py](scripts/generate_ensemble.py#L52-L81)

## 输出产物

脚本生成一组文件，其名称派生自 `--out_path` 基础路径。具体文件取决于输出格式及是否启用了全原子重建。

### 仅 Cα 输出（`--out_fmt dcd`，默认）

| 文件 | 后缀 | 描述 |
|------|--------|-------------|
| FASTA 序列 | `.seq.fasta` | 包含输入氨基酸序列的纯文本文件。 |
| Cα 轨迹 | `.ca.traj.dcd` | 包含所有生成 Cα 构象的 DCD 二进制轨迹。 |
| Cα 拓扑 | `.ca.top.pdb` | 首帧的 PDB 文件 —— 用作加载 DCD 的拓扑参考。 |

### 仅 Cα 输出（`--out_fmt numpy`）

| 文件 | 后缀 | 描述 |
|------|--------|-------------|
| FASTA 序列 | `.seq.fasta` | 包含输入氨基酸序列的纯文本文件。 |
| Cα 坐标 | `.ca.xyz.npy` | 形状为 `(n_samples, L, 3)` 的 NumPy 数组，其中 L 为序列长度。 |

### 全原子输出（设置 `-a` 时，附加于 Cα 文件之外）

| 文件 | 后缀 | 描述 |
|------|--------|-------------|
| 全原子轨迹 | `.aa.traj.dcd` | 包含由 cg2all 重建的全原子构象的 DCD 轨迹。 |
| 全原子拓扑 | `.aa.top.pdb` | 全原子 DCD 轨迹的 PDB 拓扑文件。 |

来源：[model.py](sam/model.py#L341-L393), [model.py](sam/model.py#L396-L434)

## 配置文件

`-c` 标志指向一个 YAML（或 JSON）配置文件，该文件定义了模型的每个架构和训练超参数。默认配置位于 `config/models.yaml`，并引用 `weights/v1.0/` 目录中的预训练权重。

配置中的关键部分：

| 部分 | 用途 | 关键字段 |
|---------|---------|------------|
| `generative_model` | 生成流程的顶层设置 | `data_type`, `bead_type`, `encoding_dim`, `max_len` |
| `encoder` | 欧几里得不变编码器架构与权重 | `arch`, `num_layers`, `embed_dim`, `weights`, `std_scaler_fp` |
| `decoder` | Transformer 解码器架构与权重 | `arch`, `num_layers`, `attention_type`, `weights` |
| `latent_generative_model` | DDPM 扩散调度参数 | `sched_params`（beta 调度、时间步、方差类型） |
| `latent_network` | 噪声预测（epsilon）网络架构与权重 | `arch`, `num_layers`, `attention_type`, `weights` |

<CgxTip>配置中的权重文件路径**相对于当前工作目录**，而非配置文件本身。如果从其他目录运行脚本，请在配置中使用绝对路径，或在编程方式使用 `SAM` 类时设置 `weights_parent_path` 参数。</CgxTip>

有关每个配置字段的完整解析，请参阅[配置参考](15-configuration-reference)。

来源：[models.yaml](config/models.yaml#L1-L106), [model.py](sam/model.py#L46-L131), [README.md](config/README.md#L1-L2)

## 平衡采样质量与速度

有两个参数直接影响质量与运行时间的权衡：

- **`-t` / `--n_steps`**（默认 100）：控制逆向扩散步数。默认配置中的扩散调度器使用 1,000 个时间步训练（`num_train_timesteps: 1000`），因此 `--n_steps 1000` 可产生理论最优的采样质量，而 `--n_steps 50` 则提供快速的近似结果。实践中，50–200 之间的值对大多数肽能达到良好的平衡。
- **`-b` / `--batch_size`**（默认 250）：更大的批次能提高 GPU 利用率，但会消耗更多内存。如果在 GPU 上遇到内存不足错误，请将批次大小减半后重新运行。

此关系可总结为：

| 模式 | `--n_steps` | `--batch_size` | 适用场景 |
|--------|-------------|----------------|----------|
| 快速探索 | 50 | 250 | 快速原型设计、参数扫描 |
| 均衡（默认） | 100 | 250 | 标准生产运行 |
| 高保真 | 200–1000 | 128 | 最终发表质量的系综 |

来源：[generate_ensemble.py](scripts/generate_ensemble.py#L29-L34), [models.yaml](config/models.yaml#L62-L69)

## 故障排除

| 症状 | 原因 | 解决方案 |
|---------|-------|-----|
| `ValueError: The input sequence can contain only standard amino acid letters` | `-s` 中包含非标准或小写残基代码 | 仅使用 20 个大写单字母代码 (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)。 |
| `ImportError: The cg2all library is not installed` | 未安装 cg2all 却使用了 `-a` 标志 | 安装 cg2all：`pip install git+http://github.com/huhlim/cg2all`，或移除 `-a` 标志。 |
| `ValueError: --out_fmt can only be in ('dcd',) when using --all_atom` | `--out_fmt numpy` 与 `-a` 组合使用 | 使用全原子重建时切换为 `--out_fmt dcd`（默认）。 |
| `ValueError: cg2all batch size is not an exact divisor of --n_samples` | `--n_samples` 无法被批次大小整除 | 调整 `--n_samples` 或 `--cg2all_batch_size`，使后者能整除前者。 |
| `OSError: CUDA is not available for PyTorch` | 在没有支持 CUDA 的 GPU 时使用了 `-d cuda` | 使用 `-d cpu` 或安装 CUDA 工具包及匹配的 PyTorch。 |
| `subprocess.SubprocessError: Error when running cg2all` | 全原子转换期间 cg2all 失败 | 检查 cg2all 是否正确安装；若怀疑 CUDA 问题，请尝试 `--cg2all_device cpu`。 |

来源：[generate_ensemble.py](scripts/generate_ensemble.py#L56-L81)

## 后续去向

既然你已能从命令行生成系综，接下来可以探索支撑它的架构与 API：

- **[两阶段架构概览](4-two-stage-architecture-overview)** — 了解编码器、扩散模型和解码器如何组合成完整的生成流程。
- **[SAM 模型 API 参考](14-sam-model-api-reference)** — 在你自己的 Python 代码中直接使用 `SAM` 类，以获得比 CLI 更强的控制力。
- **[通过 cg2all 进行全原子重建](17-all-atom-reconstruction-via-cg2all)** — 深入了解 Cα 轨迹如何转换为完整的原子细节。
- **[配置参考](15-configuration-reference)** — `models.yaml` 及自定义配置的完整字段级文档。