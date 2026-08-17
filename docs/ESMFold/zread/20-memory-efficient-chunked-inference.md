---
slug:20-memory-efficient-chunked-inference
blog_type:normal
---



当 GPU 内存受限时，内存高效的分块推理是使用 ESM 模型处理大型蛋白质序列的关键优化技术。该方法通过将计算分解为可管理的块来处理超出可用内存的序列，同时保持模型准确性。

## 分块注意力机制

ESM 中的核心内存优化通过轴向注意力模块中的分块注意力计算实现。系统会自动检测序列何时超过内存阈值并切换到批处理。

### RowSelfAttention 实现

`RowSelfAttention` 类在 MSA transformers 中为行级注意力实现分块处理 [esm/axial_attention.py](esm/axial_attention.py#L11-L132)。关键优化在 `_batched_forward` 方法中实现：

```python
def _batched_forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
    num_rows, num_cols, batch_size, embed_dim = x.size()
    max_rows = max(1, self.max_tokens_per_msa // num_cols)
    # 分块处理注意力权重
    for start in range(0, num_rows, max_rows):
        attn_weights = self.compute_attention_weights(
            x[start : start + max_rows], scaling,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask[:, start : start + max_rows]
        )
```

当 `num_rows * num_cols > self.max_tokens_per_msa` 时会自动触发分块逻辑，并在推理期间禁用梯度 [esm/axial_attention.py](esm/axial_attention.py#L113-L132)。

### ColumnSelfAttention 实现

类似地，`ColumnSelfAttention` 为列级注意力提供分块处理 [esm/axial_attention.py](esm/axial_attention.py#L133-L240)。这种双轴分块能够在不将整个注意力矩阵加载到内存的情况下高效处理大型多序列比对。

## 内存阈值配置

分块行为由 `max_tokens_per_msa` 参数控制，默认值为 `2^16`（65,536 个 token）[esm/axial_attention.py](esm/axial_attention.py#L19)。该阈值决定了系统何时从标准注意力切换到分块处理：

- **低于阈值**：标准注意力计算以获得最佳速度
- **高于阈值**：自动分块处理以节省内存

## 与 FSDP CPU 卸载的集成

对于极端内存限制的情况，ESM 将分块推理与 FairScale 的完全分片数据并行（FSDP）CPU 卸载相结合 [examples/esm2_infer_fairscale_fsdp_cpu_offloading.py](examples/esm2_infer_fairscale_fsdp_cpu_offloading.py#L20-L33)：

```python
fsdp_params = dict(
    mixed_precision=True,
    flatten_parameters=True,
    state_dict_device=torch.device("cpu"),  # 减少 GPU 内存使用
    cpu_offload=True,  # 启用 CPU 卸载
)
```

这种组合允许在单个 GPU 上运行如 ESM-2 15B 参数模型等大型模型，通过：
1. 分块注意力计算以减少峰值内存
2. 将优化器状态和参数卸载到 CPU
3. 使用混合精度减少内存占用

## 性能特征

分块推理引入了计算开销，但能够处理任意大型序列：

| 序列大小 | 处理模式 | 内存使用 | 速度影响 |
|----------|----------|----------|----------|
| < 65K token | 标准 | O(n²) | 基准 |
| > 65K token | 分块 | O(chunk_size²) | 慢约 10-20% |
| + FSDP | 分块 + CPU 卸载 | 最小 GPU | 慢约 30-50% |

内存节省随序列长度呈二次方缩放，使得这种方法对于宏基因组学应用和大型蛋白质家族至关重要。

<CgxTip>
当序列超过内存阈值时，分块推理在推理（torch.no_grad()）期间自动激活。无需手动配置 - 系统透明地优化内存使用。
</CgxTip>

<CgxTip>
为获得最大内存效率，请将分块推理与 FSDP CPU 卸载和混合精度结合使用。这可以在仅 16GB VRAM 的 GPU 上运行 15B 参数模型。
</CgxTip>

## 使用模式

分块推理与现有 ESM API 无缝协作：

```python
# 标准用法 - 分块自动激活
with torch.no_grad():
    results = model(large_batch_tokens, repr_layers=[48], return_contacts=True)
```

系统会自动检测内存限制并切换到分块处理，无需更改代码。对于极端内存限制，请使用示例中显示的 FSDP 模式 [examples/esm2_infer_fairscale_fsdp_cpu_offloading.py](examples/esm2_infer_fairscale_fsdp_cpu_offloading.py)。

内存高效的分块推理是一项关键优化，使大规模蛋白质语言模型推理能够在标准硬件上实现，让研究人员无需企业级 GPU 资源即可处理海量蛋白质数据集。